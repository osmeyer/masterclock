"""What each series measured at an epoch, as the two output files record it.

A pair measurement is one DAS measurement decycled and referred to its epoch
start (:func:`measure_pair`): the whole periods put back and the motion and
steering since the epoch start taken off. A triple measurement is a double
difference built from its pairs' accepted measurements.

Both give the filter step their plain values (:meth:`PairMeasurement.measured`,
:meth:`TripleMeasurement.measured`), since the domain takes no program types.
"""

from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain.double_difference import TripleValue
from masterclock.domain.filter import Measured
from masterclock.domain.phase import PHASE_PERIOD, decycle, seconds
from masterclock.domain.series import State


def _offset(measurement: DASMeasurement) -> Fraction:
    """Give a measurement's time after its epoch start, exactly.

    Parameters
    ----------
    measurement : DASMeasurement
        The raw measurement.

    Returns
    -------
    Fraction
        delta, s, from the datetimes as a whole number of microseconds; the
        float MJDs would not give it exactly.
    """
    return seconds(measurement.measurement_datetime, measurement.interpolated_datetime)


class PairMeasurement(BaseModel):
    """One pair's measurement at an epoch: a row of its measurement file.

    Parameters
    ----------
    measurement : DASMeasurement
        The raw measurement.
    cycle_count : int
        The whole periods added to the reading.
    z : int
        The decycled phase referred to the epoch start, z_E, ps.
    slip : bool, optional
        Whether the slip check corrected the cycle count; the row then
        carries S.

    Attributes
    ----------
    delta : Fraction
        The measurement time after the epoch start, s, exactly: worked out
        from the measurement's datetimes, never passed in.

    Raises
    ------
    pydantic.ValidationError
        If a field is of the wrong kind or unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement: DASMeasurement
    cycle_count: int
    z: int
    slip: bool = False

    @property
    def delta(self) -> Fraction:
        """The measurement time after the epoch start, s, exactly."""
        return _offset(self.measurement)

    def corrected(self, cycles: int) -> PairMeasurement:
        """Give the measurement with a slip check's correction made (design 11.3).

        Parameters
        ----------
        cycles : int
            The whole periods to add.

        Returns
        -------
        PairMeasurement
            The cycle count and z moved by ``cycles`` periods, marked as
            slip corrected.
        """
        return PairMeasurement(
            measurement=self.measurement,
            cycle_count=self.cycle_count + cycles,
            z=self.z + cycles * PHASE_PERIOD,
            slip=True,
        )

    def measured(self) -> Measured:
        """Give the filter step this measurement's plain values.

        Returns
        -------
        Measured
            z, the rms and the slip mark.
        """
        return Measured(z=self.z, rms=self.measurement.rms, slip=self.slip)


class TripleMeasurement(BaseModel):
    """One triple's measurement at an epoch: a row of its double-difference file.

    Parameters
    ----------
    z : int
        The double difference dd, ps.
    double_difference_sigma : float
        Its measurement sigma, ps, zero or more, as a pair's rms may be.
    components_used : {'111', '110', '101'}
        Which of (s, c), (r, s) and (s, r) gave it, in that order.
    cold : bool
        Whether one of its pairs cold-started at the epoch.

    Raises
    ------
    pydantic.ValidationError
        If a value is out of range, or a field is of the wrong kind or
        unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    z: int
    double_difference_sigma: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    components_used: Literal["111", "110", "101"]
    cold: bool

    @classmethod
    def from_value(cls, value: TripleValue) -> TripleMeasurement:
        """Build a triple's measurement from its double difference.

        Parameters
        ----------
        value : TripleValue
            The double difference the domain gave.

        Returns
        -------
        TripleMeasurement
            The same values, under the file's names.
        """
        return cls(
            z=value.z,
            double_difference_sigma=value.sigma,
            components_used=value.components_used,
            cold=value.cold,
        )

    def measured(self) -> Measured:
        """Give the filter step this measurement's plain values.

        Returns
        -------
        Measured
            z, sigma_dd and the cold mark.
        """
        return Measured(z=self.z, sigma_dd=self.double_difference_sigma, cold=self.cold)


def measure_pair(
    measurement: DASMeasurement,
    prediction: State | None,
    w: Fraction,
    anchor: int | None,
) -> PairMeasurement:
    """Decycle a DAS measurement and refer it to its epoch start (design 7).

    Parameters
    ----------
    measurement : DASMeasurement
        The raw measurement.
    prediction : State or None
        The pair's prediction at the epoch start, or ``None`` when it has
        none.
    w : Fraction
        The steering applied between the epoch start and the measurement,
        ps.
    anchor : int or None
        The last buffered measurement of a pair with no prediction, or
        ``None``.

    Returns
    -------
    PairMeasurement
        The measurement with its cycle count and z_E, from
        :func:`~masterclock.domain.phase.decycle` with the offset worked
        out from the datetimes.

    Raises
    ------
    PhaseError
        If the reading or its offset is out of range.

    Examples
    --------
    The worked epoch of the design:

    >>> from masterclock.domain.phase import exact
    >>> raw = DASMeasurement(
    ...     measurement_mjd=60941.251588, measured_phase=34579, rms=3,
    ...     switch="2B07", clock="ox23",
    ... )
    >>> prediction = State(x=1_234_567 + exact(0.0123) * 600, y=0.0123)
    >>> measure_pair(raw, prediction, Fraction(0), None).z
    1234577
    """
    delta = _offset(measurement)
    decycled = decycle(measurement.measured_phase, delta, w, prediction, anchor)
    return PairMeasurement(
        measurement=measurement, cycle_count=decycled.cycle_count, z=decycled.z
    )
