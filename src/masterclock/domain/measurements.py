"""What each series measured at an epoch, as the two output files record it.

A pair measurement is one DAS reading decycled and referred to its epoch
start (:func:`measure_pair`): the whole periods put back and the motion and
steering since the epoch start taken off. It holds the plain values it was
made from, the measurement's MJD, its phase and its rms, not the DAS's line;
its measurement time and epoch start are worked out from the MJD. A triple
measurement is a double difference built from its pairs' accepted
measurements.

Both give the filter step their plain values (:meth:`PairMeasurement.measured`,
:meth:`TripleMeasurement.measured`).
"""

from datetime import datetime, timedelta
from fractions import Fraction
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from masterclock.app.timeutil import mjd_to_datetime
from masterclock.domain.double_difference import TripleValue
from masterclock.domain.filter import FilterInput
from masterclock.domain.phase import (
    EPOCH_SECONDS,
    PHASE_MAX,
    PHASE_PERIOD,
    decycle,
    seconds,
)
from masterclock.domain.series import State

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch."""


def epoch_start(moment: datetime) -> datetime:
    """Give the start of the epoch an instant falls in.

    Parameters
    ----------
    moment : datetime
        The instant, with its timezone.

    Returns
    -------
    datetime
        The latest epoch start at or before ``moment``: epochs start at
        midnight and every :data:`~masterclock.domain.phase.EPOCH_SECONDS`
        after.

    Examples
    --------
    >>> from datetime import UTC
    >>> epoch_start(datetime(2025, 9, 23, 6, 2, 17, 203200, tzinfo=UTC))
    datetime.datetime(2025, 9, 23, 6, 0, tzinfo=datetime.timezone.utc)
    """
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + (moment - midnight) // _EPOCH * _EPOCH


class PairMeasurement(BaseModel):
    """One pair's measurement at an epoch: a row of its measurement file.

    Parameters
    ----------
    measurement_mjd : float
        The measurement time as the DAS gave it, MJD.
    measured_phase : int
        The DAS reading, ps, from 0 to
        :data:`~masterclock.domain.phase.PHASE_MAX`.
    rms : int
        Its rms, ps, at least 0.
    cycle_count : int
        The whole periods added to the reading.
    z : int
        The decycled phase referred to the epoch start, z_E, ps.
    slip : bool, optional
        Whether the slip check corrected the cycle count; the row then
        carries S.

    Attributes
    ----------
    measurement_datetime : datetime
        The measurement time, from the MJD.
    interpolated_datetime : datetime
        The start of the epoch the measurement falls in.
    delta : Fraction
        The measurement time after the epoch start, s, exactly.

    Raises
    ------
    pydantic.ValidationError
        If a value is out of range, or a field is of the wrong kind or
        unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement_mjd: Annotated[float, Field(allow_inf_nan=False)]
    measured_phase: Annotated[int, Field(ge=0, le=PHASE_MAX)]
    rms: Annotated[int, Field(ge=0)]
    cycle_count: int
    z: int
    slip: bool = False

    @property
    def measurement_datetime(self) -> datetime:
        """The measurement time, from the MJD."""
        return mjd_to_datetime(self.measurement_mjd)

    @property
    def interpolated_datetime(self) -> datetime:
        """The start of the epoch the measurement falls in."""
        return epoch_start(self.measurement_datetime)

    @property
    def delta(self) -> Fraction:
        """The measurement time after the epoch start, s, exactly."""
        return seconds(self.measurement_datetime, self.interpolated_datetime)

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
        return self.model_copy(
            update={
                "cycle_count": self.cycle_count + cycles,
                "z": self.z + cycles * PHASE_PERIOD,
                "slip": True,
            }
        )

    def filter_input(self) -> FilterInput:
        """Give the filter step this measurement's plain values.

        Returns
        -------
        FilterInput
            z, the rms and the slip mark.
        """
        return FilterInput(z=self.z, rms=self.rms, slip=self.slip)


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
    pair_cold_started : bool
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
    pair_cold_started: bool

    @classmethod
    def from_triple_value(cls, triple_value: TripleValue) -> TripleMeasurement:
        """Build a triple's measurement from its double difference.

        Parameters
        ----------
        triple_value : TripleValue
            The double difference the domain gave.

        Returns
        -------
        TripleMeasurement
            The same values, under the file's names.
        """
        return cls(
            z=triple_value.z,
            double_difference_sigma=triple_value.sigma,
            components_used=triple_value.components_used,
            pair_cold_started=triple_value.pair_cold_started,
        )

    def filter_input(self) -> FilterInput:
        """Give the filter step this measurement's plain values.

        Returns
        -------
        FilterInput
            z, sigma_dd and the cold mark.
        """
        return FilterInput(
            z=self.z,
            sigma_dd=self.double_difference_sigma,
            pair_cold_started=self.pair_cold_started,
        )


def measure_pair(
    *,
    measurement_mjd: float,
    measured_phase: int,
    rms: int,
    prediction: State | None,
    w: Fraction,
    anchor: int | None,
) -> PairMeasurement:
    """Decycle a reading and refer it to its epoch start (design 7).

    Parameters
    ----------
    measurement_mjd : float
        The measurement time as the DAS gave it, MJD.
    measured_phase : int
        The DAS reading, ps.
    rms : int
        Its rms, ps.
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
    >>> prediction = State(x=1_234_567 + exact(0.0123) * 600, y=0.0123)
    >>> measure_pair(
    ...     measurement_mjd=60941.251588, measured_phase=34579, rms=3,
    ...     prediction=prediction, w=Fraction(0), anchor=None,
    ... ).z
    1234577
    """
    measured_instant = mjd_to_datetime(measurement_mjd)
    delta = seconds(measured_instant, epoch_start(measured_instant))
    decycled = decycle(measured_phase, delta, w, prediction, anchor)
    return PairMeasurement(
        measurement_mjd=measurement_mjd,
        measured_phase=measured_phase,
        rms=rms,
        cycle_count=decycled.cycle_count,
        z=decycled.z,
    )
