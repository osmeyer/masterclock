"""Running das_processor: building each epoch, processing it, the epoch loop.

Only the run functions take the configuration. :func:`build_epoch` resolves
everything one epoch needs into an :class:`Epoch` of plain values: its
references, every pair and triple, the steering of every reference that
steers a series, and each series' settings. Everything below it receives
that epoch or plain values.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.measurements import PairMeasurement, measure_pair
from masterclock.das_processor.read_cd5m5m import DASData
from masterclock.das_processor.read_steering import read_steering
from masterclock.das_processor.registry import Existing, build_registry, refs_of
from masterclock.domain.filter import StepResult, anchor_of, filter_step, predict
from masterclock.domain.phase import EPOCH_SECONDS
from masterclock.domain.screening import Screening, screen_references
from masterclock.domain.series import (
    PairKey,
    Row,
    SeriesKey,
    SeriesParams,
    State,
    TripleKey,
)
from masterclock.domain.slips import Slips, slip_check
from masterclock.domain.steering import SteerEvent, signs, steer_u, steer_w

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch, T."""


class Epoch(BaseModel):
    """Everything one epoch needs, with the configuration resolved (design 4.3).

    Parameters
    ----------
    interpolated_datetime : AwareDatetime
        The epoch start E.
    block : DASData or None
        The epoch's DAS block; ``None`` when the DAS measured nothing.
    refs : frozenset of str
        The epoch's references, REFS(e).
    steering : dict of str to tuple of SteerEvent
        The events of every reference that steers a series, in
        (E - T, E + T], in time order.
    pairs : tuple of (str, str)
        Every pair, sorted.
    triples : tuple of (str, str, str)
        Every triple, sorted.
    params : dict of series key to SeriesParams
        Every series' settings at E.

    Raises
    ------
    pydantic.ValidationError
        If the block is of another epoch, or the settings are not for
        exactly the epoch's series.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    interpolated_datetime: AwareDatetime
    block: DASData | None
    refs: frozenset[str]
    steering: dict[str, tuple[SteerEvent, ...]]
    pairs: tuple[PairKey, ...]
    triples: tuple[TripleKey, ...]
    params: dict[SeriesKey, SeriesParams]

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a block of another epoch, or settings for other series.

        Returns
        -------
        Self
            The epoch, unchanged.

        Raises
        ------
        ValueError
            If the block's mark is not the epoch's, or the settings' keys
            are not the pairs and triples.
        """
        if (
            self.block is not None
            and self.block.interpolated_datetime != self.interpolated_datetime
        ):
            message = (
                f"the block of {self.block.interpolated_datetime} is not of the epoch"
                f" of {self.interpolated_datetime}"
            )
            raise ValueError(message)
        if set(self.params) != {*self.pairs, *self.triples}:
            message = "an epoch holds settings for exactly its pairs and triples"
            raise ValueError(message)
        return self


def build_epoch(
    mark: datetime,
    block: DASData | None,
    existing: Existing,
    config: AppConfig,
    clock_config: ClockConfig,
) -> Epoch:
    """Resolve everything an epoch needs (design 6.3).

    Parameters
    ----------
    mark : datetime
        The epoch start E.
    block : DASData or None
        The epoch's DAS block, or ``None`` when there is none.
    existing : Existing
        The series that existed before the epoch.
    config : AppConfig
        The run's settings, for the steering directory.
    clock_config : ClockConfig
        The clock configuration.

    Returns
    -------
    Epoch
        The epoch: its references, every pair and triple (see
        :func:`~masterclock.das_processor.registry.build_registry`), the
        steering of every reference any series is steered by, read over
        (E - T, E + T] (I4), and each series' settings at E.

    Raises
    ------
    ConfigError
        If a series' clock has no entry in the clock configuration.
    DataFileError
        If a steering file cannot be read.
    """
    refs = refs_of(block)
    pairs, triples = build_registry(block, refs, existing)
    keys: list[SeriesKey] = [*pairs, *triples]
    steered = sorted({mc for key in keys for mc in signs(key)})
    steering = {
        mc: read_steering(config.das.steering_path, mc, mark - _EPOCH, mark + _EPOCH)
        for mc in steered
    }
    params = {key: clock_config.params_for(key, mark) for key in keys}
    return Epoch(
        interpolated_datetime=mark,
        block=block,
        refs=refs,
        steering=steering,
        pairs=pairs,
        triples=triples,
        params=params,
    )


class PairStep(BaseModel):
    """What the pairs of an epoch gave (design 6.3).

    Parameters
    ----------
    results : dict of (str, str) to StepResult
        Each pair's row, and whether it cold-started.
    measurements : dict of (str, str) to PairMeasurement
        Each measured pair's measurement, its slip corrected.
    predictions : dict of (str, str) to State or None
        Each pair's prediction at E.
    screening : Screening
        What reference screening decided.
    slips : Slips
        What the slip check decided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    results: dict[PairKey, StepResult]
    measurements: dict[PairKey, PairMeasurement]
    predictions: dict[PairKey, State | None]
    screening: Screening
    slips: Slips


def _measured_pairs(
    epoch: Epoch,
    last: Mapping[SeriesKey, Row],
    predictions: Mapping[PairKey, State | None],
) -> dict[PairKey, PairMeasurement]:
    """Decycle every pair the epoch measured (design 7).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last : Mapping of series key to Row
        Each series' last row.
    predictions : Mapping of (str, str) to State or None
        Each pair's prediction at E.

    Returns
    -------
    dict of (str, str) to PairMeasurement
        Each measured pair's measurement, decycled against its prediction,
        or against its anchor when it has none, with the steering inside
        the epoch taken off.
    """
    if epoch.block is None:
        return {}
    mark = epoch.interpolated_datetime
    measurements = {}
    for raw in epoch.block.measurements:
        pair = (raw.reference, raw.clock)
        w = steer_w(pair, mark, epoch.steering, raw.measurement_datetime)
        anchor = anchor_of(last.get(pair))
        measurements[pair] = measure_pair(raw, predictions[pair], w, anchor)
    return measurements


def _innovations(
    measurements: Mapping[PairKey, PairMeasurement],
    predictions: Mapping[PairKey, State | None],
    last: Mapping[SeriesKey, Row],
) -> tuple[dict[PairKey, Fraction], dict[PairKey, float]]:
    """Give the pairs' innovations and scales for screening and the slip check.

    Parameters
    ----------
    measurements : Mapping of (str, str) to PairMeasurement
        The epoch's pair measurements.
    predictions : Mapping of (str, str) to State or None
        Each pair's prediction at E.
    last : Mapping of series key to Row
        Each series' last row.

    Returns
    -------
    tuple of (dict, dict)
        The innovation of every pair with a measurement and a prediction,
        exact, and the innovation scale of every pair with a prediction.
    """
    scales: dict[PairKey, float] = {}
    for pair, prediction in predictions.items():
        row = last.get(pair)
        if (
            prediction is not None
            and row is not None
            and row.innovation_scale is not None
        ):
            scales[pair] = row.innovation_scale
    innovations: dict[PairKey, Fraction] = {}
    for pair, measurement in measurements.items():
        prediction = predictions[pair]
        if prediction is not None:
            innovations[pair] = measurement.z - prediction.x
    return innovations, scales


def process_pairs(epoch: Epoch, last: Mapping[SeriesKey, Row]) -> PairStep:
    """Process an epoch's pairs: predict, decycle, screen, check slips, filter.

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last : Mapping of series key to Row
        Each series' last row; a series missing here is new.

    Returns
    -------
    PairStep
        Every pair's row and what led to it, in sorted key order.

    Raises
    ------
    FilterError
        If a row breaks a rule of a row.
    PhaseError
        If a reading or its offset is out of range.
    """
    mark = epoch.interpolated_datetime
    predictions = {
        pair: predict(last.get(pair), steer_u(pair, mark, epoch.steering))
        for pair in epoch.pairs
    }
    measurements = _measured_pairs(epoch, last, predictions)
    innovations, scales = _innovations(measurements, predictions, last)
    screening = screen_references(innovations, scales, epoch.refs)
    flags = {pair: last[pair].flags for pair in epoch.pairs if pair in last}
    slips = slip_check(innovations, scales, flags, epoch.refs, screening.excluded)
    for pair, cycles in slips.corrections.items():
        measurements[pair] = measurements[pair].corrected(cycles)
    excluded = screening.excluded | slips.excluded
    results = {}
    for pair in epoch.pairs:
        measurement = measurements.get(pair)
        results[pair] = filter_step(
            mark,
            epoch.params[pair],
            last.get(pair),
            predictions[pair],
            None if measurement is None else measurement.measured(),
            excluded=pair in excluded,
        )
    return PairStep(
        results=results,
        measurements=measurements,
        predictions=predictions,
        screening=screening,
        slips=slips,
    )
