"""Running das_processor: building each epoch, processing it, the epoch loop.

Only the run functions take the configuration. :func:`build_epoch` resolves
everything one epoch needs into an :class:`Epoch` of plain values: its
references, every pair and triple, the steering of every reference that
steers a series, and each series' settings. Everything below it receives
that epoch or plain values.
"""

from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.epochs import floor_to_ten_minutes
from masterclock.das_processor.files import (
    DayBuffer,
    DdiffRecord,
    FileKind,
    MeasRecord,
    ensure_archives,
    good_through,
    read_last_row,
    roll_back,
    write_buffer,
)
from masterclock.das_processor.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.das_processor.read_cd5m5m import DASData, read_all_blocks
from masterclock.das_processor.read_steering import read_steering
from masterclock.das_processor.registry import (
    Existing,
    build_registry,
    existing_series,
    refs_of,
    series_file,
)
from masterclock.domain.double_difference import Component, double_difference
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

_PAIR: Final[int] = 2
"""How many names a pair key holds."""

_TRIPLE: Final[int] = 3
"""How many names a triple key holds."""


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


class TripleStep(BaseModel):
    """What the triples of an epoch gave (design 12).

    Parameters
    ----------
    results : dict of (str, str, str) to StepResult
        Each triple's row, and whether it cold-started.
    measurements : dict of (str, str, str) to TripleMeasurement
        Each triple's double difference, where it has one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    results: dict[TripleKey, StepResult]
    measurements: dict[TripleKey, TripleMeasurement]


def _component(pairs: PairStep, pair: PairKey) -> Component:
    """Give a pair's part in a triple: its accepted measurement, never its estimate.

    Parameters
    ----------
    pairs : PairStep
        What the epoch's pairs gave.
    pair : (str, str)
        The pair.

    Returns
    -------
    Component
        Whether the pair's row was accepted, its z and rms when it was, its
        prediction, and whether it cold-started.
    """
    result = pairs.results.get(pair)
    prediction = pairs.predictions.get(pair)
    predicted = None if prediction is None else prediction.x
    measurement = pairs.measurements.get(pair)
    if result is None or "A" not in result.row.flags or measurement is None:
        cold = result is not None and result.cold
        return Component(accepted=False, predicted=predicted, cold=cold)
    return Component(
        accepted=True,
        z=measurement.z,
        rms=measurement.measurement.rms,
        predicted=predicted,
        cold=result.cold,
    )


def process_triples(
    epoch: Epoch, last: Mapping[SeriesKey, Row], pairs: PairStep
) -> TripleStep:
    """Process an epoch's triples: double differences, then the filter (design 12).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last : Mapping of series key to Row
        Each series' last row; a series missing here is new.
    pairs : PairStep
        What the epoch's pairs gave.

    Returns
    -------
    TripleStep
        Every triple's row and double difference, in sorted key order. A
        local triple (r, r, c) is given its self pair for both links, so
        the check that it collapses to its pair runs every epoch.

    Raises
    ------
    PhaseError
        If a local triple does not collapse to its pair.
    FilterError
        If a row breaks a rule of a row.
    """
    mark = epoch.interpolated_datetime
    results: dict[TripleKey, StepResult] = {}
    measurements: dict[TripleKey, TripleMeasurement] = {}
    for triple in epoch.triples:
        r, s, c = triple
        value = double_difference(
            triple,
            _component(pairs, (s, c)),
            _component(pairs, (r, s)),
            _component(pairs, (s, r)),
        )
        measurement = None if value is None else TripleMeasurement.from_value(value)
        if measurement is not None:
            measurements[triple] = measurement
        prediction = predict(last.get(triple), steer_u(triple, mark, epoch.steering))
        results[triple] = filter_step(
            mark,
            epoch.params[triple],
            last.get(triple),
            prediction,
            None if measurement is None else measurement.measured(),
        )
    return TripleStep(results=results, measurements=measurements)


# --------------------------------------------------------------- the epoch loop


class EpochDone(BaseModel):
    """What processing an epoch gave, for the run to log (design 6.3).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    pairs : PairStep
        What its pairs gave.
    triples : TripleStep
        What its triples gave.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    epoch: Epoch
    pairs: PairStep
    triples: TripleStep


def data_series(config: AppConfig) -> list[tuple[Path, FileKind, SeriesKey]]:
    """Give every file of the channel, with its kind and series.

    Parameters
    ----------
    config : AppConfig
        The run's settings.

    Returns
    -------
    list of (Path, FileKind, series key)
        The measurement files, then the double-difference files, each by
        series.

    Raises
    ------
    DataFileError
        If an archive cannot be listed.
    """
    processed, channel = config.processed.processed_path, config.das.rf
    existing = existing_series(processed, channel)
    series: list[tuple[Path, FileKind, SeriesKey]] = [
        (series_file(processed, channel, pair), "meas", pair)
        for pair in sorted(existing.pairs)
    ]
    series += [
        (series_file(processed, channel, triple), "ddiff", triple)
        for triple in sorted(existing.triples)
    ]
    return series


def next_epoch(config: AppConfig) -> datetime:
    """Roll every file back to the epoch they all hold, and give the next (design 6.7).

    Parameters
    ----------
    config : AppConfig
        The run's settings.

    Returns
    -------
    datetime
        One epoch after L, the oldest epoch any file is good through; with
        no file holding a whole row, the epoch containing
        ``start_from_mjd``.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, or its first row is
        damaged, so its rows cannot be placed in time.
    """
    series = data_series(config)
    good = [good_through(path, kind, key) for path, kind, key in series]
    common = min((mark for mark in good if mark is not None), default=None)
    for path, kind, key in series:
        roll_back(path, kind, key, common)
    if common is None:
        return floor_to_ten_minutes(mjd_to_datetime(config.processed.start_from_mjd))
    return common + _EPOCH


def read_last_rows(config: AppConfig) -> dict[SeriesKey, Row]:
    """Read the last row of every series from its file (I4).

    Parameters
    ----------
    config : AppConfig
        The run's settings.

    Returns
    -------
    dict of series key to Row
        Each series' last row.

    Raises
    ------
    DataFileError
        If a file cannot be read or is not sound.
    """
    return {
        key: read_last_row(path, kind, key) for path, kind, key in data_series(config)
    }


def process_epoch(
    mark: datetime,
    block: DASData | None,
    buffer: DayBuffer,
    config: AppConfig,
    clock_config: ClockConfig,
) -> EpochDone:
    """Process one epoch and add its rows to the day buffer (design 6.3).

    Parameters
    ----------
    mark : datetime
        The epoch start E.
    block : DASData or None
        The epoch's DAS block, or ``None`` when the DAS measured nothing.
    buffer : DayBuffer
        The day buffer, whose newest rows are the series' last rows; when
        it holds none, they are read from the files.
    config : AppConfig
        The run's settings.
    clock_config : ClockConfig
        The clock configuration.

    Returns
    -------
    EpochDone
        The epoch and what its pairs and triples gave.

    Raises
    ------
    MasterClockError
        If anything about the epoch cannot be read, worked out or
        formatted; the buffer then holds none of the epoch's rows.
    """
    last = dict(buffer.last) if buffer.last else read_last_rows(config)
    existing = Existing(
        pairs=frozenset((key[0], key[1]) for key in last if len(key) == _PAIR),
        triples=frozenset(
            (key[0], key[1], key[-1]) for key in last if len(key) == _TRIPLE
        ),
    )
    epoch = build_epoch(mark, block, existing, config, clock_config)
    pairs = process_pairs(epoch, last)
    triples = process_triples(epoch, last, pairs)
    records: list[tuple[SeriesKey, MeasRecord | DdiffRecord]] = [
        (
            pair,
            MeasRecord(
                measurement=pairs.measurements.get(pair), row=pairs.results[pair].row
            ),
        )
        for pair in epoch.pairs
    ]
    records += [
        (
            triple,
            DdiffRecord(
                measurement=triples.measurements.get(triple),
                row=triples.results[triple].row,
            ),
        )
        for triple in epoch.triples
    ]
    staged = DayBuffer(buffer.channel)
    processed = config.processed.processed_path
    for key, record in records:
        staged.add(series_file(processed, config.das.rf, key), key, record)
    buffer.take(staged)
    return EpochDone(epoch=epoch, pairs=pairs, triples=triples)


def run(
    config: AppConfig,
    clock_config: ClockConfig,
    steps: int | None,
    shutdown: ShutdownHandler,
) -> None:
    """Process the channel's epochs in order, up to the end of the data (design 6.3).

    Parameters
    ----------
    config : AppConfig
        The run's settings.
    clock_config : ClockConfig
        The clock configuration.
    steps : int or None
        How many epochs to process at most; ``None`` for all the data has.
    shutdown : ShutdownHandler
        Asked between epochs whether to stop.

    Raises
    ------
    MasterClockError
        If an epoch cannot be processed or a write fails; the rows of the
        day so far are then lost, and the next run computes them again.

    Notes
    -----
    The run starts one epoch after the epoch every file holds (see
    :func:`next_epoch`). An epoch with no block before the data resume is
    processed with no measurements; the run stops when no block remains,
    after ``steps`` epochs, or on a shutdown request, always between
    epochs. Rows are written after each day's 23:50 UTC epoch and when the
    run stops.
    """
    ensure_archives(config.processed.processed_path)
    mark = next_epoch(config)
    blocks = read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(mark))
    pending = _next_block(blocks, mark)
    buffer = DayBuffer(config.das.rf)
    done = 0
    while (
        pending is not None
        and not shutdown.shutdown_requested
        and (steps is None or done < steps)
    ):
        block = None
        if pending.interpolated_datetime == mark:
            block = pending
            pending = _next_block(blocks, mark + _EPOCH)
        process_epoch(mark, block, buffer, config, clock_config)
        if (mark + _EPOCH).date() != mark.date():
            write_buffer(buffer)
        mark += _EPOCH
        done += 1
    write_buffer(buffer)


def _next_block(blocks: Iterator[DASData], mark: datetime) -> DASData | None:
    """Give the next block at or after a mark, passing over any before it.

    Parameters
    ----------
    blocks : Iterator of DASData
        The DAS blocks, in order.
    mark : datetime
        The epoch the run is at.

    Returns
    -------
    DASData or None
        The next block not earlier than ``mark``; ``None`` at the end of
        the data.
    """
    for block in blocks:
        if block.interpolated_datetime >= mark:
            return block
    return None
