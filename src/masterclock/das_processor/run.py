"""Running das_processor: building each epoch, processing it, the epoch loop.

Only the run functions take the configuration. :func:`build_epoch` resolves
everything one epoch needs into an :class:`Epoch` of plain values: its
references, every pair and triple, the steering of every reference that
steers a series, and each series' settings. Everything below it receives
that epoch or plain values.
"""

from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import JOURNAL_FILE_TEMPLATE, AppConfig
from masterclock.das_processor.epochs import floor_to_ten_minutes
from masterclock.das_processor.files import (
    DayBuffer,
    DdiffRecord,
    FileKind,
    MeasRecord,
    clear_journal,
    ensure_archives,
    good_through,
    read_journal,
    read_last_row,
    roll_back,
    write_buffer,
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
from masterclock.domain.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.domain.phase import EPOCH_SECONDS, FS_PER_PS
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

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


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
        measurements[pair] = measure_pair(
            measurement_mjd=raw.measurement_mjd,
            measured_phase=raw.measured_phase,
            rms=raw.rms,
            prediction=predictions[pair],
            w=w,
            anchor=anchor,
        )
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
    predictions : dict of (str, str, str) to State or None
        Each triple's prediction at E.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    results: dict[TripleKey, StepResult]
    measurements: dict[TripleKey, TripleMeasurement]
    predictions: dict[TripleKey, State | None]


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
        rms=measurement.rms,
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
    predictions: dict[TripleKey, State | None] = {}
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
        predictions[triple] = prediction
        results[triple] = filter_step(
            mark,
            epoch.params[triple],
            last.get(triple),
            prediction,
            None if measurement is None else measurement.measured(),
        )
    return TripleStep(
        results=results, measurements=measurements, predictions=predictions
    )


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
        One epoch after L, the oldest epoch any file is good through, or
        the epoch before a write that stopped part way, when its journal is
        there; with no file holding a whole row, the epoch containing
        ``start_from_mjd``.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, or, with no write
        stopped part way, its first row is damaged, so its rows cannot be
        placed in time.

    Notes
    -----
    The journal is read before any file is checked. When it is there, a
    file whose first row is damaged is one the stopped write was creating,
    its length on the device but not its rows, and the roll-back deletes
    it, to be made again.
    """
    journal = config.processed.processed_path / JOURNAL_FILE_TEMPLATE.format(
        rf=config.das.rf
    )
    series = data_series(config)
    begun = read_journal(journal)
    stopped = begun is not None
    good = [good_through(path, kind, stopped_write=stopped) for path, kind, _ in series]
    if begun is not None:
        good.append(begun - _EPOCH)
    common = min((mark for mark in good if mark is not None), default=None)
    for path, kind, _ in series:
        roll_back(path, kind, common)
    clear_journal(journal)
    if common is None:
        return floor_to_ten_minutes(mjd_to_datetime(config.processed.start_from_mjd))
    return common + _EPOCH


def read_last_state(config: AppConfig) -> dict[SeriesKey, Row]:
    """Read every series' last row from its file (I4).

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
    return {key: read_last_row(path, kind) for path, kind, key in data_series(config)}


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
    last = dict(buffer.last) if buffer.last else read_last_state(config)
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
    done = EpochDone(epoch=epoch, pairs=pairs, triples=triples)
    log_epoch(done, last, config.das.rf)
    return done


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
    :func:`next_epoch`); while no series exists yet, at the first block,
    since an epoch before it holds no series and writes nothing, and a run
    of one epoch would otherwise never get past it. An epoch with no block
    before the data resume is processed with no measurements. The run stops
    when no block remains, after ``steps`` epochs, or on a shutdown
    request, always between epochs. Rows are written after each day's
    23:50 UTC epoch and when the run stops.
    """
    ensure_archives(config.processed.processed_path)
    mark = next_epoch(config)
    blocks = read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(mark))
    pending = _next_block(blocks, mark)
    processed, channel = config.processed.processed_path, config.das.rf
    if pending is not None and not existing_series(processed, channel).pairs:
        mark = max(mark, pending.interpolated_datetime)
    journal = processed / JOURNAL_FILE_TEMPLATE.format(rf=channel)
    buffer = DayBuffer(channel, journal)
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


# ---------------------------------------------------------------- log events


def series_name(channel: RfChannel, key: SeriesKey) -> str:
    """Name a series in the log as its file does, without the suffix.

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    key : (str, str) or (str, str, str)
        The series.

    Returns
    -------
    str
        ``das_<rf>.<names>``.

    Examples
    --------
    >>> series_name("a", ("mc2", "ox23"))
    'das_a.mc2.ox23'
    """
    return f"das_{channel}." + ".".join(key)


def _names(channel: RfChannel, keys: Iterable[PairKey]) -> str:
    """Name several pairs in the log.

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    keys : iterable of (str, str)
        The pairs.

    Returns
    -------
    str
        Their names, sorted, separated by commas.
    """
    return ", ".join(series_name(channel, key) for key in sorted(keys))


def _log_screening(pairs: PairStep, channel: RfChannel) -> None:
    """Log what screening and the slip check found (design 16.2).

    Parameters
    ----------
    pairs : PairStep
        What the epoch's pairs gave.
    channel : {'a', 'b'}
        The RF channel.
    """
    for event in pairs.screening.events:
        refs = "-".join(event.references)
        excluded = _names(channel, event.excluded)
        if event.kind == "self_missing":
            _log.warning("self-measurement of %s missing", refs)
        elif event.kind == "self_fail":
            _log.warning("self-measurement of %s failed: excluded %s", refs, excluded)
        elif event.kind == "reciprocity_fail":
            _log.warning("reciprocity of %s failed: excluded %s", refs, excluded)
        else:
            _log.warning("closure of link %s failed: excluded %s", refs, excluded)
    for slip in pairs.slips.events:
        if slip.kind == "slip_corrected":
            name = series_name(channel, slip.pairs[0])
            _log.info("%s slip corrected: %+d cycles", name, slip.cycles)
        else:
            _log.warning(
                "slip of clock %s undecided: excluded %s",
                slip.clock,
                _names(channel, slip.pairs),
            )


def _log_series(name: str, result: StepResult, last: Row | None) -> None:
    """Log what happened to one series at an epoch (design 16.2).

    Parameters
    ----------
    name : str
        The series' name.
    result : StepResult
        Its row at the epoch.
    last : Row or None
        Its last row, or ``None`` for a new series.
    """
    row = result.row
    _log.debug("%s: %s", name, row.flags)
    if "R" in row.flags and "D" not in row.flags:
        _log.warning(
            "%s rejected: innovation %.1f ps, scale %.1f ps, %d consecutive",
            name,
            row.innovation,
            row.innovation_scale,
            row.consecutive_rejects,
        )
    if result.cold:
        _log.info("%s cold start: segment %d", name, row.segment)
    if last is not None and "D" not in last.flags:
        _log_changes(name, row, last)


def _log_changes(name: str, row: Row, last: Row) -> None:
    """Log how a tracked series changed at an epoch (design 16.2).

    Parameters
    ----------
    name : str
        The series' name.
    row : Row
        Its row at the epoch.
    last : Row
        Its last row, which held a state.
    """
    if "D" in row.flags:
        _log.info("%s dormant", name)
        return
    changed = (last.time_constant, last.scale_time_constant) != (
        row.time_constant,
        row.scale_time_constant,
    )
    if changed:
        _log.info(
            "%s configuration change: M %s to %s, M_sigma %s to %s",
            name,
            last.time_constant,
            row.time_constant,
            last.scale_time_constant,
            row.scale_time_constant,
        )
    if "A" in row.flags and row.segment - last.segment == 1 + int(changed):
        _log.info("%s frequency step: segment %d", name, row.segment)
    elif "A" in row.flags and row.step_offset != last.step_offset:
        _log.info(
            "%s phase step of %d ps; step offset %d ps",
            name,
            row.step_offset - last.step_offset,
            row.step_offset,
        )


def _log_trace(name: str, row: Row, prediction: State | None) -> None:
    """Log a series' prediction and update at TRACE (design 16.2).

    Parameters
    ----------
    name : str
        The series' name.
    row : Row
        Its row at the epoch.
    prediction : State or None
        Its prediction at the epoch.
    """
    _log.trace(
        "%s: prediction %s, innovation %s, x %s, y %s, d %s",
        name,
        None if prediction is None else float(prediction.x),
        row.innovation,
        None if row.x_fs is None else f"{row.x_fs / FS_PER_PS:.3f}",
        row.y,
        row.d,
    )


def log_epoch(
    done: EpochDone,
    last: Mapping[SeriesKey, Row],
    channel: RfChannel,
) -> None:
    """Log everything an epoch did, at the design's levels (design 16.2).

    Parameters
    ----------
    done : EpochDone
        The epoch and what its pairs and triples gave.
    last : Mapping of series key to Row
        Each series' last row.
    channel : {'a', 'b'}
        The RF channel.

    Notes
    -----
    Screening, slip and reject events go at WARNING except
    corrected slips; steps, cold starts, dormancy, configuration changes
    and the epoch's counts at INFO; each series' outcome at DEBUG and its
    prediction and update at TRACE. Series are logged in key order, pairs
    first.
    """
    pairs, triples = done.pairs, done.triples
    _log_screening(pairs, channel)
    results: list[tuple[SeriesKey, StepResult, State | None]] = [
        (pair, result, pairs.predictions[pair])
        for pair, result in pairs.results.items()
    ]
    results += [
        (triple, result, triples.predictions[triple])
        for triple, result in triples.results.items()
    ]
    for key, result, prediction in results:
        name = series_name(channel, key)
        _log_series(name, result, last.get(key))
        _log_trace(name, result.row, prediction)
    accepted = sum("A" in result.row.flags for _, result, _ in results)
    _log.info(
        "epoch %s: %d pairs, %d triples, %d accepted, %d held",
        done.epoch.interpolated_datetime,
        len(pairs.results),
        len(triples.results),
        accepted,
        len(results) - accepted,
    )
