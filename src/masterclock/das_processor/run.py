"""Running das_processor: building each epoch, processing it, the epoch loop.

Only the run functions take the configuration. :func:`build_epoch` resolves
everything one epoch needs into an :class:`Epoch` of plain values: its
references, every pair and triple, the steering of every reference that
steers a series, and each series' settings, kept from the last epoch while
they cannot have changed. Everything below it receives that epoch or plain
values.
"""

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Final

from masterclock.app.log import TRACE, MasterClockLogger, get_logger
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import JOURNAL_FILE_TEMPLATE, AppConfig
from masterclock.das_processor.epochs import floor_to_ten_minutes
from masterclock.das_processor.files import (
    Cut,
    DayBuffer,
    DdiffRecord,
    FileCheck,
    FileKind,
    MeasRecord,
    check_file,
    clear_journal,
    ensure_archives,
    read_journal,
    read_last_row,
    roll_back,
    write_buffer,
)
from masterclock.das_processor.read_cd5m5m import DASData, read_all_blocks
from masterclock.das_processor.read_steering import read_steering
from masterclock.das_processor.registry import (
    ExistingSeries,
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

_NO_STEERING_INPUT: Final[tuple[Fraction, float]] = (Fraction(0), 0.0)
"""The steering input of a series no event moves: u_x and u_y both zero."""

_NO_COMPONENT: Final[Component] = Component(accepted=False)
"""The part in a triple of a pair the epoch does not hold."""


@dataclass(frozen=True, slots=True)
class Epoch:
    """Everything one epoch needs, with the configuration resolved (design 4.3).

    Parameters
    ----------
    interpolated_datetime : datetime
        The epoch start E.
    das_block : DASData or None
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
    series_params : dict of series key to SeriesParams
        Every series' settings at E.

    Raises
    ------
    ValueError
        If the block is of another epoch, or the settings are not for
        exactly the epoch's series.
    """

    interpolated_datetime: datetime
    das_block: DASData | None
    refs: frozenset[str]
    steering: dict[str, tuple[SteerEvent, ...]]
    pairs: tuple[PairKey, ...]
    triples: tuple[TripleKey, ...]
    series_params: dict[SeriesKey, SeriesParams]

    def __post_init__(self) -> None:
        """Refuse a block of another epoch, or settings for other series.

        Raises
        ------
        ValueError
            If the block's mark is not the epoch's, or the settings' keys
            are not the pairs and triples.
        """
        if (
            self.das_block is not None
            and self.das_block.interpolated_datetime != self.interpolated_datetime
        ):
            message = (
                f"the block of {self.das_block.interpolated_datetime} is not of the"
                f" epoch of {self.interpolated_datetime}"
            )
            raise ValueError(message)
        if set(self.series_params) != {*self.pairs, *self.triples}:
            message = "an epoch holds settings for exactly its pairs and triples"
            raise ValueError(message)


def build_epoch(
    epoch_start: datetime,
    das_block: DASData | None,
    earlier_series: ExistingSeries,
    config: AppConfig,
    clock_config: ClockConfig,
    last_epoch: Epoch | None = None,
) -> Epoch:
    """Resolve everything an epoch needs (design 6.3).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    das_block : DASData or None
        The epoch's DAS block, or ``None`` when there is none.
    earlier_series : ExistingSeries
        The series that existed before the epoch.
    config : AppConfig
        The run's settings, for the steering directory.
    clock_config : ClockConfig
        The clock configuration.
    last_epoch : Epoch or None, optional
        The epoch processed before this one in the run, whose settings are
        kept when they still hold; ``None`` for the run's first.

    Returns
    -------
    Epoch
        The epoch: its references, every pair and triple (see
        :func:`~masterclock.das_processor.registry.build_registry`), the
        steering of every reference any series is steered by, read over
        (E - T, E + T] (I4), and each series' settings at E. The settings
        are a copy of ``last_epoch``'s when it is earlier, had the same
        pairs and triples, and no clock's settings change after it up to E
        (see :meth:`ClockConfig.changes_between`); otherwise they are worked
        out from the clock configuration.

    Raises
    ------
    ConfigError
        If a series' clock has no entry in the clock configuration.
    DataFileError
        If a steering file cannot be read.
    """
    refs = refs_of(das_block)
    pairs, triples = build_registry(das_block, refs, earlier_series)
    series_keys: list[SeriesKey] = [*pairs, *triples]
    steering_refs = sorted(
        {mc for series_key in series_keys for mc in signs(series_key)}
    )
    steering = {
        mc: read_steering(
            config.das.steering_path, mc, epoch_start - _EPOCH, epoch_start + _EPOCH
        )
        for mc in steering_refs
    }
    if (
        last_epoch is not None
        and last_epoch.interpolated_datetime < epoch_start
        and (last_epoch.pairs, last_epoch.triples) == (pairs, triples)
        and not clock_config.changes_between(
            last_epoch.interpolated_datetime, epoch_start
        )
    ):
        series_params = dict(last_epoch.series_params)
    else:
        series_params = clock_config.params_for_series(series_keys, epoch_start)
    return Epoch(
        interpolated_datetime=epoch_start,
        das_block=das_block,
        refs=refs,
        steering=steering,
        pairs=pairs,
        triples=triples,
        series_params=series_params,
    )


@dataclass(frozen=True, slots=True)
class PairStep:
    """What the pairs of an epoch gave (design 6.3).

    Parameters
    ----------
    step_results : dict of (str, str) to StepResult
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

    step_results: dict[PairKey, StepResult]
    measurements: dict[PairKey, PairMeasurement]
    predictions: dict[PairKey, State | None]
    screening: Screening
    slips: Slips


def _steered(epoch: Epoch) -> bool:
    """Tell whether any steering event falls in an epoch's steering window.

    Parameters
    ----------
    epoch : Epoch
        The epoch.

    Returns
    -------
    bool
        Whether any reference has an event in (E - T, E + T]. When none
        has, every series' steering input and steering inside the epoch is
        zero, and is not worked out series by series.
    """
    return any(epoch.steering.values())


def _steering_input(
    series_key: SeriesKey, epoch: Epoch, *, steered: bool
) -> tuple[Fraction, float]:
    """Give a series' steering input over the epoch before E.

    Parameters
    ----------
    series_key : series key
        The series.
    epoch : Epoch
        The epoch.
    steered : bool
        Whether any event falls in the epoch's steering window (see
        :func:`_steered`).

    Returns
    -------
    tuple of (Fraction, float)
        What :func:`~masterclock.domain.steering.steer_u` gives; zero, as it
        would give, when no event falls in the window.
    """
    if not steered:
        return _NO_STEERING_INPUT
    return steer_u(series_key, epoch.interpolated_datetime, epoch.steering)


def _measured_pairs(
    epoch: Epoch,
    last_rows: Mapping[SeriesKey, Row],
    predictions: Mapping[PairKey, State | None],
) -> dict[PairKey, PairMeasurement]:
    """Decycle every pair the epoch measured (design 7).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last_rows : Mapping of series key to Row
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
    if epoch.das_block is None:
        return {}
    epoch_start = epoch.interpolated_datetime
    steered = _steered(epoch)
    measurements = {}
    for das_measurement in epoch.das_block.measurements:
        pair = (das_measurement.reference, das_measurement.clock)
        w = (
            steer_w(
                pair, epoch_start, epoch.steering, das_measurement.measurement_datetime
            )
            if steered
            else _NO_STEERING_INPUT[0]
        )
        anchor = anchor_of(last_rows.get(pair))
        measurements[pair] = measure_pair(
            measurement_mjd=das_measurement.measurement_mjd,
            measured_phase=das_measurement.measured_phase,
            rms=das_measurement.rms,
            prediction=predictions[pair],
            w=w,
            anchor=anchor,
        )
    return measurements


def _innovations(
    measurements: Mapping[PairKey, PairMeasurement],
    predictions: Mapping[PairKey, State | None],
    last_rows: Mapping[SeriesKey, Row],
) -> tuple[dict[PairKey, Fraction], dict[PairKey, float]]:
    """Give the pairs' innovations and scales for screening and the slip check.

    Parameters
    ----------
    measurements : Mapping of (str, str) to PairMeasurement
        The epoch's pair measurements.
    predictions : Mapping of (str, str) to State or None
        Each pair's prediction at E.
    last_rows : Mapping of series key to Row
        Each series' last row.

    Returns
    -------
    tuple of (dict, dict)
        The innovation of every pair with a measurement and a prediction,
        exact, and the innovation scale of every pair with a prediction.
    """
    scales: dict[PairKey, float] = {}
    for pair, prediction in predictions.items():
        last_row = last_rows.get(pair)
        if (
            prediction is not None
            and last_row is not None
            and last_row.innovation_scale is not None
        ):
            scales[pair] = last_row.innovation_scale
    innovations: dict[PairKey, Fraction] = {}
    for pair, measurement in measurements.items():
        prediction = predictions[pair]
        if prediction is not None:
            innovations[pair] = measurement.z - prediction.x
    return innovations, scales


def process_pairs(epoch: Epoch, last_rows: Mapping[SeriesKey, Row]) -> PairStep:
    """Process an epoch's pairs: predict, decycle, screen, check slips, filter.

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last_rows : Mapping of series key to Row
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
    epoch_start = epoch.interpolated_datetime
    steered = _steered(epoch)
    predictions = {
        pair: predict(
            last_rows.get(pair), _steering_input(pair, epoch, steered=steered)
        )
        for pair in epoch.pairs
    }
    measurements = _measured_pairs(epoch, last_rows, predictions)
    innovations, scales = _innovations(measurements, predictions, last_rows)
    screening = screen_references(innovations, scales, epoch.refs)
    last_flags = {
        pair: last_rows[pair].flags for pair in epoch.pairs if pair in last_rows
    }
    slips = slip_check(innovations, scales, last_flags, epoch.refs, screening.excluded)
    for pair, cycles in slips.corrections.items():
        measurements[pair] = measurements[pair].corrected(cycles)
    excluded_pairs = screening.excluded | slips.excluded
    step_results = {}
    for pair in epoch.pairs:
        measurement = measurements.get(pair)
        step_results[pair] = filter_step(
            epoch_start,
            epoch.series_params[pair],
            last_rows.get(pair),
            predictions[pair],
            None if measurement is None else measurement.filter_input(),
            excluded=pair in excluded_pairs,
        )
    return PairStep(
        step_results=step_results,
        measurements=measurements,
        predictions=predictions,
        screening=screening,
        slips=slips,
    )


@dataclass(frozen=True, slots=True)
class TripleStep:
    """What the triples of an epoch gave (design 12).

    Parameters
    ----------
    step_results : dict of (str, str, str) to StepResult
        Each triple's row, and whether it cold-started.
    measurements : dict of (str, str, str) to TripleMeasurement
        Each triple's double difference, where it has one.
    predictions : dict of (str, str, str) to State or None
        Each triple's prediction at E.
    """

    step_results: dict[TripleKey, StepResult]
    measurements: dict[TripleKey, TripleMeasurement]
    predictions: dict[TripleKey, State | None]


def _component(pair_step: PairStep, pair: PairKey) -> Component:
    """Give a pair's part in a triple: its accepted measurement, never its estimate.

    Parameters
    ----------
    pair_step : PairStep
        What the epoch's pairs gave.
    pair : (str, str)
        The pair.

    Returns
    -------
    Component
        Whether the pair's row was accepted, its z and rms when it was, its
        prediction, and whether it cold-started.
    """
    step_result = pair_step.step_results.get(pair)
    prediction = pair_step.predictions.get(pair)
    predicted = None if prediction is None else prediction.x
    measurement = pair_step.measurements.get(pair)
    if step_result is None or "A" not in step_result.row.flags or measurement is None:
        cold_started = step_result is not None and step_result.cold_started
        return Component(
            accepted=False, predicted_phase=predicted, cold_started=cold_started
        )
    return Component(
        accepted=True,
        z=measurement.z,
        rms=measurement.rms,
        predicted_phase=predicted,
        cold_started=step_result.cold_started,
    )


def process_triples(
    epoch: Epoch, last_rows: Mapping[SeriesKey, Row], pair_step: PairStep
) -> TripleStep:
    """Process an epoch's triples: double differences, then the filter (design 12).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last_rows : Mapping of series key to Row
        Each series' last row; a series missing here is new.
    pair_step : PairStep
        What the epoch's pairs gave.

    Returns
    -------
    TripleStep
        Every triple's row and double difference, in sorted key order. A
        local triple (r, r, c) is given its self pair for both links, so
        the check that it collapses to its pair runs every epoch. Each
        pair's part in the triples is worked out once for the epoch; a pair
        the epoch does not hold takes part as one not accepted, with no
        prediction.

    Raises
    ------
    PhaseError
        If a local triple does not collapse to its pair.
    FilterError
        If a row breaks a rule of a row.
    """
    epoch_start = epoch.interpolated_datetime
    steered = _steered(epoch)
    components = {pair: _component(pair_step, pair) for pair in epoch.pairs}
    step_results: dict[TripleKey, StepResult] = {}
    measurements: dict[TripleKey, TripleMeasurement] = {}
    predictions: dict[TripleKey, State | None] = {}
    for triple in epoch.triples:
        r, s, c = triple
        triple_value = double_difference(
            triple,
            components.get((s, c), _NO_COMPONENT),
            components.get((r, s), _NO_COMPONENT),
            components.get((s, r), _NO_COMPONENT),
        )
        measurement = (
            None
            if triple_value is None
            else TripleMeasurement.from_triple_value(triple_value)
        )
        if measurement is not None:
            measurements[triple] = measurement
        prediction = predict(
            last_rows.get(triple), _steering_input(triple, epoch, steered=steered)
        )
        predictions[triple] = prediction
        step_results[triple] = filter_step(
            epoch_start,
            epoch.series_params[triple],
            last_rows.get(triple),
            prediction,
            None if measurement is None else measurement.filter_input(),
        )
    return TripleStep(
        step_results=step_results, measurements=measurements, predictions=predictions
    )


# --------------------------------------------------------------- the epoch loop


@dataclass(frozen=True, slots=True)
class EpochDone:
    """What processing an epoch gave, for the run to log (design 6.3).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    pair_step : PairStep
        What its pairs gave.
    triple_step : TripleStep
        What its triples gave.
    """

    epoch: Epoch
    pair_step: PairStep
    triple_step: TripleStep


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
    processed_path, channel = config.processed.processed_path, config.das.rf
    found_series = existing_series(processed_path, channel)
    series_files: list[tuple[Path, FileKind, SeriesKey]] = [
        (series_file(processed_path, channel, pair), "meas", pair)
        for pair in sorted(found_series.pairs)
    ]
    series_files += [
        (series_file(processed_path, channel, triple), "ddiff", triple)
        for triple in sorted(found_series.triples)
    ]
    return series_files


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
    it, to be made again. Each damaged file is logged once at ERROR by the
    file check, and a roll-back that changed anything, or followed a
    stopped write, is logged once at WARNING, with the epoch, why, and how
    many files it cut, deleted and left.
    """
    journal = config.processed.processed_path / JOURNAL_FILE_TEMPLATE.format(
        rf=config.das.rf
    )
    common_epoch = _roll_back_all(config, read_journal(journal))
    clear_journal(journal)
    if common_epoch is None:
        return floor_to_ten_minutes(mjd_to_datetime(config.processed.start_from_mjd))
    return common_epoch + _EPOCH


def _roll_back_all(
    config: AppConfig, stopped_write_epoch: datetime | None
) -> datetime | None:
    """Check every file, roll them all back to one epoch, and log it once.

    Parameters
    ----------
    config : AppConfig
        The run's settings.
    stopped_write_epoch : datetime or None
        The first epoch of a write that stopped part way, from its journal;
        ``None`` when there was none.

    Returns
    -------
    datetime or None
        L, the epoch every file now ends at; ``None`` when none holds a
        whole row.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, or, with no stopped
        write, its first row is damaged.
    """
    series_files = data_series(config)
    stopped_write = stopped_write_epoch is not None
    file_checks = [
        check_file(data_file, file_kind, stopped_write=stopped_write)
        for data_file, file_kind, _ in series_files
    ]
    good_epochs = [file_check.good_through for file_check in file_checks]
    if stopped_write_epoch is not None:
        good_epochs.append(stopped_write_epoch - _EPOCH)
    common_epoch = min(
        (good_epoch for good_epoch in good_epochs if good_epoch is not None),
        default=None,
    )
    cuts = [
        roll_back(data_file, file_kind, common_epoch)
        for data_file, file_kind, _ in series_files
    ]
    if stopped_write or any(cut != "kept" for cut in cuts):
        _log_roll_back(
            config.das.rf,
            common_epoch,
            _roll_back_reason(file_checks, stopped_write),
            cuts,
        )
    return common_epoch


def _roll_back_reason(file_checks: list[FileCheck], stopped_write: bool) -> str:
    """Say why a roll-back happened, for its log entry.

    Parameters
    ----------
    file_checks : list of FileCheck
        What the file check found in each file.
    stopped_write : bool
        Whether the write journal was there.

    Returns
    -------
    str
        A write that stopped part way, damaged files, or files that ended
        at different epochs, in that order of precedence.
    """
    if stopped_write:
        return "a write that stopped part way"
    if any(file_check.damaged for file_check in file_checks):
        return "damaged files, each logged at ERROR"
    return "files that ended at different epochs"


def _log_roll_back(
    channel: RfChannel, common_epoch: datetime | None, reason: str, cuts: list[Cut]
) -> None:
    """Log a roll-back once, at WARNING (design 6.7, 16.2).

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    common_epoch : datetime or None
        The epoch every file was rolled back to; ``None`` for no row.
    reason : str
        Why, in words.
    cuts : list of {'kept', 'cut', 'deleted'}
        What the roll-back did to each file.
    """
    _log.warning(
        "rolled back every file of channel %s to %s, after %s:"
        " %d files cut, %d deleted, %d already there",
        channel,
        "no row" if common_epoch is None else common_epoch,
        reason,
        cuts.count("cut"),
        cuts.count("deleted"),
        cuts.count("kept"),
    )


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
    return {
        series_key: read_last_row(data_file, file_kind)
        for data_file, file_kind, series_key in data_series(config)
    }


def process_epoch(
    epoch_start: datetime,
    das_block: DASData | None,
    day_buffer: DayBuffer,
    config: AppConfig,
    clock_config: ClockConfig,
    last_epoch: Epoch | None = None,
) -> EpochDone:
    """Process one epoch and add its rows to the day buffer (design 6.3).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    das_block : DASData or None
        The epoch's DAS block, or ``None`` when the DAS measured nothing.
    day_buffer : DayBuffer
        The day buffer, whose newest rows are the series' last rows; when
        it holds none, they are read from the files.
    config : AppConfig
        The run's settings.
    clock_config : ClockConfig
        The clock configuration.
    last_epoch : Epoch or None, optional
        The epoch processed before this one in the run, whose settings
        :func:`build_epoch` may keep; ``None`` for the run's first.

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
    last_rows = (
        dict(day_buffer.last_rows) if day_buffer.last_rows else read_last_state(config)
    )
    earlier_series = ExistingSeries(
        pairs=frozenset(
            (series_key[0], series_key[1])
            for series_key in last_rows
            if len(series_key) == _PAIR
        ),
        triples=frozenset(
            (series_key[0], series_key[1], series_key[-1])
            for series_key in last_rows
            if len(series_key) == _TRIPLE
        ),
    )
    epoch = build_epoch(
        epoch_start, das_block, earlier_series, config, clock_config, last_epoch
    )
    pair_step = process_pairs(epoch, last_rows)
    triple_step = process_triples(epoch, last_rows, pair_step)
    series_records: list[tuple[SeriesKey, MeasRecord | DdiffRecord]] = [
        (
            pair,
            MeasRecord(
                measurement=pair_step.measurements.get(pair),
                row=pair_step.step_results[pair].row,
            ),
        )
        for pair in epoch.pairs
    ]
    series_records += [
        (
            triple,
            DdiffRecord(
                measurement=triple_step.measurements.get(triple),
                row=triple_step.step_results[triple].row,
            ),
        )
        for triple in epoch.triples
    ]
    epoch_buffer = DayBuffer(day_buffer.channel)
    processed_path = config.processed.processed_path
    for series_key, file_record in series_records:
        epoch_buffer.add(
            series_file(processed_path, config.das.rf, series_key),
            series_key,
            file_record,
        )
    day_buffer.take(epoch_buffer)
    epoch_done = EpochDone(epoch=epoch, pair_step=pair_step, triple_step=triple_step)
    log_epoch(epoch_done, last_rows, config.das.rf)
    return epoch_done


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
    epoch_start = next_epoch(config)
    das_blocks = read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(epoch_start))
    next_das_block = _next_block(das_blocks, epoch_start)
    processed_path, channel = config.processed.processed_path, config.das.rf
    if (
        next_das_block is not None
        and not existing_series(processed_path, channel).pairs
    ):
        epoch_start = max(epoch_start, next_das_block.interpolated_datetime)
    journal = processed_path / JOURNAL_FILE_TEMPLATE.format(rf=channel)
    day_buffer = DayBuffer(channel, journal)
    last_epoch: Epoch | None = None
    epochs_done = 0
    while (
        next_das_block is not None
        and not shutdown.shutdown_requested
        and (steps is None or epochs_done < steps)
    ):
        das_block = None
        if next_das_block.interpolated_datetime == epoch_start:
            das_block = next_das_block
            next_das_block = _next_block(das_blocks, epoch_start + _EPOCH)
        last_epoch = process_epoch(
            epoch_start, das_block, day_buffer, config, clock_config, last_epoch
        ).epoch
        if (epoch_start + _EPOCH).date() != epoch_start.date():
            write_buffer(day_buffer)
        epoch_start += _EPOCH
        epochs_done += 1
    write_buffer(day_buffer)


def _next_block(das_blocks: Iterator[DASData], epoch_start: datetime) -> DASData | None:
    """Give the next block at or after a mark, passing over any before it.

    Parameters
    ----------
    das_blocks : Iterator of DASData
        The DAS blocks, in order.
    epoch_start : datetime
        The epoch the run is at.

    Returns
    -------
    DASData or None
        The next block not earlier than ``epoch_start``; ``None`` at the end of
        the data.
    """
    for das_block in das_blocks:
        if das_block.interpolated_datetime >= epoch_start:
            return das_block
    return None


# ---------------------------------------------------------------- log events


def series_name(channel: RfChannel, series_key: SeriesKey) -> str:
    """Name a series in the log as its file does, without the suffix.

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    series_key : (str, str) or (str, str, str)
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
    return f"das_{channel}." + ".".join(series_key)


def _pair_names(channel: RfChannel, pair_keys: Iterable[PairKey]) -> str:
    """Name several pairs in the log.

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    pair_keys : iterable of (str, str)
        The pairs.

    Returns
    -------
    str
        Their names, sorted, separated by commas.
    """
    return ", ".join(series_name(channel, pair) for pair in sorted(pair_keys))


def _log_screening(pair_step: PairStep, channel: RfChannel) -> None:
    """Log what screening and the slip check found (design 16.2).

    Parameters
    ----------
    pair_step : PairStep
        What the epoch's pairs gave.
    channel : {'a', 'b'}
        The RF channel.
    """
    for screening_event in pair_step.screening.events:
        ref_names = "-".join(screening_event.references)
        excluded_names = _pair_names(channel, screening_event.excluded)
        if screening_event.finding == "self_missing":
            _log.warning("self-measurement of %s missing", ref_names)
        elif screening_event.finding == "self_fail":
            _log.warning(
                "self-measurement of %s failed: excluded %s", ref_names, excluded_names
            )
        elif screening_event.finding == "reciprocity_fail":
            _log.warning(
                "reciprocity of %s failed: excluded %s", ref_names, excluded_names
            )
        else:
            _log.warning(
                "closure of link %s failed: excluded %s", ref_names, excluded_names
            )
    for slip_event in pair_step.slips.events:
        if slip_event.finding == "slip_corrected":
            pair_name = series_name(channel, slip_event.pairs[0])
            _log.info("%s slip corrected: %+d cycles", pair_name, slip_event.cycles)
        else:
            _log.warning(
                "slip of clock %s undecided: excluded %s",
                slip_event.clock,
                _pair_names(channel, slip_event.pairs),
            )


def _log_series(
    series_label: str, step_result: StepResult, last_row: Row | None
) -> None:
    """Log what happened to one series at an epoch (design 16.2).

    Parameters
    ----------
    series_label : str
        The series' name.
    step_result : StepResult
        Its row at the epoch.
    last_row : Row or None
        Its last row, or ``None`` for a new series.
    """
    row = step_result.row
    _log.debug("%s: %s", series_label, row.flags)
    if "R" in row.flags and "D" not in row.flags:
        _log.warning(
            "%s rejected: innovation %.1f ps, scale %.1f ps, %d consecutive",
            series_label,
            row.innovation,
            row.innovation_scale,
            row.consecutive_rejects,
        )
    if step_result.cold_started:
        _log.info("%s cold start: segment %d", series_label, row.segment)
    if last_row is not None and "D" not in last_row.flags:
        _log_changes(series_label, row, last_row)


def _log_changes(series_label: str, row: Row, last_row: Row) -> None:
    """Log how a tracked series changed at an epoch (design 16.2).

    Parameters
    ----------
    series_label : str
        The series' name.
    row : Row
        Its row at the epoch.
    last_row : Row
        Its last row, which held a state.
    """
    if "D" in row.flags:
        _log.info("%s dormant", series_label)
        return
    settings_changed = (last_row.time_constant, last_row.scale_time_constant) != (
        row.time_constant,
        row.scale_time_constant,
    )
    if settings_changed:
        _log.info(
            "%s configuration change: M %s to %s, M_sigma %s to %s",
            series_label,
            last_row.time_constant,
            row.time_constant,
            last_row.scale_time_constant,
            row.scale_time_constant,
        )
    if "A" in row.flags and row.segment - last_row.segment == 1 + int(settings_changed):
        _log.info("%s frequency step: segment %d", series_label, row.segment)
    elif "A" in row.flags and row.step_offset != last_row.step_offset:
        _log.info(
            "%s phase step of %d ps; step offset %d ps",
            series_label,
            row.step_offset - last_row.step_offset,
            row.step_offset,
        )


def _log_trace(series_label: str, row: Row, prediction: State | None) -> None:
    """Log a series' prediction and update at TRACE (design 16.2).

    Parameters
    ----------
    series_label : str
        The series' name.
    row : Row
        Its row at the epoch.
    prediction : State or None
        Its prediction at the epoch.
    """
    _log.trace(
        "%s: prediction %s, innovation %s, x %s, y %s, d %s",
        series_label,
        None if prediction is None else float(prediction.x),
        row.innovation,
        None if row.x_fs is None else f"{row.x_fs / FS_PER_PS:.3f}",
        row.y,
        row.d,
    )


def log_epoch(
    epoch_done: EpochDone,
    last_rows: Mapping[SeriesKey, Row],
    channel: RfChannel,
) -> None:
    """Log everything an epoch did, at the design's levels (design 16.2).

    Parameters
    ----------
    epoch_done : EpochDone
        The epoch and what its pairs and triples gave.
    last_rows : Mapping of series key to Row
        Each series' last row.
    channel : {'a', 'b'}
        The RF channel.

    Notes
    -----
    Screening, slip and reject events go at WARNING except
    corrected slips; steps, cold starts, dormancy, configuration changes
    and the epoch's counts at INFO; each series' outcome at DEBUG and its
    prediction and update at TRACE. Series are logged in key order, pairs
    first. Nothing is worked out for a level the log leaves out: when
    WARNING is not logged the epoch is not looked at, and a TRACE line is
    made only when TRACE is logged.
    """
    if not _log.isEnabledFor(logging.WARNING):
        return
    trace_logged = _log.isEnabledFor(TRACE)
    pair_step, triple_step = epoch_done.pair_step, epoch_done.triple_step
    _log_screening(pair_step, channel)
    series_results: list[tuple[SeriesKey, StepResult, State | None]] = [
        (pair, step_result, pair_step.predictions[pair])
        for pair, step_result in pair_step.step_results.items()
    ]
    series_results += [
        (triple, step_result, triple_step.predictions[triple])
        for triple, step_result in triple_step.step_results.items()
    ]
    for series_key, step_result, prediction in series_results:
        series_label = series_name(channel, series_key)
        _log_series(series_label, step_result, last_rows.get(series_key))
        if trace_logged:
            _log_trace(series_label, step_result.row, prediction)
    accepted_count = sum(
        "A" in step_result.row.flags for _, step_result, _ in series_results
    )
    _log.info(
        "epoch %s: %d pairs, %d triples, %d accepted, %d held",
        epoch_done.epoch.interpolated_datetime,
        len(pair_step.step_results),
        len(triple_step.step_results),
        accepted_count,
        len(series_results) - accepted_count,
    )
