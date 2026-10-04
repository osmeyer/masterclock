"""Running das_processor: the start of a run, each epoch, the loop, the log.

A run first cuts the files back to what it can keep, then builds and
processes each epoch in turn, and logs what each epoch did.

Only the functions that read files take the configuration.
:func:`build_epoch` resolves everything one epoch needs into an
:class:`Epoch` of plain values: its references, every pair and triple, the
steering of every reference that steers a series, and each series'
settings, kept from the last epoch while they cannot have changed.
Everything below it receives that epoch or plain values.
"""

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, Protocol

from gmpy2 import mpq

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
    write_final,
)
from masterclock.das_processor.read_cd5m5m import DASData, read_all_blocks
from masterclock.das_processor.read_steering import SteeringFiles
from masterclock.das_processor.registry import (
    ExistingSeries,
    build_registry,
    existing_series,
    refs_of,
    series_file,
)
from masterclock.domain.double_difference import Component, double_difference
from masterclock.domain.filter import (
    StepResult,
    anchor_of,
    filter_step,
    predict,
    writes_row,
)
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

_NO_STEERING_INPUT: Final[tuple[mpq, float]] = (mpq(0), 0.0)
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
    locations : dict of str to int or None
        Every clock's location at E, as the clock configuration gives it.
    clocks_without_entry : frozenset of str, optional
        The clocks measured at E, or with a series before it, that the clock
        configuration has no entry for and does not ignore; their
        measurements and series are left out of the epoch, as an ignored
        clock's are.

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
    locations: dict[str, int | None]
    clocks_without_entry: frozenset[str] = frozenset()

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
    steering_files: SteeringFiles | None = None,
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
    steering_files : SteeringFiles or None, optional
        The run's steering files, read once and then as lines are added;
        ``None`` to read them afresh.

    Returns
    -------
    Epoch
        The epoch, every measurement and series of a clock the clock
        configuration has no entry for left out (see
        :func:`_configured_only`): its references, every pair and triple (see
        :func:`~masterclock.das_processor.registry.build_registry`), the
        steering of every reference any series is steered by, read over
        (E - T, E + T] (I4), and each series' settings at E. The settings
        are a copy of ``last_epoch``'s when it is earlier, had the same
        pairs and triples, and no clock's settings change after it up to E
        (see :meth:`ClockConfig.changes_between`); otherwise they are worked
        out from the clock configuration.

    Raises
    ------
    DataFileError
        If a steering file cannot be read.

    Notes
    -----
    A clock with no entry that the configuration does not ignore is logged
    at WARNING when it is first found (see :func:`_configured_only`).
    """
    das_block, earlier_series, clocks_without_entry = _configured_only(
        das_block, earlier_series, clock_config, last_epoch
    )
    refs = refs_of(das_block)
    kept_epoch = _configuration_kept(last_epoch, epoch_start, clock_config)
    locations = (
        clock_config.locations_at(epoch_start)
        if kept_epoch is None
        else dict(kept_epoch.locations)
    )
    pairs, triples = build_registry(das_block, refs, earlier_series, locations)
    series_keys: list[SeriesKey] = [*pairs, *triples]
    steering_refs = sorted(
        {mc for series_key in series_keys for mc in signs(series_key)}
    )
    if steering_files is None:
        steering_files = SteeringFiles(config.das.steering_path)
    steering = {
        mc: steering_files.events(mc, epoch_start - _EPOCH, epoch_start + _EPOCH)
        for mc in steering_refs
    }
    if kept_epoch is not None and (kept_epoch.pairs, kept_epoch.triples) == (
        pairs,
        triples,
    ):
        series_params = dict(kept_epoch.series_params)
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
        locations=locations,
        clocks_without_entry=clocks_without_entry,
    )


def _configured_only(
    das_block: DASData | None,
    earlier_series: ExistingSeries,
    clock_config: ClockConfig,
    last_epoch: Epoch | None,
) -> tuple[DASData | None, ExistingSeries, frozenset[str]]:
    """Leave out every measurement and series of a clock with no entry.

    A series takes the settings of its clock side, its last name, so a
    measurement or series whose clock the clock configuration lacks cannot
    be worked out. It is left out, and the clock is logged once at WARNING
    when it is first found so, in the run or after an epoch without it,
    unless the configuration ignores it: then nothing is logged.

    Parameters
    ----------
    das_block : DASData or None
        The epoch's DAS block, or ``None``.
    earlier_series : ExistingSeries
        The series that existed before the epoch.
    clock_config : ClockConfig
        The clock configuration.
    last_epoch : Epoch or None
        The epoch processed before this one in the run, whose clocks
        without an entry were logged already; ``None`` for the run's first.

    Returns
    -------
    tuple of (DASData or None, ExistingSeries, frozenset of str)
        The block without the measurements of clocks with no entry, ``None``
        when none is left; the series without those of such clocks; and
        those clocks, less the ones the configuration ignores.
    """
    configured = frozenset(clock_config.clocks)
    das_block, block_clocks = _configured_block(das_block, configured)
    earlier_series, series_clocks = _configured_series(earlier_series, configured)
    clocks_without_entry = (block_clocks | series_clocks) - frozenset(
        clock_config.ignore
    )
    logged = frozenset() if last_epoch is None else last_epoch.clocks_without_entry
    for clock in sorted(clocks_without_entry - logged):
        _log.warning(
            "clock %s has no entry in the clock configuration: its measurements are"
            " ignored",
            clock,
        )
    return das_block, earlier_series, clocks_without_entry


def _configured_block(
    das_block: DASData | None, configured: frozenset[str]
) -> tuple[DASData | None, frozenset[str]]:
    """Leave out of a block every measurement of a clock with no entry.

    Parameters
    ----------
    das_block : DASData or None
        The epoch's DAS block, or ``None``.
    configured : frozenset of str
        The clocks the clock configuration names.

    Returns
    -------
    tuple of (DASData or None, frozenset of str)
        The block without those measurements, ``None`` when none is left,
        and the clocks left out.
    """
    if das_block is None:
        return None, frozenset()
    kept = tuple(
        das_measurement
        for das_measurement in das_block.measurements
        if das_measurement.clock in configured
    )
    if len(kept) == len(das_block.measurements):
        return das_block, frozenset()
    left_out = (
        frozenset(das_measurement.clock for das_measurement in das_block.measurements)
        - configured
    )
    if not kept:
        return None, left_out
    return das_block.model_copy(update={"measurements": kept}), left_out


def _configured_series(
    earlier_series: ExistingSeries, configured: frozenset[str]
) -> tuple[ExistingSeries, frozenset[str]]:
    """Leave out every series whose clock side has no entry.

    Parameters
    ----------
    earlier_series : ExistingSeries
        The series that existed before the epoch.
    configured : frozenset of str
        The clocks the clock configuration names.

    Returns
    -------
    tuple of (ExistingSeries, frozenset of str)
        The series whose last name the configuration names, and the clocks
        of those left out.
    """
    series_keys: list[SeriesKey] = [*earlier_series.pairs, *earlier_series.triples]
    left_out = frozenset(series_key[-1] for series_key in series_keys) - configured
    if not left_out:
        return earlier_series, left_out
    return (
        ExistingSeries(
            pairs=frozenset(
                pair for pair in earlier_series.pairs if pair[-1] in configured
            ),
            triples=frozenset(
                triple for triple in earlier_series.triples if triple[-1] in configured
            ),
        ),
        left_out,
    )


def _configuration_kept(
    last_epoch: Epoch | None, epoch_start: datetime, clock_config: ClockConfig
) -> Epoch | None:
    """Give the last epoch when what the clock configuration gave it still holds.

    Parameters
    ----------
    last_epoch : Epoch or None
        The epoch processed before this one in the run, or ``None``.
    epoch_start : datetime
        The epoch start E.
    clock_config : ClockConfig
        The clock configuration.

    Returns
    -------
    Epoch or None
        ``last_epoch`` when it is earlier than E and no clock's entry takes
        effect after it, up to E (see
        :meth:`~masterclock.das_processor.clock_config.ClockConfig.changes_between`),
        so every clock's settings and location are as they were; otherwise
        ``None``.
    """
    if (
        last_epoch is None
        or last_epoch.interpolated_datetime >= epoch_start
        or clock_config.changes_between(last_epoch.interpolated_datetime, epoch_start)
    ):
        return None
    return last_epoch


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
    series_key: SeriesKey,
    epoch_start: datetime,
    steering: Mapping[str, tuple[SteerEvent, ...]],
) -> tuple[mpq, float]:
    """Give a series' steering input over the epoch before E.

    Parameters
    ----------
    series_key : series key
        The series.
    epoch_start : datetime
        The epoch start E.
    steering : Mapping of str to tuple of SteerEvent
        The epoch's steering events; empty, or holding no event, when no
        event falls in the epoch's steering window.

    Returns
    -------
    tuple of (mpq, float)
        What :func:`~masterclock.domain.steering.steer_u` gives; zero, as it
        would give, when no event falls in the window.
    """
    if not any(steering.values()):
        return _NO_STEERING_INPUT
    return steer_u(series_key, epoch_start, steering)


class PairReading(NamedTuple):
    """One DAS measurement of a pair, as the pairs' work needs it.

    Parameters
    ----------
    pair : (str, str)
        The pair measured.
    measurement_mjd : float
        When, as the DAS gave it.
    measured_phase : int
        The reading, ps.
    rms : int
        Its rms, ps.
    measurement_datetime : datetime
        When, as a datetime.
    """

    pair: PairKey
    measurement_mjd: float
    measured_phase: int
    rms: int
    measurement_datetime: datetime


def pair_readings(das_block: DASData | None) -> tuple[PairReading, ...]:
    """Give a block's measurements as pair readings, in the block's order.

    Parameters
    ----------
    das_block : DASData or None
        The epoch's DAS block, or ``None``.

    Returns
    -------
    tuple of PairReading
        One per measurement; none without a block.
    """
    if das_block is None:
        return ()
    return tuple(
        PairReading(
            (das_measurement.reference, das_measurement.clock),
            das_measurement.measurement_mjd,
            das_measurement.measured_phase,
            das_measurement.rms,
            das_measurement.measurement_datetime,
        )
        for das_measurement in das_block.measurements
    )


def _measured_pairs(
    epoch_start: datetime,
    steering: Mapping[str, tuple[SteerEvent, ...]],
    readings: Iterable[PairReading],
    last_rows: Mapping[SeriesKey, Row],
    predictions: Mapping[PairKey, State | None],
) -> dict[PairKey, PairMeasurement]:
    """Decycle every pair the epoch measured (design 7).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    steering : Mapping of str to tuple of SteerEvent
        The epoch's steering events.
    readings : Iterable of PairReading
        The epoch's readings.
    last_rows : Mapping of series key to Row
        Each series' last row.
    predictions : Mapping of (str, str) to State or None
        Each pair's prediction at E.

    Returns
    -------
    dict of (str, str) to PairMeasurement
        Each measured pair's measurement, in the readings' order, decycled
        against its prediction, or against its anchor when it has none,
        with the steering inside the epoch taken off.

    Raises
    ------
    PhaseError
        If a reading or its offset is out of range.
    """
    steered = any(steering.values())
    measurements = {}
    for reading in readings:
        pair = reading.pair
        w = (
            steer_w(pair, epoch_start, steering, reading.measurement_datetime)
            if steered
            else _NO_STEERING_INPUT[0]
        )
        measurements[pair] = measure_pair(
            measurement_mjd=reading.measurement_mjd,
            measured_phase=reading.measured_phase,
            rms=reading.rms,
            prediction=predictions[pair],
            w=w,
            anchor=anchor_of(last_rows.get(pair)),
        )
    return measurements


def _innovations(
    measurements: Mapping[PairKey, PairMeasurement],
    predictions: Mapping[PairKey, State | None],
    last_rows: Mapping[SeriesKey, Row],
) -> tuple[dict[PairKey, mpq], dict[PairKey, float]]:
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
        exact, in the measurements' order, and the innovation scale of
        every pair with a prediction, in the predictions' order.
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
    innovations: dict[PairKey, mpq] = {}
    for pair, measurement in measurements.items():
        prediction = predictions[pair]
        if prediction is not None:
            innovations[pair] = measurement.z - prediction.x
    return innovations, scales


@dataclass(frozen=True, slots=True)
class PairStart:
    """What some pairs give before screening (design 7, 8.3).

    Parameters
    ----------
    predictions : dict of (str, str) to State or None
        Each pair's prediction at E, in the pairs' order.
    measurements : dict of (str, str) to PairMeasurement
        Each measured pair's measurement, in the readings' order.
    innovations : dict of (str, str) to mpq
        Each innovation screening and the slip check use.
    scales : dict of (str, str) to float
        Each innovation scale they use.
    last_flags : dict of (str, str) to str
        The flags of each pair's last row, in the pairs' order.
    """

    predictions: dict[PairKey, State | None]
    measurements: dict[PairKey, PairMeasurement]
    innovations: dict[PairKey, mpq]
    scales: dict[PairKey, float]
    last_flags: dict[PairKey, str]


def start_pairs(
    epoch_start: datetime,
    steering: Mapping[str, tuple[SteerEvent, ...]],
    pairs: Iterable[PairKey],
    readings: Iterable[PairReading],
    last_rows: Mapping[SeriesKey, Row],
) -> PairStart:
    """Predict and decycle some pairs: their part of an epoch before screening.

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    steering : Mapping of str to tuple of SteerEvent
        The epoch's steering events.
    pairs : Iterable of (str, str)
        The pairs, sorted.
    readings : Iterable of PairReading
        Their readings, in the block's order.
    last_rows : Mapping of series key to Row
        Each series' row of the epoch before E.

    Returns
    -------
    PairStart
        Each pair's prediction, each measurement, and the innovations,
        scales and last flags screening and the slip check use.

    Raises
    ------
    PhaseError
        If a reading or its offset is out of range.
    """
    pairs = tuple(pairs)
    predictions = {
        pair: predict(last_rows.get(pair), _steering_input(pair, epoch_start, steering))
        for pair in pairs
    }
    measurements = _measured_pairs(
        epoch_start, steering, readings, last_rows, predictions
    )
    innovations, scales = _innovations(measurements, predictions, last_rows)
    last_flags = {pair: last_rows[pair].flags for pair in pairs if pair in last_rows}
    return PairStart(
        predictions=predictions,
        measurements=measurements,
        innovations=innovations,
        scales=scales,
        last_flags=last_flags,
    )


def finish_pairs(
    epoch_start: datetime,
    series_params: Mapping[SeriesKey, SeriesParams],
    last_rows: Mapping[SeriesKey, Row],
    last_segments: Mapping[SeriesKey, int],
    pair_start: PairStart,
    corrections: Mapping[PairKey, int],
    excluded: frozenset[PairKey],
) -> tuple[dict[PairKey, PairMeasurement], dict[PairKey, StepResult]]:
    """Correct and filter some pairs: their part of an epoch after screening.

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    series_params : Mapping of series key to SeriesParams
        Each pair's settings at E.
    last_rows : Mapping of series key to Row
        Each series' row of the epoch before E.
    last_segments : Mapping of series key to int
        The segment of the newest row of each series that starts again
        (see :func:`~masterclock.domain.filter.carry`).
    pair_start : PairStart
        What :func:`start_pairs` gave for the pairs, whose predictions name
        them in order.
    corrections : Mapping of (str, str) to int
        The slip check's correction of each pair it corrected, cycles.
    excluded : frozenset of (str, str)
        The pairs screening or the slip check excluded.

    Returns
    -------
    tuple of (dict, dict)
        Each measurement, its slip corrected, and each pair's row and
        whether it cold-started, in the pairs' order.

    Raises
    ------
    FilterError
        If a row breaks a rule of a row.
    """
    measurements = dict(pair_start.measurements)
    for pair, cycles in corrections.items():
        measurements[pair] = measurements[pair].corrected(cycles)
    step_results = {}
    for pair, prediction in pair_start.predictions.items():
        measurement = measurements.get(pair)
        step_results[pair] = filter_step(
            epoch_start,
            series_params[pair],
            last_rows.get(pair),
            prediction,
            None if measurement is None else measurement.filter_input(),
            excluded=pair in excluded,
            last_segment=last_segments.get(pair),
        )
    return measurements, step_results


def process_pairs(
    epoch: Epoch,
    last_rows: Mapping[SeriesKey, Row],
    last_segments: Mapping[SeriesKey, int] | None = None,
) -> PairStep:
    """Process an epoch's pairs: predict, decycle, screen, check slips, filter.

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last_rows : Mapping of series key to Row
        Each series' row of the epoch before E; a series missing here is
        new, or starts again.
    last_segments : Mapping of series key to int or None, optional
        The segment of the newest row of each series that starts again
        after epochs it had no row for; such a series starts in the next
        segment (see :func:`~masterclock.domain.filter.carry`); none when
        ``None``.

    Returns
    -------
    PairStep
        Every pair's row and what led to it, in sorted key order: the
        pairs started (see :func:`start_pairs`), screened and checked for
        slips across the epoch, then finished (see :func:`finish_pairs`).

    Raises
    ------
    FilterError
        If a row breaks a rule of a row.
    PhaseError
        If a reading or its offset is out of range.
    """
    epoch_start = epoch.interpolated_datetime
    pair_start = start_pairs(
        epoch_start,
        epoch.steering,
        epoch.pairs,
        pair_readings(epoch.das_block),
        last_rows,
    )
    screening, slips = screen_pairs(
        pair_start.innovations, pair_start.scales, pair_start.last_flags, epoch.refs
    )
    measurements, step_results = finish_pairs(
        epoch_start,
        epoch.series_params,
        last_rows,
        last_segments or {},
        pair_start,
        slips.corrections,
        screening.excluded | slips.excluded,
    )
    return PairStep(
        step_results=step_results,
        measurements=measurements,
        predictions=pair_start.predictions,
        screening=screening,
        slips=slips,
    )


def screen_pairs(
    innovations: Mapping[PairKey, mpq],
    scales: Mapping[PairKey, float],
    last_flags: Mapping[PairKey, str],
    refs: frozenset[str],
) -> tuple[Screening, Slips]:
    """Screen an epoch's references and check its pairs for slips (design 10, 11).

    Parameters
    ----------
    innovations : Mapping of (str, str) to mpq
        Every pair's innovation.
    scales : Mapping of (str, str) to float
        Every pair's innovation scale.
    last_flags : Mapping of (str, str) to str
        The flags of every pair's last row.
    refs : frozenset of str
        The epoch's references.

    Returns
    -------
    tuple of (Screening, Slips)
        What screening decided, and what the slip check decided with the
        pairs screening excluded left out.
    """
    screening = screen_references(innovations, scales, refs)
    slips = slip_check(innovations, scales, last_flags, refs, screening.excluded)
    return screening, slips


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


def component_of(
    step_result: StepResult | None,
    prediction: State | None,
    measurement: PairMeasurement | None,
) -> Component:
    """Give a pair's part in a triple: its accepted measurement, never its estimate.

    Parameters
    ----------
    step_result : StepResult or None
        The pair's row at the epoch; ``None`` for a pair the epoch does
        not hold.
    prediction : State or None
        Its prediction at the epoch.
    measurement : PairMeasurement or None
        Its measurement, its slip corrected; ``None`` when it has none.

    Returns
    -------
    Component
        Whether the pair's row was accepted, its z and rms when it was, its
        prediction, and whether it cold-started.
    """
    predicted = None if prediction is None else prediction.x
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


def _component(pair_step: PairStep, pair: PairKey) -> Component:
    """Give a pair's part in a triple, from what the epoch's pairs gave.

    Parameters
    ----------
    pair_step : PairStep
        What the epoch's pairs gave.
    pair : (str, str)
        The pair.

    Returns
    -------
    Component
        What :func:`component_of` gives for the pair; for a pair the epoch
        does not hold, one not accepted, with no prediction.
    """
    return component_of(
        pair_step.step_results.get(pair),
        pair_step.predictions.get(pair),
        pair_step.measurements.get(pair),
    )


def process_triples(
    epoch: Epoch,
    last_rows: Mapping[SeriesKey, Row],
    pair_step: PairStep,
    last_segments: Mapping[SeriesKey, int] | None = None,
) -> TripleStep:
    """Process an epoch's triples: double differences, then the filter (design 12).

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    last_rows : Mapping of series key to Row
        Each series' row of the epoch before E; a series missing here is
        new, or starts again.
    pair_step : PairStep
        What the epoch's pairs gave.
    last_segments : Mapping of series key to int or None, optional
        The segment of the newest row of each series that starts again
        (see :func:`process_pairs`); none when ``None``.

    Returns
    -------
    TripleStep
        What :func:`work_triples` gives for every triple, each pair's part
        worked out once for the epoch.

    Raises
    ------
    PhaseError
        If a local triple does not collapse to its pair.
    FilterError
        If a row breaks a rule of a row.
    """
    return work_triples(
        epoch.interpolated_datetime,
        epoch.steering,
        epoch.triples,
        epoch.series_params,
        last_rows,
        last_segments or {},
        {pair: _component(pair_step, pair) for pair in epoch.pairs},
    )


def work_triples(
    epoch_start: datetime,
    steering: Mapping[str, tuple[SteerEvent, ...]],
    triples: Iterable[TripleKey],
    series_params: Mapping[SeriesKey, SeriesParams],
    last_rows: Mapping[SeriesKey, Row],
    last_segments: Mapping[SeriesKey, int],
    components: Mapping[PairKey, Component],
) -> TripleStep:
    """Work some triples of an epoch from the pairs' parts (design 12).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    steering : Mapping of str to tuple of SteerEvent
        The epoch's steering events.
    triples : Iterable of (str, str, str)
        The triples, sorted.
    series_params : Mapping of series key to SeriesParams
        Each triple's settings at E.
    last_rows : Mapping of series key to Row
        Each series' row of the epoch before E.
    last_segments : Mapping of series key to int
        The segment of the newest row of each series that starts again.
    components : Mapping of (str, str) to Component
        Each pair's part in the triples (see :func:`component_of`).

    Returns
    -------
    TripleStep
        Every triple's row and double difference, in the triples' order. A
        local triple (r, r, c) is given its self pair for both links, so
        the check that it collapses to its pair runs every epoch; a pair
        with no part given takes part as one not accepted, with no
        prediction.

    Raises
    ------
    PhaseError
        If a local triple does not collapse to its pair.
    FilterError
        If a row breaks a rule of a row.
    """
    step_results: dict[TripleKey, StepResult] = {}
    measurements: dict[TripleKey, TripleMeasurement] = {}
    predictions: dict[TripleKey, State | None] = {}
    for triple in triples:
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
            last_rows.get(triple), _steering_input(triple, epoch_start, steering)
        )
        predictions[triple] = prediction
        step_results[triple] = filter_step(
            epoch_start,
            series_params[triple],
            last_rows.get(triple),
            prediction,
            None if measurement is None else measurement.filter_input(),
            last_segment=last_segments.get(triple),
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
    """Cut back what cannot be kept, and give the next epoch (design 6.7).

    Parameters
    ----------
    config : AppConfig
        The run's settings.

    Returns
    -------
    datetime
        One epoch after the newest epoch any file is good through; with no
        file holding a whole row, the epoch containing ``start_from_mjd``.
        A file may end earlier: its series was left out of the later
        epochs, or was dormant with no measurement, or a damaged end was
        cut off, and it starts cold when it is next in an epoch.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, or, with no write
        stopped part way, its first row is damaged, so its rows cannot be
        placed in time.

    Notes
    -----
    The journal is read before any file is checked. When it is there, a
    write stopped part way, and every file is cut back to before the
    write's first epoch; a file whose first row is damaged is then one the
    write was creating, its length on the device but not its rows, and it
    is deleted, to be made again. A damaged file is cut back to its last
    good row. Each damaged file is logged once at ERROR by the file check,
    and a roll-back that changed anything, or followed a stopped write, is
    logged once at WARNING, with why, how many files it cut, deleted and
    left, and the epoch the run goes on after. Once the files are cut back,
    the journal is deleted.
    """
    journal = config.processed.processed_path / JOURNAL_FILE_TEMPLATE.format(
        rf=config.das.rf
    )
    newest_epoch = _roll_back_all(config, read_journal(journal))
    clear_journal(journal)
    if newest_epoch is None:
        return floor_to_ten_minutes(mjd_to_datetime(config.processed.start_from_mjd))
    return newest_epoch + _EPOCH


def _roll_back_all(
    config: AppConfig, stopped_write_epoch: datetime | None
) -> datetime | None:
    """Check every file, cut each back to what it can keep, and log it once.

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
        The newest epoch any file now ends at; ``None`` when none holds a
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
    kept_epochs = _kept_epochs(file_checks, stopped_write_epoch)
    cuts = [
        roll_back(data_file, file_kind, kept_epoch)
        for (data_file, file_kind, _), kept_epoch in zip(
            series_files, kept_epochs, strict=True
        )
    ]
    newest_epoch = max(
        (kept_epoch for kept_epoch in kept_epochs if kept_epoch is not None),
        default=None,
    )
    if stopped_write or any(cut != "kept" for cut in cuts):
        _log_roll_back(
            config.das.rf,
            newest_epoch,
            "a write that stopped part way"
            if stopped_write
            else "damaged files, each logged at ERROR",
            cuts,
        )
    return newest_epoch


def _kept_epochs(
    file_checks: list[FileCheck], stopped_write_epoch: datetime | None
) -> list[datetime | None]:
    """Give the last epoch each file keeps.

    Parameters
    ----------
    file_checks : list of FileCheck
        What the file check found in each file.
    stopped_write_epoch : datetime or None
        The first epoch of a write that stopped part way; ``None`` when
        there was none.

    Returns
    -------
    list of datetime or None
        For each file, its last good row's epoch, and no later than the
        epoch before ``stopped_write_epoch``; ``None`` for a file that keeps
        no row.
    """
    kept_epochs = [file_check.good_through for file_check in file_checks]
    if stopped_write_epoch is None:
        return kept_epochs
    return [
        None if kept_epoch is None else min(kept_epoch, stopped_write_epoch - _EPOCH)
        for kept_epoch in kept_epochs
    ]


def _log_roll_back(
    channel: RfChannel, newest_epoch: datetime | None, reason: str, cuts: list[Cut]
) -> None:
    """Log a roll-back once, at WARNING (design 6.7, 16.2).

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel.
    newest_epoch : datetime or None
        The newest epoch any file ends at after it; ``None`` for no row.
    reason : str
        Why, in words.
    cuts : list of {'kept', 'cut', 'deleted'}
        What the roll-back did to each file.
    """
    _log.warning(
        "cut back the files of channel %s after %s: %d files cut, %d deleted,"
        " %d left; the newest row is now of %s",
        channel,
        reason,
        cuts.count("cut"),
        cuts.count("deleted"),
        cuts.count("kept"),
        "no epoch" if newest_epoch is None else newest_epoch,
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


def rows_before(
    newest_rows: Mapping[SeriesKey, Row], epoch_start: datetime
) -> tuple[dict[SeriesKey, Row], dict[SeriesKey, int]]:
    """Split the series' newest rows into last rows and series that start again.

    Parameters
    ----------
    newest_rows : Mapping of series key to Row
        Each series' newest row.
    epoch_start : datetime
        The epoch start E.

    Returns
    -------
    tuple of (dict, dict)
        The rows of the epoch before E, each its series' last row; and for
        every series whose newest row is older, its segment. Such a series
        had no row for the epoch before E, so it starts cold at E, as a new
        series does, in the segment after its newest row's.
    """
    last_rows: dict[SeriesKey, Row] = {}
    last_segments: dict[SeriesKey, int] = {}
    for series_key, newest_row in newest_rows.items():
        if newest_row.interpolated_datetime == epoch_start - _EPOCH:
            last_rows[series_key] = newest_row
        else:
            last_segments[series_key] = newest_row.segment
    return last_rows, last_segments


def process_epoch(
    epoch_start: datetime,
    das_block: DASData | None,
    day_buffer: DayBuffer,
    config: AppConfig,
    clock_config: ClockConfig,
    last_epoch: Epoch | None = None,
    steering_files: SteeringFiles | None = None,
) -> EpochDone:
    """Process one epoch, add its rows to the day buffer, and log it (design 6.3).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    das_block : DASData or None
        The epoch's DAS block, or ``None`` when the DAS measured nothing.
    day_buffer : DayBuffer
        The day buffer, whose newest rows are the series' last rows; when
        it holds none, they are read from the files into it.
    config : AppConfig
        The run's settings.
    clock_config : ClockConfig
        The clock configuration.
    last_epoch : Epoch or None, optional
        The epoch processed before this one in the run, whose settings
        :func:`build_epoch` may keep; ``None`` for the run's first.
    steering_files : SteeringFiles or None, optional
        The run's steering files; ``None`` to read them afresh.

    Returns
    -------
    EpochDone
        The epoch and what its pairs and triples gave. Every series' row
        is added but a dormant one with no measurement (see
        :func:`~masterclock.domain.filter.writes_row`).

    Raises
    ------
    MasterClockError
        If anything about the epoch cannot be read, worked out or
        formatted; the buffer then holds none of the epoch's rows.
    """
    if not day_buffer.last_rows:
        day_buffer.last_rows.update(read_last_state(config))
    newest_rows = dict(day_buffer.last_rows)
    earlier_series = ExistingSeries(
        pairs=frozenset(
            (series_key[0], series_key[1])
            for series_key in newest_rows
            if len(series_key) == _PAIR
        ),
        triples=frozenset(
            (series_key[0], series_key[1], series_key[-1])
            for series_key in newest_rows
            if len(series_key) == _TRIPLE
        ),
    )
    last_rows, last_segments = rows_before(newest_rows, epoch_start)
    epoch = build_epoch(
        epoch_start,
        das_block,
        earlier_series,
        config,
        clock_config,
        last_epoch,
        steering_files,
    )
    pair_step = process_pairs(epoch, last_rows, last_segments)
    triple_step = process_triples(epoch, last_rows, pair_step, last_segments)
    epoch_buffer = DayBuffer(day_buffer.channel)
    processed_path = config.processed.processed_path
    for series_key, file_record in _file_records(epoch, pair_step, triple_step):
        epoch_buffer.add(
            series_file(processed_path, config.das.rf, series_key),
            series_key,
            file_record,
        )
    day_buffer.take(epoch_buffer)
    epoch_done = EpochDone(epoch=epoch, pair_step=pair_step, triple_step=triple_step)
    log_epoch(epoch_done, last_rows, config.das.rf)
    return epoch_done


def _file_records(
    epoch: Epoch, pair_step: PairStep, triple_step: TripleStep
) -> list[tuple[SeriesKey, MeasRecord | DdiffRecord]]:
    """Give the record of each series of an epoch whose row is written.

    Parameters
    ----------
    epoch : Epoch
        The epoch.
    pair_step : PairStep
        What its pairs gave.
    triple_step : TripleStep
        What its triples gave.

    Returns
    -------
    list of (series key, MeasRecord or DdiffRecord)
        The pairs' records, then the triples', each in key order; a series
        dormant with no measurement has none (see
        :func:`~masterclock.domain.filter.writes_row`).
    """
    series_records: list[tuple[SeriesKey, MeasRecord | DdiffRecord]] = [
        (
            pair,
            MeasRecord(
                measurement=pair_step.measurements.get(pair),
                row=pair_step.step_results[pair].row,
            ),
        )
        for pair in epoch.pairs
        if writes_row(pair_step.step_results[pair].row)
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
        if writes_row(triple_step.step_results[triple].row)
    ]
    return series_records


class EpochProcessor(Protocol):
    """Something that processes an epoch as :func:`process_epoch` does.

    One is :class:`~masterclock.das_processor.workers.WorkerPool`, which
    works the series in worker processes.
    """

    def process_epoch(
        self,
        epoch_start: datetime,
        das_block: DASData | None,
        day_buffer: DayBuffer,
        config: AppConfig,
        clock_config: ClockConfig,
        last_epoch: Epoch | None = None,
        steering_files: SteeringFiles | None = None,
    ) -> Epoch:
        """Process one epoch and add its rows to the day buffer.

        Parameters
        ----------
        epoch_start : datetime
            The epoch start E.
        das_block : DASData or None
            The epoch's DAS block, or ``None`` when the DAS measured nothing.
        day_buffer : DayBuffer
            The day buffer.
        config : AppConfig
            The run's settings.
        clock_config : ClockConfig
            The clock configuration.
        last_epoch : Epoch or None, optional
            The epoch processed before this one in the run.
        steering_files : SteeringFiles or None, optional
            The run's steering files.

        Returns
        -------
        Epoch
            The epoch, as :func:`process_epoch` gives it in its result.
        """


def run(
    config: AppConfig,
    clock_config: ClockConfig,
    steps: int | None,
    shutdown: ShutdownHandler,
    epoch_processor: EpochProcessor | None = None,
) -> None:
    """Process the channel's epochs in order, up to the end of the data (design 6.3).

    Parameters
    ----------
    config : AppConfig
        The run's settings.
    clock_config : ClockConfig
        The clock configuration.
    steps : int or None
        How many epochs that write a row to process at most; ``None`` for
        all the data has.
    shutdown : ShutdownHandler
        Asked between epochs whether to stop.
    epoch_processor : EpochProcessor or None, optional
        What processes each epoch; ``None``, the default, for
        :func:`process_epoch` in this process.

    Raises
    ------
    MasterClockError
        If an epoch cannot be processed or a write fails. When the run had
        written its journal, the journal is left, so the next run cuts
        every file back to before this run's first epoch and computes its
        rows again; otherwise no rows were written, though the files may
        have been cut back at the start (see :func:`next_epoch`).

    Notes
    -----
    The run starts one epoch after the newest epoch any file holds (see
    :func:`next_epoch`); while no series exists yet, at the first block,
    since an epoch before it holds no series and writes nothing, and a run
    of one epoch would otherwise never get past it. An epoch with no block
    before the data resume is processed with no measurements. An epoch
    that writes no row is not counted as a step: once a gap epoch writes
    none, no later epoch of the gap writes any, so a run of one step at a
    time goes on to the next epoch that does, as a run in one go does. The
    run stops when no block remains, after ``steps`` epochs that wrote
    rows, or on a shutdown request, always between epochs. Rows are
    written after each day's 23:50 UTC epoch and when the run stops, and
    flushed to the device only when the run stops (:func:`write_final`).
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
    steering_files = SteeringFiles(config.das.steering_path)
    epochs_done = 0
    while (
        next_das_block is not None
        and not shutdown.shutdown_requested
        and (steps is None or epochs_done < steps)
    ):
        rows_before_epoch = day_buffer.rows_added
        das_block = None
        if next_das_block.interpolated_datetime == epoch_start:
            das_block = next_das_block
            next_das_block = _next_block(das_blocks, epoch_start + _EPOCH)
        if epoch_processor is None:
            last_epoch = process_epoch(
                epoch_start,
                das_block,
                day_buffer,
                config,
                clock_config,
                last_epoch,
                steering_files,
            ).epoch
        else:
            last_epoch = epoch_processor.process_epoch(
                epoch_start,
                das_block,
                day_buffer,
                config,
                clock_config,
                last_epoch,
                steering_files,
            )
        if (epoch_start + _EPOCH).date() != epoch_start.date():
            write_buffer(day_buffer)
        epoch_start += _EPOCH
        epochs_done += day_buffer.rows_added > rows_before_epoch
    write_final(day_buffer)


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


def log_screening(screening: Screening, slips: Slips, channel: RfChannel) -> None:
    """Log what screening and the slip check found (design 16.2).

    Parameters
    ----------
    screening : Screening
        What screening decided.
    slips : Slips
        What the slip check decided.
    channel : {'a', 'b'}
        The RF channel.
    """
    for screening_event in screening.events:
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
    for slip_event in slips.events:
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
        Its row of the epoch before, or ``None`` for a new series or one
        that starts again.
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
    Screening, slip and reject events go at WARNING; corrected slips,
    phase and frequency steps, cold starts, dormancy, a series that stops,
    configuration changes and the epoch's counts of rows written at INFO;
    each series' outcome at DEBUG and its prediction and update at TRACE.
    Series are logged in key order, pairs first. Nothing is worked out for
    a level the log leaves out: when WARNING is not logged the epoch is not
    looked at, and a TRACE line is made only when TRACE is logged.
    """
    if not _log.isEnabledFor(logging.WARNING):
        return
    pair_step, triple_step = epoch_done.pair_step, epoch_done.triple_step
    log_screening(pair_step.screening, pair_step.slips, channel)
    accepted_count = log_series_results(
        pair_step.step_results, pair_step.predictions, last_rows, channel
    )
    accepted_count += log_series_results(
        triple_step.step_results, triple_step.predictions, last_rows, channel
    )
    log_counts(
        epoch_done.epoch.interpolated_datetime,
        _written_count(pair_step.step_results),
        _written_count(triple_step.step_results),
        accepted_count,
    )


def _written_count[KeyT: SeriesKey](step_results: Mapping[KeyT, StepResult]) -> int:
    """Count the rows of an epoch's series that are written.

    Parameters
    ----------
    step_results : Mapping of series key to StepResult
        Each series' row at the epoch.

    Returns
    -------
    int
        How many of the rows are written (see
        :func:`~masterclock.domain.filter.writes_row`).
    """
    return sum(writes_row(step_result.row) for step_result in step_results.values())


def log_series_results[KeyT: SeriesKey](
    step_results: Mapping[KeyT, StepResult],
    predictions: Mapping[KeyT, State | None],
    last_rows: Mapping[SeriesKey, Row],
    channel: RfChannel,
) -> int:
    """Log each series' outcome at an epoch, in the order given (design 16.2).

    Parameters
    ----------
    step_results : Mapping of series key to StepResult
        Each series' row at the epoch, in the order to log them.
    predictions : Mapping of series key to State or None
        Each series' prediction at the epoch.
    last_rows : Mapping of series key to Row
        Each series' last row.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    int
        How many of the rows were accepted.

    Notes
    -----
    A series' TRACE line is made only when TRACE is logged. A row that is
    not written is not logged; when the series had a row at the epoch
    before, that it stops is logged once, at INFO.
    """
    trace_logged = _log.isEnabledFor(TRACE)
    accepted_count = 0
    for series_key, step_result in step_results.items():
        series_label = series_name(channel, series_key)
        if not writes_row(step_result.row):
            if series_key in last_rows:
                _log.info("%s stops: no row until it is measured again", series_label)
            continue
        _log_series(series_label, step_result, last_rows.get(series_key))
        if trace_logged:
            _log_trace(series_label, step_result.row, predictions[series_key])
        accepted_count += "A" in step_result.row.flags
    return accepted_count


def log_counts(
    epoch_start: datetime, pair_count: int, triple_count: int, accepted_count: int
) -> None:
    """Log an epoch's counts at INFO (design 16.2).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    pair_count : int
        How many pairs wrote a row at the epoch.
    triple_count : int
        How many triples did.
    accepted_count : int
        How many of their rows were accepted; the rest are held.
    """
    _log.info(
        "epoch %s: %d pairs, %d triples, %d accepted, %d held",
        epoch_start,
        pair_count,
        triple_count,
        accepted_count,
        pair_count + triple_count - accepted_count,
    )
