"""Working an epoch's series in worker processes (design 6.8).

Each worker process owns a fixed share of the series, the pairs and the
triples whose name falls to it (:func:`owner_of`), and keeps their newest
rows and settings from epoch to epoch, with each pair's newest z for a
disabled row to carry, so a series' state never crosses between processes.
The main process builds each epoch, screens it and checks it for slips, and
writes the rows; the workers do the rest of each series' work, with the
same functions a run without workers uses
(:mod:`masterclock.das_processor.run`), so every series is worked out the
same way whichever process works it.

An epoch takes three exchanges with every worker:

1. The main process sends each worker its pairs, triples, readings and any
   settings it does not hold yet; each worker predicts and decycles its
   pairs and sends back their innovations, scales and last flags.
2. The main process screens the epoch and checks it for slips with all of
   them, and sends each worker its pairs' corrections and exclusions; each
   worker filters its pairs and sends back their lines and their parts in
   the triples.
3. The main process sends every pair's part to every worker; each worker
   works its triples and sends back their lines.

A worker never writes the log. It keeps what its series log, with the
series each record is for, and the main process writes the records in the
order a run without workers writes them: screening first, then the pairs
and the triples, each in key order, then the epoch's counts. A failure in
a worker is logged there, sent back with its records, and raised again by
the main process; a worker that stops answering is a failure too.

Everything passes between the processes in Python's pickle format, which
only this program's own processes write and read.
"""

import contextlib
import functools
import logging
import multiprocessing
import pickle  # nosec B403 - only between this run's own processes
import signal
import zlib
from collections.abc import Mapping
from datetime import datetime
from heapq import merge
from multiprocessing.connection import Connection
from operator import itemgetter
from pathlib import Path
from types import TracebackType
from typing import Final, NamedTuple, NoReturn, Self

from gmpy2 import mpq

from masterclock.app.exceptions import MasterClockError
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.das_processor import run
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.exceptions import WorkerError
from masterclock.das_processor.files import (
    DayBuffer,
    DdiffRecord,
    FileKind,
    MeasRecord,
    read_last_record,
    record_line,
    record_z,
)
from masterclock.das_processor.read_cd5m5m import DASData
from masterclock.das_processor.read_steering import SteeringFiles
from masterclock.das_processor.registry import (
    ExistingSeries,
    existing_series,
    series_file,
)
from masterclock.domain.double_difference import Component
from masterclock.domain.filter import StepResult, writes_row
from masterclock.domain.series import (
    PairKey,
    Row,
    SeriesKey,
    SeriesParams,
    State,
    TripleKey,
)
from masterclock.domain.steering import SteerEvent

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

_CONTEXT: Final = multiprocessing.get_context("forkserver")
"""How worker processes start: each from a clean server process, never as a
copy of the main process with its open files and lock."""

_PAIR: Final[int] = 2
"""How many names a pair key holds."""

_STOP_WAIT: Final[float] = 10.0
"""How long the main process waits for a worker to end before ending it, s."""

ANSWER_WAIT: float = 600.0
"""How long the main process waits for each worker's answer, s.

A worker's part of one exchange takes well under a second, so this is
only a limit for a worker that hangs while still running: the time between
two scheduled runs, by which a run that is still waiting has already run
into the next one. Not Final, so a test can make it short.
"""


@functools.cache
def owner_of(series_key: SeriesKey, num_workers: int) -> int:
    """Give the worker that owns a series.

    Parameters
    ----------
    series_key : series key
        The series.
    num_workers : int
        How many workers there are.

    Returns
    -------
    int
        From 0 to ``num_workers - 1``, from the series' name alone, so a
        series has the same owner in every run with as many workers.

    Examples
    --------
    >>> owner_of(("mc1", "ox23"), 1)
    0
    """
    return zlib.crc32(".".join(series_key).encode("ascii")) % num_workers


class EpochTask(NamedTuple):
    """What a worker needs for its share of an epoch.

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    steering : dict of str to tuple of SteerEvent
        The epoch's steering events.
    pairs : tuple of (str, str)
        The worker's pairs in the epoch, sorted.
    triples : tuple of (str, str, str)
        Its triples in the epoch, sorted.
    new_params : dict of series key to SeriesParams
        The settings of its series that it does not hold yet, or that
        changed.
    readings : tuple of PairReading
        Its pairs' readings, in the block's order.
    """

    epoch_start: datetime
    steering: dict[str, tuple[SteerEvent, ...]]
    pairs: tuple[PairKey, ...]
    triples: tuple[TripleKey, ...]
    new_params: dict[SeriesKey, SeriesParams]
    readings: tuple[run.PairReading, ...]


class PairsStarted(NamedTuple):
    """What a worker's pairs give for screening and the slip check.

    Parameters
    ----------
    innovations : dict of (str, str) to mpq
        Their innovations, in the readings' order.
    scales : dict of (str, str) to float
        Their innovation scales, in the pairs' order.
    last_flags : dict of (str, str) to str
        Their last rows' flags, in the pairs' order.
    """

    innovations: dict[PairKey, mpq]
    scales: dict[PairKey, float]
    last_flags: dict[PairKey, str]


class SeriesDone(NamedTuple):
    """What a worker's series gave at an epoch.

    Parameters
    ----------
    lines : list of (series key, str)
        Each line made, with its newline, and its series, in key order;
        a series whose row is not written has none.
    components : dict of (str, str) to Component
        Each pair's part in the triples; empty for triples.
    accepted_count : int
        How many of the rows were accepted.
    log_records : list of (series key, list of logging.LogRecord)
        What each series logged, in key order; empty when WARNING is not
        logged, or the records were left to the logging set up already.
    """

    lines: list[tuple[SeriesKey, str]]
    components: dict[PairKey, Component]
    accepted_count: int
    log_records: list[tuple[SeriesKey, list[logging.LogRecord]]]


class RecordCapture(logging.Handler):
    """Keep log records in a list, each message made, to be sent on."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep a record, its message made so it holds nothing but text to send.

        Parameters
        ----------
        record : logging.LogRecord
            The record.
        """
        record.msg = record.getMessage()
        record.args = None
        record.exc_info = None
        record.exc_text = None
        self.records.append(record)

    def take(self) -> list[logging.LogRecord]:
        """Give the records kept so far, and keep none.

        Returns
        -------
        list of logging.LogRecord
            The records, in the order they were logged.
        """
        records, self.records = self.records, []
        return records


class SeriesShard:
    """A share of the series: their newest rows, settings and epoch's work.

    Each pair's newest z is kept beside its newest row, for a disabled
    pair's row to carry.

    Parameters
    ----------
    processed_path : Path
        The processed directory, where a series' file is read the first
        time the share meets the series.
    channel : {'a', 'b'}
        The RF channel.
    capture : RecordCapture or None, optional
        Where the share's log records go, to be sent on with the series
        each is for; ``None`` leaves them to the logging set up already.
    """

    def __init__(
        self,
        processed_path: Path,
        channel: RfChannel,
        capture: RecordCapture | None = None,
    ) -> None:
        """Start with no series."""
        self._processed_path = processed_path
        self._channel: RfChannel = channel
        self._capture = capture
        self._newest_rows: dict[SeriesKey, Row] = {}
        self._newest_z: dict[PairKey, int | None] = {}
        self._series_params: dict[SeriesKey, SeriesParams] = {}
        self._task: EpochTask | None = None
        self._last_rows: dict[SeriesKey, Row] = {}
        self._last_segments: dict[SeriesKey, int] = {}
        self._pair_start: run.PairStart | None = None

    def start_pairs(self, task: EpochTask) -> PairsStarted:
        """Begin an epoch: predict and decycle the share's pairs.

        Parameters
        ----------
        task : EpochTask
            The share's part of the epoch.

        Returns
        -------
        PairsStarted
            What screening and the slip check need of the pairs.

        Raises
        ------
        DataFileError
            If a series' file cannot be read or is not sound.
        PhaseError
            If a reading or its offset is out of range.
        """
        self._series_params.update(task.new_params)
        self._task = task
        series_keys: list[SeriesKey] = [*task.pairs, *task.triples]
        newest_rows = {
            series_key: self._newest_row(series_key) for series_key in series_keys
        }
        self._last_rows, self._last_segments = run.rows_before(
            {
                series_key: newest_row
                for series_key, newest_row in newest_rows.items()
                if newest_row is not None
            },
            task.epoch_start,
        )
        self._pair_start = run.start_pairs(
            task.epoch_start,
            task.steering,
            task.pairs,
            task.readings,
            self._last_rows,
            self._series_params,
            self._newest_z,
        )
        return PairsStarted(
            innovations=self._pair_start.innovations,
            scales=self._pair_start.scales,
            last_flags=self._pair_start.last_flags,
        )

    def finish_pairs(
        self, corrections: Mapping[PairKey, int], excluded: frozenset[PairKey]
    ) -> SeriesDone:
        """Correct and filter the share's pairs, and make their lines.

        Parameters
        ----------
        corrections : Mapping of (str, str) to int
            The slip check's corrections of the share's pairs, cycles.
        excluded : frozenset of (str, str)
            The share's pairs screening or the slip check excluded.

        Returns
        -------
        SeriesDone
            Each pair's line and part in the triples.

        Raises
        ------
        WorkerError
            If no epoch was begun.
        FilterError
            If a row breaks a rule of a row.
        DataFileError
            If a value does not fit its column.
        """
        task, pair_start = self._started()
        measurements, step_results = run.finish_pairs(
            task.epoch_start,
            self._series_params,
            self._last_rows,
            self._last_segments,
            pair_start,
            corrections,
            excluded,
        )
        components = {
            pair: run.component_of(
                step_result, pair_start.predictions[pair], measurements.get(pair)
            )
            for pair, step_result in step_results.items()
        }
        return self._done(
            step_results,
            pair_start.predictions,
            {
                pair: run.pair_record(
                    pair, step_result, measurements, pair_start.disabled_readings
                )
                for pair, step_result in step_results.items()
                if writes_row(step_result.row)
            },
            components,
        )

    def work_triples(self, components: Mapping[PairKey, Component]) -> SeriesDone:
        """Work the share's triples from every pair's part, and make their lines.

        Parameters
        ----------
        components : Mapping of (str, str) to Component
            Every pair's part in the triples.

        Returns
        -------
        SeriesDone
            Each triple's line.

        Raises
        ------
        WorkerError
            If no epoch was begun.
        PhaseError
            If a local triple does not collapse to its pair.
        FilterError
            If a row breaks a rule of a row.
        DataFileError
            If a value does not fit its column.
        """
        task, _ = self._started()
        triple_step = run.work_triples(
            task.epoch_start,
            task.steering,
            task.triples,
            self._series_params,
            self._last_rows,
            self._last_segments,
            components,
        )
        return self._done(
            triple_step.step_results,
            triple_step.predictions,
            {
                triple: DdiffRecord(
                    triple_step.measurements.get(triple), step_result.row
                )
                for triple, step_result in triple_step.step_results.items()
                if writes_row(step_result.row)
            },
            {},
        )

    def _started(self) -> tuple[EpochTask, run.PairStart]:
        """Give the epoch begun by :meth:`start_pairs`.

        Returns
        -------
        tuple of (EpochTask, PairStart)
            The epoch's task and what its pairs gave before screening.

        Raises
        ------
        WorkerError
            If no epoch was begun.
        """
        if self._task is None or self._pair_start is None:
            _fail("a worker was asked to go on with an epoch it did not begin")
        return self._task, self._pair_start

    def _newest_row(self, series_key: SeriesKey) -> Row | None:
        """Give a series' newest row, read from its file the first time.

        Parameters
        ----------
        series_key : series key
            The series.

        Returns
        -------
        Row or None
            Its newest row; ``None`` for a series with no file yet. A pair's
            newest z is kept with it.

        Raises
        ------
        DataFileError
            If its file cannot be read or is not sound.
        """
        newest_row = self._newest_rows.get(series_key)
        if newest_row is None:
            data_file = series_file(self._processed_path, self._channel, series_key)
            if data_file.exists():
                file_kind: FileKind = "meas" if len(series_key) == _PAIR else "ddiff"
                newest_record = read_last_record(data_file, file_kind)
                newest_row = newest_record.row
                self._keep_newest(series_key, newest_record)
        return newest_row

    def _keep_newest(
        self, series_key: SeriesKey, newest_record: MeasRecord | DdiffRecord
    ) -> None:
        """Keep a series' newest row, and a pair's newest z.

        Parameters
        ----------
        series_key : series key
            The series.
        newest_record : MeasRecord or DdiffRecord
            Its newest record, as its file holds it.
        """
        self._newest_rows[series_key] = newest_record.row
        if isinstance(newest_record, MeasRecord):
            self._newest_z[(series_key[0], series_key[1])] = record_z(newest_record)

    def _done[KeyT: SeriesKey](
        self,
        step_results: Mapping[KeyT, StepResult],
        predictions: Mapping[KeyT, State | None],
        file_records: Mapping[KeyT, MeasRecord | DdiffRecord],
        components: dict[PairKey, Component],
    ) -> SeriesDone:
        """Make the series' lines, keep their newest rows, and log their outcomes.

        Parameters
        ----------
        step_results : Mapping of series key to StepResult
            Each series' row, in key order.
        predictions : Mapping of series key to State or None
            Each series' prediction.
        file_records : Mapping of series key to MeasRecord or DdiffRecord
            The record of each series whose row is written, in key order.
        components : dict of (str, str) to Component
            The pairs' parts in the triples, passed on.

        Returns
        -------
        SeriesDone
            The lines, parts, accepted count and log records.

        Raises
        ------
        DataFileError
            If a value does not fit its column.
        """
        lines: list[tuple[SeriesKey, str]] = []
        for series_key, file_record in file_records.items():
            line_text, kept_row = record_line(file_record)
            lines.append((series_key, line_text))
            self._newest_rows[series_key] = kept_row
            if isinstance(file_record, MeasRecord):
                self._newest_z[(series_key[0], series_key[1])] = record_z(file_record)
        accepted_count = sum("A" in done.row.flags for done in step_results.values())
        log_records: list[tuple[SeriesKey, list[logging.LogRecord]]] = []
        if _log.isEnabledFor(logging.WARNING):
            for series_key, step_result in step_results.items():
                run.log_series_results(
                    {series_key: step_result},
                    predictions,
                    self._last_rows,
                    self._channel,
                )
                if self._capture is not None:
                    log_records.append((series_key, self._capture.take()))
        return SeriesDone(lines, components, accepted_count, log_records)


def answer_tasks(connection: Connection, shard: SeriesShard) -> None:
    """Work a share's part of each epoch the main process sends, until told to stop.

    Parameters
    ----------
    connection : Connection
        The worker's end of its pipe to the main process.
    shard : SeriesShard
        The worker's share of the series.

    Raises
    ------
    MasterClockError
        If a series' work fails.
    """
    while (task := connection.recv()) is not None:
        connection.send(("ok", shard.start_pairs(task)))
        corrections, excluded = connection.recv()
        connection.send(("ok", shard.finish_pairs(corrections, excluded)))
        connection.send(("ok", shard.work_triples(connection.recv())))


def serve(
    connection: Connection, processed_path: Path, channel: RfChannel, log_level: int
) -> None:
    """Be a worker: answer the main process until it says to stop.

    Run in the worker process. Stop signals and interrupts are left to the
    main process, which stops between epochs and then stops the workers.

    Parameters
    ----------
    connection : Connection
        The worker's end of its pipe to the main process.
    processed_path : Path
        The processed directory.
    channel : {'a', 'b'}
        The RF channel.
    log_level : int
        The main process's logging level, so the worker logs what it would.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    capture = RecordCapture()
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(capture)
    root.setLevel(log_level)
    try:
        answer_tasks(connection, SeriesShard(processed_path, channel, capture))
    except MasterClockError as exc:
        connection.send(("error", exc, capture.take()))
    except Exception as exc:  # any other failure is sent back too
        message = f"a worker failed: {type(exc).__name__}: {exc}"
        _log.error(message)
        connection.send(("error", WorkerError(message), capture.take()))


class WorkerPool:
    """Worker processes that work an epoch's series for the main process.

    Use as a context manager: the workers start on entry and stop on exit.

    Parameters
    ----------
    num_workers : int
        How many worker processes.
    processed_path : Path
        The processed directory.
    channel : {'a', 'b'}
        The RF channel.
    """

    def __init__(self, num_workers: int, processed_path: Path, channel: RfChannel):
        """Hold the settings; the workers start on entry."""
        self._num_workers = num_workers
        self._processed_path = processed_path
        self._channel: RfChannel = channel
        self._connections: list[Connection] = []
        self._processes: list[multiprocessing.process.BaseProcess] = []
        self._sent_params: list[dict[SeriesKey, SeriesParams]] = []
        self._series: tuple[set[PairKey], set[TripleKey]] | None = None

    def __enter__(self) -> Self:
        """Start the workers.

        Returns
        -------
        WorkerPool
            The pool.
        """
        log_level = logging.getLogger().getEffectiveLevel()
        for _ in range(self._num_workers):
            main_end, worker_end = _CONTEXT.Pipe()
            process = _CONTEXT.Process(
                target=serve,
                args=(worker_end, self._processed_path, self._channel, log_level),
                daemon=True,
            )
            process.start()
            worker_end.close()
            self._connections.append(main_end)
            self._processes.append(process)
            self._sent_params.append({})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the workers, waiting for each to end, and ending any that does not.

        Parameters
        ----------
        exc_type : type of BaseException or None
            The exception leaving the block, if any.
        exc_value : BaseException or None
            The exception, if any.
        traceback : TracebackType or None
            Its traceback, if any.
        """
        for connection in self._connections:
            with contextlib.suppress(OSError):
                connection.send(None)
        for process in self._processes:
            process.join(_STOP_WAIT)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self._connections:
            connection.close()

    def process_epoch(
        self,
        epoch_start: datetime,
        das_block: DASData | None,
        day_buffer: DayBuffer,
        config: AppConfig,
        clock_config: ClockConfig,
        last_epoch: run.Epoch | None = None,
        steering_files: SteeringFiles | None = None,
    ) -> run.Epoch:
        """Process one epoch with the workers, add its rows to the buffer, and log it.

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
            The run's steering files; ``None`` to read them afresh.

        Returns
        -------
        Epoch
            The epoch.

        Raises
        ------
        MasterClockError
            If anything about the epoch cannot be read, worked out or
            formatted, in the main process or a worker; the buffer then
            holds none of the epoch's rows.
        """
        pair_keys, triple_keys = self._known_series()
        epoch = run.build_epoch(
            epoch_start,
            das_block,
            ExistingSeries(pairs=frozenset(pair_keys), triples=frozenset(triple_keys)),
            config,
            clock_config,
            last_epoch,
            steering_files,
        )
        readings = run.pair_readings(epoch.das_block)
        pairs_started = self._start_pairs(epoch, readings)
        screening, slips = run.screen_pairs(
            _joined([started.innovations for started in pairs_started]),
            _joined([started.scales for started in pairs_started]),
            _joined([started.last_flags for started in pairs_started]),
            epoch.refs,
        )
        logging_on = _log.isEnabledFor(logging.WARNING)
        if logging_on:
            run.log_screening(screening, slips, self._channel)
        pairs_done = self._finish_pairs(
            slips.corrections, screening.excluded | slips.excluded
        )
        components: dict[PairKey, Component] = {}
        for done in pairs_done:
            components.update(done.components)
        component_bytes = pickle.dumps(components)
        for worker in range(self._num_workers):
            self._send(worker, component_bytes)
        triples_done = self._answers(SeriesDone)
        epoch_buffer = self._buffered(epoch_start, [*pairs_done, *triples_done])
        written_pairs, written_triples = _written(pairs_done), _written(triples_done)
        if logging_on:
            _write_records(pairs_done)
            _write_records(triples_done)
            run.log_counts(
                epoch_start,
                len(written_pairs),
                len(written_triples),
                sum(done.accepted_count for done in (*pairs_done, *triples_done)),
            )
        day_buffer.take(epoch_buffer)
        self._note_written([*written_pairs, *written_triples])
        return epoch

    def _note_written(self, series_keys: list[SeriesKey]) -> None:
        """Add the series that wrote a row to the series known so far.

        Parameters
        ----------
        series_keys : list of series key
            The series that wrote a row at the epoch.

        Raises
        ------
        DataFileError
            If an archive cannot be listed, the first time.
        """
        pair_keys, triple_keys = self._known_series()
        for series_key in series_keys:
            if len(series_key) == _PAIR:
                pair_keys.add((series_key[0], series_key[1]))
            else:
                triple_keys.add((series_key[0], series_key[1], series_key[-1]))

    def _finish_pairs(
        self, corrections: Mapping[PairKey, int], excluded: frozenset[PairKey]
    ) -> list[SeriesDone]:
        """Send each worker its pairs' corrections and exclusions; give what they did.

        Parameters
        ----------
        corrections : Mapping of (str, str) to int
            The slip check's corrections, cycles.
        excluded : frozenset of (str, str)
            The pairs screening or the slip check excluded.

        Returns
        -------
        list of SeriesDone
            Each worker's pairs' lines and parts in the triples, in worker
            order.

        Raises
        ------
        MasterClockError
            If a worker failed or stopped.
        """
        for worker in range(self._num_workers):
            self._send(
                worker,
                (
                    {
                        pair: cycles
                        for pair, cycles in corrections.items()
                        if owner_of(pair, self._num_workers) == worker
                    },
                    frozenset(
                        pair
                        for pair in excluded
                        if owner_of(pair, self._num_workers) == worker
                    ),
                ),
            )
        return self._answers(SeriesDone)

    def _buffered(
        self, epoch_start: datetime, workers_done: list[SeriesDone]
    ) -> DayBuffer:
        """Put the workers' lines of an epoch in a buffer of their own.

        Parameters
        ----------
        epoch_start : datetime
            The epoch start E.
        workers_done : list of SeriesDone
            What the workers' series gave.

        Returns
        -------
        DayBuffer
            The epoch's lines, each in its series' file.

        Raises
        ------
        DataFileError
            If a path is given another series.
        """
        epoch_buffer = DayBuffer()
        for done in workers_done:
            for series_key, line_text in done.lines:
                epoch_buffer.add_line(
                    series_file(self._processed_path, self._channel, series_key),
                    "meas" if len(series_key) == _PAIR else "ddiff",
                    series_key,
                    line_text,
                    epoch_start,
                )
        return epoch_buffer

    def _known_series(self) -> tuple[set[PairKey], set[TripleKey]]:
        """Give the series with a row so far, read from the files' names the first time.

        Returns
        -------
        tuple of (set, set)
            The pairs and the triples, which the caller adds each series to
            once it writes a row, as a run without workers knows a series by
            its newest row.

        Raises
        ------
        DataFileError
            If an archive cannot be listed.
        """
        if self._series is None:
            found = existing_series(self._processed_path, self._channel)
            self._series = (set(found.pairs), set(found.triples))
        return self._series

    def _start_pairs(
        self, epoch: run.Epoch, readings: tuple[run.PairReading, ...]
    ) -> list[PairsStarted]:
        """Send each worker its part of an epoch, and give what its pairs gave.

        Parameters
        ----------
        epoch : Epoch
            The epoch.
        readings : tuple of PairReading
            The epoch's readings, in the block's order.

        Returns
        -------
        list of PairsStarted
            Each worker's answer, in worker order.

        Raises
        ------
        MasterClockError
            If a worker failed or stopped.
        """
        for worker in range(self._num_workers):
            pairs = tuple(
                pair
                for pair in epoch.pairs
                if owner_of(pair, self._num_workers) == worker
            )
            triples = tuple(
                triple
                for triple in epoch.triples
                if owner_of(triple, self._num_workers) == worker
            )
            sent_params = self._sent_params[worker]
            series_keys: list[SeriesKey] = [*pairs, *triples]
            new_params = {
                series_key: epoch.series_params[series_key]
                for series_key in series_keys
                if sent_params.get(series_key) is not epoch.series_params[series_key]
            }
            sent_params.update(new_params)
            self._send(
                worker,
                EpochTask(
                    epoch_start=epoch.interpolated_datetime,
                    steering=epoch.steering,
                    pairs=pairs,
                    triples=triples,
                    new_params=new_params,
                    readings=tuple(
                        reading
                        for reading in readings
                        if owner_of(reading.pair, self._num_workers) == worker
                    ),
                ),
            )
        return self._answers(PairsStarted)

    def _send(self, worker: int, message: object) -> None:
        """Send a worker a message, already pickled or to be pickled here.

        Parameters
        ----------
        worker : int
            The worker.
        message : object
            The message, or its pickled bytes, to send the same bytes to
            several workers without pickling them again.

        Raises
        ------
        WorkerError
            If the worker's end of the pipe is closed: it stopped.
        """
        message_bytes = message if isinstance(message, bytes) else pickle.dumps(message)
        try:
            self._connections[worker].send_bytes(message_bytes)
        except OSError as exc:
            _fail(f"worker {worker} stopped answering: {exc!r}")

    def _answers[AnswerT](self, answer_type: type[AnswerT]) -> list[AnswerT]:
        """Wait for every worker's answer.

        Parameters
        ----------
        answer_type : type
            The kind of answer the exchange gives.

        Returns
        -------
        list
            Each worker's answer, in worker order.

        Raises
        ------
        MasterClockError
            The failure a worker sent back, its log records written first;
            or a :class:`WorkerError` for a worker that stopped answering,
            gave no answer within :data:`ANSWER_WAIT`, or gave another kind
            of answer.
        """
        answers = []
        for worker, connection in enumerate(self._connections):
            try:
                if not connection.poll(ANSWER_WAIT):
                    _fail(f"worker {worker} gave no answer within {ANSWER_WAIT:g} s")
                answer = connection.recv()
            except (
                EOFError,
                OSError,
            ) as exc:
                _fail(f"worker {worker} stopped answering: {exc!r}")
            if answer[0] == "error":
                _, failure, log_records = answer
                for log_record in log_records:
                    logging.getLogger(log_record.name).handle(log_record)
                raise failure
            if not isinstance(answer[1], answer_type):
                _fail(
                    f"worker {worker} gave {answer[1]!r}, not a {answer_type.__name__}"
                )
            answers.append(answer[1])
        return answers


def _joined[ValueT](parts: list[dict[PairKey, ValueT]]) -> dict[PairKey, ValueT]:
    """Join the workers' parts into one mapping.

    Screening and the slip check read their mappings by key and sort what
    they report, so the order of the joined mapping changes nothing.

    Parameters
    ----------
    parts : list of dict of (str, str) to value
        Each worker's part.

    Returns
    -------
    dict of (str, str) to value
        Every key some part holds.
    """
    joined: dict[PairKey, ValueT] = {}
    for part in parts:
        joined.update(part)
    return joined


def _written(workers_done: list[SeriesDone]) -> list[SeriesKey]:
    """Give the series whose lines the workers sent back.

    Parameters
    ----------
    workers_done : list of SeriesDone
        Each worker's answer.

    Returns
    -------
    list of series key
        Every series that wrote a row at the epoch.
    """
    return [series_key for done in workers_done for series_key, _ in done.lines]


def _write_records(workers_done: list[SeriesDone]) -> None:
    """Write the workers' log records, series by series in key order.

    Parameters
    ----------
    workers_done : list of SeriesDone
        Each worker's answer, its records in key order.
    """
    for _, log_records in merge(
        *(done.log_records for done in workers_done), key=itemgetter(0)
    ):
        for log_record in log_records:
            logging.getLogger(log_record.name).handle(log_record)


def _fail(message: str) -> NoReturn:
    """Log and raise a worker failure.

    Parameters
    ----------
    message : str
        What went wrong.

    Raises
    ------
    WorkerError
        Always.
    """
    _log.error(message)
    raise WorkerError(message)
