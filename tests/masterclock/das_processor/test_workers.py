"""Tests for src/masterclock/das_processor/workers.py.

The rules covered: every series has one owner among the workers, from its
name alone; a run with worker processes writes byte-identical data files to
a run without them, with one worker or several, in one batch or one epoch
per run, a series that stops, a clock with no entry and clocks disabled
for a time included, and logs
the same records in the same order; the pool counts a series as existing
once it writes a row; the command line's num_workers starts the workers.

A worker answers each exchange of an epoch, reading a series' file the first
time it meets the series, and logs with its records kept, each with its
series; nothing is logged when WARNING is not; a failure of the project's
kinds is sent back as it is with its records, any other as a WorkerError,
logged; a worker asked to go on with an epoch it did not begin refuses. A
record kept by a worker holds only its formatted text, so it can be sent,
and a series shard with nowhere to keep records logs them straight away. The
main process raises a failure a worker sent back after writing its records,
and a WorkerError for a worker that stopped, whether found in sending or
receiving, or that gave another kind of answer; on leaving, it asks each
worker to stop, tolerates one already gone, and ends one that does not stop.
"""

import logging
import signal
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.log import TRACE
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import main, run, workers
from masterclock.das_processor.clock_config import ClockConfig, read_clock_config
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.exceptions import DataFileError, WorkerError
from masterclock.das_processor.files import (
    DayBuffer,
    ensure_archives,
    parse_meas_row,
    read_last_row,
)
from masterclock.das_processor.read_cd5m5m import DASMeasurement, read_all_blocks
from masterclock.das_processor.read_steering import STEERING_FILE_TEMPLATE
from masterclock.das_processor.registry import ExistingSeries, series_file
from masterclock.domain.exceptions import PhaseError
from masterclock.domain.phase import PHASE_PERIOD

T: Final = timedelta(minutes=10)
"""One epoch."""

FIRST_EPOCH: Final = datetime(2025, 9, 23, 23, 0, tzinfo=UTC)
"""The first epoch of the invented deployment, an hour before midnight."""

EPOCH_COUNT: Final = 12
"""How many epochs the invented deployment's DAS files hold."""

CLOCK_CONFIG_YAML: Final = (
    "rejects_before_restart: 4\n"
    "reject_fraction_epochs: 25.0\n"
    "reject_fraction_limit: 0.5\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 10.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  cesium: {filter_states: 2, time_constant: 5.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  mc: {filter_states: 1, scale_time_constant: 5.0,"
    " initial_innovation_scale: 3.0, gap_limit: 8}\n"
    "clocks:\n"
    "  mc1: [{type: mc, location: 1}]\n"
    "  mc2: [{type: mc, location: 1}]\n"
    "  hm1: [{type: maser, location: 1}]\n"
    "  cs1: [{type: cesium, location: 2}]\n"
)
"""An invented clock configuration: two references and two clocks."""

PAIR_RATES: Final[dict[tuple[str, str], int]] = {
    ("mc1", "mc1"): 0,
    ("mc1", "mc2"): 3,
    ("mc2", "mc1"): -3,
    ("mc2", "mc2"): 0,
    ("mc1", "hm1"): 40,
    ("mc2", "hm1"): 37,
    ("mc2", "cs1"): -25,
}
"""Each measured pair's rate, ps per 100 s."""

PHASE_JUMP: Final = (8, ("mc1", "hm1"), 90_000)
"""At this epoch this pair jumps by this many hundredths of a ps: rejects, a step."""

MISSING_READING: Final = (5, ("mc2", "cs1"))
"""This pair is not measured at this epoch."""

STOPPED_READINGS: Final = (3, ("mc2", "hm1"))
"""This pair is not measured from this epoch on, so its file stops."""

UNCONFIGURED_PAIR: Final = ("mc1", "xx9")
"""A pair measured every epoch whose clock has no entry in the configuration."""


def write_deployment(
    deployment_directory: Path, clock_config_yaml: str = CLOCK_CONFIG_YAML
) -> AppConfig:
    """Write the invented deployment's inputs in a new directory; give its config."""
    for directory_name in ("das", "steering", "processed"):
        (deployment_directory / directory_name).mkdir(parents=True)
    (deployment_directory / "clock_config.yaml").write_text(
        clock_config_yaml, encoding="utf-8"
    )
    das_lines_by_day: dict[int, list[str]] = {}
    for epoch_index in range(EPOCH_COUNT):
        epoch_start = FIRST_EPOCH + epoch_index * T
        measured_rates = [*PAIR_RATES.items(), (UNCONFIGURED_PAIR, 0)]
        for pair_index, (pair, rate) in enumerate(measured_rates):
            if (epoch_index, pair) == MISSING_READING or (
                epoch_index >= STOPPED_READINGS[0] and pair == STOPPED_READINGS[1]
            ):
                continue
            seconds_into_epoch = 20 + 10 * pair_index
            phase = 1_000 * pair_index + rate * (epoch_index * 600 + seconds_into_epoch)
            if (epoch_index >= PHASE_JUMP[0]) and pair == PHASE_JUMP[1]:
                phase += PHASE_JUMP[2]
            das_measurement = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(
                        epoch_start + timedelta(seconds=seconds_into_epoch)
                    ),
                    6,
                ),
                measured_phase=(phase // 100) % PHASE_PERIOD,
                rms=3,
                switch=f"{pair[0][-1]}A{pair_index:02d}",
                clock=pair[1],
            )
            das_lines_by_day.setdefault(int(datetime_to_mjd(epoch_start)), []).append(
                f"{das_measurement}\n"
            )
    for data_day, das_lines in das_lines_by_day.items():
        (deployment_directory / "das" / f"cd5m5m_{data_day}.dat").write_text(
            "".join(das_lines), encoding="ascii"
        )
    steered_at = datetime_to_mjd(FIRST_EPOCH + 4 * T + timedelta(seconds=90))
    (
        deployment_directory / "steering" / STEERING_FILE_TEMPLATE.format(mc="mc1")
    ).write_text(f"{steered_at:.6f} 2.0 0.0001\n", encoding="ascii")
    return AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": deployment_directory / "das",
                "steering_path": deployment_directory / "steering",
            },
            "processed": {
                "processed_path": deployment_directory / "processed",
                "start_from_mjd": datetime_to_mjd(FIRST_EPOCH),
                "clock_config_file": deployment_directory / "clock_config.yaml",
                "num_workers": None,
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )


def archived_files(config: AppConfig) -> dict[str, bytes]:
    """Give every data file's bytes, by its path under the processed directory."""
    processed_root = config.processed.processed_path
    return {
        str(data_file.relative_to(processed_root)): data_file.read_bytes()
        for data_file in sorted(processed_root.rglob("das_a.*.dat"))
    }


def clock_config_of(config: AppConfig) -> ClockConfig:
    """Read a deployment's clock configuration."""
    return read_clock_config(config.processed.clock_config_file)


def run_without_workers(
    deployment_directory: Path, clock_config_yaml: str = CLOCK_CONFIG_YAML
) -> AppConfig:
    """Run the invented deployment in one batch, without workers; give its config."""
    config = write_deployment(deployment_directory, clock_config_yaml)
    run.run(config, clock_config_of(config), None, ShutdownHandler())
    return config


def run_with_workers(
    deployment_directory: Path,
    num_workers: int,
    steps: int | None = None,
    clock_config_yaml: str = CLOCK_CONFIG_YAML,
) -> AppConfig:
    """Run the invented deployment with worker processes; give its config."""
    config = write_deployment(deployment_directory, clock_config_yaml)
    with workers.WorkerPool(
        num_workers, config.processed.processed_path, config.das.rf
    ) as worker_pool:
        run.run(config, clock_config_of(config), steps, ShutdownHandler(), worker_pool)
    return config


@pytest.fixture(autouse=True)
def restore_logging_and_signals() -> Iterator[None]:
    """Put the root logger and the signal handlers back as they were after each test."""
    root_logger = logging.getLogger()
    saved_level, saved_handlers = root_logger.level, list(root_logger.handlers)
    saved_signals = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }
    yield
    root_logger.handlers[:] = saved_handlers
    root_logger.setLevel(saved_level)
    for signal_number, handler in saved_signals.items():
        signal.signal(signal_number, handler)


# ------------------------------------------------------------------ owners


@pytest.mark.parametrize("num_workers", [1, 2, 3, 5])
def test_every_series_has_one_owner(num_workers: int) -> None:
    """Give each series one worker, the same every time, every worker used."""
    series_keys = [
        *PAIR_RATES,
        *((r, s, c) for r in ("mc1", "mc2") for s, c in PAIR_RATES),
    ]
    owners = [workers.owner_of(series_key, num_workers) for series_key in series_keys]
    assert all(0 <= owner < num_workers for owner in owners)
    assert owners == [workers.owner_of(key, num_workers) for key in series_keys]
    assert set(owners) == set(range(num_workers))


# ------------------------------------------------- runs with worker processes


@pytest.mark.parametrize("num_workers", [1, 3])
def test_workers_write_the_files_a_run_without_them_writes(
    tmp_path: Path, num_workers: int
) -> None:
    """Give byte-identical data files with workers as without them (I5, U28)."""
    without_files = archived_files(run_without_workers(tmp_path / "without"))
    with_files = archived_files(run_with_workers(tmp_path / "with", num_workers))
    assert sum(name.startswith("meas/") for name in without_files) == len(PAIR_RATES)
    assert sum(name.startswith("ddiff/") for name in without_files) == 12
    stopped_file = series_file(Path(), "a", STOPPED_READINGS[1])
    stopped_bytes = without_files[str(stopped_file)]
    last_line = stopped_bytes.splitlines()[-1].decode()
    last_row_epoch = parse_meas_row(last_line).row.interpolated_datetime
    assert last_row_epoch == FIRST_EPOCH + (EPOCH_COUNT - 2) * T
    assert with_files == without_files


def test_workers_one_epoch_per_run_write_the_same_files(tmp_path: Path) -> None:
    """Give the batch's data files from one epoch per run with workers (I5, U28)."""
    without_files = archived_files(run_without_workers(tmp_path / "without"))
    config = write_deployment(tmp_path / "stepped")
    for _ in range(EPOCH_COUNT + 1):
        with workers.WorkerPool(
            2, config.processed.processed_path, config.das.rf
        ) as worker_pool:
            run.run(config, clock_config_of(config), 1, ShutdownHandler(), worker_pool)
    assert without_files
    assert archived_files(config) == without_files


EPOCH_6_MJD: Final = datetime_to_mjd(FIRST_EPOCH + 6 * T)
"""The MJD of the seventh epoch's start."""

EPOCH_10_MJD: Final = datetime_to_mjd(FIRST_EPOCH + 10 * T)
"""The MJD of the eleventh epoch's start."""

EPOCH_11_MJD: Final = datetime_to_mjd(FIRST_EPOCH + 11 * T)
"""The MJD of the twelfth epoch's start."""

DISABLED_CLOCK_CONFIG_YAML: Final = CLOCK_CONFIG_YAML.replace(
    "  mc2: [{type: mc, location: 1}]\n",
    "  mc2:\n"
    "    - {type: mc, location: 1}\n"
    f"    - {{effective_mjd: {EPOCH_10_MJD}, disabled: true}}\n"
    f"    - {{effective_mjd: {EPOCH_11_MJD}, disabled: false}}\n",
).replace(
    "  cs1: [{type: cesium, location: 2}]\n",
    "  cs1:\n"
    "    - {type: cesium, location: 2}\n"
    f"    - {{effective_mjd: {EPOCH_6_MJD}, enabled: false}}\n"
    f"    - {{effective_mjd: {EPOCH_10_MJD}, enabled: true}}\n",
)
"""The clock configuration, with cs1 disabled for four epochs and mc2 for one."""


def test_workers_write_the_files_of_disabled_clocks_a_run_without_them_writes(
    tmp_path: Path,
) -> None:
    """Give the same files with workers, in one go or stepped, clocks disabled (U29)."""
    without_files = archived_files(
        run_without_workers(tmp_path / "without", DISABLED_CLOCK_CONFIG_YAML)
    )
    disabled_counts = {
        file_name: disabled_count
        for file_name, file_bytes in without_files.items()
        if (
            disabled_count := sum(
                row_line.rsplit(", ", 1)[-1].strip() == "O"
                for row_line in file_bytes.decode().splitlines()
            )
        )
    }
    assert disabled_counts == {
        "meas/das_a.mc2.cs1.dat": 5,
        "meas/das_a.mc1.mc2.dat": 1,
        "meas/das_a.mc2.mc1.dat": 1,
        "meas/das_a.mc2.mc2.dat": 1,
    }
    with_files = archived_files(
        run_with_workers(
            tmp_path / "with", 3, clock_config_yaml=DISABLED_CLOCK_CONFIG_YAML
        )
    )
    assert with_files == without_files
    config = write_deployment(tmp_path / "stepped", DISABLED_CLOCK_CONFIG_YAML)
    for _ in range(EPOCH_COUNT + 1):
        with workers.WorkerPool(
            2, config.processed.processed_path, config.das.rf
        ) as worker_pool:
            run.run(config, clock_config_of(config), 1, ShutdownHandler(), worker_pool)
    assert archived_files(config) == without_files


def test_workers_log_what_a_run_without_them_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the same records in the same order with workers as without, at TRACE."""
    logged: list[list[tuple[str, int, str]]] = []
    for deployment_name, num_workers in (("without", None), ("with", 3)):
        caplog.clear()
        with caplog.at_level(TRACE):
            if num_workers is None:
                run_without_workers(tmp_path / deployment_name)
            else:
                run_with_workers(tmp_path / deployment_name, num_workers)
        logged.append(
            [
                (log_record.name, log_record.levelno, log_record.getMessage())
                for log_record in caplog.records
            ]
        )
    levels = {levelno for _, levelno, _ in logged[0]}
    assert {TRACE, logging.DEBUG, logging.INFO, logging.WARNING} <= levels
    assert logged[1] == logged[0]


def test_the_pool_knows_a_series_once_it_writes_a_row(tmp_path: Path) -> None:
    """Count as existing only the series that wrote a row, as a run without workers."""
    config = write_deployment(tmp_path)
    das_block = next(
        read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(FIRST_EPOCH))
    )
    day_buffer = DayBuffer()
    with workers.WorkerPool(
        1, config.processed.processed_path, config.das.rf
    ) as worker_pool:
        epoch = worker_pool.process_epoch(
            FIRST_EPOCH, das_block, day_buffer, config, clock_config_of(config)
        )
        known_pairs, known_triples = worker_pool._known_series()
    assert epoch.triples
    assert (known_pairs, known_triples) == (set(epoch.pairs), set())
    assert day_buffer.rows_added == len(epoch.pairs)


def test_the_command_line_starts_the_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run with --num-workers through the entry point, and write the same files."""
    without_files = archived_files(run_without_workers(tmp_path / "without"))
    pool_epochs: list[datetime] = []
    real_process_epoch = workers.WorkerPool.process_epoch

    def count_epoch(
        worker_pool: workers.WorkerPool, epoch_start: datetime, *args: object
    ) -> run.Epoch:
        """Note the epoch, then process it with the workers."""
        pool_epochs.append(epoch_start)
        return real_process_epoch(worker_pool, epoch_start, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(workers.WorkerPool, "process_epoch", count_epoch)
    config = write_deployment(tmp_path / "with")
    argv = [
        "--rf", "a",
        "--cd5m5m-path", str(config.das.cd5m5m_path),
        "--steering-path", str(config.das.steering_path),
        "--processed-path", str(config.processed.processed_path),
        "--clock-config-file", str(config.processed.clock_config_file),
        "--start-from-mjd", f"{config.processed.start_from_mjd:.6f}",
        "--num-workers", "2",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
    ]  # fmt: skip
    assert main(argv) == 0
    assert len(pool_epochs) == EPOCH_COUNT
    assert archived_files(config) == without_files


def test_a_worker_that_stopped_stops_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Raise and log WorkerError when a worker process is gone."""
    config = write_deployment(tmp_path)
    das_block = next(
        read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(FIRST_EPOCH))
    )
    with workers.WorkerPool(
        1, config.processed.processed_path, config.das.rf
    ) as worker_pool:
        (worker_process,) = worker_pool._processes
        worker_process.kill()
        worker_process.join()
        with pytest.raises(WorkerError, match="worker 0 stopped answering"):
            worker_pool.process_epoch(
                FIRST_EPOCH,
                das_block,
                DayBuffer(),
                config,
                clock_config_of(config),
            )
    assert [
        (log_record.levelname, UNCONFIGURED_PAIR[1] in log_record.getMessage())
        for log_record in caplog.records
    ] == [("WARNING", True), ("ERROR", False)]


# ------------------------------------------------- a worker, in this process


def first_task(config: AppConfig, epoch_start: datetime) -> workers.EpochTask:
    """Give a single worker's task for an epoch of the invented deployment."""
    das_block = next(
        read_all_blocks(config.das.cd5m5m_path, datetime_to_mjd(epoch_start))
    )
    epoch = run.build_epoch(
        epoch_start,
        das_block,
        ExistingSeries(
            pairs=frozenset(PAIR_RATES),
            triples=frozenset(),
        ),
        config,
        clock_config_of(config),
    )
    return workers.EpochTask(
        epoch_start=epoch_start,
        steering=epoch.steering,
        pairs=epoch.pairs,
        triples=epoch.triples,
        new_params=dict(epoch.series_params),
        readings=run.pair_readings(epoch.das_block),
    )


def serve_queued(
    config: AppConfig, messages: list[object], log_level: int
) -> list[tuple[object, ...]]:
    """Run a worker in this process on messages queued for it; give its answers.

    A last None, the stop message, is always queued, so the worker ends
    even when the messages run out, rather than waiting for one that never
    comes.
    """
    main_end, worker_end = Pipe()
    for message in [*messages, None]:
        main_end.send(message)
    workers.serve(worker_end, config.processed.processed_path, "a", log_level)
    answers = []
    while main_end.poll():
        answers.append(main_end.recv())
    return answers


def test_a_worker_answers_each_exchange_of_an_epoch(tmp_path: Path) -> None:
    """Answer an epoch's three exchanges, reading files, records kept by series."""
    config = run_without_workers(tmp_path)
    task = first_task(config, FIRST_EPOCH + EPOCH_COUNT * T - T)
    task = task._replace(epoch_start=FIRST_EPOCH + EPOCH_COUNT * T, readings=())
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    answers = serve_queued(config, [task, ({}, frozenset()), {}, None], logging.DEBUG)
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    assert [answer[0] for answer in answers] == ["ok", "ok", "ok"]
    started, pairs_done, triples_done = (answer[1] for answer in answers)
    written_pairs = [pair for pair in task.pairs if pair != STOPPED_READINGS[1]]
    assert isinstance(started, workers.PairsStarted)
    assert started.last_flags.keys() == set(written_pairs)
    assert isinstance(pairs_done, workers.SeriesDone)
    assert [series_key for series_key, _ in pairs_done.lines] == written_pairs
    assert [series_key for series_key, _ in pairs_done.log_records] == list(task.pairs)
    assert [
        series_key for series_key, log_records in pairs_done.log_records if log_records
    ] == written_pairs
    assert isinstance(triples_done, workers.SeriesDone)
    tracked_triples = [
        triple
        for triple in task.triples
        if (
            triple_file := series_file(config.processed.processed_path, "a", triple)
        ).exists()
        and "D" not in (last_row := read_last_row(triple_file, "ddiff")).flags
        and last_row.interpolated_datetime == task.epoch_start - T
    ]
    assert tracked_triples
    assert [series_key for series_key, _ in triples_done.lines] == tracked_triples
    assert triples_done.components == {}


def test_a_worker_logs_nothing_when_warning_is_not_logged(tmp_path: Path) -> None:
    """Keep no records, and work out none, at a level above WARNING."""
    config = write_deployment(tmp_path)
    task = first_task(config, FIRST_EPOCH)
    answers = serve_queued(config, [task, ({}, frozenset()), {}, None], logging.ERROR)
    series_done = [answer[1] for answer in answers[1:]]
    assert all(isinstance(done, workers.SeriesDone) for done in series_done)
    assert [done.log_records for done in series_done] == [[], []]  # type: ignore[attr-defined]


def test_a_worker_sends_back_a_failure_of_the_project_s_kinds(tmp_path: Path) -> None:
    """Send back a failure as it is, with the records that logged it."""
    config = write_deployment(tmp_path)
    task = first_task(config, FIRST_EPOCH)
    ensure_archives(config.processed.processed_path)
    series_file(config.processed.processed_path, "a", task.pairs[0]).write_text(
        "not a data file\n"
    )
    (answer,) = serve_queued(config, [task], logging.DEBUG)
    status, failure, log_records = answer
    assert status == "error"
    assert isinstance(failure, DataFileError)
    assert isinstance(log_records, list)
    assert [log_record.getMessage() for log_record in log_records] == [str(failure)]


def test_a_worker_sends_back_any_other_failure_as_a_worker_error(
    tmp_path: Path,
) -> None:
    """Send back a failure of another kind as a WorkerError, having logged it."""
    config = write_deployment(tmp_path)
    (answer,) = serve_queued(config, ["not a task"], logging.DEBUG)
    status, failure, log_records = answer
    assert status == "error"
    assert isinstance(failure, WorkerError)
    assert isinstance(log_records, list)
    assert str(failure).startswith("a worker failed: AttributeError: ")
    assert [log_record.levelname for log_record in log_records] == ["ERROR"]


def test_a_captured_record_holds_only_text() -> None:
    """Keep a record with its message made and nothing that cannot be sent."""
    capture = workers.RecordCapture()
    try:
        message = "phase %d out of range"
        raise ValueError(message)
    except ValueError:
        log_record = logging.LogRecord(
            "masterclock.domain.phase",
            logging.ERROR,
            __file__,
            1,
            message,
            (200_000,),
            sys.exc_info(),
        )
    capture.handle(log_record)
    (kept,) = capture.take()
    assert (kept.msg, kept.args, kept.exc_info) == (
        "phase 200000 out of range",
        None,
        None,
    )
    sending_end, receiving_end = Pipe()
    sending_end.send(kept)
    assert receiving_end.recv().getMessage() == "phase 200000 out of range"
    assert capture.take() == []


def test_a_shard_left_to_the_logging_set_up_keeps_no_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a shard's series straight away when it has nowhere to keep records."""
    config = run_without_workers(tmp_path)
    shard = workers.SeriesShard(config.processed.processed_path, "a")
    task = first_task(config, FIRST_EPOCH + EPOCH_COUNT * T - T)
    task = task._replace(epoch_start=FIRST_EPOCH + EPOCH_COUNT * T, readings=())
    shard.start_pairs(task)
    with caplog.at_level(logging.DEBUG):
        pairs_done = shard.finish_pairs({}, frozenset())
    assert pairs_done.log_records == []
    assert len(caplog.records) >= len(task.pairs)


def test_a_shard_reads_a_series_file_only_the_first_time(tmp_path: Path) -> None:
    """Keep each series' newest row, so a later epoch needs no file."""
    config = run_without_workers(tmp_path)
    shard = workers.SeriesShard(config.processed.processed_path, "a")
    task = first_task(config, FIRST_EPOCH + EPOCH_COUNT * T - T)
    next_epoch = FIRST_EPOCH + EPOCH_COUNT * T
    shard.start_pairs(task._replace(epoch_start=next_epoch, readings=()))
    shard.finish_pairs({}, frozenset())
    shard.work_triples({})
    data_files = sorted(config.processed.processed_path.rglob("das_a.*.dat"))
    for data_file in data_files:
        data_file.unlink()
    started = shard.start_pairs(task._replace(epoch_start=next_epoch + T, readings=()))
    assert len(data_files) == len(task.pairs) + len(task.triples)
    assert started.last_flags.keys() == set(task.pairs) - {STOPPED_READINGS[1]}


def test_a_shard_refuses_to_go_on_with_an_epoch_it_did_not_begin(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Raise and log WorkerError when asked to finish an epoch never begun."""
    shard = workers.SeriesShard(tmp_path, "a")
    with pytest.raises(WorkerError, match="did not begin"):
        shard.finish_pairs({}, frozenset())
    with pytest.raises(WorkerError, match="did not begin"):
        shard.work_triples({})
    assert [log_record.levelname for log_record in caplog.records] == ["ERROR"] * 2


# ------------------------------------------- the main process's side, alone


def pool_on_pipes(
    num_workers: int,
) -> tuple[workers.WorkerPool, list[Connection[object, object]]]:
    """Give a pool whose workers are pipe ends this test answers for."""
    worker_pool = workers.WorkerPool(num_workers, Path("/unused"), "a")
    worker_ends = []
    for _ in range(num_workers):
        main_end, worker_end = Pipe()
        worker_pool._connections.append(main_end)
        worker_pool._sent_params.append({})
        worker_ends.append(worker_end)
    return worker_pool, worker_ends


def test_a_failure_sent_back_is_raised_after_its_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Write the worker's records, then raise the failure it sent back."""
    worker_pool, (worker_end,) = pool_on_pipes(1)
    log_record = logging.LogRecord(
        "masterclock.domain.phase", logging.ERROR, __file__, 1, "bad phase", None, None
    )
    worker_end.send(("error", PhaseError("bad phase"), [log_record]))
    with pytest.raises(PhaseError, match="bad phase"):
        worker_pool._answers(workers.SeriesDone)
    assert [record.getMessage() for record in caplog.records] == ["bad phase"]


def test_a_worker_that_closed_its_pipe_has_stopped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raise and log WorkerError for a worker gone, in sending or in receiving."""
    worker_pool, (worker_end,) = pool_on_pipes(1)
    worker_end.close()
    with pytest.raises(WorkerError, match="worker 0 stopped answering"):
        worker_pool._answers(workers.SeriesDone)
    worker_pool._connections[0].close()
    with pytest.raises(WorkerError, match="worker 0 stopped answering"):
        worker_pool._send(0, None)
    assert [log_record.levelname for log_record in caplog.records] == ["ERROR"] * 2


def test_a_worker_that_gives_no_answer_in_time_has_stopped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Raise and log WorkerError for a worker still running but silent too long."""
    monkeypatch.setattr(workers, "ANSWER_WAIT", 0.05)
    worker_pool, (_worker_end,) = pool_on_pipes(1)
    with pytest.raises(WorkerError, match=r"worker 0 gave no answer within 0\.05 s"):
        worker_pool._answers(workers.SeriesDone)
    assert [log_record.levelname for log_record in caplog.records] == ["ERROR"]


def test_a_worker_must_give_the_answer_the_exchange_asks_for() -> None:
    """Raise WorkerError for an answer of another kind."""
    worker_pool, (worker_end,) = pool_on_pipes(1)
    worker_end.send(("ok", "not a SeriesDone"))
    with pytest.raises(WorkerError, match="not a SeriesDone"):
        worker_pool._answers(workers.SeriesDone)


class StuckProcess:
    """A worker process that does not end until it is ended."""

    def __init__(self) -> None:
        """Start alive."""
        self.alive = True
        self.joins: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        """Wait, noting for how long."""
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        """Tell whether it is still running."""
        return self.alive

    def terminate(self) -> None:
        """End it."""
        self.alive = False


def test_leaving_ends_a_worker_that_does_not_stop() -> None:
    """Ask each worker to stop, pass over one gone, and end one still running."""
    worker_pool, worker_ends = pool_on_pipes(2)
    worker_ends[1].close()
    stuck_process = StuckProcess()
    worker_pool._processes = [stuck_process]  # type: ignore[list-item]
    worker_pool.__exit__(None, None, None)
    assert worker_ends[0].recv() is None
    assert stuck_process.joins == [workers._STOP_WAIT, None]
    assert not stuck_process.alive
    assert all(connection.closed for connection in worker_pool._connections)
