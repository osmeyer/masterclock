"""Tests that das_processor recovers from any failure byte for byte (U21, I8).

The rules covered: an error at any point of computing an epoch or
preparing a write changes no file; a stop at any point of the write, before
or after any write or fsync, a torn line included, is undone by the next
run, whose data files end byte-identical to those of a run that never
stopped; rows a power failure loses from one file before the final write,
which alone flushes the data files, are made again from the run's first
epoch; and when a file whose end is torn has a line damaged by hand in its
middle, every file is cut back before that line and the next run's data
files end byte-identical to those of a run that never stopped. A file of
whole rows whose last row is good is not scanned (design 5.7), so damage in
its middle alone is not looked for.
"""

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import pytest

from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import files, registry, run
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import DASMeasurement

FIRST_EPOCH_START: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch: four before midnight, so the run writes twice."""

T: Final = timedelta(minutes=10)
"""One epoch."""

EPOCH_COUNT: Final = 7
"""How many epochs the invented data cover."""

CLOCK_CONFIG_YAML: Final = (
    "rejects_before_restart: 4\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 10.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  mc: {filter_states: 1, scale_time_constant: 5.0,"
    " initial_innovation_scale: 3.0, gap_limit: 8}\n"
    "clocks:\n"
    "  mc1: [{type: mc, location: 1}]\n"
    "  mc2: [{type: mc, location: 1}]\n"
    "  hm1: [{type: maser, location: 1}]\n"
    "  hm2: [{type: maser, location: 1}]\n"
)
"""An invented clock configuration."""

MEASURED_PAIRS: Final = (
    ("mc1", "mc1", 100, 0),
    ("mc1", "mc2", 5_000, 1),
    ("mc2", "mc1", 195_000, -1),
    ("mc2", "mc2", 200, 0),
    ("mc1", "hm1", 70_000, 25),
)
"""Each pair measured: reference, clock, first phase, rate in ps per 100 s."""


type Tear = Literal["first row", "half"]
"""Where a write stopped part way is torn."""

TORN_BYTE_COUNTS: Final[dict[Tear, Callable[[int], int]]] = {
    "first row": lambda _length: 7,
    "half": lambda length: length // 2 + 7,
}
"""How many bytes of a torn write reach the file: a few, inside the first
line, so a file that held rows ends in a torn line after them; or about half,
so whole new rows come before the torn line."""


class Stop(BaseException):
    """A stop the program cannot catch, as a crash or a SIGKILL is."""


def make_deployment(deployment_directory: Path) -> AppConfig:
    """Write an invented deployment in ``deployment_directory``; give its settings."""
    for directory_name in ("das", "steering", "processed"):
        (deployment_directory / directory_name).mkdir(parents=True)
    (deployment_directory / "clock_config.yaml").write_text(
        CLOCK_CONFIG_YAML, encoding="utf-8"
    )
    das_lines_by_day: dict[int, list[str]] = {}
    for epoch_index in range(EPOCH_COUNT):
        epoch_start = FIRST_EPOCH_START + epoch_index * T
        for pair_index, (reference, clock, first_phase, rate_ps_per_100_s) in enumerate(
            MEASURED_PAIRS
        ):
            seconds_into_epoch = 20 + 10 * pair_index
            das_measurement = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(
                        epoch_start + timedelta(seconds=seconds_into_epoch)
                    ),
                    6,
                ),
                measured_phase=(
                    first_phase
                    + rate_ps_per_100_s
                    * (epoch_index * 600 + seconds_into_epoch)
                    // 100
                )
                % 200_000,
                rms=3,
                switch=f"{reference[-1]}A{pair_index:02d}",
                clock=clock,
            )
            das_lines_by_day.setdefault(int(datetime_to_mjd(epoch_start)), []).append(
                f"{das_measurement}\n"
            )
    for data_day, das_lines in das_lines_by_day.items():
        (deployment_directory / "das" / f"cd5m5m_{data_day}.dat").write_text(
            "".join(das_lines), encoding="ascii"
        )
    return AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": deployment_directory / "das",
                "steering_path": deployment_directory / "steering",
            },
            "processed": {
                "processed_path": deployment_directory / "processed",
                "start_from_mjd": datetime_to_mjd(FIRST_EPOCH_START),
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


def run_once(config: AppConfig, steps: int | None = None) -> None:
    """Run the channel once."""
    clock_config = read_clock_config(config.processed.clock_config_file)
    run.run(config, clock_config, steps, ShutdownHandler())


@pytest.fixture(scope="module")
def uninterrupted_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Give the data files of a run that never stops."""
    config = make_deployment(tmp_path_factory.mktemp("uninterrupted"))
    run_once(config)
    return archived_files(config)


# --------------------------------------------------- errors before any write

FAILURE_POINTS: Final[tuple[tuple[object, str], ...]] = (
    (run, "build_epoch"),
    (run, "process_pairs"),
    (run, "process_triples"),
    (files.DayBuffer, "add"),
    (files, "_check_existing"),
    (files, "_check_new"),
    (files, "_check_space"),
)
"""Points of computing an epoch and preparing a write, where an error is raised."""


def failing_at(
    patched_object: object, attribute_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make ``patched_object.attribute_name`` raise a DataFileError when called."""

    def raise_injected(*_args: object, **_kwargs: object) -> None:
        """Raise as a failure there would."""
        message = f"injected at {attribute_name}"
        raise DataFileError(message)

    monkeypatch.setattr(patched_object, attribute_name, raise_injected)


@pytest.mark.parametrize(("patched_object", "attribute_name"), FAILURE_POINTS)
def test_an_error_while_computing_or_preparing_changes_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted_files: dict[str, bytes],
    patched_object: object,
    attribute_name: str,
) -> None:
    """Leave every file as it was; the next run gives the uninterrupted files (U21).

    The next run is left out after the new-file check fails, since the
    reading added for that check changes what the run writes.
    """
    config = make_deployment(tmp_path)
    run_once(config, steps=4)
    files_before = archived_files(config)
    assert files_before
    with monkeypatch.context() as scoped_monkeypatch:
        failing_at(patched_object, attribute_name, scoped_monkeypatch)
        if attribute_name == "_check_new":
            (tmp_path / "das" / "cd5m5m_60942.dat").write_text(
                (tmp_path / "das" / "cd5m5m_60942.dat").read_text()
                + str(
                    DASMeasurement(
                        measurement_mjd=round(
                            datetime_to_mjd(FIRST_EPOCH_START + 6 * T) + 0.0009, 6
                        ),
                        measured_phase=1,
                        rms=3,
                        switch="2A09",
                        clock="hm2",
                    )
                )
                + "\n"
            )
        with pytest.raises(DataFileError, match="injected"):
            run_once(config)
    assert archived_files(config) == files_before
    if attribute_name != "_check_new":
        run_once(config)
        assert archived_files(config) == uninterrupted_files


# ------------------------------------------------------- stops while writing


class WriteEvents:
    """Count the write step's writes and fsyncs, and stop at one of them."""

    def __init__(
        self, stop_event: int | None, *, stop_before: bool, tear: Tear | None
    ) -> None:
        """Stop at event ``stop_event``, before or after it, tearing a write."""
        self.event_count = 0
        self.stop_event = stop_event
        self.stop_before = stop_before
        self.tear = tear

    def event(
        self,
        write_or_fsync: Callable[[], None],
        partial_write: Callable[[Tear], None] | None = None,
    ) -> None:
        """Do one write or fsync, or stop there."""
        event_index = self.event_count
        self.event_count += 1
        if event_index != self.stop_event:
            write_or_fsync()
            return
        if self.stop_before:
            if self.tear is not None and partial_write is not None:
                partial_write(self.tear)
            raise Stop
        write_or_fsync()
        raise Stop


def route_writes_through(
    monkeypatch: pytest.MonkeyPatch, write_events: WriteEvents
) -> None:
    """Route the write step's writes and fsyncs through ``write_events``."""
    real_open, real_fsync = Path.open, os.fsync

    class RoutedFile:
        """A data file opened for the write step."""

        def __init__(self, wrapped_file: object) -> None:
            """Wrap ``wrapped_file``."""
            self.wrapped_file = wrapped_file

        def __enter__(self) -> RoutedFile:
            """Enter the wrapped file."""
            self.wrapped_file.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *exit_details: object) -> None:
            """Close the wrapped file."""
            self.wrapped_file.__exit__(*exit_details)  # type: ignore[attr-defined]

        def write(self, file_bytes: bytes) -> None:
            """Write, or stop part way."""
            write_events.event(
                lambda: self.wrapped_file.write(file_bytes),  # type: ignore[attr-defined]
                lambda tear: self.wrapped_file.write(  # type: ignore[attr-defined]
                    file_bytes[: TORN_BYTE_COUNTS[tear](len(file_bytes))]
                ),
            )

        def flush(self) -> None:
            """Flush the wrapped file."""
            self.wrapped_file.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            """Give the wrapped file's descriptor."""
            return self.wrapped_file.fileno()  # type: ignore[attr-defined, no-any-return]

    def open_routed(
        opened_path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        """Open a file, routing a data file opened to append or create."""
        wrapped_file = real_open(opened_path, mode, *args, **kwargs)  # type: ignore[call-overload]
        return RoutedFile(wrapped_file) if mode in {"ab", "xb"} else wrapped_file

    def fsync_or_stop(fd: int) -> None:
        """Flush a descriptor, or stop there."""
        write_events.event(lambda: real_fsync(fd))

    monkeypatch.setattr(Path, "open", open_routed)
    monkeypatch.setattr(os, "fsync", fsync_or_stop)


def test_counting_the_write_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count writes and fsyncs of a run, so every one of them is stopped at below."""
    config = make_deployment(tmp_path)
    write_events = WriteEvents(None, stop_before=False, tear=None)
    with monkeypatch.context() as scoped_monkeypatch:
        route_writes_through(scoped_monkeypatch, write_events)
        run_once(config)
    assert write_events.event_count == WRITE_EVENTS


WRITE_EVENTS: Final = 52
"""How many writes and fsyncs a run of the invented data makes.

The processed directory is flushed once, after the two archive directories
are made in it. The first of the two writes, one per day, writes and
flushes the journal and its directory, and each writes every data file.
The final write then flushes every data file the run wrote and the
directories of the files it created, and flushes the journal's directory
after deleting it.
"""


def every_stop() -> Iterator[tuple[int, bool, Tear | None]]:
    """Give every stop: each event, before it (torn each way or not) and after it."""
    for event_index in range(WRITE_EVENTS):
        yield event_index, True, None
        for tear in TORN_BYTE_COUNTS:
            yield event_index, True, tear
        yield event_index, False, None


@pytest.mark.parametrize(("stop_event", "stop_before", "tear"), list(every_stop()))
def test_a_stop_while_writing_is_undone_by_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted_files: dict[str, bytes],
    stop_event: int,
    stop_before: bool,
    tear: Tear | None,
) -> None:
    """Give the uninterrupted run's files after a stop anywhere in the write (U21)."""
    config = make_deployment(tmp_path)
    with monkeypatch.context() as scoped_monkeypatch:
        route_writes_through(
            scoped_monkeypatch,
            WriteEvents(stop_event, stop_before=stop_before, tear=tear),
        )
        with pytest.raises(Stop):
            run_once(config)
    run_once(config)
    assert archived_files(config) == uninterrupted_files


def test_rows_a_power_failure_loses_before_the_final_write_are_made_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted_files: dict[str, bytes],
) -> None:
    """Cut every file back to the run's first epoch when a file lost its last rows.

    The day's write is not flushed, so a power failure before the final
    write can leave one file without rows the others kept; the journal,
    flushed at the run's first write, makes the next run redo the whole run.
    """
    config = make_deployment(tmp_path)
    run_once(config, steps=2)
    synced_files = archived_files(config)
    with monkeypatch.context() as scoped_monkeypatch:

        def stop(_day_buffer: files.DayBuffer) -> None:
            """Stop as a power failure would, before anything is flushed."""
            raise Stop

        scoped_monkeypatch.setattr(run, "write_final", stop)
        with pytest.raises(Stop):
            run_once(config)
    lost_file = config.processed.processed_path / "meas" / "das_a.mc1.hm1.dat"
    lost_key = "meas/das_a.mc1.hm1.dat"
    assert archived_files(config)[lost_key] != synced_files[lost_key]
    lost_file.write_bytes(synced_files[lost_key])
    run_once(config)
    assert archived_files(config) == uninterrupted_files


# ---------------------------------------------------------- damage by hand


@pytest.mark.parametrize("file_kind", ["meas", "ddiff"])
def test_a_line_damaged_by_hand_cuts_every_file_and_the_run_is_redone(
    tmp_path: Path, uninterrupted_files: dict[str, bytes], file_kind: str
) -> None:
    """Cut every file before a damaged line; give the uninterrupted files (U21, U26).

    The damaged file is torn at its end too, so the file check finds it.
    """
    config = make_deployment(tmp_path)
    epochs_run, damaged_row = (5, 3) if file_kind == "meas" else (5, 1)
    run_once(config, steps=epochs_run)
    series_key = ("mc1", "hm1") if file_kind == "meas" else ("mc1", "mc1", "mc2")
    data_file = registry.series_file(config.processed.processed_path, "a", series_key)
    line_size = files.WIDTHS[file_kind] + 1  # type: ignore[index]
    header_size = len(files.header(file_kind))  # type: ignore[arg-type]
    file_bytes = bytearray(data_file.read_bytes())
    file_bytes[header_size + damaged_row * line_size + 5] = ord("#")
    data_file.write_bytes(bytes(file_bytes[:-line_size] + b"torn"))
    run_once(config)
    assert archived_files(config) == uninterrupted_files
