"""Tests that das_processor recovers from any failure byte for byte (U21, I8).

The rules covered: an error at any point of computing an epoch or
preparing a write changes no file; a stop at any point of the write, before
or after any write or fsync, a torn line included, is undone by the next
run, whose data files end byte-identical to those of a run that never
stopped; and a line damaged by hand in the middle of a file is redone from
its epoch, with the same result.
"""

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import pytest

from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import files, run
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import DASMeasurement

START: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch: four before midnight, so the run writes twice."""

T: Final = timedelta(minutes=10)
"""One epoch."""

EPOCHS: Final = 7
"""How many epochs the invented data cover."""

CLOCKS: Final = (
    "rejects_before_restart: 4\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 10.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  mc: {filter_states: 1, scale_time_constant: 5.0,"
    " initial_innovation_scale: 3.0, gap_limit: 8}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  mc2: [{type: mc}]\n"
    "  hm1: [{type: maser}]\n"
    "  hm2: [{type: maser}]\n"
)
"""An invented clock configuration."""

PAIRS: Final = (
    ("mc1", "mc1", 100, 0),
    ("mc1", "mc2", 5_000, 1),
    ("mc2", "mc1", 195_000, -1),
    ("mc2", "mc2", 200, 0),
    ("mc1", "hm1", 70_000, 25),
)
"""Each pair measured: reference, clock, first phase, rate in ps per 100 s."""


type Tear = Literal["first row", "half"]
"""Where a write stopped part way is torn."""

TEARS: Final[dict[Tear, Callable[[int], int]]] = {
    "first row": lambda _length: 7,
    "half": lambda length: length // 2 + 7,
}
"""How many bytes of a torn write reach the file: a few, inside the first
line, so a file that held rows ends in a torn line after them; or about half,
so whole new rows come before the torn line."""


class Stop(BaseException):
    """A stop the program cannot catch, as a crash or a SIGKILL is."""


def deployment(directory: Path) -> AppConfig:
    """Write an invented deployment in ``directory``, and give its settings."""
    for name in ("das", "steering", "processed"):
        (directory / name).mkdir(parents=True)
    (directory / "clock_config.yaml").write_text(CLOCKS, encoding="utf-8")
    days: dict[int, list[str]] = {}
    for index in range(EPOCHS):
        mark = START + index * T
        for slot, (reference, clock, phase, rate) in enumerate(PAIRS):
            offset = 20 + 10 * slot
            raw = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(mark + timedelta(seconds=offset)), 6
                ),
                measured_phase=(phase + rate * (index * 600 + offset) // 100) % 200_000,
                rms=3,
                switch=f"{reference[-1]}A{slot:02d}",
                clock=clock,
            )
            days.setdefault(int(datetime_to_mjd(mark)), []).append(f"{raw}\n")
    for day, lines in days.items():
        (directory / "das" / f"cd5m5m_{day}.dat").write_text(
            "".join(lines), encoding="ascii"
        )
    return AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": directory / "das",
                "steering_path": directory / "steering",
            },
            "processed": {
                "processed_path": directory / "processed",
                "redo_from_mjd": None,
                "start_from_mjd": datetime_to_mjd(START),
                "clock_config_file": directory / "clock_config.yaml",
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )


def archive(config: AppConfig) -> dict[str, bytes]:
    """Give every data file's bytes, by its path under the processed directory."""
    root = config.processed.processed_path
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("das_a.*.dat"))
    }


def run_once(config: AppConfig, steps: int | None = None) -> None:
    """Run the channel once."""
    clocks = read_clock_config(config.processed.clock_config_file)
    run.run(config, clocks, steps, ShutdownHandler())


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """Give the data files of a run that never stops."""
    config = deployment(tmp_path_factory.mktemp("uninterrupted"))
    run_once(config)
    return archive(config)


# --------------------------------------------------- errors before any write

POINTS: Final[tuple[tuple[object, str], ...]] = (
    (run, "build_epoch"),
    (run, "process_pairs"),
    (run, "process_triples"),
    (files.DayBuffer, "add"),
    (files, "_check_existing"),
    (files, "_check_new"),
    (files, "_check_space"),
)
"""Points of computing an epoch and preparing a write, where an error is raised."""


def failing_at(owner: object, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``owner.name`` raise a MasterClockError when called."""

    def fail(*_args: object, **_kwargs: object) -> None:
        """Raise as a failure there would."""
        message = f"injected at {name}"
        raise DataFileError(message)

    monkeypatch.setattr(owner, name, fail)


@pytest.mark.parametrize(("owner", "name"), POINTS)
def test_an_error_while_computing_or_preparing_changes_no_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
    owner: object,
    name: str,
) -> None:
    """Leave every file as it was, then let the next run finish alike (U21)."""
    config = deployment(tmp_path)
    run_once(config, steps=4)
    before = archive(config)
    assert before
    with monkeypatch.context() as patch:
        failing_at(owner, name, patch)
        if name == "_check_new":
            (tmp_path / "das" / "cd5m5m_60942.dat").write_text(
                (tmp_path / "das" / "cd5m5m_60942.dat").read_text()
                + str(
                    DASMeasurement(
                        measurement_mjd=round(
                            datetime_to_mjd(START + 6 * T) + 0.0009, 6
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
    assert archive(config) == before
    if name != "_check_new":
        run_once(config)
        assert archive(config) == uninterrupted


# ------------------------------------------------------- stops while writing


class Writes:
    """Count the write step's writes and fsyncs, and stop at one of them."""

    def __init__(self, stop_at: int | None, *, before: bool, torn: Tear | None) -> None:
        """Stop at event ``stop_at``, before or after it, tearing a write."""
        self.count = 0
        self.stop_at = stop_at
        self.before = before
        self.torn = torn

    def event(
        self, do: Callable[[], None], part: Callable[[Tear], None] | None = None
    ) -> None:
        """Do one write or fsync, or stop there."""
        here = self.count
        self.count += 1
        if here != self.stop_at:
            do()
            return
        if self.before:
            if self.torn is not None and part is not None:
                part(self.torn)
            raise Stop
        do()
        raise Stop


def stopping(monkeypatch: pytest.MonkeyPatch, writes: Writes) -> None:
    """Route the write step's writes and fsyncs through ``writes``."""
    real_open, real_fsync = Path.open, os.fsync

    class File:
        """A data file opened for the write step."""

        def __init__(self, inner: object) -> None:
            """Wrap ``inner``."""
            self.inner = inner

        def __enter__(self) -> File:
            """Enter the wrapped file."""
            self.inner.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *details: object) -> None:
            """Close the wrapped file."""
            self.inner.__exit__(*details)  # type: ignore[attr-defined]

        def write(self, data: bytes) -> None:
            """Write, or stop part way."""
            writes.event(
                lambda: self.inner.write(data),  # type: ignore[attr-defined]
                lambda tear: self.inner.write(data[: TEARS[tear](len(data))]),  # type: ignore[attr-defined]
            )

        def flush(self) -> None:
            """Flush the wrapped file."""
            self.inner.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            """Give the wrapped file's descriptor."""
            return self.inner.fileno()  # type: ignore[attr-defined, no-any-return]

    def opened(path: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        """Open a file, routing a data file opened to append or create."""
        inner = real_open(path, mode, *args, **kwargs)  # type: ignore[call-overload]
        return File(inner) if mode in {"ab", "xb"} else inner

    def synced(fd: int) -> None:
        """Flush a descriptor, or stop there."""
        writes.event(lambda: real_fsync(fd))

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(os, "fsync", synced)


def test_counting_the_write_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count writes and fsyncs of a run, so every one of them is stopped at below."""
    config = deployment(tmp_path)
    writes = Writes(None, before=False, torn=None)
    with monkeypatch.context() as patch:
        stopping(patch, writes)
        run_once(config)
    assert writes.count == WRITE_EVENTS


WRITE_EVENTS: Final = 39
"""How many writes and fsyncs a run of the invented data makes.

The archives' directories are flushed once when made. Each of the two
writes, one per day, writes and flushes the journal and its directory,
writes and flushes the 7 files, flushes the directories of the files it
created (both, the first day), and flushes the journal's directory after
deleting it.
"""


def stops() -> Iterator[tuple[int, bool, Tear | None]]:
    """Give every stop: each event, before it (torn each way or not) and after it."""
    for at in range(WRITE_EVENTS):
        yield at, True, None
        for tear in TEARS:
            yield at, True, tear
        yield at, False, None


@pytest.mark.parametrize(("at", "before", "torn"), list(stops()))
def test_a_stop_while_writing_is_undone_by_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
    at: int,
    before: bool,
    torn: Tear | None,
) -> None:
    """Give the uninterrupted run's files after a stop anywhere in the write (U21)."""
    config = deployment(tmp_path)
    with monkeypatch.context() as patch:
        stopping(patch, Writes(at, before=before, torn=torn))
        with pytest.raises(Stop):
            run_once(config)
    run_once(config)
    assert archive(config) == uninterrupted


# ---------------------------------------------------------- damage by hand


@pytest.mark.parametrize("kind", ["meas", "ddiff"])
def test_a_line_damaged_by_hand_is_redone_from_its_epoch(
    tmp_path: Path, uninterrupted: dict[str, bytes], kind: str
) -> None:
    """Redo every file from a damaged line's epoch, with the same result (U21, U26)."""
    config = deployment(tmp_path)
    run_once(config)
    name = "das_a.mc1.hm1.dat" if kind == "meas" else "das_a.mc1.mc1.hm1.dat"
    path = config.processed.processed_path / kind / name
    size = files.WIDTHS[kind] + 1  # type: ignore[index]
    data = bytearray(path.read_bytes())
    line = files.HEADER_LINES[kind] + 3  # type: ignore[index]
    data[line * size + 5] = ord("#")
    path.write_bytes(bytes(data[:-size] + b"torn"))
    run_once(config)
    assert archive(config) == uninterrupted
