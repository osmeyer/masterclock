"""Tests for src/masterclock/das_processor/__init__.py.

The rules covered: das_processor's main runs the channel from its settings
and exits 0; a required setting given by neither source is a usage error,
the full help and exit 2; a setting that cannot be used, found before
logging starts, is reported on standard error with exit 1; any later
MasterClockError exits 1, logged once where it was raised; a second run of
the same channel is refused by the run lock; a redo deletes the rows from
its epoch before the run, which computes them again; and logging set to
None logs nothing.

A usage error ends with the missing setting; the logging settings reach the
logging; and a redo computes its rows again with the settings in force now.
"""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

from masterclock import das_processor
from masterclock.app.lock import RunLock
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import files
from masterclock.das_processor.read_cd5m5m import DASMeasurement

START: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch of the runs below."""

T: Final = timedelta(minutes=10)
"""One epoch."""

CLOCKS: Final = (
    "rejects_before_restart: 6\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 40}\n"
    "  mc: {filter_states: 1, scale_time_constant: 30.0,"
    " initial_innovation_scale: 2.0, gap_limit: 40}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  nav23: [{type: maser}]\n"
)
"""An invented clock configuration."""


@pytest.fixture(autouse=True)
def root_logger() -> Iterator[None]:
    """Put the root logger back as it was after each test."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in root.handlers:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def deployment(tmp_path: Path, epochs: int = 4) -> list[str]:
    """Make an invented deployment with DAS data, and give its command line."""
    for name in ("das", "steering", "processed"):
        (tmp_path / name).mkdir()
    (tmp_path / "clock_config.yaml").write_text(CLOCKS, encoding="utf-8")
    lines = []
    for index in range(epochs):
        start = datetime_to_mjd(START + index * T)
        for offset, (clock, phase) in enumerate((("mc1", 1000), ("nav23", 50_000))):
            raw = DASMeasurement(
                measurement_mjd=round(start + (offset + 1) * 2e-5, 6),
                measured_phase=phase,
                rms=3,
                switch=f"1A0{offset + 1}",
                clock=clock,
            )
            lines.append(f"{raw}\n")
    day = int(datetime_to_mjd(START))
    (tmp_path / "das" / f"cd5m5m_{day}.dat").write_text(
        "".join(lines), encoding="ascii"
    )
    return [
        "--rf", "a",
        "--cd5m5m-path", str(tmp_path / "das"),
        "--steering-path", str(tmp_path / "steering"),
        "--processed-path", str(tmp_path / "processed"),
        "--clock-config-file", str(tmp_path / "clock_config.yaml"),
        "--start-from-mjd", f"{datetime_to_mjd(START):.6f}",
        "--log-file", "None",
        "--log-level", "INFO",
        "--backup-count", "None",
    ]  # fmt: skip


def archive(tmp_path: Path) -> dict[str, bytes]:
    """Give every data file's bytes by name."""
    return {
        path.name: path.read_bytes()
        for path in sorted((tmp_path / "processed").rglob("das_a.*.dat"))
    }


def test_a_run_exits_zero_and_writes_the_archives(tmp_path: Path) -> None:
    """Run the channel from its settings to the end of the data, exit 0."""
    argv = deployment(tmp_path)
    assert das_processor.main(argv) == 0
    assert sorted(archive(tmp_path)) == [
        "das_a.mc1.mc1.dat",
        "das_a.mc1.mc1.nav23.dat",
        "das_a.mc1.nav23.dat",
    ]


def test_a_missing_setting_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the full help and exit 2 for a setting given by neither source."""
    argv = deployment(tmp_path)
    index = argv.index("--rf")
    with pytest.raises(SystemExit) as stopped:
        das_processor.main(argv[:index] + argv[index + 2 :])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err
    assert "--rf" in err


def test_a_setting_that_cannot_be_used_exits_one_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a bad path on standard error, before logging starts, with exit 1."""
    argv = deployment(tmp_path)
    argv[argv.index("--cd5m5m-path") + 1] = str(tmp_path / "missing")
    assert das_processor.main(argv) == 1
    assert "das_processor: error:" in capsys.readouterr().err


def test_an_error_during_the_run_exits_one_logged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Exit 1 for a MasterClockError, logged once, where it was raised."""
    argv = deployment(tmp_path)
    (tmp_path / "clock_config.yaml").write_text(CLOCKS.replace("nav23", "hm9"))
    assert das_processor.main(argv) == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "nav23" in errors[0].getMessage()


def test_a_second_run_of_the_channel_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse to run while another run holds the channel's lock."""
    argv = deployment(tmp_path)
    with RunLock(tmp_path / "processed", "das_processor_a.lock"):
        assert das_processor.main(argv) == 1
    assert archive(tmp_path) == {}
    assert any(
        "lock" in r.getMessage() for r in caplog.records if r.levelname == "ERROR"
    )


def test_a_redo_deletes_the_rows_from_its_epoch_before_the_run(tmp_path: Path) -> None:
    """Redo from an MJD before running, so the rows are computed again alike (6.5)."""
    argv = deployment(tmp_path)
    assert das_processor.main(argv) == 0
    before = archive(tmp_path)
    redo = [*argv, "--redo-from-mjd", f"{datetime_to_mjd(START + 2 * T):.6f}"]
    assert das_processor.main(redo) == 0
    assert archive(tmp_path) == before


def test_logging_set_to_none_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log no record of a run whose log level is None."""
    argv = deployment(tmp_path)
    argv[argv.index("--log-level") + 1] = "None"
    with caplog.at_level(logging.DEBUG):
        assert das_processor.main(argv) == 0
    assert caplog.records == []


def test_the_steps_option_limits_the_run(tmp_path: Path) -> None:
    """Process --steps N epochs and stop."""
    argv = deployment(tmp_path)
    assert das_processor.main([*argv, "--steps", "2"]) == 0
    size = len(archive(tmp_path)["das_a.mc1.mc1.dat"])
    assert das_processor.main(argv) == 0
    assert len(archive(tmp_path)["das_a.mc1.mc1.dat"]) > size


def test_a_usage_error_names_the_missing_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End a usage error with the setting neither source gave, word for word."""
    argv = deployment(tmp_path)
    index = argv.index("--rf")
    with pytest.raises(SystemExit):
        das_processor.main(argv[:index] + argv[index + 2 :])
    assert capsys.readouterr().err.endswith(
        "das_processor: error: these settings must be provided by the config file"
        " or the command line: [DAS] rf (--rf)\n"
    )


def test_the_logging_settings_reach_the_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start logging with the run's level, log file and backup count."""
    argv = deployment(tmp_path)
    argv[argv.index("--log-file") + 1] = str(tmp_path / "run.log")
    argv[argv.index("--backup-count") + 1] = "3"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        das_processor, "configure_logging", lambda **kwargs: calls.append(kwargs)
    )
    assert das_processor.main(argv) == 0
    assert calls == [
        {"level": logging.INFO, "log_file": tmp_path / "run.log", "backup_count": 3}
    ]


def test_a_redo_recomputes_its_rows_with_the_settings_now(tmp_path: Path) -> None:
    """Compute the rows from the redo's epoch again, with today's clock settings."""
    argv = deployment(tmp_path)
    assert das_processor.main(argv) == 0
    before = archive(tmp_path)["das_a.mc1.nav23.dat"]
    (tmp_path / "clock_config.yaml").write_text(
        CLOCKS.replace("time_constant: 100.0", "time_constant: 50.0"), encoding="utf-8"
    )
    redo = [*argv, "--redo-from-mjd", f"{datetime_to_mjd(START + 2 * T):.6f}"]
    assert das_processor.main(redo) == 0
    after = archive(tmp_path)["das_a.mc1.nav23.dat"]
    assert len(after) == len(before)
    assert after != before
    size = files.MEAS_WIDTH + 1
    kept = (files.MEAS_HEADER_LINES + 2) * size
    assert after[:kept] == before[:kept]
