"""Tests for src/masterclock/das_processor/__init__.py.

The rules covered: das_processor's main runs the channel from its settings
and exits 0; a required setting given by neither source is a usage error,
the full help and exit 2, logged too when logging can start; logging
starts from its own settings first, so a setting or path that cannot be
used is logged at ERROR with exit 1, printed on standard error instead
only when it keeps logging from starting or logging is set to None; any
later
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

FIRST_EPOCH_START: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch of the runs below."""

T: Final = timedelta(minutes=10)
"""One epoch."""

CLOCK_CONFIG_YAML: Final = (
    "rejects_before_restart: 6\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 40}\n"
    "  mc: {filter_states: 1, scale_time_constant: 30.0,"
    " initial_innovation_scale: 2.0, gap_limit: 40}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  ox23: [{type: maser}]\n"
)
"""An invented clock configuration."""


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Put the root logger back as it was after each test."""
    root_logger = logging.getLogger()
    saved_level, saved_handlers = root_logger.level, list(root_logger.handlers)
    yield
    for handler in root_logger.handlers:
        if handler not in saved_handlers:
            root_logger.removeHandler(handler)
    root_logger.setLevel(saved_level)


def make_deployment(tmp_path: Path, epoch_count: int = 4) -> list[str]:
    """Make an invented deployment with DAS data, and give its command line."""
    for directory_name in ("das", "steering", "processed"):
        (tmp_path / directory_name).mkdir()
    (tmp_path / "clock_config.yaml").write_text(CLOCK_CONFIG_YAML, encoding="utf-8")
    das_lines = []
    for epoch_index in range(epoch_count):
        epoch_start_mjd = datetime_to_mjd(FIRST_EPOCH_START + epoch_index * T)
        for clock_index, (clock_name, measured_phase) in enumerate(
            (("mc1", 1000), ("ox23", 50_000))
        ):
            das_measurement = DASMeasurement(
                measurement_mjd=round(epoch_start_mjd + (clock_index + 1) * 2e-5, 6),
                measured_phase=measured_phase,
                rms=3,
                switch=f"1A0{clock_index + 1}",
                clock=clock_name,
            )
            das_lines.append(f"{das_measurement}\n")
    data_day = int(datetime_to_mjd(FIRST_EPOCH_START))
    (tmp_path / "das" / f"cd5m5m_{data_day}.dat").write_text(
        "".join(das_lines), encoding="ascii"
    )
    return [
        "--rf", "a",
        "--cd5m5m-path", str(tmp_path / "das"),
        "--steering-path", str(tmp_path / "steering"),
        "--processed-path", str(tmp_path / "processed"),
        "--clock-config-file", str(tmp_path / "clock_config.yaml"),
        "--start-from-mjd", f"{datetime_to_mjd(FIRST_EPOCH_START):.6f}",
        "--log-file", "None",
        "--log-level", "INFO",
        "--backup-count", "None",
    ]  # fmt: skip


def archived_files(tmp_path: Path) -> dict[str, bytes]:
    """Give every data file's bytes by name."""
    return {
        data_file.name: data_file.read_bytes()
        for data_file in sorted((tmp_path / "processed").rglob("das_a.*.dat"))
    }


def test_a_run_exits_zero_and_writes_the_archives(tmp_path: Path) -> None:
    """Run the channel from its settings to the end of the data, exit 0."""
    argv = make_deployment(tmp_path)
    assert das_processor.main(argv) == 0
    assert sorted(archived_files(tmp_path)) == [
        "das_a.mc1.mc1.dat",
        "das_a.mc1.mc1.mc1.dat",
        "das_a.mc1.mc1.ox23.dat",
        "das_a.mc1.ox23.dat",
    ]


def test_a_missing_setting_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the full help and exit 2 for a setting given by neither source."""
    argv = make_deployment(tmp_path)
    rf_index = argv.index("--rf")
    with pytest.raises(SystemExit) as system_exit:
        das_processor.main(argv[:rf_index] + argv[rf_index + 2 :])
    assert system_exit.value.code == 2
    stderr_text = capsys.readouterr().err
    assert "usage:" in stderr_text
    assert "--rf" in stderr_text


def test_a_setting_that_cannot_be_used_is_logged_and_exits_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a bad path at ERROR, logging already started, and exit 1."""
    argv = make_deployment(tmp_path)
    argv[argv.index("--cd5m5m-path") + 1] = str(tmp_path / "missing")
    assert das_processor.main(argv) == 1
    error_messages = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "ERROR"
    ]
    assert len(error_messages) == 1
    assert error_messages[0].startswith("[DAS] cd5m5m_path: ")


def test_with_logging_silenced_a_bad_setting_is_still_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the error on standard error when the log level None logs nothing."""
    argv = make_deployment(tmp_path)
    argv[argv.index("--cd5m5m-path") + 1] = str(tmp_path / "missing")
    argv[argv.index("--log-level") + 1] = "None"
    assert das_processor.main(argv) == 1
    assert capsys.readouterr().err.startswith("das_processor: error: [DAS] cd5m5m_path")


def test_a_logging_setting_that_cannot_be_used_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print a log file that cannot be opened on standard error: nothing can log."""
    argv = make_deployment(tmp_path)
    (tmp_path / "a file").write_text("")
    argv[argv.index("--log-file") + 1] = str(tmp_path / "a file" / "run.log")
    assert das_processor.main(argv) == 1
    assert capsys.readouterr().err.startswith("das_processor: error: ")


def test_a_usage_error_is_logged_when_logging_can_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a setting neither source gives at ERROR, as well as print the help."""
    argv = make_deployment(tmp_path)
    rf_index = argv.index("--rf")
    with pytest.raises(SystemExit):
        das_processor.main(argv[:rf_index] + argv[rf_index + 2 :])
    error_messages = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "ERROR"
    ]
    assert error_messages == [
        "these settings must be provided by the config file or the command line:"
        " [DAS] rf (--rf)"
    ]


def test_an_error_during_the_run_exits_one_logged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Exit 1 for a MasterClockError, logged once, where it was raised."""
    argv = make_deployment(tmp_path)
    (tmp_path / "clock_config.yaml").write_text(
        CLOCK_CONFIG_YAML.replace("ox23", "hm9")
    )
    assert das_processor.main(argv) == 1
    error_records = [
        log_record for log_record in caplog.records if log_record.levelname == "ERROR"
    ]
    assert len(error_records) == 1
    assert "ox23" in error_records[0].getMessage()


def test_a_second_run_of_the_channel_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse to run while another run holds the channel's lock."""
    argv = make_deployment(tmp_path)
    with RunLock(tmp_path / "processed", "das_processor_a.lock"):
        assert das_processor.main(argv) == 1
    assert archived_files(tmp_path) == {}
    assert any(
        "lock" in log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "ERROR"
    )


def test_a_redo_deletes_the_rows_from_its_epoch_before_the_run(tmp_path: Path) -> None:
    """Redo from an MJD before running, so the rows are computed again alike (6.5)."""
    argv = make_deployment(tmp_path)
    assert das_processor.main(argv) == 0
    files_before_redo = archived_files(tmp_path)
    redo_argv = [
        *argv,
        "--redo-from-mjd",
        f"{datetime_to_mjd(FIRST_EPOCH_START + 2 * T):.6f}",
    ]
    assert das_processor.main(redo_argv) == 0
    assert archived_files(tmp_path) == files_before_redo


def test_logging_set_to_none_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log no record of a run whose log level is None."""
    argv = make_deployment(tmp_path)
    argv[argv.index("--log-level") + 1] = "None"
    with caplog.at_level(logging.DEBUG):
        assert das_processor.main(argv) == 0
    assert caplog.records == []


def test_the_steps_option_limits_the_run(tmp_path: Path) -> None:
    """Process --steps N epochs and stop."""
    argv = make_deployment(tmp_path)
    assert das_processor.main([*argv, "--steps", "2"]) == 0
    length_after_two_epochs = len(archived_files(tmp_path)["das_a.mc1.mc1.dat"])
    assert das_processor.main(argv) == 0
    assert len(archived_files(tmp_path)["das_a.mc1.mc1.dat"]) > length_after_two_epochs


def test_a_usage_error_names_the_missing_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End a usage error with the setting neither source gave, word for word."""
    argv = make_deployment(tmp_path)
    rf_index = argv.index("--rf")
    with pytest.raises(SystemExit):
        das_processor.main(argv[:rf_index] + argv[rf_index + 2 :])
    assert capsys.readouterr().err.endswith(
        "das_processor: error: these settings must be provided by the config file"
        " or the command line: [DAS] rf (--rf)\n"
    )


def test_the_logging_settings_reach_the_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start logging with the run's level, log file and backup count."""
    argv = make_deployment(tmp_path)
    argv[argv.index("--log-file") + 1] = str(tmp_path / "run.log")
    argv[argv.index("--backup-count") + 1] = "3"
    configure_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        das_processor,
        "configure_logging",
        lambda **kwargs: configure_calls.append(kwargs),
    )
    assert das_processor.main(argv) == 0
    assert configure_calls == [
        {
            "root_level": logging.INFO,
            "log_file": tmp_path / "run.log",
            "backup_count": 3,
        }
    ]


def test_a_redo_recomputes_its_rows_with_the_settings_now(tmp_path: Path) -> None:
    """Compute the rows from the redo's epoch again, with today's clock settings."""
    argv = make_deployment(tmp_path)
    assert das_processor.main(argv) == 0
    file_before_redo = archived_files(tmp_path)["das_a.mc1.ox23.dat"]
    (tmp_path / "clock_config.yaml").write_text(
        CLOCK_CONFIG_YAML.replace("time_constant: 100.0", "time_constant: 50.0"),
        encoding="utf-8",
    )
    redo_argv = [
        *argv,
        "--redo-from-mjd",
        f"{datetime_to_mjd(FIRST_EPOCH_START + 2 * T):.6f}",
    ]
    assert das_processor.main(redo_argv) == 0
    file_after_redo = archived_files(tmp_path)["das_a.mc1.ox23.dat"]
    assert len(file_after_redo) == len(file_before_redo)
    assert file_after_redo != file_before_redo
    line_size = files.MEAS_WIDTH + 1
    kept_length = (files.MEAS_HEADER_LINES + 2) * line_size
    assert file_after_redo[:kept_length] == file_before_redo[:kept_length]


def test_a_logging_setting_left_out_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the full help and exit 2 for a logging setting neither source gives."""
    argv = make_deployment(tmp_path)
    log_level_index = argv.index("--log-level")
    with pytest.raises(SystemExit) as system_exit:
        das_processor.main(argv[:log_level_index] + argv[log_level_index + 2 :])
    assert system_exit.value.code == 2
    assert capsys.readouterr().err.endswith(
        "das_processor: error: these settings must be provided by the config file"
        " or the command line: [LOGGING] log_level (--log-level)\n"
    )
