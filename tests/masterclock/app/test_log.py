"""Tests for src/masterclock/app/log.py.

The rules covered: importing the module registers the TRACE level and the
project's logger class; every record is written on one line, stamped with
its UTC time and MJD, except for a traceback or a stack; trace and todo log
at their levels, with no stack unless one is asked for, take the keywords
the standard methods take, and name their real caller, a check that does
not run under mutmut, whose wrapper around each function adds a frame;
get_logger refuses a logger of the wrong class, the root logger included;
configure_logging attaches a stream handler, and a file handler that rotates
at midnight UTC in a directory it makes when missing, and replaces what its
previous call attached; and a log file that cannot be made raises OSError
with the new stream handler still attached.
"""

import io
import logging
import os
import re
import sys
from collections.abc import Iterator
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import log
from masterclock.app.exceptions import LoggingError
from masterclock.app.timeutil import unix_to_mjd

# 2026-07-12 03:14:15.926 UTC, an invented instant.
RECORD_CREATED_UNIX: Final = 1_783_826_055.926
STAMP_PATTERN: Final = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} UTC, MJD \d{5}\.\d{6}"

not_under_mutmut: Final = pytest.mark.skipif(
    "MUTANT_UNDER_TEST" in os.environ,
    reason="mutmut puts a function around each function it changes, which"
    " adds a frame, so the caller a record names is not the test",
)
"""Skip a test of the caller a record names while mutmut runs."""


@pytest.fixture(autouse=True)
def restore_root() -> Iterator[None]:
    """Put the root logger's handlers and level back after each test."""
    root_logger = logging.getLogger()
    saved_handlers, saved_level = list(root_logger.handlers), root_logger.level
    yield
    for active_handler in log._active_handlers:
        root_logger.removeHandler(active_handler)
        active_handler.close()
    log._active_handlers.clear()
    root_logger.handlers[:] = saved_handlers
    root_logger.setLevel(saved_level)


class KeepingHandler(logging.Handler):
    """Keep every record it is given, for a test to look at."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


def info_record(
    message_text: str, created_unix: float = RECORD_CREATED_UNIX
) -> logging.LogRecord:
    """Return an INFO record with the given message and creation time."""
    log_record = logging.LogRecord(
        "invented.name", logging.INFO, "f.py", 1, message_text, None, None
    )
    log_record.created = created_unix
    return log_record


def test_importing_registers_trace_and_the_logger_class() -> None:
    """Name level 5 TRACE and make new loggers MasterClockLoggers."""
    assert log.TRACE == 5
    assert logging.getLevelName(log.TRACE) == "TRACE"
    assert logging.getLoggerClass() is log.MasterClockLogger


@pytest.mark.parametrize(
    ("message_text", "folded_text"),
    [
        ("already one line", "already one line"),
        (
            "1 error\n  paths.output\n    Input should be 'a'",
            "1 error; paths.output; Input should be 'a'",
        ),
        ("a\r\nb\rc\x0bd\x0ce\u2028f", "a; b; c; d; e; f"),
        ("\n  first\n\n   \nsecond  \n", "first; second"),
        ("", ""),
        ("  \n \n", ""),
    ],
)
def test_one_line_folds_known_texts(message_text: str, folded_text: str) -> None:
    """Strip each line, drop blank ones, and join the rest with '; '."""
    assert log.one_line(message_text) == folded_text


@given(st.text())
def test_one_line_always_gives_one_line(message_text: str) -> None:
    """Give a result with no line breaks of any kind, whatever the text."""
    folded_text = log.one_line(message_text)
    assert len(folded_text.splitlines()) <= 1
    assert folded_text == folded_text.strip()


@given(st.text().map(str.strip).filter(lambda text: len(text.splitlines()) == 1))
def test_one_line_leaves_a_single_line_alone(message_text: str) -> None:
    """Return a single stripped line unchanged."""
    assert log.one_line(message_text) == message_text


def test_the_time_is_written_in_utc_with_its_mjd() -> None:
    """Write the creation time in UTC to the millisecond, then its MJD."""
    formatter = log.UtcMjdFormatter()
    assert formatter.formatTime(info_record("m")) == (
        "2026-07-12 03:14:15.926 UTC, MJD 61233.134907"
    )
    assert (
        formatter.formatTime(info_record("m"), "%H:%M") == "03:14 UTC, MJD 61233.134907"
    )
    assert f"{unix_to_mjd(RECORD_CREATED_UNIX):.6f}" == "61233.134907"


def test_milliseconds_are_truncated_not_rounded() -> None:
    """Drop the microseconds past the millisecond instead of rounding them."""
    formatter = log.UtcMjdFormatter()
    time_stamp = formatter.formatTime(info_record("m", created_unix=1_783_826_055.9269))
    assert time_stamp.startswith("2026-07-12 03:14:15.926 UTC")


def test_a_record_is_written_on_one_line() -> None:
    """Fold the whole rendered record, and leave the record itself unchanged."""
    formatter = log.UtcMjdFormatter(log.DEFAULT_FORMAT)
    log_record = info_record("two\n  lines\n\n  here")
    assert formatter.format(log_record) == (
        "2026-07-12 03:14:15.926 UTC, MJD 61233.134907 | INFO | invented.name:"
        " two; lines; here"
    )
    assert log_record.msg == "two\n  lines\n\n  here"


def test_a_traceback_keeps_its_lines() -> None:
    """Write the record on one line and the traceback after it, unfolded."""
    formatter = log.UtcMjdFormatter(log.DEFAULT_FORMAT)
    log_record = info_record("failed\nbadly")
    try:
        raise ValueError("invented failure")
    except ValueError:
        log_record.exc_info = sys.exc_info()
    output_lines = formatter.format(log_record).splitlines()
    assert output_lines[0].endswith("| INFO | invented.name: failed; badly")
    assert output_lines[1] == "Traceback (most recent call last):"
    assert output_lines[-1] == "ValueError: invented failure"


def test_get_logger_gives_the_project_logger() -> None:
    """Return a MasterClockLogger, the same one each time for a name."""
    project_logger = log.get_logger("tests.log.first")
    assert isinstance(project_logger, log.MasterClockLogger)
    assert log.get_logger("tests.log.first") is project_logger


def test_get_logger_refuses_a_logger_of_another_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise LoggingError for a name registered as a plain Logger."""
    monkeypatch.setitem(
        logging.Logger.manager.loggerDict,
        "tests.log.plain",
        logging.Logger("tests.log.plain"),
    )
    with pytest.raises(LoggingError) as raised:
        log.get_logger("tests.log.plain")
    assert str(raised.value) == (
        "logger 'tests.log.plain' is a Logger, not a MasterClockLogger"
    )


def logger_with_kept(logger_level: int) -> tuple[log.MasterClockLogger, KeepingHandler]:
    """Return a project logger at ``logger_level`` and the handler keeping records."""
    project_logger = log.get_logger("tests.log.levels")
    keeping_handler = KeepingHandler()
    project_logger.handlers[:] = [keeping_handler]
    project_logger.propagate = False
    project_logger.setLevel(logger_level)
    return project_logger, keeping_handler


def test_trace_logs_at_trace() -> None:
    """Log at TRACE, interpolating arguments, with no stack unless asked."""
    project_logger, keeping_handler = logger_with_kept(log.TRACE)
    project_logger.trace("value %s", 7)
    (log_record,) = keeping_handler.records
    assert log_record.levelno == log.TRACE
    assert log_record.levelname == "TRACE"
    assert log_record.getMessage() == "value 7"
    assert log_record.stack_info is None


def test_todo_logs_at_debug_with_its_prefix() -> None:
    """Log at DEBUG with 'TODO: ' in front, and no stack unless asked."""
    project_logger, keeping_handler = logger_with_kept(logging.DEBUG)
    project_logger.todo("finish %s", "this")
    (log_record,) = keeping_handler.records
    assert log_record.levelno == logging.DEBUG
    assert log_record.getMessage() == "TODO: finish this"
    assert log_record.stack_info is None


@not_under_mutmut
def test_trace_and_todo_name_their_caller() -> None:
    """Name the function that called trace or todo, and its file."""
    project_logger, keeping_handler = logger_with_kept(log.TRACE)
    project_logger.trace("traced")
    project_logger.todo("to do")
    assert [
        (log_record.funcName, log_record.pathname)
        for log_record in keeping_handler.records
    ] == [
        ("test_trace_and_todo_name_their_caller", __file__),
        ("test_trace_and_todo_name_their_caller", __file__),
    ]


@pytest.mark.parametrize("method_name", ["trace", "todo"])
def test_trace_and_todo_take_the_standard_keywords(method_name: str) -> None:
    """Pass exc_info, stack_info and extra through, as the standard methods do."""
    project_logger, keeping_handler = logger_with_kept(log.TRACE)
    log_method = getattr(project_logger, method_name)
    try:
        raise ValueError("invented failure")
    except ValueError:
        log_method("with traceback", exc_info=True)
    log_method("with stack", stack_info=True)
    log_method("with extra", extra={"invented_key": 3})
    traced, stacked, extended = keeping_handler.records
    assert traced.exc_info is not None
    assert traced.exc_info[0] is ValueError
    assert stacked.stack_info is not None
    assert stacked.stack_info.startswith("Stack (most recent call last):")
    assert extended.__dict__["invented_key"] == 3


def log_one_frame_up(project_logger: log.MasterClockLogger) -> None:
    """Log from here as though from the function that called this one."""
    project_logger.trace("reported", stacklevel=2)
    project_logger.todo("reported", stacklevel=2)


@not_under_mutmut
def test_trace_and_todo_take_a_stacklevel() -> None:
    """Name the caller the given number of frames up, as the standard methods do."""
    project_logger, keeping_handler = logger_with_kept(log.TRACE)
    log_one_frame_up(project_logger)
    assert [log_record.funcName for log_record in keeping_handler.records] == [
        "test_trace_and_todo_take_a_stacklevel",
        "test_trace_and_todo_take_a_stacklevel",
    ]


def test_get_logger_refuses_the_root_logger() -> None:
    """Raise LoggingError for the names of the standard root logger."""
    for root_name in ("root", ""):
        with pytest.raises(LoggingError, match="is a RootLogger"):
            log.get_logger(root_name)


def test_trace_and_todo_are_dropped_above_their_levels() -> None:
    """Write nothing for trace above TRACE, or for todo above DEBUG."""
    project_logger, keeping_handler = logger_with_kept(logging.DEBUG)
    project_logger.trace("hidden")
    project_logger.setLevel(logging.INFO)
    project_logger.todo("hidden")
    assert keeping_handler.records == []


def test_configure_logging_attaches_a_one_line_stream_handler() -> None:
    """Attach a stream handler with the project format, and set the level."""
    output_stream = io.StringIO()
    stream_handler = log.configure_logging(
        root_level=logging.WARNING, stream=output_stream
    )
    root_logger = logging.getLogger()
    assert stream_handler in root_logger.handlers
    assert stream_handler.stream is output_stream
    assert isinstance(stream_handler.formatter, log.UtcMjdFormatter)
    assert root_logger.level == logging.WARNING
    log.get_logger("tests.log.stream").warning("seen\n  twice")
    log.get_logger("tests.log.stream").info("not seen")
    assert re.fullmatch(
        STAMP_PATTERN + r" \| WARNING \| tests\.log\.stream: seen; twice\n",
        output_stream.getvalue(),
    )


def test_configure_logging_writes_to_standard_error_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use the standard error stream in force when no stream is given."""
    stream_handler = log.configure_logging()
    assert stream_handler.stream is sys.stderr
    log.get_logger("tests.log.stderr").info("to stderr")
    assert "| INFO | tests.log.stderr: to stderr" in capsys.readouterr().err


def test_configure_logging_replaces_its_own_handlers() -> None:
    """Remove and close the handlers of the previous call, and keep others."""
    root_logger = logging.getLogger()
    other_handler = KeepingHandler()
    root_logger.addHandler(other_handler)
    first_handler = log.configure_logging(stream=io.StringIO())
    second_handler = log.configure_logging(stream=io.StringIO())
    assert first_handler not in root_logger.handlers
    assert second_handler in root_logger.handlers
    assert other_handler in root_logger.handlers
    assert log._active_handlers == [second_handler]


def file_handlers() -> list[TimedRotatingFileHandler]:
    """Return the rotating file handlers on the root logger."""
    return [
        root_handler
        for root_handler in logging.getLogger().handlers
        if isinstance(root_handler, TimedRotatingFileHandler)
    ]


@pytest.mark.parametrize(("backup_count", "kept_backups"), [(None, 0), (0, 0), (7, 7)])
def test_the_log_file_rotates_at_midnight_utc(
    tmp_path: Path, backup_count: int | None, kept_backups: int
) -> None:
    """Create the missing directories and rotate daily at midnight UTC."""
    log_file = tmp_path / "logs" / "deeper" / "run.log"
    log.configure_logging(
        stream=io.StringIO(), log_file=log_file, backup_count=backup_count
    )
    (file_handler,) = file_handlers()
    assert file_handler.when == "MIDNIGHT"
    assert file_handler.utc
    assert file_handler.backupCount == kept_backups
    assert file_handler.encoding == "utf-8"
    log.get_logger("tests.log.file").info("into\nthe file")
    file_handler.flush()
    assert re.fullmatch(
        STAMP_PATTERN + r" \| INFO \| tests\.log\.file: into; the file\n",
        log_file.read_text(encoding="utf-8"),
    )


def test_a_second_call_closes_the_previous_log_file(tmp_path: Path) -> None:
    """Close the earlier file handler when configuring again."""
    log.configure_logging(stream=io.StringIO(), log_file=tmp_path / "a.log")
    (first_file_handler,) = file_handlers()
    log.configure_logging(stream=io.StringIO(), log_file=tmp_path / "b.log")
    (second_file_handler,) = file_handlers()
    assert first_file_handler is not second_file_handler
    assert first_file_handler.stream is None
    assert Path(second_file_handler.baseFilename) == tmp_path / "b.log"


def test_a_log_file_that_cannot_be_made_is_refused(tmp_path: Path) -> None:
    """Raise OSError, leaving only the new stream handler attached."""
    blocking_file = tmp_path / "a_file"
    blocking_file.write_text("x", encoding="utf-8")
    output_stream = io.StringIO()
    with pytest.raises(OSError, match="a_file"):
        log.configure_logging(stream=output_stream, log_file=blocking_file / "run.log")
    (only_handler,) = log._active_handlers
    assert isinstance(only_handler, logging.StreamHandler)
    assert only_handler.stream is output_stream
    assert file_handlers() == []
