"""Tests for src/masterclock/app/log.py.

The rules covered: importing the module registers the TRACE level and the
project's logger class; every record is written on one line, stamped with
its UTC time and MJD, except for a traceback or a stack; trace and todo log
at their levels and name their real caller; get_logger refuses a logger of
the wrong class; configure_logging attaches a stream handler, and a file
handler that rotates at midnight UTC, and replaces what its previous call
attached.
"""

import io
import logging
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
CREATED: Final = 1_783_826_055.926
STAMP: Final = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} UTC, MJD \d{5}\.\d{6}"


@pytest.fixture(autouse=True)
def restore_root() -> Iterator[None]:
    """Put the root logger's handlers and level back after each test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in log._active_handlers:
        root.removeHandler(handler)
        handler.close()
    log._active_handlers.clear()
    root.handlers[:] = handlers
    root.setLevel(level)


class Kept(logging.Handler):
    """Keep every record it is given, for a test to look at."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


def record(message: str, created: float = CREATED) -> logging.LogRecord:
    """Return an INFO record with the given message and creation time."""
    made = logging.LogRecord(
        "invented.name", logging.INFO, "f.py", 1, message, None, None
    )
    made.created = created
    return made


def test_importing_registers_trace_and_the_logger_class() -> None:
    """Name level 5 TRACE and make new loggers MasterClockLoggers."""
    assert log.TRACE == 5
    assert logging.getLevelName(log.TRACE) == "TRACE"
    assert logging.getLoggerClass() is log.MasterClockLogger


@pytest.mark.parametrize(
    ("text", "folded"),
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
def test_one_line_folds_known_texts(text: str, folded: str) -> None:
    """Strip each line, drop blank ones, and join the rest with '; '."""
    assert log.one_line(text) == folded


@given(st.text())
def test_one_line_always_gives_one_line(text: str) -> None:
    """Give a result with no line breaks of any kind, whatever the text."""
    folded = log.one_line(text)
    assert len(folded.splitlines()) <= 1
    assert folded == folded.strip()


@given(st.text().map(str.strip).filter(lambda text: len(text.splitlines()) == 1))
def test_one_line_leaves_a_single_line_alone(text: str) -> None:
    """Return a single stripped line unchanged."""
    assert log.one_line(text) == text


def test_the_time_is_written_in_utc_with_its_mjd() -> None:
    """Write the creation time in UTC to the millisecond, then its MJD."""
    formatter = log.UtcMjdFormatter()
    assert formatter.formatTime(record("m")) == (
        "2026-07-12 03:14:15.926 UTC, MJD 61233.134907"
    )
    assert formatter.formatTime(record("m"), "%H:%M") == "03:14 UTC, MJD 61233.134907"
    assert f"{unix_to_mjd(CREATED):.6f}" == "61233.134907"


def test_milliseconds_are_truncated_not_rounded() -> None:
    """Drop the microseconds past the millisecond instead of rounding them."""
    formatter = log.UtcMjdFormatter()
    stamp = formatter.formatTime(record("m", created=1_783_826_055.9269))
    assert stamp.startswith("2026-07-12 03:14:15.926 UTC")


def test_a_record_is_written_on_one_line() -> None:
    """Fold the whole rendered record, and leave the record itself unchanged."""
    formatter = log.UtcMjdFormatter(log.DEFAULT_FORMAT)
    made = record("two\n  lines\n\n  here")
    assert formatter.format(made) == (
        "2026-07-12 03:14:15.926 UTC, MJD 61233.134907 | INFO | invented.name:"
        " two; lines; here"
    )
    assert made.msg == "two\n  lines\n\n  here"


def test_a_traceback_keeps_its_lines() -> None:
    """Write the record on one line and the traceback after it, unfolded."""
    formatter = log.UtcMjdFormatter(log.DEFAULT_FORMAT)
    made = record("failed\nbadly")
    try:
        raise ValueError("invented failure")
    except ValueError:
        made.exc_info = sys.exc_info()
    lines = formatter.format(made).splitlines()
    assert lines[0].endswith("| INFO | invented.name: failed; badly")
    assert lines[1] == "Traceback (most recent call last):"
    assert lines[-1] == "ValueError: invented failure"


def test_get_logger_gives_the_project_logger() -> None:
    """Return a MasterClockLogger, the same one each time for a name."""
    first = log.get_logger("tests.log.first")
    assert isinstance(first, log.MasterClockLogger)
    assert log.get_logger("tests.log.first") is first


def test_get_logger_refuses_a_logger_of_another_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise LoggingError for a name registered as a plain Logger."""
    monkeypatch.setitem(
        logging.Logger.manager.loggerDict,
        "tests.log.plain",
        logging.Logger("tests.log.plain"),
    )
    with pytest.raises(LoggingError, match=r"'tests\.log\.plain' is a Logger"):
        log.get_logger("tests.log.plain")


def logger_with_kept(level: int) -> tuple[log.MasterClockLogger, Kept]:
    """Return a project logger at ``level`` and the handler keeping its records."""
    logger = log.get_logger("tests.log.levels")
    kept = Kept()
    logger.handlers[:] = [kept]
    logger.propagate = False
    logger.setLevel(level)
    return logger, kept


def test_trace_logs_at_trace_and_names_its_caller() -> None:
    """Log at TRACE, interpolating arguments, as from the calling function."""
    logger, kept = logger_with_kept(log.TRACE)
    logger.trace("value %s", 7)
    (made,) = kept.records
    assert made.levelno == log.TRACE
    assert made.levelname == "TRACE"
    assert made.getMessage() == "value 7"
    assert made.funcName == "test_trace_logs_at_trace_and_names_its_caller"
    assert made.pathname == __file__


def test_todo_logs_at_debug_with_its_prefix_and_names_its_caller() -> None:
    """Log at DEBUG with 'TODO: ' in front, as from the calling function."""
    logger, kept = logger_with_kept(logging.DEBUG)
    logger.todo("finish %s", "this")
    (made,) = kept.records
    assert made.levelno == logging.DEBUG
    assert made.getMessage() == "TODO: finish this"
    assert (
        made.funcName == "test_todo_logs_at_debug_with_its_prefix_and_names_its_caller"
    )
    assert made.pathname == __file__


@pytest.mark.parametrize("method", ["trace", "todo"])
def test_trace_and_todo_take_the_standard_keywords(method: str) -> None:
    """Pass exc_info, stack_info and extra through, as the standard methods do."""
    logger, kept = logger_with_kept(log.TRACE)
    call = getattr(logger, method)
    try:
        raise ValueError("invented failure")
    except ValueError:
        call("with traceback", exc_info=True)
    call("with stack", stack_info=True)
    call("with extra", extra={"invented_key": 3})
    traced, stacked, extended = kept.records
    assert traced.exc_info is not None
    assert traced.exc_info[0] is ValueError
    assert stacked.stack_info is not None
    assert stacked.stack_info.startswith("Stack (most recent call last):")
    assert extended.__dict__["invented_key"] == 3


def report(logger: log.MasterClockLogger) -> None:
    """Log from here as though from the function that called this one."""
    logger.trace("reported", stacklevel=2)
    logger.todo("reported", stacklevel=2)


def test_trace_and_todo_take_a_stacklevel() -> None:
    """Name the caller the given number of frames up, as the standard methods do."""
    logger, kept = logger_with_kept(log.TRACE)
    report(logger)
    assert [made.funcName for made in kept.records] == [
        "test_trace_and_todo_take_a_stacklevel",
        "test_trace_and_todo_take_a_stacklevel",
    ]


def test_get_logger_refuses_the_root_logger() -> None:
    """Raise LoggingError for the names of the standard root logger."""
    for name in ("root", ""):
        with pytest.raises(LoggingError, match="is a RootLogger"):
            log.get_logger(name)


def test_trace_and_todo_are_dropped_above_their_levels() -> None:
    """Write nothing for trace above TRACE, or for todo above DEBUG."""
    logger, kept = logger_with_kept(logging.DEBUG)
    logger.trace("hidden")
    logger.setLevel(logging.INFO)
    logger.todo("hidden")
    assert kept.records == []


def test_configure_logging_attaches_a_one_line_stream_handler() -> None:
    """Attach a stream handler with the project format, and set the level."""
    stream = io.StringIO()
    handler = log.configure_logging(level=logging.WARNING, stream=stream)
    root = logging.getLogger()
    assert handler in root.handlers
    assert handler.stream is stream
    assert isinstance(handler.formatter, log.UtcMjdFormatter)
    assert root.level == logging.WARNING
    log.get_logger("tests.log.stream").warning("seen\n  twice")
    log.get_logger("tests.log.stream").info("not seen")
    assert re.fullmatch(
        STAMP + r" \| WARNING \| tests\.log\.stream: seen; twice\n", stream.getvalue()
    )


def test_configure_logging_writes_to_standard_error_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use the standard error stream in force when no stream is given."""
    handler = log.configure_logging()
    assert handler.stream is sys.stderr
    log.get_logger("tests.log.stderr").info("to stderr")
    assert "| INFO | tests.log.stderr: to stderr" in capsys.readouterr().err


def test_configure_logging_replaces_its_own_handlers() -> None:
    """Remove and close the handlers of the previous call, and keep others."""
    root = logging.getLogger()
    other = Kept()
    root.addHandler(other)
    first = log.configure_logging(stream=io.StringIO())
    second = log.configure_logging(stream=io.StringIO())
    assert first not in root.handlers
    assert second in root.handlers
    assert other in root.handlers
    assert log._active_handlers == [second]


def file_handlers() -> list[TimedRotatingFileHandler]:
    """Return the rotating file handlers on the root logger."""
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, TimedRotatingFileHandler)
    ]


@pytest.mark.parametrize(("backup_count", "kept"), [(None, 0), (0, 0), (7, 7)])
def test_the_log_file_rotates_at_midnight_utc(
    tmp_path: Path, backup_count: int | None, kept: int
) -> None:
    """Create the missing directories and rotate daily at midnight UTC."""
    log_file = tmp_path / "logs" / "deeper" / "run.log"
    log.configure_logging(
        stream=io.StringIO(), log_file=log_file, backup_count=backup_count
    )
    (handler,) = file_handlers()
    assert handler.when == "MIDNIGHT"
    assert handler.utc
    assert handler.backupCount == kept
    assert handler.encoding == "utf-8"
    log.get_logger("tests.log.file").info("into\nthe file")
    handler.flush()
    assert re.fullmatch(
        STAMP + r" \| INFO \| tests\.log\.file: into; the file\n",
        log_file.read_text(encoding="utf-8"),
    )


def test_a_second_call_closes_the_previous_log_file(tmp_path: Path) -> None:
    """Close the earlier file handler when configuring again."""
    log.configure_logging(stream=io.StringIO(), log_file=tmp_path / "a.log")
    (first,) = file_handlers()
    log.configure_logging(stream=io.StringIO(), log_file=tmp_path / "b.log")
    (second,) = file_handlers()
    assert first is not second
    assert first.stream is None
    assert Path(second.baseFilename) == tmp_path / "b.log"


def test_a_log_file_that_cannot_be_made_is_refused(tmp_path: Path) -> None:
    """Raise OSError, leaving only the new stream handler attached."""
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    stream = io.StringIO()
    with pytest.raises(OSError, match="a_file"):
        log.configure_logging(stream=stream, log_file=blocker / "run.log")
    (only,) = log._active_handlers
    assert isinstance(only, logging.StreamHandler)
    assert only.stream is stream
    assert file_handlers() == []
