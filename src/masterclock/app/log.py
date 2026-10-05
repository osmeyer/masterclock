"""Application logging built on the standard :mod:`logging` module.

Customizations over stock :mod:`logging`:

* Timestamps are rendered in UTC and labelled ``UTC`` (rather than with a
  numeric timezone offset), followed by a comma and the same instant expressed
  as a Modified Julian Day (MJD) to six decimal places
  (see :class:`UtcMjdFormatter`).
* A record occupies one line however long its message ran, folded by the
  formatter rather than by whoever wrote it (see :func:`one_line`). The
  exceptions are a traceback and a stack given with ``stack_info``, which
  keep their several lines.
* A ``TRACE`` level (numeric value :data:`TRACE` = 5, below ``DEBUG``) is
  registered, with a matching :meth:`MasterClockLogger.trace` method.
* TODO messages are emitted at ``DEBUG`` level with a ``"TODO: "`` prefix via
  :meth:`MasterClockLogger.todo`.

Importing this module registers the ``TRACE`` level name and installs
:class:`MasterClockLogger` as the logger class, so loggers obtained afterwards
via :func:`logging.getLogger` (or this module's :func:`get_logger`) support
the extra methods.

The Modified Julian Day is defined as ``JD - 2400000.5``; MJD 0 began at
midnight on 1858-11-17 UTC. For a POSIX timestamp ``t`` (seconds since the
Unix epoch), ``MJD = t / 86400 + 40587``.
"""

import logging
import sys
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from typing import TYPE_CHECKING, Final, TextIO, override

from masterclock.app.exceptions import LoggingError
from masterclock.app.timeutil import unix_to_mjd

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import TracebackType

TRACE: Final[int] = 5
"""Numeric value of the TRACE log level, finer-grained than :data:`logging.DEBUG`."""

DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
"""Log record format used by :func:`configure_logging`."""

LINE_JOIN: Final[str] = "; "
"""What stands between the lines of a message folded into one record."""

type ExcInfo = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)
"""What the ``exc_info`` keyword of a logging call accepts."""


def one_line(message_text: str) -> str:
    r"""Fold text onto a single line.

    Parameters
    ----------
    message_text : str
        The text, which may run to several lines.

    Returns
    -------
    str
        The same text with each line stripped and joined by
        :data:`LINE_JOIN`, and blank lines dropped.

    Notes
    -----
    Each line is stripped before joining because the text most often folded
    is an indented report from a validation library: the indentation helps
    on several lines but only adds spaces on one.

    Examples
    --------
    >>> one_line("1 validation error\n  paths.output\n    Input should be 'a'")
    "1 validation error; paths.output; Input should be 'a'"
    >>> one_line("already one line")
    'already one line'
    """
    return LINE_JOIN.join(
        stripped
        for text_line in message_text.splitlines()
        if (stripped := text_line.strip())
    )


_KEEP_ALL_BACKUPS: Final[int] = 0
"""``backupCount`` that makes the rotating handler keep every rotated file."""

_active_handlers: Final[list[logging.Handler]] = []
"""The handlers the last :func:`configure_logging` call put on the root."""


class MasterClockLogger(logging.Logger):
    """Logger with the project's extra logging methods.

    Adds :meth:`trace` (logs at the custom :data:`TRACE` level) and
    :meth:`todo` (logs at ``DEBUG`` with a ``"TODO: "`` prefix). Installed as
    the process-wide logger class when :mod:`masterclock.app.log`
    is imported, so :func:`logging.getLogger` returns instances of this class
    for loggers created after that point.
    """

    def trace(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Log ``msg % args`` at the :data:`TRACE` level.

        Takes the same keywords as :meth:`logging.Logger.debug`, with the
        same meanings.

        Parameters
        ----------
        msg : object
            The message, optionally containing %-style placeholders.
        *args : object
            Values interpolated into ``msg`` by the logging machinery.
        exc_info : ExcInfo, optional
            Exception information to add to the record, as for
            :meth:`logging.Logger.debug`.
        stack_info : bool, optional
            Whether to add the current stack to the record.
        stacklevel : int, optional
            Which caller the record names, counted in stack frames; 1, the
            default, names the function that called this method.
        extra : Mapping[str, object] or None, optional
            Attributes to add to the record.
        """
        if self.isEnabledFor(TRACE):
            self._log(
                TRACE,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
            )

    def todo(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Log ``msg % args`` at ``DEBUG`` level, prefixed with ``"TODO: "``.

        Use for work-remaining markers that should surface in debug output.
        Takes the same keywords as :meth:`logging.Logger.debug`, with the
        same meanings.

        Parameters
        ----------
        msg : object
            The message, optionally containing %-style placeholders.
        *args : object
            Values interpolated into the message by the logging machinery.
        exc_info : ExcInfo, optional
            Exception information to add to the record, as for
            :meth:`logging.Logger.debug`.
        stack_info : bool, optional
            Whether to add the current stack to the record.
        stacklevel : int, optional
            Which caller the record names, counted in stack frames; 1, the
            default, names the function that called this method.
        extra : Mapping[str, object] or None, optional
            Attributes to add to the record.
        """
        if self.isEnabledFor(logging.DEBUG):
            self._log(
                logging.DEBUG,
                f"TODO: {msg}",
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
            )


class UtcMjdFormatter(logging.Formatter):
    """Formatter whose records are one line, UTC-labelled and paired with an MJD.

    Renders ``%(asctime)s`` as the record's creation time in UTC - labelled
    ``UTC`` instead of a timezone offset - followed by a comma and the same
    instant as a Modified Julian Day to six decimal places, for example
    ``2026-07-12 03:14:15.926 UTC, MJD 61233.134907``.

    A record is folded onto one line here, rather than by whoever wrote it,
    so every record in the log is one line whatever its message held: a
    validation library's indented report, an operating system message with a
    newline in it, a quoted line of data. A log whose records can run to
    several lines cannot be read a line at a time.

    A traceback, and a stack given with ``stack_info``, are left as they are.
    The base class appends them after the record's own line, and they are
    only readable on their several lines.
    """

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format the record's creation time as ``<utc datetime> UTC, MJD <mjd>``.

        Parameters
        ----------
        record : logging.LogRecord
            The record whose ``created`` timestamp is being rendered.
        datefmt : str or None, optional
            :meth:`~datetime.datetime.strftime` format for the datetime
            portion. Defaults to ``%Y-%m-%d %H:%M:%S`` followed by the
            milliseconds, truncated rather than rounded.

        Returns
        -------
        str
            The UTC datetime labelled ``UTC``, a comma, and the Modified
            Julian Day to six decimal places labelled ``MJD``.

        Notes
        -----
        Overrides a standard library method, which is why its name is not in
        this project's style.
        """
        created_utc = datetime.fromtimestamp(record.created, tz=UTC)
        rendered = (
            created_utc.strftime(datefmt)
            if datefmt is not None
            else created_utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        )
        return f"{rendered} UTC, MJD {unix_to_mjd(record.created):.6f}"

    @override
    def formatMessage(self, record: logging.LogRecord) -> str:
        """Render the record on one line, whatever the message ran to.

        Parameters
        ----------
        record : logging.LogRecord
            The record being rendered, its message already interpolated.

        Returns
        -------
        str
            The formatted record, folded onto a single line.

        Notes
        -----
        Overrides a standard library method, which is why its name is not in
        this project's style.

        The whole rendered line is folded rather than the message alone, which
        needs no care about which part ran long: every other part - the
        timestamp, the level, the logger name - is one line already, so
        folding them changes nothing. Nothing on the record is changed
        either, so a second handler formatting the same record sees what the
        first did.
        """
        return one_line(super().formatMessage(record))


def get_logger(logger_name: str) -> MasterClockLogger:
    """Return the application logger with the given name.

    Parameters
    ----------
    logger_name : str
        The logger name, typically ``__name__``.

    Returns
    -------
    MasterClockLogger
        The logger registered under ``logger_name``.

    Raises
    ------
    LoggingError
        If a plain :class:`logging.Logger` was already registered under
        ``logger_name`` before :mod:`masterclock.app.log` was imported,
        or if ``logger_name`` is ``"root"`` or empty, which both give the standard
        root logger.
    """
    logger = logging.getLogger(logger_name)
    if not isinstance(logger, MasterClockLogger):
        raise LoggingError(
            f"logger {logger_name!r} is a {type(logger).__name__},"
            " not a MasterClockLogger"
        )
    return logger


def configure_logging(
    root_level: int = logging.INFO,
    stream: TextIO | None = None,
    log_file: Path | None = None,
    backup_count: int | None = None,
) -> logging.Handler:
    """Attach a UTC/MJD-formatted handler to the root logger.

    Installs a :class:`logging.StreamHandler` when there is no ``log_file``;
    when there is, a :class:`logging.handlers.TimedRotatingFileHandler` that
    rolls the file over at midnight UTC, keeping ``backup_count`` rotated
    days (or all of them when it is ``None``), in its place, so the log goes
    to the file alone and a scheduler keeping a run's output gets none of
    it. The stream handler is attached first and removed only once the file
    is open, so a failure to open it can still be logged. Each handler uses
    :class:`UtcMjdFormatter` with :data:`DEFAULT_FORMAT`, and the root
    logger's level is set. Calling this again removes and closes the
    handler the previous call installed instead of stacking more.

    Parameters
    ----------
    root_level : int, optional
        Root logger level. Defaults to :data:`logging.INFO`.
    stream : TextIO or None, optional
        Destination stream for the stream handler. Defaults to
        :data:`sys.stderr`.
    log_file : Path or None, optional
        Path of the log file to also write to, rotated daily at midnight UTC.
        Its parent directory is created (with any missing parents) when it
        does not exist, since the handler creates the file but not the
        directory above it. ``None`` (the default) installs no file handler.
    backup_count : int or None, optional
        Number of rotated daily log files to keep before the oldest is
        discarded. ``None`` (the default) keeps every rotated file, and so
        does ``0``, as it does for the standard handler. Only meaningful when
        ``log_file`` is given.

    Returns
    -------
    logging.Handler
        The handler the records go to: the file handler when ``log_file``
        is given, the stream handler otherwise.

    Raises
    ------
    OSError
        If ``log_file``'s parent directory does not exist and cannot be
        created, or the log file itself cannot be opened - for example when
        a parent is not writable, or a non-directory occupies the path. By
        then the previous call's handlers are removed and the new stream
        handler is attached, so the failure itself can still be logged.
    """
    root = logging.getLogger()
    for previous in _active_handlers:
        root.removeHandler(previous)
        previous.close()
    _active_handlers.clear()

    formatter = UtcMjdFormatter(DEFAULT_FORMAT)
    stream_handler: logging.StreamHandler[TextIO] = logging.StreamHandler(
        stream if stream is not None else sys.stderr
    )
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    _active_handlers.append(stream_handler)

    if log_file is not None:
        # The handler opens the file but does not create its directory, so a
        # log_file under a missing directory would stop the run before it
        # starts.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            utc=True,
            backupCount=_KEEP_ALL_BACKUPS if backup_count is None else backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        root.removeHandler(stream_handler)
        _active_handlers[:] = [file_handler]
        root.setLevel(root_level)
        return file_handler

    root.setLevel(root_level)
    return stream_handler


logging.addLevelName(TRACE, "TRACE")
logging.setLoggerClass(MasterClockLogger)
