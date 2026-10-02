"""The steering files: what was done to each reference clock, and when.

Each reference has one steering file, ``steer_<mc>.dat`` in the steering
directory, shared by both RF channels. An adapter outside das_processor
appends one line per steering event, in time order:

    <MJD> <dx_ps> <dy_ps_per_s>

dx and dy are the changes applied to the reference's own phase and rate.
Steering is logged when it is applied, so every event of an epoch is in the
file before that epoch's DAS lines are.

A reference with no file has never been steered. A file that is there must
be sound in every line: one that cannot be read, a line that does not
parse, and a line earlier than the one before it are each refused, wherever
the line lies, since the file can no longer be trusted. As in the DAS files,
numbers are plain decimals only, and a last line with no newline makes the
file damaged.
"""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Final, NoReturn

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import FIRST_DAY, LAST_DAY
from masterclock.domain.references import REFERENCE_PATTERN
from masterclock.domain.steering import SteerEvent

STEERING_FILE_TEMPLATE: Final[str] = "steer_{mc}.dat"
"""The name of a reference's steering file, ``mc`` its name."""

_MJD: Final[re.Pattern[str]] = re.compile(r"[0-9]+\.[0-9]+")
"""How an MJD is written: digits, a point, digits."""

_CHANGE: Final[re.Pattern[str]] = re.compile(
    r"[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)
"""How a phase or rate change is written: a plain decimal, with an exponent or not."""

_COLUMNS: Final[int] = 3
"""How many columns a line has."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def read_steering(
    steering_path: Path, mc: str, after: datetime, through: datetime
) -> tuple[SteerEvent, ...]:
    """Give a reference's steering events in (after, through] (design 5.6).

    Parameters
    ----------
    steering_path : Path
        The steering directory.
    mc : str
        The reference.
    after : datetime
        Events at this instant or before are left out; must carry a
        timezone.
    through : datetime
        Events after this instant are left out; must carry a timezone.

    Returns
    -------
    tuple of SteerEvent
        The events applied after ``after`` and at or before ``through``,
        in time order; none when the reference has no steering file.

    Raises
    ------
    DataFileError
        If ``mc`` is not a reference's name, or the file is there but
        cannot be read as ASCII, has a line that does not parse or is
        earlier than the line before it, or ends without a newline.
    """
    if re.fullmatch(REFERENCE_PATTERN, mc) is None:
        _fail(f"steering is read for references only, not {mc}")
    path = steering_path / STEERING_FILE_TEMPLATE.format(mc=mc)
    text = _read(path)
    if text is None:
        return ()
    return tuple(
        event
        for event in _events(path, text)
        if after < event.applied_datetime <= through
    )


def _read(path: Path) -> str | None:
    """Read a steering file's text.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    str or None
        The text; ``None`` when there is no such file.

    Raises
    ------
    DataFileError
        If the file is there but cannot be read as ASCII.
    """
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        _fail(f"cannot read steering file {path}: {exc}", exc)


def _events(path: Path, text: str) -> list[SteerEvent]:
    """Read every event of a steering file, checking each line.

    Parameters
    ----------
    path : Path
        The file, for the message.
    text : str
        Its text.

    Returns
    -------
    list of SteerEvent
        Every event, in file order.

    Raises
    ------
    DataFileError
        If a line has no newline, does not parse, or is earlier than the
        line before it.
    """
    events: list[SteerEvent] = []
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        where = f"steering file {path}: line {number}"
        if not line.endswith("\n"):
            _fail(f"{where} has no newline")
        event = _parse(where, line[:-1])
        if events and event.applied_datetime < events[-1].applied_datetime:
            _fail(f"{where} is earlier than the line before")
        events.append(event)
    return events


def _parse(where: str, line: str) -> SteerEvent:
    """Read one line of a steering file.

    Parameters
    ----------
    where : str
        The file and line, for the message.
    line : str
        The line, without its newline.

    Returns
    -------
    SteerEvent
        The event.

    Raises
    ------
    DataFileError
        If the line is not three columns of an MJD on a data day and two
        finite plain decimals.
    """
    columns = line.split()
    if len(columns) != _COLUMNS:
        _fail(f"{where} has {len(columns)} columns, not {_COLUMNS}: {line!r}")
    mjd_text, dx_text, dy_text = columns
    if _MJD.fullmatch(mjd_text) is None:
        _fail(f"{where}: {mjd_text!r} is not an MJD on a data day")
    mjd = float(mjd_text)
    if not FIRST_DAY <= mjd < LAST_DAY + 1:
        _fail(f"{where}: {mjd_text!r} is not an MJD on a data day")
    dx, dy = _change(where, dx_text), _change(where, dy_text)
    return SteerEvent(applied_datetime=mjd_to_datetime(mjd), dx=dx, dy=dy)


def _change(where: str, text: str) -> float:
    """Read a phase or rate change.

    Parameters
    ----------
    where : str
        The file and line, for the message.
    text : str
        The column.

    Returns
    -------
    float
        The change.

    Raises
    ------
    DataFileError
        If the column is not a plain decimal, or its value is not finite.
    """
    value = float(text) if _CHANGE.fullmatch(text) else math.nan
    if not math.isfinite(value):
        _fail(f"{where}: {text!r} is not a finite plain decimal")
    return value


def _fail(message: str, cause: Exception | None = None) -> NoReturn:
    """Log and raise a steering file error.

    Parameters
    ----------
    message : str
        What is wrong.
    cause : Exception or None, optional
        The error it came from, if any.

    Raises
    ------
    DataFileError
        Always.
    """
    _log.error(message)
    raise DataFileError(message) from cause
