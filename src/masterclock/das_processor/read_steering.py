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
    steering_path: Path, mc: str, window_start: datetime, window_end: datetime
) -> tuple[SteerEvent, ...]:
    """Give a reference's steering events in (after, through] (design 5.6).

    Parameters
    ----------
    steering_path : Path
        The steering directory.
    mc : str
        The reference.
    window_start : datetime
        Events at this instant or before are left out; must carry a
        timezone.
    window_end : datetime
        Events after this instant are left out; must carry a timezone.

    Returns
    -------
    tuple of SteerEvent
        The events applied after ``window_start`` and at or before ``window_end``,
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
    steering_file = steering_path / STEERING_FILE_TEMPLATE.format(mc=mc)
    steering_text = _read_steering_text(steering_file)
    if steering_text is None:
        return ()
    return tuple(
        steer_event
        for steer_event in _steer_events(steering_file, steering_text)
        if window_start < steer_event.applied_datetime <= window_end
    )


def _read_steering_text(steering_file: Path) -> str | None:
    """Read a steering file's text.

    Parameters
    ----------
    steering_file : Path
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
        return steering_file.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        _fail(f"cannot read steering file {steering_file}: {exc}", exc)


def _steer_events(steering_file: Path, steering_text: str) -> list[SteerEvent]:
    """Read every event of a steering file, checking each line.

    Parameters
    ----------
    steering_file : Path
        The file, for the message.
    steering_text : str
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
    steer_events: list[SteerEvent] = []
    for line_number, line in enumerate(
        steering_text.splitlines(keepends=True), start=1
    ):
        line_place = f"steering file {steering_file}: line {line_number}"
        if not line.endswith("\n"):
            _fail(f"{line_place} has no newline")
        steer_event = _parse_steering_line(line_place, line[:-1])
        if (
            steer_events
            and steer_event.applied_datetime < steer_events[-1].applied_datetime
        ):
            _fail(f"{line_place} is earlier than the line before")
        steer_events.append(steer_event)
    return steer_events


def _parse_steering_line(line_place: str, line: str) -> SteerEvent:
    """Read one line of a steering file.

    Parameters
    ----------
    line_place : str
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
    line_columns = line.split()
    if len(line_columns) != _COLUMNS:
        _fail(f"{line_place} has {len(line_columns)} columns, not {_COLUMNS}: {line!r}")
    mjd_text, dx_text, dy_text = line_columns
    if _MJD.fullmatch(mjd_text) is None:
        _fail(f"{line_place}: {mjd_text!r} is not an MJD on a data day")
    mjd = float(mjd_text)
    if not FIRST_DAY <= mjd < LAST_DAY + 1:
        _fail(f"{line_place}: {mjd_text!r} is not an MJD on a data day")
    dx, dy = _parse_change(line_place, dx_text), _parse_change(line_place, dy_text)
    return SteerEvent(applied_datetime=mjd_to_datetime(mjd), dx=dx, dy=dy)


def _parse_change(line_place: str, column_text: str) -> float:
    """Read a phase or rate change.

    Parameters
    ----------
    line_place : str
        The file and line, for the message.
    column_text : str
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
    change = float(column_text) if _CHANGE.fullmatch(column_text) else math.nan
    if not math.isfinite(change):
        _fail(f"{line_place}: {column_text!r} is not a finite plain decimal")
    return change


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
