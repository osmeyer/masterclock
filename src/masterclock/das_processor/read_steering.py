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
the line lies, since the file can no longer be trusted. The MJD is digits, a
point and digits; a change is a finite decimal number, signed or not, with an
exponent or not. As in the DAS files, a last line with no newline makes the
file damaged. Each event read is checked by a pydantic model
(:class:`SteerEventFields`) before the program uses it.

A run keeps the events it has read (:class:`SteeringFiles`): each line is
read and checked once, and an epoch reads only the lines appended since. A
file that is replaced, or is shorter than what was read, is read again from
its start.
"""

import bisect
import math
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, NoReturn

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

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
"""How a phase or rate change is written: a decimal, signed or not, exponent or not."""

_COLUMNS: Final[int] = 3
"""How many columns a line has."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


class SteerEventFields(BaseModel):
    """A steering event's fields as read from a steering file, checked.

    The fields are a :class:`~masterclock.domain.steering.SteerEvent`'s, with
    the same names, order and meanings.

    Raises
    ------
    pydantic.ValidationError
        If the instant has no timezone, a change is not a finite number, or
        a field is missing or unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    applied_datetime: AwareDatetime
    dx: Annotated[float, Field(allow_inf_nan=False)]
    dy: Annotated[float, Field(allow_inf_nan=False)]

    def event(self) -> SteerEvent:
        """Give the steering event the fields make.

        Returns
        -------
        SteerEvent
            The event.
        """
        return SteerEvent(**dict(self))


@dataclass(slots=True)
class _KeptFile:
    """What has been read of one steering file.

    Parameters
    ----------
    file_identity : tuple of (int, int)
        The device and inode of the file read, so a replaced file shows.
    bytes_read : int
        How much of the file has been read, from its start.
    lines_read : int
        How many lines that holds, so a later line is named by its number.
    events : list of SteerEvent
        Every event read, in time order.
    instants : list of datetime
        When each event was applied, in the same order, to search.
    """

    file_identity: tuple[int, int]
    bytes_read: int = 0
    lines_read: int = 0
    events: list[SteerEvent] = field(default_factory=list)
    instants: list[datetime] = field(default_factory=list)


class SteeringFiles:
    """Each reference's steering events, read once in a run and then as lines are added.

    Parameters
    ----------
    steering_path : Path
        The steering directory.
    """

    def __init__(self, steering_path: Path) -> None:
        """Start with nothing read.

        Parameters
        ----------
        steering_path : Path
            The steering directory.
        """
        self.steering_path = steering_path
        self._kept_files: dict[str, _KeptFile] = {}

    def events(
        self, mc: str, window_start: datetime, window_end: datetime
    ) -> tuple[SteerEvent, ...]:
        """Give a reference's events in (window_start, window_end] (design 5.6).

        The lines appended to the file since it was last read are read and
        checked first; a file replaced, or shorter than what was read, is
        read again from its start. A read that is refused keeps nothing of
        it.

        Parameters
        ----------
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
            The events applied after ``window_start`` and at or before
            ``window_end``, in time order; none when the reference has no
            steering file.

        Raises
        ------
        DataFileError
            If ``mc`` is not a reference's name, or the file is there but is
            not a regular file or cannot be read as ASCII, has a line that
            does not parse or is earlier than the line before it, or ends
            without a newline.
        """
        if re.fullmatch(REFERENCE_PATTERN, mc) is None:
            _fail(f"steering is read for references only, not {mc}")
        kept_file = self._read_new_lines(
            mc, self.steering_path / STEERING_FILE_TEMPLATE.format(mc=mc)
        )
        if kept_file is None:
            return ()
        first = bisect.bisect_right(kept_file.instants, window_start)
        last = bisect.bisect_right(kept_file.instants, window_end)
        return tuple(kept_file.events[first:last])

    def _read_new_lines(self, mc: str, steering_file: Path) -> _KeptFile | None:
        """Bring what is kept of a reference's file up to the file's end.

        Parameters
        ----------
        mc : str
            The reference.
        steering_file : Path
            Its steering file.

        Returns
        -------
        _KeptFile or None
            Everything read of the file; ``None`` when there is no file.

        Raises
        ------
        DataFileError
            If the file is not a regular file, cannot be read as ASCII, or a
            new line is damaged or out of time order.
        """
        try:
            file_status = steering_file.stat()
        except FileNotFoundError:
            self._kept_files.pop(mc, None)
            return None
        except OSError as exc:
            _fail(f"cannot read steering file {steering_file}: {exc}", exc)
        if not stat.S_ISREG(file_status.st_mode):
            _fail(f"cannot read steering file {steering_file}: not a regular file")
        file_identity = (file_status.st_dev, file_status.st_ino)
        kept_file = self._kept_files.get(mc)
        if (
            kept_file is None
            or kept_file.file_identity != file_identity
            or file_status.st_size < kept_file.bytes_read
        ):
            kept_file = _KeptFile(file_identity=file_identity)
        if file_status.st_size > kept_file.bytes_read:
            new_bytes = _read_from(steering_file, kept_file.bytes_read)
            new_events = _steer_events(
                steering_file,
                _ascii(steering_file, new_bytes),
                kept_file.lines_read,
                kept_file.events[-1] if kept_file.events else None,
            )
            kept_file.bytes_read += len(new_bytes)
            kept_file.lines_read += len(new_events)
            kept_file.events += new_events
            kept_file.instants += [
                steer_event.applied_datetime for steer_event in new_events
            ]
        self._kept_files[mc] = kept_file
        return kept_file


def _read_from(steering_file: Path, offset: int) -> bytes:
    """Read a steering file from a byte offset to its end.

    Parameters
    ----------
    steering_file : Path
        The file.
    offset : int
        Where to start, bytes.

    Returns
    -------
    bytes
        The rest of the file.

    Raises
    ------
    DataFileError
        If the file cannot be read.
    """
    try:
        with steering_file.open("rb") as steering_stream:
            steering_stream.seek(offset)
            return steering_stream.read()
    except OSError as exc:
        _fail(f"cannot read steering file {steering_file}: {exc}", exc)


def _ascii(steering_file: Path, file_bytes: bytes) -> str:
    """Decode a steering file's bytes as ASCII.

    Parameters
    ----------
    steering_file : Path
        The file, for the message.
    file_bytes : bytes
        Bytes read from it.

    Returns
    -------
    str
        The text.

    Raises
    ------
    DataFileError
        If the bytes are not ASCII.
    """
    try:
        return file_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        _fail(f"cannot read steering file {steering_file}: {exc}", exc)


def _steer_events(
    steering_file: Path,
    steering_text: str,
    lines_before: int,
    last_event: SteerEvent | None,
) -> list[SteerEvent]:
    """Read every event of new steering lines, checking each line.

    Parameters
    ----------
    steering_file : Path
        The file, for the message.
    steering_text : str
        The new lines' text.
    lines_before : int
        How many lines of the file come before them.
    last_event : SteerEvent or None
        The event of the line before them; ``None`` at the file's start.

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
        steering_text.splitlines(keepends=True), start=lines_before + 1
    ):
        line_place = f"steering file {steering_file}: line {line_number}"
        if not line.endswith("\n"):
            _fail(f"{line_place} has no newline")
        steer_event = _parse_steering_line(line_place, line[:-1])
        if (
            last_event is not None
            and steer_event.applied_datetime < last_event.applied_datetime
        ):
            _fail(f"{line_place} is earlier than the line before")
        steer_events.append(steer_event)
        last_event = steer_event
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
        finite decimal numbers.
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
    return SteerEventFields(applied_datetime=mjd_to_datetime(mjd), dx=dx, dy=dy).event()


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
        If the column is not a decimal number as :data:`_CHANGE` spells it,
        or its value is not finite.
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
