"""The two output files: their columns, headers and rows.

Every series has one file: a pair a measurement file, a triple a
double-difference file. Each line of a file, header lines included, is the
same width W, worked out from the file's column table, and ends with a
newline. The header says in words what the file is and what each column
holds, and warns that only das_processor may write the file; the program
never reads it back.

A row is the values of one epoch, each right-justified in a column exactly
as wide as its values need and separated by ``", "``; ``-`` marks an empty
field. Floats are written as ``{:+.16e}``, seventeen significant digits,
which give every float back exactly; the estimator's phase, held in whole
femtoseconds, is written in ps with three decimals. A value too wide for its
column is refused before anything is written.

A row read back (:func:`parse_meas_row`, :func:`parse_ddiff_row`) is
formatted again and must give the same line, so a line is accepted only in
the one form das_processor writes. The measurement file holds no
innovation, so a measurement row reads back without one.

Rows are buffered and written a day at a time (:func:`write_buffer`), every
check made before the first byte, under a write journal that names the
first epoch being written. A run that starts checks every file and rolls
them all back to the oldest epoch they all hold, or to before the epoch a
journal names (:func:`roll_back`, :func:`read_journal`).
"""

import math
import os
import re
import shutil
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Final, Literal, NamedTuple, NoReturn, Self

from pydantic import BaseModel, ConfigDict, model_validator

from masterclock.app.exceptions import describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.config import DDIFF_SUBDIRECTORY, MEAS_SUBDIRECTORY
from masterclock.das_processor.epochs import floor_to_ten_minutes, format_epoch
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import RMS_WIDTH
from masterclock.domain.exceptions import FilterError, PhaseError
from masterclock.domain.measurements import PairMeasurement, TripleMeasurement
from masterclock.domain.phase import EPOCH_SECONDS, FS_PER_PS
from masterclock.domain.references import REFERENCE_PATTERN
from masterclock.domain.series import (
    Reject,
    Row,
    SeriesKey,
)

type FileKind = Literal["meas", "ddiff"]
"""The kind of an output file: a pair's measurements or a triple's."""


class Column(NamedTuple):
    """One column of an output file.

    Parameters
    ----------
    name : str
        The column's name, as the header gives it.
    width : int
        How many characters its values take, right-justified.
    meaning : str
        What it holds, with its unit, as the header gives it.
    """

    name: str
    width: int
    meaning: str


_EPOCH_COLUMNS: Final[tuple[Column, ...]] = (
    Column("interpolated_datetime", 25, "epoch start E, UTC"),
    Column("interpolated_mjd", 13, "epoch start E, MJD"),
)
"""The columns every row starts with: its epoch."""

_STATE_COLUMNS: Final[tuple[Column, ...]] = (
    Column("x", 20, "estimated phase at E, ps, to the femtosecond"),
    Column("y", 23, "estimated rate, ps/s"),
    Column("d", 23, "estimated drift, ps/s^2; 0 for a 1- or 2-state estimator"),
    Column("innovation_scale", 23, "innovation scale, ps"),
    Column("segment", 9, "segment number"),
    Column("step_offset", 16, "sum of phase steps in this segment, ps"),
    Column("epochs_in_segment", 9, "rows since the segment started"),
    Column("epochs_since_accept", 9, "rows since the last accepted measurement"),
    Column("consecutive_rejects", 9, "consecutive counted rejects"),
    Column("reject1_mjd", 13, "reject buffer, oldest: epoch start, MJD"),
    Column("reject1_innovation", 23, "reject buffer, oldest: innovation, ps"),
    Column("reject2_mjd", 13, "reject buffer, middle: epoch start, MJD"),
    Column("reject2_innovation", 23, "reject buffer, middle: innovation, ps"),
    Column("reject3_mjd", 13, "reject buffer, newest: epoch start, MJD"),
    Column("reject3_innovation", 23, "reject buffer, newest: innovation, ps"),
    Column("filter_states", 1, "estimator states: 1, 2 or 3"),
    Column("time_constant", 23, "estimator time constant, epochs"),
    Column("scale_time_constant", 23, "innovation-scale averaging constant, epochs"),
    Column(
        "flags",
        8,
        "A accepted, R rejected, X excluded, P predicted, D dormant,"
        " S slip corrected, N new segment, U unsettled",
    ),
)
"""The columns every row ends with: the estimator's state and counters."""

MEAS_COLUMNS: Final[tuple[Column, ...]] = (
    *_EPOCH_COLUMNS,
    Column("measurement_datetime", 32, "measurement time, UTC"),
    Column("measurement_mjd", 13, "measurement time, MJD"),
    Column("measured_phase", 6, "raw phase from the DAS, ps"),
    Column("rms", RMS_WIDTH, "RMS from the DAS, ps"),
    Column("cycle_count", 12, "whole periods added in decycling"),
    Column("z", 16, "decycled phase interpolated to E, ps"),
    *_STATE_COLUMNS,
)
"""The columns of a measurement file, in order (design 5.4)."""

DDIFF_COLUMNS: Final[tuple[Column, ...]] = (
    *_EPOCH_COLUMNS,
    Column("z", 16, "double difference dd at E, ps"),
    Column("innovation", 23, "innovation: z minus the prediction, ps"),
    Column("double_difference_sigma", 23, "measurement sigma of dd, ps"),
    Column("components_used", 3, "components used: (s,c) (r,s) (s,r)"),
    *_STATE_COLUMNS,
)
"""The columns of a double-difference file, in order (design 5.5)."""

SEPARATOR: Final[str] = ", "
"""What stands between two columns."""

EMPTY: Final[str] = "-"
"""What an empty field holds."""

_PAIR: Final[int] = 2
"""How many names a pair key holds."""

_TRIPLE: Final[int] = 3
"""How many names a triple key holds."""

_PREAMBLE: Final[int] = 6
"""How many header lines come before the column lines."""


def _width(columns: tuple[Column, ...]) -> int:
    """Work out a row's width from its columns.

    Parameters
    ----------
    columns : tuple of Column
        The columns.

    Returns
    -------
    int
        The columns' widths and the separators between them.
    """
    return sum(column.width for column in columns) + len(SEPARATOR) * (len(columns) - 1)


MEAS_WIDTH: Final[int] = _width(MEAS_COLUMNS)
"""The width W of every line of a measurement file, newline not counted."""

DDIFF_WIDTH: Final[int] = _width(DDIFF_COLUMNS)
"""The width W of every line of a double-difference file, newline not counted."""

MEAS_HEADER_LINES: Final[int] = _PREAMBLE + len(MEAS_COLUMNS)
"""How many header lines a measurement file has."""

DDIFF_HEADER_LINES: Final[int] = _PREAMBLE + len(DDIFF_COLUMNS)
"""How many header lines a double-difference file has."""

WIDTHS: Final[dict[FileKind, int]] = {"meas": MEAS_WIDTH, "ddiff": DDIFF_WIDTH}
"""The line width of each kind of file."""

HEADER_LINES: Final[dict[FileKind, int]] = {
    "meas": MEAS_HEADER_LINES,
    "ddiff": DDIFF_HEADER_LINES,
}
"""How many header lines each kind of file has."""

_X_TEXT: Final[re.Pattern[str]] = re.compile(r"-?[0-9]+\.[0-9]{3}")
"""How the estimator's phase is written: ps with three decimals."""

_MJD_DECIMALS: Final[int] = 6
"""Decimal places of an MJD column."""

_HALF_EPOCH: Final[timedelta] = timedelta(minutes=5)
"""Half an epoch, to read an MJD back to its nearest mark."""

_TITLES: Final[dict[FileKind, str]] = {
    "meas": "measurement",
    "ddiff": "double-difference",
}
"""What each kind of file is called in its header."""

_WARNING: Final[str] = (
    "# WARNING: do not modify this file. Only das_processor may write it;"
    " any other change damages the archive."
)
"""The header's second line."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def _fail(message: str, cause: Exception | None = None) -> NoReturn:
    """Log and raise a data file error.

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


# ------------------------------------------------------------------ records


class MeasRecord(BaseModel):
    """One row of a measurement file: a pair's measurement and its row.

    Parameters
    ----------
    measurement : PairMeasurement or None
        The pair's measurement at the epoch, or ``None`` when there was
        none.
    row : Row
        The pair's row at the epoch; its innovation is not written.

    Raises
    ------
    DataFileError
        If there is a measurement and the row is P or the other way round,
        the measurement is of another epoch than the row's, or the
        measurement's slip correction and the row's S do not go together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement: PairMeasurement | None
    row: Row

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a measurement that does not belong with its row.

        Returns
        -------
        Self
            The record, unchanged.
        """
        _check_measured(self.measurement is not None, self.row)
        if self.measurement is None:
            if "S" in self.row.flags:
                _fail(f"row of {self.row.interpolated_datetime}: S with no measurement")
            return self
        if self.measurement.interpolated_datetime != self.row.interpolated_datetime:
            _fail(
                f"row of {self.row.interpolated_datetime}: its measurement is of"
                f" the epoch of {self.measurement.interpolated_datetime}"
            )
        if self.measurement.slip != ("S" in self.row.flags):
            _fail(
                f"row of {self.row.interpolated_datetime}: a slip correction"
                " goes with flag S and S with a slip correction"
            )
        return self


class DdiffRecord(BaseModel):
    """One row of a double-difference file: a triple's measurement and its row.

    Parameters
    ----------
    measurement : TripleMeasurement or None
        The triple's double difference at the epoch, or ``None`` when there
        was none. Its cold mark is not written.
    row : Row
        The triple's row at the epoch.

    Raises
    ------
    DataFileError
        If there is a measurement and the row is P or the other way round,
        or the row carries S, which a triple never does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement: TripleMeasurement | None
    row: Row

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a measurement that does not belong with its row.

        Returns
        -------
        Self
            The record, unchanged.
        """
        _check_measured(self.measurement is not None, self.row)
        if "S" in self.row.flags:
            _fail(f"row of {self.row.interpolated_datetime}: a triple never carries S")
        return self


def _check_measured(measured: bool, row: Row) -> None:
    """Refuse a measurement on a P row, or none on another.

    Parameters
    ----------
    measured : bool
        Whether the record has a measurement.
    row : Row
        Its row.

    Raises
    ------
    DataFileError
        If ``measured`` and the row is P, or neither.
    """
    if measured == ("P" in row.flags):
        _fail(
            f"row of {row.interpolated_datetime}: a row has a measurement"
            f" exactly when it is not P; flags {row.flags!r}"
        )


# ------------------------------------------------------------------ headers


def header(kind: FileKind, channel: RfChannel, key: SeriesKey) -> str:
    """Give the header of a series' file (design 5.2, 5.4, 5.5).

    Parameters
    ----------
    kind : {'meas', 'ddiff'}
        The kind of file.
    channel : {'a', 'b'}
        The RF channel.
    key : (str, str) or (str, str, str)
        The pair of a measurement file, the triple of a double-difference
        file.

    Returns
    -------
    str
        The header lines, each padded with spaces to the file's width and
        ending with a newline: the kind of file and its format, a warning
        not to modify it, the channel and the series, the series in words,
        the line format, then one line per column with its name and
        meaning.

    Raises
    ------
    DataFileError
        If ``key`` is not a pair for a measurement file or a triple for a
        double-difference file.
    """
    columns = MEAS_COLUMNS if kind == "meas" else DDIFF_COLUMNS
    lines = [
        f"# das_processor {_TITLES[kind]} file, format 1",
        _WARNING,
        f"# RF channel {channel}. {_named(kind, key)}",
        f"# {_described(kind, key)}",
        f"# One row per 10-minute epoch; '{EMPTY}' marks an empty field.",
        f"# Columns: right-justified, fixed width, separated by '{SEPARATOR}'.",
        *(f"#   {column.name:<24}{column.meaning}" for column in columns),
    ]
    width = WIDTHS[kind]
    return "".join(f"{line.ljust(width)}\n" for line in lines)


def _named(kind: FileKind, key: SeriesKey) -> str:
    """Name a series for the header.

    Parameters
    ----------
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    str
        ``Pair (a, b).`` or ``Triple (r, s, c).``.

    Raises
    ------
    DataFileError
        If the key's length does not fit the kind of file.
    """
    size, series = (_PAIR, "pair") if kind == "meas" else (_TRIPLE, "triple")
    if len(key) != size:
        _fail(f"a {_TITLES[kind]} file is for a {series}: {key}")
    return f"{series.capitalize()} ({', '.join(key)})."


def _described(kind: FileKind, key: SeriesKey) -> str:
    """Say in words what a series measures.

    Parameters
    ----------
    kind : {'meas', 'ddiff'}
        The kind of file.
    key : (str, str) or (str, str, str)
        The series, of the length the kind needs.

    Returns
    -------
    str
        For a pair, the reference measured against itself, a reference or
        a clock; for a triple, the clock against its local reference, or
        against a remote one through its local one.
    """
    if kind == "meas":
        a, b = key[0], key[1]
        if a == b:
            return f"Reference {a} measured against itself."
        other = "reference" if re.fullmatch(REFERENCE_PATTERN, b) else "clock"
        return f"Reference {a} measured against {other} {b}."
    r, s, c = key[0], key[1], key[-1]
    if r == s:
        return f"Clock {c} against its local reference {r}."
    return f"Clock {c} against remote reference {r}, through local reference {s}."


# ------------------------------------------------------------- formatting


def _x_text(x_fs: int) -> str:
    """Write a phase held in whole femtoseconds as ps with three decimals.

    Parameters
    ----------
    x_fs : int
        The phase, fs.

    Returns
    -------
    str
        The phase, ps, e.g. ``1234574.457`` or ``-0.007``.
    """
    whole, part = divmod(abs(x_fs), FS_PER_PS)
    return f"{'-' if x_fs < 0 else ''}{whole}.{part:03d}"


def _float_text(value: float | None) -> str | None:
    """Write a float column, or nothing.

    Parameters
    ----------
    value : float or None
        The value.

    Returns
    -------
    str or None
        ``{:+.16e}``, or ``None`` for an empty field.
    """
    return None if value is None else f"{value:+.16e}"


def _mark_texts(mark: datetime) -> tuple[str, str]:
    """Write an epoch as its UTC mark and its MJD.

    Parameters
    ----------
    mark : datetime
        The ten-minute mark.

    Returns
    -------
    tuple of (str, str)
        The mark and its MJD to six places.
    """
    return format_epoch(mark, datetime_to_mjd(mark), 0, _MJD_DECIMALS)


def _state_texts(row: Row) -> list[str | None]:
    """Write the state and counter columns of a row.

    Parameters
    ----------
    row : Row
        The row.

    Returns
    -------
    list of (str or None)
        One text per state column, ``None`` for an empty field.
    """
    texts: list[str | None] = [
        None if row.x_fs is None else _x_text(row.x_fs),
        _float_text(row.y),
        _float_text(row.d),
        _float_text(row.innovation_scale),
        str(row.segment),
        str(row.step_offset),
        str(row.epochs_in_segment),
        str(row.epochs_since_accept),
        str(row.consecutive_rejects),
    ]
    for index in range(3):
        if index < len(row.rejects):
            when, value = row.rejects[index]
            texts += [_mark_texts(when)[1], _float_text(value)]
        else:
            texts += [None, None]
    texts += [
        str(row.filter_states),
        _float_text(row.time_constant),
        _float_text(row.scale_time_constant),
        row.flags,
    ]
    return texts


def _joined(
    columns: tuple[Column, ...], texts: list[str | None], mark: datetime
) -> str:
    """Lay out a row's texts in their columns.

    Parameters
    ----------
    columns : tuple of Column
        The file's columns.
    texts : list of (str or None)
        One text per column; ``None`` for an empty field.
    mark : datetime
        The row's epoch, for the message.

    Returns
    -------
    str
        The line, without its newline.

    Raises
    ------
    DataFileError
        If a value is wider than its column.
    """
    fields = []
    for column, text in zip(columns, texts, strict=True):
        value = EMPTY if text is None else text
        if len(value) > column.width:
            _fail(
                f"row of {mark}: {column.name} {value!r} does not fit"
                f" its {column.width} characters"
            )
        fields.append(value.rjust(column.width))
    return SEPARATOR.join(fields)


def format_meas_row(record: MeasRecord) -> str:
    """Write a measurement file row (design 5.4).

    Parameters
    ----------
    record : MeasRecord
        The pair's measurement and row at an epoch.

    Returns
    -------
    str
        The line, :data:`MEAS_WIDTH` wide, without its newline.

    Raises
    ------
    DataFileError
        If a value is wider than its column.
    """
    row = record.row
    pair = record.measurement
    texts: list[str | None] = list(_mark_texts(row.interpolated_datetime))
    if pair is None:
        texts += [None] * 6
    else:
        texts += [
            pair.measurement_datetime.isoformat(sep=" ", timespec="microseconds"),
            f"{pair.measurement_mjd:.{_MJD_DECIMALS}f}",
            str(pair.measured_phase),
            str(pair.rms),
            str(pair.cycle_count),
            str(pair.z),
        ]
    return _joined(MEAS_COLUMNS, texts + _state_texts(row), row.interpolated_datetime)


def format_ddiff_row(record: DdiffRecord) -> str:
    """Write a double-difference file row (design 5.5).

    Parameters
    ----------
    record : DdiffRecord
        The triple's measurement and row at an epoch.

    Returns
    -------
    str
        The line, :data:`DDIFF_WIDTH` wide, without its newline.

    Raises
    ------
    DataFileError
        If a value is wider than its column.
    """
    row = record.row
    triple = record.measurement
    texts: list[str | None] = list(_mark_texts(row.interpolated_datetime))
    if triple is None:
        texts += [None, _float_text(row.innovation), None, None]
    else:
        texts += [
            str(triple.z),
            _float_text(row.innovation),
            _float_text(triple.double_difference_sigma),
            triple.components_used,
        ]
    return _joined(DDIFF_COLUMNS, texts + _state_texts(row), row.interpolated_datetime)


# ---------------------------------------------------------------- parsing


def _fields(line: str, columns: tuple[Column, ...]) -> list[str | None]:
    """Split a row into its fields.

    Parameters
    ----------
    line : str
        The line, without its newline.
    columns : tuple of Column
        The file's columns.

    Returns
    -------
    list of (str or None)
        Each field without its padding; ``None`` for an empty one.

    Raises
    ------
    ValueError
        If the line does not have one field per column.
    """
    fields = line.split(SEPARATOR)
    if len(fields) != len(columns):
        message = f"{len(fields)} fields, not {len(columns)}"
        raise ValueError(message)
    texts = [field.strip() for field in fields]
    return [None if text == EMPTY else text for text in texts]


def _given(text: str | None) -> str:
    """Give a field that must not be empty.

    Parameters
    ----------
    text : str or None
        The field.

    Returns
    -------
    str
        The field.

    Raises
    ------
    ValueError
        If it is empty.
    """
    if text is None:
        message = "a field that is never empty is empty"
        raise ValueError(message)
    return text


def _x_value(text: str) -> int:
    """Read the estimator's phase, ps with three decimals, as whole fs.

    Parameters
    ----------
    text : str
        The field.

    Returns
    -------
    int
        The phase, fs.

    Raises
    ------
    ValueError
        If the field is not written as ps with three decimals.
    """
    if _X_TEXT.fullmatch(text) is None:
        message = f"phase {text!r} is not ps with three decimals"
        raise ValueError(message)
    whole, part = text.lstrip("-").split(".")
    value = int(whole) * FS_PER_PS + int(part)
    return -value if text.startswith("-") else value


def _mark_value(text: str) -> datetime:
    """Read an MJD column back to the ten-minute mark it was written from.

    Parameters
    ----------
    text : str
        The field.

    Returns
    -------
    datetime
        The nearest mark; formatting it again shows whether it was one.
    """
    return floor_to_ten_minutes(mjd_to_datetime(float(text)) + _HALF_EPOCH)


def _float_value(text: str) -> float:
    """Read a float column.

    Parameters
    ----------
    text : str
        The field.

    Returns
    -------
    float
        The value.

    Raises
    ------
    ValueError
        If the field is not a number, or not a finite one, which no column
        holds.
    """
    value = float(text)
    if not math.isfinite(value):
        message = f"{text!r} is not a finite number"
        raise ValueError(message)
    return value


def _optional_float(text: str | None) -> float | None:
    """Read a float column that may be empty.

    Parameters
    ----------
    text : str or None
        The field.

    Returns
    -------
    float or None
        The value, or ``None``.
    """
    return None if text is None else _float_value(text)


def _row(mark: datetime, innovation: float | None, texts: list[str | None]) -> Row:
    """Build a row from its epoch, its innovation and its state columns.

    Parameters
    ----------
    mark : datetime
        The epoch start.
    innovation : float or None
        The innovation.
    texts : list of (str or None)
        The state columns' fields.

    Returns
    -------
    Row
        The row.

    Raises
    ------
    ValueError
        If a field is not of its kind or not finite, or the fields make no
        valid row: a pydantic ValidationError, which is not logged, so a
        file check can read a damaged line quietly.
    """
    x, y, d, scale, segment, step, in_segment, since, rejected = texts[:9]
    rejects: list[Reject] = []
    for index in range(3):
        when, value = texts[9 + 2 * index], texts[10 + 2 * index]
        if when is not None or value is not None:
            rejects.append((_mark_value(_given(when)), _float_value(_given(value))))
    states, time_constant, scale_constant, flags = texts[15:]
    return Row.model_validate(
        {
            "interpolated_datetime": mark,
            "innovation": innovation,
            "x_fs": None if x is None else _x_value(x),
            "y": _optional_float(y),
            "d": _optional_float(d),
            "innovation_scale": _optional_float(scale),
            "segment": int(_given(segment)),
            "step_offset": int(_given(step)),
            "epochs_in_segment": int(_given(in_segment)),
            "epochs_since_accept": int(_given(since)),
            "consecutive_rejects": int(_given(rejected)),
            "rejects": tuple(rejects),
            "filter_states": int(_given(states)),
            "time_constant": _optional_float(time_constant),
            "scale_time_constant": _float_value(_given(scale_constant)),
            "flags": _given(flags),
        }
    )


def _pair_measurement(texts: list[str | None], flags: str) -> PairMeasurement | None:
    """Build a pair's measurement from its columns.

    Parameters
    ----------
    texts : list of (str or None)
        The six measurement columns' fields.
    flags : str
        The row's flags, whose S marks a slip correction.

    Returns
    -------
    PairMeasurement or None
        The measurement, or ``None`` when every field is empty.

    Raises
    ------
    ValueError
        If a field is not of its kind, or the fields do not make a
        measurement.
    """
    if all(text is None for text in texts):
        return None
    _, mjd, phase, rms, cycles, z = (_given(text) for text in texts)
    return PairMeasurement(
        measurement_mjd=_float_value(mjd),
        measured_phase=int(phase),
        rms=int(rms),
        cycle_count=int(cycles),
        z=int(z),
        slip="S" in flags,
    )


def _attempt[RecordT: (MeasRecord, DdiffRecord)](
    line: str, build: Callable[[], RecordT], again: Callable[[RecordT], str]
) -> tuple[RecordT | None, str, Exception | None]:
    """Build a record from a line, and check it gives the line back.

    Parameters
    ----------
    line : str
        The line.
    build : callable
        Builds the record from the line.
    again : callable
        Formats a record.

    Returns
    -------
    tuple of (record or None, str, Exception or None)
        The record, or ``None`` with what is wrong with the line and the
        error that showed it. Nothing is logged here; a record's own
        check logs the rule a line breaks.
    """
    try:
        record = build()
    except (
        ValueError,
        FilterError,
        PhaseError,
        DataFileError,
    ) as exc:
        return None, f"row {line[:25]!r} does not parse: {describe_error(exc)}", exc
    if again(record) != line:
        return (
            None,
            f"row {line[:25]!r} is not written as das_processor writes it",
            None,
        )
    return record, "", None


def _parsed[RecordT: (MeasRecord, DdiffRecord)](
    line: str, build: Callable[[], RecordT], again: Callable[[RecordT], str]
) -> RecordT:
    """Build a record from a line, and check it gives the line back.

    Parameters
    ----------
    line : str
        The line.
    build : callable
        Builds the record from the line.
    again : callable
        Formats a record.

    Returns
    -------
    MeasRecord or DdiffRecord
        The record.

    Raises
    ------
    DataFileError
        If the line does not make a record, or the record does not format
        back to exactly the line.
    """
    record, problem, cause = _attempt(line, build, again)
    if record is None:
        _fail(problem, cause)
    return record


def _meas_record(line: str) -> MeasRecord:
    """Build a measurement file record from a line, unchecked against it.

    Parameters
    ----------
    line : str
        The line.

    Returns
    -------
    MeasRecord
        The record.
    """
    texts = _fields(line, MEAS_COLUMNS)
    row = _row(datetime.fromisoformat(_given(texts[0])), None, texts[8:])
    return MeasRecord(measurement=_pair_measurement(texts[2:8], row.flags), row=row)


def _ddiff_record(line: str) -> DdiffRecord:
    """Build a double-difference file record from a line, unchecked against it.

    Parameters
    ----------
    line : str
        The line.

    Returns
    -------
    DdiffRecord
        The record.
    """
    texts = _fields(line, DDIFF_COLUMNS)
    z, innovation, sigma, used = texts[2:6]
    row = _row(
        datetime.fromisoformat(_given(texts[0])),
        _optional_float(innovation),
        texts[6:],
    )
    triple = None
    if z is not None or sigma is not None or used is not None:
        triple = TripleMeasurement.model_validate(
            {
                "z": int(_given(z)),
                "double_difference_sigma": _float_value(_given(sigma)),
                "components_used": _given(used),
                "cold": False,
            }
        )
    return DdiffRecord(measurement=triple, row=row)


def parse_meas_row(line: str) -> MeasRecord:
    """Read a measurement file row (design 5.4).

    Parameters
    ----------
    line : str
        The line, without its newline.

    Returns
    -------
    MeasRecord
        The measurement and the row; the row has no innovation, which the
        file does not hold.

    Raises
    ------
    DataFileError
        If the line does not parse, makes no valid record, or is not
        exactly what :func:`format_meas_row` gives for that record.
    """
    return _parsed(line, lambda: _meas_record(line), format_meas_row)


def parse_ddiff_row(line: str) -> DdiffRecord:
    """Read a double-difference file row (design 5.5).

    Parameters
    ----------
    line : str
        The line, without its newline.

    Returns
    -------
    DdiffRecord
        The measurement and the row; the measurement is not marked cold,
        which the file does not hold.

    Raises
    ------
    DataFileError
        If the line does not parse, makes no valid record, or is not
        exactly what :func:`format_ddiff_row` gives for that record.
    """
    return _parsed(line, lambda: _ddiff_record(line), format_ddiff_row)


# ------------------------------------------------- file check and last row


def _parse_line(text: str, kind: FileKind) -> MeasRecord | DdiffRecord:
    """Read a row of either kind of file.

    Parameters
    ----------
    text : str
        The line, without its newline.
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    MeasRecord or DdiffRecord
        The record.

    Raises
    ------
    DataFileError
        If the line is not a row of that file.
    """
    if kind == "meas":
        return parse_meas_row(text)
    return parse_ddiff_row(text)


def row_epoch(line: bytes, kind: FileKind) -> datetime | None:
    """Give the epoch of a good row (design 5.7).

    Parameters
    ----------
    line : bytes
        One line slot of a file, with its newline if it has one.
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    datetime or None
        The row's epoch when the slot is a whole line, ending in its
        newline, that is ASCII and parses as a row of the file; otherwise
        ``None``. A header line never parses as a row. A damaged line is
        reported by the roll-back that removes it; only a line that parses
        but breaks a record's rules, such as a measurement on a P row, is
        also logged, by the record's check.
    """
    if not line.endswith(b"\n"):
        return None
    try:
        text = line[:-1].decode("ascii")
    except UnicodeDecodeError:
        return None
    record: MeasRecord | DdiffRecord | None
    if kind == "meas":
        record = _attempt(text, lambda: _meas_record(text), format_meas_row)[0]
    else:
        record = _attempt(text, lambda: _ddiff_record(text), format_ddiff_row)[0]
    return None if record is None else record.row.interpolated_datetime


def _slot(file: BinaryIO, index: int, size: int) -> bytes:
    """Read one line slot of a file.

    Parameters
    ----------
    file : BinaryIO
        The open file.
    index : int
        The slot, from 0 at the first header line.
    size : int
        A line's size, newline included.

    Returns
    -------
    bytes
        The slot's bytes.
    """
    file.seek(index * size)
    return file.read(size)


def good_through(path: Path, kind: FileKind) -> datetime | None:
    """Give the epoch of a file's last good row (design 5.2, 5.7).

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    datetime or None
        For a sound file, its length its header plus whole rows and its
        last row good, that row's epoch, from one short read. Otherwise the
        epoch of the row before its first line that is not a good row.
        ``None`` when it holds no whole row.

    Raises
    ------
    DataFileError
        If the file cannot be read, or its first row is not good, so its
        rows cannot be placed in time.
    """
    size, header_lines = WIDTHS[kind] + 1, HEADER_LINES[kind]
    try:
        with path.open("rb") as file:
            length = file.seek(0, os.SEEK_END)
            rows = length // size - header_lines
            if rows < 1:
                return None
            if length % size == 0:
                last = row_epoch(_slot(file, header_lines + rows - 1, size), kind)
                if last is not None:
                    return last
            good = None
            for index in range(rows):
                epoch = row_epoch(_slot(file, header_lines + index, size), kind)
                if epoch is None:
                    break
                good = epoch
    except OSError as exc:
        _fail(f"cannot read data file {path}: {exc}", exc)
    if good is None:
        _fail(f"{path} has a damaged first row, so its rows cannot be placed in time")
    return good


def read_last_row(path: Path, kind: FileKind) -> Row:
    """Read the last row of a sound file (design 5.7).

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    Row
        Its last row.

    Raises
    ------
    DataFileError
        If the file cannot be read, is not sound, or its last row is not a
        row of the file.
    """
    return read_last_record(path, kind).row


def read_last_record(path: Path, kind: FileKind) -> MeasRecord | DdiffRecord:
    """Read the last record of a sound file: its measurement and row.

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    MeasRecord or DdiffRecord
        Its last record.

    Raises
    ------
    DataFileError
        If the file cannot be read, is not sound, or its last row is not a
        row of the file.
    """
    size, header_lines = WIDTHS[kind] + 1, HEADER_LINES[kind]
    try:
        with path.open("rb") as file:
            length = file.seek(0, os.SEEK_END)
            rows = length // size - header_lines
            if rows < 1 or length % size != 0:
                _fail(f"data file {path} is not sound: {length} bytes")
            line = _slot(file, header_lines + rows - 1, size)
    except OSError as exc:
        _fail(f"cannot read data file {path}: {exc}", exc)
    try:
        text = line[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        _fail(f"data file {path} is not sound: its last row is not ASCII", exc)
    return _parse_line(text, kind)


# --------------------------------------------------- day buffer and write

_KIND_ORDER: Final[dict[FileKind, int]] = {"meas": 0, "ddiff": 1}
"""The order the kinds of file are written in: measurement files first."""


class DayBuffer:
    """The rows computed since the last write, for every file (design 5.8).

    Rows are written a UTC day at a time. Until then each file's rows are
    held here as text, and each series' newest row as a row, which the next
    epoch takes as the series' last row.

    Parameters
    ----------
    channel : {'a', 'b'}
        The RF channel, whose name a new file's header gives.

    Attributes
    ----------
    texts : dict of Path to str
        Each file's lines since the last write, newlines included.
    last : dict of series key to Row
        Each series' newest row, as a later run would read it back.
    start : datetime or None
        The earliest epoch of the rows since the last write; ``None`` when
        there are none.
    """

    def __init__(self, channel: RfChannel, journal: Path | None = None) -> None:
        """Start an empty buffer.

        Parameters
        ----------
        channel : {'a', 'b'}
            The RF channel.
        journal : Path or None, optional
            The write journal :func:`write_buffer` keeps while it writes; no
            journal is kept when ``None``.
        """
        self.channel: RfChannel = channel
        self.journal = journal
        self.start: datetime | None = None
        self.texts: dict[Path, str] = {}
        self.last: dict[SeriesKey, Row] = {}
        self._series: dict[Path, tuple[FileKind, SeriesKey]] = {}

    def add(self, path: Path, key: SeriesKey, record: MeasRecord | DdiffRecord) -> None:
        """Add a series' record for an epoch.

        Parameters
        ----------
        path : Path
            The series' file.
        key : (str, str) or (str, str, str)
            The series.
        record : MeasRecord or DdiffRecord
            Its measurement and row at the epoch.

        Raises
        ------
        DataFileError
            If a value does not fit its column, the line does not read back
            as written, or ``path`` was given another series or kind of
            file before.

        Notes
        -----
        The record is formatted and read back, and the row read back is
        kept as the series' newest: the row a later run would read from the
        file, so the next epoch is the same whether it takes the row from
        here or from the file (I5).
        """
        kind: FileKind = "meas" if isinstance(record, MeasRecord) else "ddiff"
        if self._series.setdefault(path, (kind, key)) != (kind, key):
            _fail(f"{path} holds the {self._series[path]} series, not {(kind, key)}")
        if isinstance(record, MeasRecord):
            line = format_meas_row(record)
            back = parse_meas_row(line).row
        else:
            line = format_ddiff_row(record)
            back = parse_ddiff_row(line).row
        self.texts[path] = self.texts.get(path, "") + line + "\n"
        self.last[key] = back
        self._started(back.interpolated_datetime)

    def _started(self, mark: datetime | None) -> None:
        """Note an epoch of a buffered row, keeping the earliest.

        Parameters
        ----------
        mark : datetime or None
            The epoch; ``None`` changes nothing.
        """
        if mark is not None and (self.start is None or mark < self.start):
            self.start = mark

    def take(self, other: DayBuffer) -> None:
        """Move another buffer's rows into this one, all of them or none.

        Parameters
        ----------
        other : DayBuffer
            A buffer of rows added after this one's, such as one epoch's.

        Raises
        ------
        DataFileError
            If a path of ``other`` is another series' or kind's in this
            buffer; this buffer is then unchanged.
        """
        for path in other.texts:
            series = other.series_of(path)
            if self._series.get(path, series) != series:
                _fail(f"{path} holds the {self._series[path]} series, not {series}")
        for path, text in other.texts.items():
            self._series[path] = other.series_of(path)
            self.texts[path] = self.texts.get(path, "") + text
        self.last.update(other.last)
        self._started(other.start)

    def series_of(self, path: Path) -> tuple[FileKind, SeriesKey]:
        """Give the kind of file and the series a buffered path is for.

        Parameters
        ----------
        path : Path
            A path the buffer holds text for.

        Returns
        -------
        tuple of (FileKind, series key)
            Its kind of file and series.
        """
        return self._series[path]


def write_buffer(buffer: DayBuffer) -> None:
    """Write every file's buffered rows: all of them, or none (design 5.8).

    Parameters
    ----------
    buffer : DayBuffer
        The buffer; its texts are emptied after the write, its newest rows
        kept.

    Raises
    ------
    DataFileError
        Before any file is opened, if a buffered text is not ASCII, an
        existing file is not a regular file this process can write or its
        length is not its header plus whole rows, a new file's directory is
        not one this process can write into or a file of that name is
        already there, a write journal is already there, or the free space
        does not cover every byte to be written; nothing is changed then.
        While writing, if the device fails.

    Notes
    -----
    When the buffer has a journal, the first epoch of its rows is written
    to it and flushed before any data file is opened, and the journal is
    deleted after the last flush. A run that finds the journal knows this
    write stopped part way, whichever files it reached or created, and
    rolls every file back to before that epoch (see :func:`read_journal`).
    """
    data = _prepared(buffer)
    if not data:
        return
    if buffer.journal is not None and buffer.start is not None:
        _write_journal(buffer.journal, buffer.start)
    order = sorted(data, key=lambda path: _write_order(buffer, path))
    new_directories: set[Path] = set()
    for path in order:
        is_new = not os.path.lexists(path)
        try:
            with path.open("xb" if is_new else "ab") as file:
                file.write(data[path])
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            _fail(f"cannot write data file {path}: {exc}", exc)
        if is_new:
            new_directories.add(path.parent)
    for directory in sorted(new_directories):
        _sync_directory(directory)
    if buffer.journal is not None:
        _delete(buffer.journal)
    buffer.texts.clear()
    buffer.start = None


def _write_order(buffer: DayBuffer, path: Path) -> tuple[int, SeriesKey]:
    """Give a file's place in the write: measurement files first, by series.

    Parameters
    ----------
    buffer : DayBuffer
        The buffer.
    path : Path
        A buffered file.

    Returns
    -------
    tuple of (int, series key)
        Its kind's rank and its series.
    """
    kind, key = buffer.series_of(path)
    return _KIND_ORDER[kind], key


def _prepared(buffer: DayBuffer) -> dict[Path, bytes]:
    """Check every file can be written, and give the bytes to write (design 5.8).

    Parameters
    ----------
    buffer : DayBuffer
        The buffer.

    Returns
    -------
    dict of Path to bytes
        The bytes to append to each file; a new file's begin with its
        header.

    Raises
    ------
    DataFileError
        If any check of the prepare step fails; no file is opened.
    """
    data: dict[Path, bytes] = {}
    for path, text in buffer.texts.items():
        kind, key = buffer.series_of(path)
        if os.path.lexists(path):
            _check_existing(path, kind)
            prefix = ""
        else:
            _check_new(path)
            prefix = header(kind, buffer.channel, key)
        try:
            data[path] = (prefix + text).encode("ascii")
        except UnicodeEncodeError as exc:
            _fail(f"the rows for {path} are not ASCII", exc)
    if buffer.journal is not None and data:
        if os.path.lexists(buffer.journal):
            _fail(f"write journal {buffer.journal} is there: a write is still open")
        _check_new(buffer.journal)
    _check_space(data)
    return data


def _check_existing(path: Path, kind: FileKind) -> None:
    """Refuse an existing file that cannot be appended to soundly.

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        Its kind.

    Raises
    ------
    DataFileError
        If it is not a regular file, this process cannot write it, or its
        length is not its header plus whole rows.
    """
    if path.is_symlink() or not path.is_file():
        _fail(f"data file {path} is not a regular file")
    if not os.access(path, os.W_OK):
        _fail(f"data file {path} cannot be written")
    size = WIDTHS[kind] + 1
    length = path.stat().st_size
    if length % size != 0 or length // size <= HEADER_LINES[kind]:
        _fail(f"data file {path} is not sound: {length} bytes")


def _check_new(path: Path) -> None:
    """Refuse a new file whose directory cannot be written into.

    Parameters
    ----------
    path : Path
        The file, which does not exist yet.

    Raises
    ------
    DataFileError
        If its directory is not a directory this process can write into.
    """
    directory = path.parent
    if not directory.is_dir() or not os.access(directory, os.W_OK | os.X_OK):
        _fail(f"file {path} cannot be created in {directory}")


def _check_space(data: dict[Path, bytes]) -> None:
    """Refuse a write the free space does not cover.

    Parameters
    ----------
    data : dict of Path to bytes
        The bytes to write to each file.

    Raises
    ------
    DataFileError
        If, on any device, the bytes to write are more than its free space.
    """
    needed: dict[int, tuple[Path, int]] = {}
    for path, chunk in data.items():
        device = path.parent.stat().st_dev
        directory, total = needed.get(device, (path.parent, 0))
        needed[device] = (directory, total + len(chunk))
    for directory, total in needed.values():
        free = shutil.disk_usage(directory).free
        if total > free:
            _fail(f"{total} bytes to write in {directory}, only {free} free")


def _sync_directory(directory: Path) -> None:
    """Flush a directory's entries to the device, so a new file's name is kept.

    Parameters
    ----------
    directory : Path
        The directory.

    Raises
    ------
    DataFileError
        If the device fails.
    """
    try:
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
    except OSError as exc:
        _fail(f"cannot flush directory {directory}: {exc}", exc)


# ----------------------------------------------------------- roll-back, redo

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch, T."""

type _Cut = Literal["kept", "cut", "deleted"]
"""What keeping a file's rows through an epoch did to it."""


def _keep_through(path: Path, kind: FileKind, through: datetime | None) -> _Cut:
    """Keep a file's rows up to and including an epoch, and remove the rest.

    Parameters
    ----------
    path : Path
        The file, whose rows up to ``through`` are good.
    kind : {'meas', 'ddiff'}
        Its kind.
    through : datetime or None
        The last epoch to keep; ``None`` to keep none.

    Returns
    -------
    {'kept', 'cut', 'deleted'}
        Whether the file was left as it was, truncated just after its row
        for ``through``, or deleted because it had no row at or before it.

    Raises
    ------
    DataFileError
        If the file cannot be read, changed or deleted, its first row is
        not good, or the row for ``through`` found by counting one row per
        epoch from the first is missing or of another epoch.
    """
    size, header_lines = WIDTHS[kind] + 1, HEADER_LINES[kind]
    try:
        with path.open("rb") as file:
            length = file.seek(0, os.SEEK_END)
            rows = length // size - header_lines
            first = (
                row_epoch(_slot(file, header_lines, size), kind) if rows > 0 else None
            )
            if through is None or first is None or through < first:
                keep = 0
            else:
                keep = (through - first) // _EPOCH + 1
                if keep > rows:
                    _fail(f"{path} has no row for {through}")
                found = row_epoch(_slot(file, header_lines + keep - 1, size), kind)
                if found != through:
                    _fail(
                        f"{path} does not hold one row per epoch: {found} for {through}"
                    )
    except OSError as exc:
        _fail(f"cannot read data file {path}: {exc}", exc)
    if keep == 0:
        _delete(path)
        return "deleted"
    end = (header_lines + keep) * size
    if end == length:
        return "kept"
    _truncate(path, end)
    return "cut"


def _delete(path: Path) -> None:
    """Delete a file, and flush its directory so the deletion is kept.

    Parameters
    ----------
    path : Path
        The file.

    Raises
    ------
    DataFileError
        If the file cannot be deleted or the device fails.
    """
    try:
        path.unlink()
    except OSError as exc:
        _fail(f"cannot delete {path}: {exc}", exc)
    _sync_directory(path.parent)


def _truncate(path: Path, end: int) -> None:
    """Cut a data file to a length, and flush it.

    Parameters
    ----------
    path : Path
        The file.
    end : int
        Its new length, bytes.

    Raises
    ------
    DataFileError
        If the file cannot be changed or the device fails.
    """
    try:
        with path.open("r+b") as file:
            file.truncate(end)
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        _fail(f"cannot cut data file {path}: {exc}", exc)


def roll_back(path: Path, kind: FileKind, common: datetime | None) -> None:
    """Roll a file back to the epoch every file holds (design 6.7).

    Parameters
    ----------
    path : Path
        The file, good through ``common`` or later.
    kind : {'meas', 'ddiff'}
        Its kind.
    common : datetime or None
        L, the oldest epoch any file of the channel is good through;
        ``None`` when no file holds a whole row.

    Raises
    ------
    DataFileError
        If the file cannot be read, changed or deleted, or does not hold
        one row per epoch.

    Notes
    -----
    The file is truncated just after its row for ``common``, which also
    removes any damaged or torn line after it, or deleted when it has no
    row at or before ``common``. A file that already ends there is left as
    it is. Each file changed is logged at WARNING with its path and the
    epoch.
    """
    done = _keep_through(path, kind, common)
    if done == "cut":
        _log.warning("data file %s rolled back to %s", path, common)
    elif done == "deleted":
        _log.warning("data file %s deleted: no row at or before %s", path, common)


def redo_from(series: Iterable[tuple[Path, FileKind]], mark: datetime) -> None:
    """Delete every row at or after an epoch from every file (design 6.5).

    Parameters
    ----------
    series : iterable of (Path, FileKind)
        Every file of the channel, measurement and double-difference, with
        its kind.
    mark : datetime
        The epoch to reprocess from.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, its first row is not
        good, or it does not hold one row per epoch.

    Notes
    -----
    Each file is truncated just before its first row at or after ``mark``,
    and deleted when it has no earlier row. A file is never kept past its
    last good row, so a damaged one is cut there instead, and the roll-back
    that follows (see :func:`roll_back`) brings every file to one epoch.
    Running it again after an interruption finishes the deletion: a file
    already cut is left as it is.
    """
    for path, kind in series:
        good = good_through(path, kind)
        through = mark - _EPOCH if good is None else min(mark - _EPOCH, good)
        done = _keep_through(path, kind, None if good is None else through)
        if done != "kept":
            _log.info("data file %s %s for a redo from %s", path, done, mark)


def ensure_archives(processed_path: Path) -> None:
    """Make the two archive directories when they are missing (design 5.1).

    Parameters
    ----------
    processed_path : Path
        The directory holding the archives.

    Raises
    ------
    DataFileError
        If a directory cannot be made, or its name is taken by something
        that is not a directory.

    Notes
    -----
    A directory made here is flushed into ``processed_path``, so the files
    written into it later are found after a crash.
    """
    made = False
    for name in (MEAS_SUBDIRECTORY, DDIFF_SUBDIRECTORY):
        directory = processed_path / name
        if directory.is_dir():
            continue
        try:
            directory.mkdir()
        except OSError as exc:
            _fail(f"cannot make archive {directory}: {exc}", exc)
        made = True
    if made:
        _sync_directory(processed_path)


# ---------------------------------------------------------- the write journal


def _write_journal(journal: Path, start: datetime) -> None:
    """Write the journal of a write about to start, and flush it.

    Parameters
    ----------
    journal : Path
        The journal, which is not there yet.
    start : datetime
        The first epoch of the rows to write.

    Raises
    ------
    DataFileError
        If the journal cannot be written or the device fails.
    """
    try:
        with journal.open("xb") as file:
            file.write(f"{start.isoformat()}\n".encode("ascii"))
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        _fail(f"cannot write journal {journal}: {exc}", exc)
    _sync_directory(journal.parent)


def read_journal(journal: Path) -> datetime | None:
    """Give the first epoch of a write that stopped part way (design 6.7).

    Parameters
    ----------
    journal : Path
        The channel's write journal.

    Returns
    -------
    datetime or None
        The first epoch the stopped write was writing; ``None`` when there
        is no journal, or it is not whole, which means it was not flushed
        and so no data file was opened.

    Raises
    ------
    DataFileError
        If the journal is there but cannot be read.
    """
    if not os.path.lexists(journal):
        return None
    try:
        text = journal.read_bytes()
    except OSError as exc:
        _fail(f"cannot read journal {journal}: {exc}", exc)
    try:
        start = datetime.fromisoformat(text.decode("ascii").rstrip("\n"))
    except (
        UnicodeDecodeError,
        ValueError,
    ):
        return None
    return start if start.tzinfo is not None and text.endswith(b"\n") else None


def clear_journal(journal: Path) -> None:
    """Delete the write journal once its stopped write is undone.

    Parameters
    ----------
    journal : Path
        The channel's write journal; nothing is done when it is not there.

    Raises
    ------
    DataFileError
        If it cannot be deleted or the device fails.
    """
    if os.path.lexists(journal):
        _delete(journal)
