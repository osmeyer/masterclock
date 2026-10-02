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
"""

import os
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Final, Literal, NamedTuple, NoReturn, Self

from pydantic import BaseModel, ConfigDict, model_validator

from masterclock.app.exceptions import describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.epochs import floor_to_ten_minutes, format_epoch
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.measurements import PairMeasurement, TripleMeasurement
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain.exceptions import FilterError, PhaseError
from masterclock.domain.phase import FS_PER_PS
from masterclock.domain.references import REFERENCE_PATTERN
from masterclock.domain.series import (
    PairKey,
    Reject,
    Row,
    SeriesKey,
    build_row,
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
    Column("rms", 6, "RMS from the DAS, ps"),
    Column("switch", 4, "switch position from the DAS"),
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
        if (
            self.measurement.measurement.interpolated_datetime
            != self.row.interpolated_datetime
        ):
            _fail(
                f"row of {self.row.interpolated_datetime}: its measurement is of"
                f" the epoch of {self.measurement.measurement.interpolated_datetime}"
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
    key : (str, str) or (str, str, str)
        The series.

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
        texts += [None] * 7
    else:
        raw = pair.measurement
        texts += [
            raw.measurement_datetime.isoformat(sep=" ", timespec="microseconds"),
            f"{raw.measurement_mjd:.{_MJD_DECIMALS}f}",
            str(raw.measured_phase),
            str(raw.rms),
            raw.switch,
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
    return None if text is None else float(text)


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
        If a field is not of its kind.
    FilterError
        If the fields make no valid row.
    """
    x, y, d, scale, segment, step, in_segment, since, rejected = texts[:9]
    rejects: list[Reject] = []
    for index in range(3):
        when, value = texts[9 + 2 * index], texts[10 + 2 * index]
        if when is not None or value is not None:
            rejects.append((_mark_value(_given(when)), float(_given(value))))
    states, time_constant, scale_constant, flags = texts[15:]
    return build_row(
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
            "scale_time_constant": float(_given(scale_constant)),
            "flags": _given(flags),
        }
    )


def _pair_measurement(
    texts: list[str | None], key: PairKey, flags: str
) -> PairMeasurement | None:
    """Build a pair's measurement from its columns.

    Parameters
    ----------
    texts : list of (str or None)
        The seven measurement columns' fields.
    key : (str, str)
        The file's pair.
    flags : str
        The row's flags, whose S marks a slip correction.

    Returns
    -------
    PairMeasurement or None
        The measurement, or ``None`` when every field is empty.

    Raises
    ------
    ValueError
        If a field is not of its kind, the fields do not make a DAS
        measurement, or it is against another reference than the pair's.
    """
    if all(text is None for text in texts):
        return None
    _, mjd, phase, rms, switch, cycles, z = (_given(text) for text in texts)
    raw = DASMeasurement(
        measurement_mjd=float(mjd),
        measured_phase=int(phase),
        rms=int(rms),
        switch=switch,
        clock=key[1],
    )
    if raw.reference != key[0]:
        message = f"the row's measurement is against {raw.reference}, not {key[0]}"
        raise ValueError(message)
    return PairMeasurement(
        measurement=raw, cycle_count=int(cycles), z=int(z), slip="S" in flags
    )


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
    try:
        record = build()
    except (
        ValueError,
        FilterError,
        PhaseError,
    ) as exc:
        _fail(f"row {line[:25]!r} does not parse: {describe_error(exc)}", exc)
    if again(record) != line:
        _fail(f"row {line[:25]!r} is not written as das_processor writes it")
    return record


def parse_meas_row(line: str, key: PairKey) -> MeasRecord:
    """Read a measurement file row (design 5.4).

    Parameters
    ----------
    line : str
        The line, without its newline.
    key : (str, str)
        The file's pair.

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

    def build() -> MeasRecord:
        """Build the record from the line's fields."""
        texts = _fields(line, MEAS_COLUMNS)
        row = _row(datetime.fromisoformat(_given(texts[0])), None, texts[9:])
        return MeasRecord(
            measurement=_pair_measurement(texts[2:9], key, row.flags), row=row
        )

    return _parsed(line, build, format_meas_row)


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

    def build() -> DdiffRecord:
        """Build the record from the line's fields."""
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
                    "double_difference_sigma": float(_given(sigma)),
                    "components_used": _given(used),
                    "cold": False,
                }
            )
        return DdiffRecord(measurement=triple, row=row)

    return _parsed(line, build, format_ddiff_row)


# ------------------------------------------------- file check and last row


def _parse_line(text: str, kind: FileKind, key: SeriesKey) -> MeasRecord | DdiffRecord:
    """Read a row of either kind of file.

    Parameters
    ----------
    text : str
        The line, without its newline.
    kind : {'meas', 'ddiff'}
        The kind of file.
    key : (str, str) or (str, str, str)
        The file's series.

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
        return parse_meas_row(text, (key[0], key[1]))
    return parse_ddiff_row(text)


def row_epoch(line: bytes, kind: FileKind, key: SeriesKey) -> datetime | None:
    """Give the epoch of a good row (design 5.7).

    Parameters
    ----------
    line : bytes
        One line slot of a file, with its newline if it has one.
    kind : {'meas', 'ddiff'}
        The kind of file.
    key : (str, str) or (str, str, str)
        The file's series.

    Returns
    -------
    datetime or None
        The row's epoch when the slot is a whole line, ending in its
        newline, that is ASCII and parses as a row of the file; otherwise
        ``None``. A header line never parses as a row.
    """
    if not line.endswith(b"\n"):
        return None
    try:
        return _parse_line(
            line[:-1].decode("ascii"), kind, key
        ).row.interpolated_datetime
    except (
        UnicodeDecodeError,
        DataFileError,
    ):
        return None


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


def good_through(path: Path, kind: FileKind, key: SeriesKey) -> datetime | None:
    """Give the epoch of a file's last good row (design 5.2, 5.7).

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        The kind of file.
    key : (str, str) or (str, str, str)
        The file's series.

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
                last = row_epoch(_slot(file, header_lines + rows - 1, size), kind, key)
                if last is not None:
                    return last
            good = None
            for index in range(rows):
                epoch = row_epoch(_slot(file, header_lines + index, size), kind, key)
                if epoch is None:
                    break
                good = epoch
    except OSError as exc:
        _fail(f"cannot read data file {path}: {exc}", exc)
    if good is None:
        _fail(f"{path} has a damaged first row, so its rows cannot be placed in time")
    return good


def read_last_row(path: Path, kind: FileKind, key: SeriesKey) -> Row:
    """Read the last row of a sound file (design 5.7).

    Parameters
    ----------
    path : Path
        The file.
    kind : {'meas', 'ddiff'}
        The kind of file.
    key : (str, str) or (str, str, str)
        The file's series.

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
    return _parse_line(text, kind, key).row
