"""The two output files: their columns, headers and rows.

Every series has one file: a pair a measurement file, a triple a
double-difference file. A file is its header, then its rows. The header
says in words what the kind of file is and what each column holds,
numbered, and warns that only das_processor may write the file; each
header line is as long as its text, and every file of a kind has the same
header, so the same size. The file's name says which series it holds.
Every row is the same width W, worked out from the file's column table.
Every line ends with a newline. The program never reads the header back.

A row is the values of one epoch, each right-justified in a column exactly
as wide as its values need and separated by ``", "``; ``-`` marks an empty
field. Floats are written as ``{:+.16e}``, seventeen significant digits,
which give every float back exactly; the estimator's phase, held in whole
femtoseconds, is written in ps with three decimals. A value too wide for its
column is refused before anything is written.

A row read back (:func:`parse_meas_row`, :func:`parse_ddiff_row`) is data
from outside the program: its values are checked by pydantic models, the
row against every rule a row keeps, and it is formatted again and must give
the same line, so a line is accepted only in the one form das_processor
writes. The measurement file holds no innovation, so a measurement row
reads back without one. A disabled pair's row (O) holds its reading with no
cycle count and the z of the pair's newest row, or none.

Rows are buffered and written a day at a time (:func:`write_buffer`), every
check made before the first byte, under a write journal that names the
first epoch the run writes. The files are flushed to the device only by the
run's final write (:func:`write_final`), which then deletes the journal. A
run that starts checks every file, and cuts every file back to before the
epoch a journal names and to the last good row of any damaged file, so the
files stay in step (:func:`cut_epoch`, :func:`roll_back`,
:func:`read_journal`). Files may end at different epochs: a series writes no
row for an epoch it is not in, nor while it is dormant with no measurement
or disabled with no reading.
"""

import dataclasses
import math
import os
import re
import shutil
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, BinaryIO, Final, Literal, NamedTuple, NoReturn, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from masterclock.app.exceptions import describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.config import DDIFF_SUBDIRECTORY, MEAS_SUBDIRECTORY
from masterclock.das_processor.epochs import floor_to_ten_minutes, format_epoch
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import RMS_WIDTH
from masterclock.domain.exceptions import FilterError, PhaseError
from masterclock.domain.measurements import (
    DisabledReading,
    PairMeasurement,
    TripleMeasurement,
)
from masterclock.domain.phase import EPOCH_SECONDS, FS_PER_PS, PHASE_MAX
from masterclock.domain.series import (
    FilterStates,
    PairKey,
    Reject,
    Row,
    SeriesKey,
    check_row,
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
    Column(
        "epochs_since_accept",
        9,
        "rows since the last accepted measurement, not counting dormant rows"
        " that buffer a measurement",
    ),
    Column("consecutive_rejects", 9, "consecutive counted rejects"),
    Column("reject1_mjd", 13, "reject buffer, oldest: epoch start, MJD"),
    Column(
        "reject1_innovation",
        23,
        "reject buffer, oldest: innovation, ps; while dormant, a measurement z, ps",
    ),
    Column("reject2_mjd", 13, "reject buffer, middle: epoch start, MJD"),
    Column(
        "reject2_innovation",
        23,
        "reject buffer, middle: innovation, ps; while dormant, a measurement z, ps",
    ),
    Column("reject3_mjd", 13, "reject buffer, newest: epoch start, MJD"),
    Column(
        "reject3_innovation",
        23,
        "reject buffer, newest: innovation, ps; while dormant, a measurement z, ps",
    ),
    Column("filter_states", 1, "estimator states: 1, 2 or 3"),
    Column("time_constant", 23, "estimator time constant, epochs"),
    Column("scale_time_constant", 23, "innovation-scale averaging constant, epochs"),
    Column(
        "flags",
        8,
        "A accepted, R rejected, X excluded, P predicted, O disabled,"
        " D dormant, S slip corrected, N new segment, U unsettled",
    ),
)
"""The columns every row ends with: the estimator's state and counters."""

MEAS_COLUMNS: Final[tuple[Column, ...]] = (
    *_EPOCH_COLUMNS,
    Column("measurement_datetime", 32, "measurement time, UTC"),
    Column("measurement_mjd", 13, "measurement time, MJD"),
    Column("measured_phase", 6, "raw phase from the DAS, ps"),
    Column("rms", RMS_WIDTH, "RMS from the DAS, ps"),
    Column("cycle_count", 12, "whole periods added in decycling; - when disabled"),
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

_PREAMBLE: Final[int] = 4
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
"""The width W of every row of a measurement file, newline not counted."""

DDIFF_WIDTH: Final[int] = _width(DDIFF_COLUMNS)
"""The width W of every row of a double-difference file, newline not counted."""

MEAS_HEADER_LINES: Final[int] = _PREAMBLE + len(MEAS_COLUMNS)
"""How many header lines a measurement file has."""

DDIFF_HEADER_LINES: Final[int] = _PREAMBLE + len(DDIFF_COLUMNS)
"""How many header lines a double-difference file has."""

WIDTHS: Final[dict[FileKind, int]] = {"meas": MEAS_WIDTH, "ddiff": DDIFF_WIDTH}
"""The row width of each kind of file."""

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


_QUIET: Final[ContextVar[bool]] = ContextVar("_QUIET", default=False)
"""Whether a refusal goes unlogged: while the file check reads a line, so the
check gives one explanation for a damaged file, not one per rule broken."""


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
    if not _QUIET.get():
        _log.error(message)
    raise DataFileError(message) from cause


# ------------------------------------------------------------------ records


@dataclass(frozen=True, slots=True)
class MeasRecord:
    """One row of a measurement file: a pair's measurement and its row.

    Parameters
    ----------
    measurement : PairMeasurement or DisabledReading or None
        The pair's measurement at the epoch; its reading while it is
        disabled; or ``None`` when there was none.
    row : Row
        The pair's row at the epoch; its innovation is not written.

    Raises
    ------
    DataFileError
        If there is a measurement and the row is P or the other way round,
        the measurement is of another epoch than the row's, it is a
        disabled reading and the row is not O or the other way round, or
        the measurement's slip correction and the row's S do not go
        together.
    """

    measurement: PairMeasurement | DisabledReading | None
    row: Row

    def __post_init__(self) -> None:
        """Refuse a measurement that does not belong with its row.

        Raises
        ------
        DataFileError
            As the class says.
        """
        _check_measured(self.measurement is not None, self.row)
        if self.measurement is None:
            if "S" in self.row.flags:
                _fail(f"row of {self.row.interpolated_datetime}: S with no measurement")
            return
        if self.measurement.interpolated_datetime != self.row.interpolated_datetime:
            _fail(
                f"row of {self.row.interpolated_datetime}: its measurement is of"
                f" the epoch of {self.measurement.interpolated_datetime}"
            )
        if isinstance(self.measurement, DisabledReading) != ("O" in self.row.flags):
            _fail(
                f"row of {self.row.interpolated_datetime}: a disabled reading goes"
                f" with flag O and O with a disabled reading; flags {self.row.flags!r}"
            )
        if isinstance(self.measurement, DisabledReading):
            return
        if self.measurement.slip != ("S" in self.row.flags):
            _fail(
                f"row of {self.row.interpolated_datetime}: a slip correction"
                " goes with flag S and S with a slip correction"
            )


@dataclass(frozen=True, slots=True)
class DdiffRecord:
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

    measurement: TripleMeasurement | None
    row: Row

    def __post_init__(self) -> None:
        """Refuse a measurement that does not belong with its row.

        Raises
        ------
        DataFileError
            As the class says.
        """
        _check_measured(self.measurement is not None, self.row)
        if "S" in self.row.flags:
            _fail(f"row of {self.row.interpolated_datetime}: a triple never carries S")


def _check_measured(has_measurement: bool, row: Row) -> None:
    """Refuse a measurement on a P row, or none on another.

    Parameters
    ----------
    has_measurement : bool
        Whether the record has a measurement.
    row : Row
        Its row.

    Raises
    ------
    DataFileError
        If ``has_measurement`` and the row is P, or neither.
    """
    if has_measurement == ("P" in row.flags):
        _fail(
            f"row of {row.interpolated_datetime}: a row has a measurement"
            f" exactly when it is not P; flags {row.flags!r}"
        )


# ------------------------------------------------------------------ headers


def header(file_kind: FileKind) -> str:
    """Give the header of a file of a kind (design 5.2, 5.4, 5.5).

    Parameters
    ----------
    file_kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    str
        The header lines, each as long as its text and ending with a
        newline: the kind of file and its format, a warning not to modify
        it, the line format, then one line per column with its number, from
        1, its name and its meaning. It is the same for every file of the
        kind; the file's name says which series it holds.
    """
    columns = MEAS_COLUMNS if file_kind == "meas" else DDIFF_COLUMNS
    header_lines_text = [
        f"# das_processor {_TITLES[file_kind]} file, format 1",
        _WARNING,
        f"# One row per 10-minute epoch; '{EMPTY}' marks an empty field.",
        f"# Columns: right-justified, fixed width, separated by '{SEPARATOR}'.",
        *(
            f"#  {number:>2}  {column.name:<24}{column.meaning}"
            for number, column in enumerate(columns, start=1)
        ),
    ]
    return "".join(f"{header_line}\n" for header_line in header_lines_text)


HEADER_SIZES: Final[dict[FileKind, int]] = {
    file_kind: len(header(file_kind)) for file_kind in ("meas", "ddiff")
}
"""The size H of each kind of file's header, bytes."""


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
    whole_ps, fs_digits = divmod(abs(x_fs), FS_PER_PS)
    return f"{'-' if x_fs < 0 else ''}{whole_ps}.{fs_digits:03d}"


def _float_text(number: float | None) -> str | None:
    """Write a float column, or nothing.

    Parameters
    ----------
    number : float or None
        The value.

    Returns
    -------
    str or None
        ``{:+.16e}``, or ``None`` for an empty field.
    """
    return None if number is None else f"{number:+.16e}"


def _mark_texts(epoch_start: datetime) -> tuple[str, str]:
    """Write an epoch as its UTC mark and its MJD.

    Parameters
    ----------
    epoch_start : datetime
        The ten-minute mark.

    Returns
    -------
    tuple of (str, str)
        The mark and its MJD to six places.
    """
    return format_epoch(epoch_start, datetime_to_mjd(epoch_start), 0, _MJD_DECIMALS)


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
    state_texts: list[str | None] = [
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
    for reject_index in range(3):
        if reject_index < len(row.rejects):
            reject_epoch, reject_innovation = row.rejects[reject_index]
            state_texts += [
                _mark_texts(reject_epoch)[1],
                _float_text(reject_innovation),
            ]
        else:
            state_texts += [None, None]
    state_texts += [
        str(row.filter_states),
        _float_text(row.time_constant),
        _float_text(row.scale_time_constant),
        row.flags,
    ]
    return state_texts


def _row_template(columns: tuple[Column, ...]) -> str:
    """Give the format string that lays out a row of the given columns.

    Parameters
    ----------
    columns : tuple of Column
        A file's columns.

    Returns
    -------
    str
        Each column right-justified to its width, separated by
        :data:`SEPARATOR`.

    Examples
    --------
    >>> _row_template((Column("a", 3, ""), Column("b", 2, "")))
    '{:>3}, {:>2}'
    """
    return SEPARATOR.join(f"{{:>{column.width}}}" for column in columns)


_ROW_TEMPLATES: Final[dict[tuple[Column, ...], tuple[str, int]]] = {
    columns: (_row_template(columns), _width(columns))
    for columns in (MEAS_COLUMNS, DDIFF_COLUMNS)
}
"""Each file's row format string and row width."""


def _joined(
    columns: tuple[Column, ...], column_texts: list[str | None], epoch_start: datetime
) -> str:
    """Lay out a row's texts in their columns.

    Parameters
    ----------
    columns : tuple of Column
        The file's columns.
    column_texts : list of (str or None)
        One text per column; ``None`` for an empty field.
    epoch_start : datetime
        The row's epoch, for the message.

    Returns
    -------
    str
        The line, without its newline.

    Raises
    ------
    DataFileError
        If a value is wider than its column.

    Notes
    -----
    Padding never shortens a text, so the line is the row's width exactly
    when every value fits; only a line of another width is searched for the
    value that does not.
    """
    field_texts = [EMPTY if text is None else text for text in column_texts]
    row_template, row_width = _ROW_TEMPLATES[columns]
    row_line = row_template.format(*field_texts)
    if len(row_line) != row_width:
        column, field_text = next(
            (column, field_text)
            for column, field_text in zip(columns, field_texts, strict=True)
            if len(field_text) > column.width
        )
        _fail(
            f"row of {epoch_start}: {column.name} {field_text!r} does not fit"
            f" its {column.width} characters"
        )
    return row_line


def format_meas_row(meas_record: MeasRecord) -> str:
    """Write a measurement file row (design 5.4).

    Parameters
    ----------
    meas_record : MeasRecord
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
    row = meas_record.row
    pair_measurement = meas_record.measurement
    column_texts: list[str | None] = list(_mark_texts(row.interpolated_datetime))
    if pair_measurement is None:
        column_texts += [None] * 6
    elif isinstance(pair_measurement, DisabledReading):
        column_texts += [
            *_reading_texts(pair_measurement),
            None,
            None if pair_measurement.z is None else str(pair_measurement.z),
        ]
    else:
        column_texts += [
            *_reading_texts(pair_measurement),
            str(pair_measurement.cycle_count),
            str(pair_measurement.z),
        ]
    return _joined(
        MEAS_COLUMNS, column_texts + _state_texts(row), row.interpolated_datetime
    )


def _reading_texts(reading: PairMeasurement | DisabledReading) -> list[str | None]:
    """Write a reading's time, MJD, phase and rms columns.

    Parameters
    ----------
    reading : PairMeasurement or DisabledReading
        The reading.

    Returns
    -------
    list of (str or None)
        The four columns' texts.
    """
    return [
        reading.measurement_datetime.isoformat(sep=" ", timespec="microseconds"),
        f"{reading.measurement_mjd:.{_MJD_DECIMALS}f}",
        str(reading.measured_phase),
        str(reading.rms),
    ]


def format_ddiff_row(ddiff_record: DdiffRecord) -> str:
    """Write a double-difference file row (design 5.5).

    Parameters
    ----------
    ddiff_record : DdiffRecord
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
    row = ddiff_record.row
    triple_measurement = ddiff_record.measurement
    column_texts: list[str | None] = list(_mark_texts(row.interpolated_datetime))
    if triple_measurement is None:
        column_texts += [None, _float_text(row.innovation), None, None]
    else:
        column_texts += [
            str(triple_measurement.z),
            _float_text(row.innovation),
            _float_text(triple_measurement.double_difference_sigma),
            triple_measurement.components_used,
        ]
    return _joined(
        DDIFF_COLUMNS, column_texts + _state_texts(row), row.interpolated_datetime
    )


# ---------------------------------------------------------------- parsing


class RowFields(BaseModel):
    """A row's fields as read back from a file, checked before the row is used.

    The fields are a :class:`~masterclock.domain.series.Row`'s, with the same
    names, order and meanings, each of exactly its kind; the row they make
    must keep every rule a row keeps
    (:func:`~masterclock.domain.series.check_row`).

    Raises
    ------
    pydantic.ValidationError
        If a field is of the wrong kind, missing or unknown, a datetime has
        no timezone, or the row breaks a rule of a row, a counter below 0
        among them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    interpolated_datetime: AwareDatetime
    innovation: float | None
    x_fs: int | None
    y: float | None
    d: float | None
    innovation_scale: float | None
    segment: int
    step_offset: int
    epochs_in_segment: int
    epochs_since_accept: int
    consecutive_rejects: int
    rejects: tuple[tuple[AwareDatetime, float], ...]
    filter_states: FilterStates
    time_constant: float | None
    scale_time_constant: float
    flags: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse fields that make no valid row.

        Returns
        -------
        Self
            The fields, unchanged.

        Raises
        ------
        ValueError
            Naming the first rule of a row the fields break.
        """
        check_row(self.row())
        return self

    def row(self) -> Row:
        """Give the row the fields make.

        Returns
        -------
        Row
            The row.
        """
        return Row(**dict(self))


class PairMeasurementFields(BaseModel):
    """A pair measurement's fields as read back from a measurement file.

    The fields are a :class:`~masterclock.domain.measurements.PairMeasurement`'s,
    with the same names, order and meanings.

    Raises
    ------
    pydantic.ValidationError
        If a field is of the wrong kind, missing or unknown, the MJD is not
        finite, the reading is outside 0 to
        :data:`~masterclock.domain.phase.PHASE_MAX`, or the rms is below 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement_mjd: Annotated[float, Field(allow_inf_nan=False)]
    measured_phase: Annotated[int, Field(ge=0, le=PHASE_MAX)]
    rms: Annotated[int, Field(ge=0)]
    cycle_count: int
    z: int
    slip: bool

    def measurement(self) -> PairMeasurement:
        """Give the measurement the fields make.

        Returns
        -------
        PairMeasurement
            The measurement.
        """
        return PairMeasurement(**dict(self))


class DisabledReadingFields(BaseModel):
    """A disabled pair's reading as read back from a measurement file.

    The fields are a :class:`~masterclock.domain.measurements.DisabledReading`'s,
    with the same names, order and meanings.

    Raises
    ------
    pydantic.ValidationError
        If a field is of the wrong kind, missing or unknown, the MJD is not
        finite, the reading is outside 0 to
        :data:`~masterclock.domain.phase.PHASE_MAX`, or the rms is below 0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    measurement_mjd: Annotated[float, Field(allow_inf_nan=False)]
    measured_phase: Annotated[int, Field(ge=0, le=PHASE_MAX)]
    rms: Annotated[int, Field(ge=0)]
    z: int | None

    def reading(self) -> DisabledReading:
        """Give the reading the fields make.

        Returns
        -------
        DisabledReading
            The reading.
        """
        return DisabledReading(**dict(self))


class TripleMeasurementFields(BaseModel):
    """A triple measurement's fields as read back from a double-difference file.

    The fields are a
    :class:`~masterclock.domain.measurements.TripleMeasurement`'s, with the
    same names, order and meanings.

    Raises
    ------
    pydantic.ValidationError
        If a field is of the wrong kind, missing or unknown, the sigma is
        below 0 or not finite, or the components are not one of 111, 110
        and 101.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    z: int
    double_difference_sigma: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    components_used: Literal["111", "110", "101"]
    pair_cold_started: bool

    def measurement(self) -> TripleMeasurement:
        """Give the measurement the fields make.

        Returns
        -------
        TripleMeasurement
            The measurement.
        """
        return TripleMeasurement(**dict(self))


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
    split_fields = line.split(SEPARATOR)
    if len(split_fields) != len(columns):
        message = f"{len(split_fields)} fields, not {len(columns)}"
        raise ValueError(message)
    field_texts = [split_field.strip() for split_field in split_fields]
    return [None if field_text == EMPTY else field_text for field_text in field_texts]


def _given(field_text: str | None) -> str:
    """Give a field that must not be empty.

    Parameters
    ----------
    field_text : str or None
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
    if field_text is None:
        message = "a field that is never empty is empty"
        raise ValueError(message)
    return field_text


def _x_value(field_text: str) -> int:
    """Read the estimator's phase, ps with three decimals, as whole fs.

    Parameters
    ----------
    field_text : str
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
    if _X_TEXT.fullmatch(field_text) is None:
        message = f"phase {field_text!r} is not ps with three decimals"
        raise ValueError(message)
    whole_ps, fs_digits = field_text.lstrip("-").split(".")
    magnitude_fs = int(whole_ps) * FS_PER_PS + int(fs_digits)
    return -magnitude_fs if field_text.startswith("-") else magnitude_fs


def _mark_value(mjd_text: str) -> datetime:
    """Read an MJD column back to the ten-minute mark it was written from.

    Parameters
    ----------
    mjd_text : str
        The field.

    Returns
    -------
    datetime
        The nearest mark; formatting it again shows whether it was one.
    """
    return floor_to_ten_minutes(mjd_to_datetime(float(mjd_text)) + _HALF_EPOCH)


def _float_value(field_text: str) -> float:
    """Read a float column.

    Parameters
    ----------
    field_text : str
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
    number = float(field_text)
    if not math.isfinite(number):
        message = f"{field_text!r} is not a finite number"
        raise ValueError(message)
    return number


def _optional_float(field_text: str | None) -> float | None:
    """Read a float column that may be empty.

    Parameters
    ----------
    field_text : str or None
        The field.

    Returns
    -------
    float or None
        The value, or ``None``.
    """
    return None if field_text is None else _float_value(field_text)


def _row(
    epoch_start: datetime, innovation: float | None, state_texts: list[str | None]
) -> Row:
    """Build a row from its epoch, its innovation and its state columns.

    Parameters
    ----------
    epoch_start : datetime
        The epoch start.
    innovation : float or None
        The innovation.
    state_texts : list of (str or None)
        The state columns' fields.

    Returns
    -------
    Row
        The row.

    Raises
    ------
    ValueError
        If a field is not of its kind or not finite, or the fields make no
        valid row (see :class:`RowFields`): a pydantic ValidationError, which
        is not logged, so a file check can read a damaged line quietly.
    """
    (
        x,
        y,
        d,
        innovation_scale,
        segment,
        step_offset,
        epochs_in_segment,
        epochs_since_accept,
        consecutive_rejects,
    ) = state_texts[:9]
    rejects: list[Reject] = []
    for reject_index in range(3):
        reject_mjd_text, reject_innovation_text = (
            state_texts[9 + 2 * reject_index],
            state_texts[10 + 2 * reject_index],
        )
        if reject_mjd_text is not None or reject_innovation_text is not None:
            rejects.append(
                (
                    _mark_value(_given(reject_mjd_text)),
                    _float_value(_given(reject_innovation_text)),
                )
            )
    filter_states, time_constant, scale_time_constant, flags = state_texts[15:]
    return RowFields.model_validate(
        {
            "interpolated_datetime": epoch_start,
            "innovation": innovation,
            "x_fs": None if x is None else _x_value(x),
            "y": _optional_float(y),
            "d": _optional_float(d),
            "innovation_scale": _optional_float(innovation_scale),
            "segment": int(_given(segment)),
            "step_offset": int(_given(step_offset)),
            "epochs_in_segment": int(_given(epochs_in_segment)),
            "epochs_since_accept": int(_given(epochs_since_accept)),
            "consecutive_rejects": int(_given(consecutive_rejects)),
            "rejects": tuple(rejects),
            "filter_states": int(_given(filter_states)),
            "time_constant": _optional_float(time_constant),
            "scale_time_constant": _float_value(_given(scale_time_constant)),
            "flags": _given(flags),
        }
    ).row()


def _pair_measurement(
    measurement_texts: list[str | None], flags: str
) -> PairMeasurement | DisabledReading | None:
    """Build a pair's measurement from its columns.

    Parameters
    ----------
    measurement_texts : list of (str or None)
        The six measurement columns' fields.
    flags : str
        The row's flags, whose S marks a slip correction and O a disabled
        pair.

    Returns
    -------
    PairMeasurement or DisabledReading or None
        The measurement; for an O row, the disabled reading; ``None`` when
        every field is empty.

    Raises
    ------
    ValueError
        If a field is not of its kind, or the fields do not make a
        measurement: on an O row a cycle count is given, or on another
        row a field is empty.
    """
    if all(field_text is None for field_text in measurement_texts):
        return None
    if "O" in flags:
        return _disabled_reading(measurement_texts)
    _, mjd_text, phase_text, rms_text, cycles_text, z_text = (
        _given(field_text) for field_text in measurement_texts
    )
    return PairMeasurementFields.model_validate(
        {
            "measurement_mjd": _float_value(mjd_text),
            "measured_phase": int(phase_text),
            "rms": int(rms_text),
            "cycle_count": int(cycles_text),
            "z": int(z_text),
            "slip": "S" in flags,
        }
    ).measurement()


def _disabled_reading(measurement_texts: list[str | None]) -> DisabledReading:
    """Build a disabled pair's reading from its columns.

    Parameters
    ----------
    measurement_texts : list of (str or None)
        The six measurement columns' fields.

    Returns
    -------
    DisabledReading
        The reading, with the z it carries or none.

    Raises
    ------
    ValueError
        If a field is not of its kind, a reading field is empty, or a cycle
        count is given, which a reading never decycled has none of.
    """
    _, mjd_text, phase_text, rms_text, cycles_text, z_text = measurement_texts
    if cycles_text is not None:
        message = "a disabled pair's row has no cycle count"
        raise ValueError(message)
    return DisabledReadingFields.model_validate(
        {
            "measurement_mjd": _float_value(_given(mjd_text)),
            "measured_phase": int(_given(phase_text)),
            "rms": int(_given(rms_text)),
            "z": None if z_text is None else int(z_text),
        }
    ).reading()


def _attempt[RecordT: (MeasRecord, DdiffRecord)](
    line: str,
    build_record: Callable[[], RecordT],
    format_record: Callable[[RecordT], str],
) -> tuple[RecordT | None, str, Exception | None]:
    """Build a record from a line, and check it gives the line back.

    Parameters
    ----------
    line : str
        The line.
    build_record : callable
        Builds the record from the line.
    format_record : callable
        Formats a record.

    Returns
    -------
    tuple of (record or None, str, Exception or None)
        The record, or ``None`` with what is wrong with the line and the
        error that showed it. A record that formats back to another line,
        or cannot be formatted back at all, as when the line fits a field
        wider than its column by narrowing another, is not written as
        das_processor writes it. Nothing is logged, a record's own checks
        and the formatting included: the caller says what is wrong.
    """
    not_written_so = f"row {line[:25]!r} is not written as das_processor writes it"
    quiet_token = _QUIET.set(True)
    try:
        parsed_record = build_record()
    except (
        ValueError,
        FilterError,
        PhaseError,
        DataFileError,
    ) as exc:
        _QUIET.reset(quiet_token)
        return None, f"row {line[:25]!r} does not parse: {describe_error(exc)}", exc
    try:
        line_back = format_record(parsed_record)
    except DataFileError as exc:
        return None, not_written_so, exc
    finally:
        _QUIET.reset(quiet_token)
    if line_back != line:
        return None, not_written_so, None
    return parsed_record, "", None


def _parsed[RecordT: (MeasRecord, DdiffRecord)](
    line: str,
    build_record: Callable[[], RecordT],
    format_record: Callable[[RecordT], str],
) -> RecordT:
    """Build a record from a line, and check it gives the line back.

    Parameters
    ----------
    line : str
        The line.
    build_record : callable
        Builds the record from the line.
    format_record : callable
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
    parsed_record, parse_problem, parse_error = _attempt(
        line, build_record, format_record
    )
    if parsed_record is None:
        _fail(parse_problem, parse_error)
    return parsed_record


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

    Raises
    ------
    ValueError
        If a field is not of its kind, or the fields make no valid row or
        measurement.
    DataFileError
        If the measurement does not belong with the row.
    """
    field_texts = _fields(line, MEAS_COLUMNS)
    row = _row(datetime.fromisoformat(_given(field_texts[0])), None, field_texts[8:])
    return MeasRecord(
        measurement=_pair_measurement(field_texts[2:8], row.flags), row=row
    )


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

    Raises
    ------
    ValueError
        If a field is not of its kind, or the fields make no valid row or
        measurement.
    DataFileError
        If the measurement does not belong with the row.
    """
    field_texts = _fields(line, DDIFF_COLUMNS)
    z_text, innovation_text, sigma_text, components_used = field_texts[2:6]
    row = _row(
        datetime.fromisoformat(_given(field_texts[0])),
        _optional_float(innovation_text),
        field_texts[6:],
    )
    triple_measurement = None
    if z_text is not None or sigma_text is not None or components_used is not None:
        triple_measurement = TripleMeasurementFields.model_validate(
            {
                "z": int(_given(z_text)),
                "double_difference_sigma": _float_value(_given(sigma_text)),
                "components_used": _given(components_used),
                "pair_cold_started": False,
            }
        ).measurement()
    return DdiffRecord(measurement=triple_measurement, row=row)


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


def _parse_line(row_line: str, file_kind: FileKind) -> MeasRecord | DdiffRecord:
    """Read a row of either kind of file.

    Parameters
    ----------
    row_line : str
        The line, without its newline.
    file_kind : {'meas', 'ddiff'}
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
    if file_kind == "meas":
        return parse_meas_row(row_line)
    return parse_ddiff_row(row_line)


def row_epoch(slot: bytes, file_kind: FileKind) -> datetime | None:
    """Give the epoch of a good row (design 5.7).

    Parameters
    ----------
    slot : bytes
        One row slot of a file, with its newline if it has one.
    file_kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    datetime or None
        The row's epoch when the slot is a whole line, ending in its
        newline, that is ASCII and parses as a row of the file; otherwise
        ``None``. A header line never parses as a row. Nothing is logged.
    """
    epoch, _ = _examined(slot, file_kind)
    return epoch


def _examined(slot: bytes, file_kind: FileKind) -> tuple[datetime | None, str]:
    """Give a row slot's epoch, or what is wrong with it.

    Parameters
    ----------
    slot : bytes
        One row slot of a file.
    file_kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    tuple of (datetime or None, str)
        The row's epoch and ``""`` for a good row; otherwise ``None`` and
        the reason, in words. Nothing is logged.
    """
    if not slot.endswith(b"\n"):
        return None, "the line is cut short, with no newline"
    try:
        row_line = slot[:-1].decode("ascii")
    except UnicodeDecodeError:
        return None, "the line is not ASCII text"
    file_record: MeasRecord | DdiffRecord | None
    if file_kind == "meas":
        file_record, damage_reason, _ = _attempt(
            row_line, lambda: _meas_record(row_line), format_meas_row
        )
    else:
        file_record, damage_reason, _ = _attempt(
            row_line, lambda: _ddiff_record(row_line), format_ddiff_row
        )
    if file_record is None:
        return None, damage_reason
    return file_record.row.interpolated_datetime, ""


def _slot(
    open_file: BinaryIO, header_end: int, slot_index: int, line_size: int
) -> bytes:
    """Read one row slot of a file.

    Parameters
    ----------
    open_file : BinaryIO
        The open file.
    header_end : int
        Where its header ends, bytes from its start.
    slot_index : int
        The slot, from 0 at the first row.
    line_size : int
        A row's size, newline included.

    Returns
    -------
    bytes
        The slot's bytes.
    """
    open_file.seek(header_end + slot_index * line_size)
    return open_file.read(line_size)


def _not_sound(data_file: Path, file_length: int) -> str:
    """Say that a file is not its header and whole rows.

    Parameters
    ----------
    data_file : Path
        The file.
    file_length : int
        Its length, bytes.

    Returns
    -------
    str
        The message.
    """
    return (
        f"data file {data_file} is not sound: its {file_length} bytes are not"
        " its header and whole rows"
    )


def _row_slots(file_length: int, file_kind: FileKind) -> tuple[int, int]:
    """Count a file's whole row slots after its header.

    Parameters
    ----------
    file_length : int
        The file's length, bytes.
    file_kind : {'meas', 'ddiff'}
        Its kind.

    Returns
    -------
    tuple of (int, int)
        How many whole rows fit after the header, and the bytes left over
        after them, which a sound file has none of; none of either for a
        file no longer than its header.
    """
    return divmod(max(file_length - HEADER_SIZES[file_kind], 0), WIDTHS[file_kind] + 1)


class FileCheck(NamedTuple):
    """What the file check found in one file.

    Parameters
    ----------
    good_through : datetime or None
        The epoch of its last good row; ``None`` when it holds none.
    damaged : bool
        Whether it holds no whole row, or anything but its header and good
        whole rows.
    """

    good_through: datetime | None
    damaged: bool


def good_through(
    data_file: Path, file_kind: FileKind, *, stopped_write: bool = False
) -> datetime | None:
    """Give the epoch of a file's last good row (design 5.2, 5.7).

    Parameters
    ----------
    data_file : Path
        The file.
    file_kind : {'meas', 'ddiff'}
        The kind of file.
    stopped_write : bool, optional
        Whether a write stopped part way, as its journal shows (see
        :func:`check_file`).

    Returns
    -------
    datetime or None
        As :func:`check_file` gives it.

    Raises
    ------
    DataFileError
        As :func:`check_file` raises it.
    """
    return check_file(data_file, file_kind, stopped_write=stopped_write).good_through


def check_file(
    data_file: Path, file_kind: FileKind, *, stopped_write: bool = False
) -> FileCheck:
    """Check a file, and say how far it is good (design 5.2, 5.7).

    Parameters
    ----------
    data_file : Path
        The file.
    file_kind : {'meas', 'ddiff'}
        The kind of file.
    stopped_write : bool, optional
        Whether a write stopped part way, as its journal shows; a file it
        was creating may then hold its length but not its rows.

    Returns
    -------
    FileCheck
        For a sound file, one whose length is its header plus whole rows
        and whose last row is good: that row's epoch, found with one short
        read, and ``damaged`` false.
        Otherwise the epoch of the row before its first line that is not a
        good row; after a stopped write,
        ``None`` when the file holds no whole row or its first row is not
        good. A damaged file is logged once at ERROR, naming where it is
        damaged and why.

    Raises
    ------
    DataFileError
        If the file cannot be read, or, unless a write stopped part way,
        it holds no whole row or its first row is not good, so its rows
        cannot be placed in time.
    """
    line_size = WIDTHS[file_kind] + 1
    try:
        with data_file.open("rb") as open_file:
            header_end = HEADER_SIZES[file_kind]
            row_slots, left_over = _row_slots(open_file.seek(0, os.SEEK_END), file_kind)
            if row_slots < 1:
                good_epoch, damage_reason = None, "it holds no whole row"
            elif left_over == 0 and (
                last_epoch := row_epoch(
                    _slot(open_file, header_end, row_slots - 1, line_size), file_kind
                )
            ):
                return FileCheck(good_through=last_epoch, damaged=False)
            else:
                good_epoch, damage_reason = _first_damage(
                    open_file, row_slots, header_end, line_size, file_kind
                )
    except OSError as exc:
        _fail(f"cannot read data file {data_file}: {exc}", exc)
    if good_epoch is None and not stopped_write:
        _fail(
            f"{data_file} has a damaged first row, so its rows cannot be placed in"
            f" time: {damage_reason}"
        )
    damage_place = (
        "from its first row" if good_epoch is None else f"after its row of {good_epoch}"
    )
    _log.error("data file %s is damaged %s: %s", data_file, damage_place, damage_reason)
    return FileCheck(good_through=good_epoch, damaged=True)


def _first_damage(
    open_file: BinaryIO,
    row_slots: int,
    header_end: int,
    line_size: int,
    file_kind: FileKind,
) -> tuple[datetime | None, str]:
    """Find a damaged file's first line that is not a good row.

    Parameters
    ----------
    open_file : BinaryIO
        The file, open.
    row_slots : int
        Its whole row slots after the header.
    header_end : int
        Where its header ends, bytes from its start.
    line_size : int
        Its row width, newline included.
    file_kind : {'meas', 'ddiff'}
        Its kind.

    Returns
    -------
    tuple of (datetime or None, str)
        The epoch of the last good row before it, ``None`` when there is
        none, and what is wrong: with every whole row good, the last line,
        cut short.
    """
    good_epoch = None
    for slot_index in range(row_slots):
        slot_epoch, damage_reason = _examined(
            _slot(open_file, header_end, slot_index, line_size), file_kind
        )
        if slot_epoch is None:
            return good_epoch, damage_reason
        good_epoch = slot_epoch
    return good_epoch, "its last line is cut short"


def cut_epoch(
    file_checks: Iterable[FileCheck], latest_epoch: datetime | None
) -> datetime | None:
    """Give the last epoch every file of a channel may keep, so they stay in step.

    Parameters
    ----------
    file_checks : iterable of FileCheck
        What the file check found in each file of the channel.
    latest_epoch : datetime or None
        The latest epoch a file may keep whatever its damage: the epoch
        before a stopped write's or a redo's first; ``None`` for no such
        limit.

    Returns
    -------
    datetime or None
        The earliest of ``latest_epoch`` and the last good row of every
        damaged file; ``None`` when there is neither, and every file keeps
        all its good rows. A damaged file with no good row, which only a
        stopped write leaves, sets no limit: it holds nothing to keep.
    """
    limits = [
        file_check.good_through for file_check in file_checks if file_check.damaged
    ]
    limits.append(latest_epoch)
    return min((epoch for epoch in limits if epoch is not None), default=None)


def read_last_row(data_file: Path, file_kind: FileKind) -> Row:
    """Read the last row of a sound file (design 5.7).

    Parameters
    ----------
    data_file : Path
        The file.
    file_kind : {'meas', 'ddiff'}
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
    return read_last_record(data_file, file_kind).row


def read_last_record(data_file: Path, file_kind: FileKind) -> MeasRecord | DdiffRecord:
    """Read the last record of a sound file: its measurement and row.

    Parameters
    ----------
    data_file : Path
        The file.
    file_kind : {'meas', 'ddiff'}
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
    line_size = WIDTHS[file_kind] + 1
    try:
        with data_file.open("rb") as open_file:
            file_length = open_file.seek(0, os.SEEK_END)
            header_end = HEADER_SIZES[file_kind]
            row_slots, left_over = _row_slots(file_length, file_kind)
            if row_slots < 1 or left_over != 0:
                _fail(_not_sound(data_file, file_length))
            last_slot = _slot(open_file, header_end, row_slots - 1, line_size)
    except OSError as exc:
        _fail(f"cannot read data file {data_file}: {exc}", exc)
    try:
        last_line = last_slot[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        _fail(f"data file {data_file} is not sound: its last row is not ASCII", exc)
    return _parse_line(last_line, file_kind)


# --------------------------------------------------- day buffer and write

_FILE_KIND_ORDER: Final[dict[FileKind, int]] = {"meas": 0, "ddiff": 1}
"""The order the kinds of file are written in: measurement files first."""


def record_z(file_record: MeasRecord) -> int | None:
    """Give the z a pair's record holds, which a disabled row after it carries.

    Parameters
    ----------
    file_record : MeasRecord
        A pair's measurement and row at an epoch.

    Returns
    -------
    int or None
        The z of its measurement or of its disabled reading; ``None`` for
        a row without one.
    """
    return None if file_record.measurement is None else file_record.measurement.z


def record_line(file_record: MeasRecord | DdiffRecord) -> tuple[str, Row]:
    """Write a series' record as its file's line, and give the row to keep.

    Parameters
    ----------
    file_record : MeasRecord or DdiffRecord
        A series' measurement and row at an epoch.

    Returns
    -------
    tuple of (str, Row)
        The line, with its newline, and the row a later run would read back
        from it: the row as it is, less what the file does not hold, a
        measurement row's innovation.

    Raises
    ------
    DataFileError
        If a value is wider than its column.
    """
    if isinstance(file_record, MeasRecord):
        return (
            format_meas_row(file_record) + "\n",
            dataclasses.replace(file_record.row, innovation=None),
        )
    return format_ddiff_row(file_record) + "\n", file_record.row


class DayBuffer:
    """The rows computed since the last write, for every file (design 5.8).

    Rows are written a UTC day at a time. Until then each file's rows are
    held here as a list of lines, joined only when they are written, and
    each series' newest row as a row, which the next epoch takes as the
    series' last row.

    Parameters
    ----------
    journal : Path or None, optional
        The write journal kept from the first write to the final one (see
        :func:`write_buffer`); no journal is kept when ``None``.

    Attributes
    ----------
    file_lines : dict of Path to list of str
        Each file's lines since the last write, in order, each with its
        newline.
    last_rows : dict of series key to Row
        Each series' newest row, as a later run would read it back.
    last_z : dict of (str, str) to int or None
        Each pair's newest z, as a later run would read it back; ``None``
        for a row without one. A disabled pair's row carries it.
    earliest_epoch : datetime or None
        The earliest epoch of the rows since the last write; ``None`` when
        there are none.
    rows_added : int
        How many rows were added since the buffer was made, written or not.
    journal_written : bool
        Whether the journal was written by a write not yet followed by the
        final write.
    unflushed_files : set of Path
        The files written since the last final write, which a power
        failure could still undo.
    unflushed_directories : set of Path
        The directories of the files among them that the writes created.
    """

    def __init__(self, journal: Path | None = None) -> None:
        """Start an empty buffer.

        Parameters
        ----------
        journal : Path or None, optional
            The write journal kept from the first write to the final one
            (see :func:`write_buffer`); no journal is kept when ``None``.
        """
        self.journal = journal
        self.earliest_epoch: datetime | None = None
        self.rows_added = 0
        self.file_lines: dict[Path, list[str]] = {}
        self.last_rows: dict[SeriesKey, Row] = {}
        self.last_z: dict[PairKey, int | None] = {}
        self.journal_written = False
        self.unflushed_files: set[Path] = set()
        self.unflushed_directories: set[Path] = set()
        self._file_series: dict[Path, tuple[FileKind, SeriesKey]] = {}

    def add(
        self,
        data_file: Path,
        series_key: SeriesKey,
        file_record: MeasRecord | DdiffRecord,
    ) -> None:
        """Add a series' record for an epoch.

        Parameters
        ----------
        data_file : Path
            The series' file.
        series_key : (str, str) or (str, str, str)
            The series.
        file_record : MeasRecord or DdiffRecord
            Its measurement and row at the epoch.

        Raises
        ------
        DataFileError
            If a value does not fit its column, or ``data_file`` was given
            another series or kind of file before.

        Notes
        -----
        The row kept as the series' newest is the row a later run would read
        from the file, so the next epoch is the same whether it takes the
        row from here or from the file (I5). The line is not read back to
        get it: every value is written in a form that gives it back exactly,
        which the round-trip tests show, so the row is kept as it is, less
        what the file does not hold: a measurement row's innovation.
        """
        file_kind: FileKind = "meas" if isinstance(file_record, MeasRecord) else "ddiff"
        self._check_series(data_file, file_kind, series_key)
        line_text, kept_row = record_line(file_record)
        self.file_lines.setdefault(data_file, []).append(line_text)
        self.last_rows[series_key] = kept_row
        if isinstance(file_record, MeasRecord):
            self.last_z[(series_key[0], series_key[1])] = record_z(file_record)
        self._started(kept_row.interpolated_datetime)

    def add_line(
        self,
        data_file: Path,
        file_kind: FileKind,
        series_key: SeriesKey,
        line_text: str,
        epoch_start: datetime,
    ) -> None:
        """Add a series' line for an epoch, made elsewhere by :func:`record_line`.

        The series' newest row is not kept here: whoever made the line keeps
        it, as a worker process does for its own series.

        Parameters
        ----------
        data_file : Path
            The series' file.
        file_kind : {'meas', 'ddiff'}
            The kind of file.
        series_key : (str, str) or (str, str, str)
            The series.
        line_text : str
            The row's line, with its newline.
        epoch_start : datetime
            The row's epoch.

        Raises
        ------
        DataFileError
            If ``data_file`` was given another series or kind of file before.
        """
        self._check_series(data_file, file_kind, series_key)
        self.file_lines.setdefault(data_file, []).append(line_text)
        self._started(epoch_start)

    def _check_series(
        self, data_file: Path, file_kind: FileKind, series_key: SeriesKey
    ) -> None:
        """Note the series a path is for, refusing a path given another before.

        Parameters
        ----------
        data_file : Path
            The series' file.
        file_kind : {'meas', 'ddiff'}
            The kind of file.
        series_key : (str, str) or (str, str, str)
            The series.

        Raises
        ------
        DataFileError
            If ``data_file`` was given another series or kind of file before.
        """
        if self._file_series.setdefault(data_file, (file_kind, series_key)) != (
            file_kind,
            series_key,
        ):
            _fail(
                f"{data_file} holds the {self._file_series[data_file]} series,"
                f" not {(file_kind, series_key)}"
            )

    def _started(self, epoch_start: datetime | None, row_count: int = 1) -> None:
        """Note rows added and the epoch of one, keeping the earliest epoch.

        Parameters
        ----------
        epoch_start : datetime or None
            The epoch; ``None`` leaves the earliest epoch as it is.
        row_count : int, optional
            How many rows were added.
        """
        self.rows_added += row_count
        if epoch_start is not None and (
            self.earliest_epoch is None or epoch_start < self.earliest_epoch
        ):
            self.earliest_epoch = epoch_start

    def take(self, newer_buffer: DayBuffer) -> None:
        """Move another buffer's rows into this one, all of them or none.

        Parameters
        ----------
        newer_buffer : DayBuffer
            A buffer of rows added after this one's, such as one epoch's.

        Raises
        ------
        DataFileError
            If a path of ``newer_buffer`` is another series' or kind's in this
            buffer; this buffer is then unchanged.
        """
        for data_file in newer_buffer.file_lines:
            file_series = newer_buffer.series_of(data_file)
            if self._file_series.get(data_file, file_series) != file_series:
                _fail(
                    f"{data_file} holds the {self._file_series[data_file]} series,"
                    f" not {file_series}"
                )
        for data_file, new_lines in newer_buffer.file_lines.items():
            self._file_series[data_file] = newer_buffer.series_of(data_file)
            self.file_lines.setdefault(data_file, []).extend(new_lines)
        self.last_rows.update(newer_buffer.last_rows)
        self.last_z.update(newer_buffer.last_z)
        self._started(newer_buffer.earliest_epoch, newer_buffer.rows_added)

    def series_of(self, data_file: Path) -> tuple[FileKind, SeriesKey]:
        """Give the kind of file and the series a buffered path is for.

        Parameters
        ----------
        data_file : Path
            A path the buffer holds text for.

        Returns
        -------
        tuple of (FileKind, series key)
            Its kind of file and series.
        """
        return self._file_series[data_file]


def write_buffer(day_buffer: DayBuffer) -> None:
    """Write every file's buffered rows: all of them, or none (design 5.8).

    Parameters
    ----------
    day_buffer : DayBuffer
        The buffer; its texts are emptied after the write, its newest rows
        kept, and the files written noted as not yet flushed.

    Raises
    ------
    DataFileError
        Before any file is opened, if a buffered text is not ASCII, an
        existing file is not a regular file this process can write or its
        length is not its header plus one or more whole rows, a new file's
        directory is not one this process can write into, a write journal
        is already there at the first write,
        or the free space does not cover every byte to be written; nothing
        is changed then. While writing, if the device fails.

    Notes
    -----
    The data files are written but not flushed to the device; only
    :func:`write_final` flushes them, once, at the end of the run. When the
    buffer has a journal, the first write writes the first epoch of its
    rows to it and flushes it before any data file is opened, and the
    journal stays until :func:`write_final` has flushed every file. A run
    that finds the journal knows a run stopped before its files were all
    flushed, whichever files it reached or created, and rolls every file
    back to before that epoch (see :func:`read_journal`).
    """
    file_bytes = _prepared(day_buffer)
    if not file_bytes:
        return
    if (
        day_buffer.journal is not None
        and not day_buffer.journal_written
        and day_buffer.earliest_epoch is not None
    ):
        _write_journal(day_buffer.journal, day_buffer.earliest_epoch)
        day_buffer.journal_written = True
    write_order = sorted(
        file_bytes, key=lambda data_file: _write_order(day_buffer, data_file)
    )
    for data_file in write_order:
        is_new = not os.path.lexists(data_file)
        try:
            with data_file.open("xb" if is_new else "ab") as open_file:
                open_file.write(file_bytes[data_file])
        except OSError as exc:
            _fail(f"cannot write data file {data_file}: {exc}", exc)
        day_buffer.unflushed_files.add(data_file)
        if is_new:
            day_buffer.unflushed_directories.add(data_file.parent)
    day_buffer.file_lines.clear()
    day_buffer.earliest_epoch = None


def write_final(day_buffer: DayBuffer) -> None:
    """Write the last rows, flush every file written since, and end the journal.

    Parameters
    ----------
    day_buffer : DayBuffer
        The buffer; written as by :func:`write_buffer`, then every file the
        writes reached is flushed.

    Raises
    ------
    DataFileError
        As :func:`write_buffer` does, or if a file or directory cannot be
        flushed; the journal is then left, so the next run rolls back.

    Notes
    -----
    Each file is flushed once, measurement files first, then the
    directories of the files the writes created, and the journal is deleted
    last, so it is there until every row it covers is on the device.
    """
    write_buffer(day_buffer)
    flush_order = sorted(
        day_buffer.unflushed_files,
        key=lambda data_file: _write_order(day_buffer, data_file),
    )
    for data_file in flush_order:
        try:
            with data_file.open("rb") as open_file:
                os.fsync(open_file.fileno())
        except OSError as exc:
            _fail(f"cannot flush data file {data_file}: {exc}", exc)
    for new_directory in sorted(day_buffer.unflushed_directories):
        _sync_directory(new_directory)
    day_buffer.unflushed_files.clear()
    day_buffer.unflushed_directories.clear()
    if day_buffer.journal is not None and day_buffer.journal_written:
        _delete(day_buffer.journal)
        day_buffer.journal_written = False


def _write_order(day_buffer: DayBuffer, data_file: Path) -> tuple[int, SeriesKey]:
    """Give a file's place in the write: measurement files first, by series.

    Parameters
    ----------
    day_buffer : DayBuffer
        The buffer.
    data_file : Path
        A buffered file.

    Returns
    -------
    tuple of (int, series key)
        Its kind's rank and its series.
    """
    file_kind, series_key = day_buffer.series_of(data_file)
    return _FILE_KIND_ORDER[file_kind], series_key


def _prepared(day_buffer: DayBuffer) -> dict[Path, bytes]:
    """Check every file can be written, and give the bytes to write (design 5.8).

    Parameters
    ----------
    day_buffer : DayBuffer
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
    file_bytes: dict[Path, bytes] = {}
    for data_file, file_lines in day_buffer.file_lines.items():
        file_kind, _ = day_buffer.series_of(data_file)
        if os.path.lexists(data_file):
            _check_existing(data_file, file_kind)
            header_text = ""
        else:
            _check_new(data_file)
            header_text = header(file_kind)
        try:
            file_bytes[data_file] = (header_text + "".join(file_lines)).encode("ascii")
        except UnicodeEncodeError as exc:
            _fail(f"the rows for {data_file} are not ASCII", exc)
    if day_buffer.journal is not None and file_bytes and not day_buffer.journal_written:
        if os.path.lexists(day_buffer.journal):
            _fail(f"write journal {day_buffer.journal} is there: a write is still open")
        _check_new(day_buffer.journal)
    _check_space(file_bytes)
    return file_bytes


def _check_existing(data_file: Path, file_kind: FileKind) -> None:
    """Refuse an existing file that cannot be appended to soundly.

    Parameters
    ----------
    data_file : Path
        The file.
    file_kind : {'meas', 'ddiff'}
        Its kind.

    Raises
    ------
    DataFileError
        If it is not a regular file, this process cannot write it, or its
        length is not its header and one or more whole rows.
    """
    if data_file.is_symlink() or not data_file.is_file():
        _fail(f"data file {data_file} is not a regular file")
    if not os.access(data_file, os.W_OK):
        _fail(f"data file {data_file} cannot be written")
    file_length = data_file.stat().st_size
    row_slots, left_over = _row_slots(file_length, file_kind)
    if row_slots < 1 or left_over != 0:
        _fail(_not_sound(data_file, file_length))


def _check_new(new_file: Path) -> None:
    """Refuse a new file whose directory cannot be written into.

    Parameters
    ----------
    new_file : Path
        The file, which does not exist yet.

    Raises
    ------
    DataFileError
        If its directory is not a directory this process can write into.
    """
    parent_directory = new_file.parent
    if not parent_directory.is_dir() or not os.access(
        parent_directory, os.W_OK | os.X_OK
    ):
        _fail(f"file {new_file} cannot be created in {parent_directory}")


def _check_space(file_bytes: dict[Path, bytes]) -> None:
    """Refuse a write the free space does not cover.

    Parameters
    ----------
    file_bytes : dict of Path to bytes
        The bytes to write to each file.

    Raises
    ------
    DataFileError
        If, on any device, the bytes to write are more than its free space.
    """
    bytes_by_device: dict[int, tuple[Path, int]] = {}
    for data_file, file_chunk in file_bytes.items():
        device = data_file.parent.stat().st_dev
        device_directory, byte_total = bytes_by_device.get(
            device, (data_file.parent, 0)
        )
        bytes_by_device[device] = (device_directory, byte_total + len(file_chunk))
    for device_directory, byte_total in bytes_by_device.values():
        free_bytes = shutil.disk_usage(device_directory).free
        if byte_total > free_bytes:
            _fail(
                f"{byte_total} bytes to write in {device_directory},"
                f" only {free_bytes} free"
            )


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
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        _fail(f"cannot flush directory {directory}: {exc}", exc)


# ----------------------------------------------------------- roll-back, redo

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch, T."""

type Cut = Literal["kept", "cut", "deleted"]
"""What keeping a file's rows through an epoch did to it."""


def _keep_through(
    data_file: Path, file_kind: FileKind, last_kept_epoch: datetime | None
) -> Cut:
    """Keep a file's rows up to and including an epoch, and remove the rest.

    Parameters
    ----------
    data_file : Path
        The file, whose rows up to ``last_kept_epoch`` are good.
    file_kind : {'meas', 'ddiff'}
        Its kind.
    last_kept_epoch : datetime or None
        The last epoch to keep; ``None`` to keep none.

    Returns
    -------
    {'kept', 'cut', 'deleted'}
        Whether the file was left as it was, truncated just after its row
        for ``last_kept_epoch``, or deleted because it had no row at or before it.

    Raises
    ------
    DataFileError
        If the file cannot be read, changed or deleted.

    Notes
    -----
    A file's rows are in time order, though an epoch may have none: a
    series writes no row while it is left out of the epochs, nor while it
    is dormant with no measurement or disabled with no reading. The rows
    kept are found by a binary search over the row slots, a slot that is not a
    good row counting as after ``last_kept_epoch``: every row up to it is
    good, so the damage lies after it.
    """
    line_size = WIDTHS[file_kind] + 1
    try:
        with data_file.open("rb") as open_file:
            file_length = open_file.seek(0, os.SEEK_END)
            header_end = HEADER_SIZES[file_kind]
            row_slots, _ = _row_slots(file_length, file_kind)
            kept_rows = 0
            if last_kept_epoch is not None:
                kept_rows = _rows_through(
                    open_file,
                    row_slots,
                    header_end,
                    line_size,
                    file_kind,
                    last_kept_epoch,
                )
    except OSError as exc:
        _fail(f"cannot read data file {data_file}: {exc}", exc)
    if kept_rows == 0:
        _delete(data_file)
        return "deleted"
    new_length = header_end + kept_rows * line_size
    if new_length == file_length:
        return "kept"
    _truncate(data_file, new_length)
    return "cut"


def _rows_through(
    open_file: BinaryIO,
    row_slots: int,
    header_end: int,
    line_size: int,
    file_kind: FileKind,
    last_kept_epoch: datetime,
) -> int:
    """Count a file's rows up to and including an epoch, by binary search.

    Parameters
    ----------
    open_file : BinaryIO
        The file, open.
    row_slots : int
        Its whole row slots after the header.
    header_end : int
        Where its header ends, bytes from its start.
    line_size : int
        Its row width, newline included.
    file_kind : {'meas', 'ddiff'}
        Its kind.
    last_kept_epoch : datetime
        The last epoch to keep.

    Returns
    -------
    int
        How many slots from the first hold good rows at or before
        ``last_kept_epoch``.
    """
    low, high = 0, row_slots
    while low < high:
        middle = (low + high) // 2
        middle_epoch = row_epoch(
            _slot(open_file, header_end, middle, line_size), file_kind
        )
        if middle_epoch is not None and middle_epoch <= last_kept_epoch:
            low = middle + 1
        else:
            high = middle
    return low


def _delete(deleted_file: Path) -> None:
    """Delete a file, and flush its directory so the deletion is kept.

    Parameters
    ----------
    deleted_file : Path
        The file.

    Raises
    ------
    DataFileError
        If the file cannot be deleted or the device fails.
    """
    try:
        deleted_file.unlink()
    except OSError as exc:
        _fail(f"cannot delete {deleted_file}: {exc}", exc)
    _sync_directory(deleted_file.parent)


def _truncate(data_file: Path, new_length: int) -> None:
    """Cut a data file to a length, and flush it.

    Parameters
    ----------
    data_file : Path
        The file.
    new_length : int
        Its new length, bytes.

    Raises
    ------
    DataFileError
        If the file cannot be changed or the device fails.
    """
    try:
        with data_file.open("r+b") as open_file:
            open_file.truncate(new_length)
            open_file.flush()
            os.fsync(open_file.fileno())
    except OSError as exc:
        _fail(f"cannot cut data file {data_file}: {exc}", exc)


def roll_back(
    data_file: Path, file_kind: FileKind, last_kept_epoch: datetime | None
) -> Cut:
    """Cut a file back to its rows up to an epoch (design 6.7).

    Parameters
    ----------
    data_file : Path
        The file, its rows good up to ``last_kept_epoch``.
    file_kind : {'meas', 'ddiff'}
        Its kind.
    last_kept_epoch : datetime or None
        The last epoch to keep: the file's last good row, or the epoch
        before a write that stopped part way if that is earlier; ``None``
        to keep none.

    Returns
    -------
    {'kept', 'cut', 'deleted'}
        What was done to the file; nothing is logged here, since the run
        logs the whole roll-back once.

    Raises
    ------
    DataFileError
        If the file cannot be read, changed or deleted.

    Notes
    -----
    The file is truncated just after its last row at or before
    ``last_kept_epoch``, which also removes any damaged or torn line after
    it, or deleted when it has no such row. A file that already ends there
    is left as it is.
    """
    return _keep_through(data_file, file_kind, last_kept_epoch)


def redo_from(
    data_files: Iterable[tuple[Path, FileKind]],
    redo_epoch: datetime,
    channel: RfChannel,
) -> None:
    """Delete every row at or after an epoch from every file (design 6.5).

    Parameters
    ----------
    data_files : iterable of (Path, FileKind)
        Every file of the channel, measurement and double-difference, with
        its kind.
    redo_epoch : datetime
        The epoch to reprocess from.
    channel : {'a', 'b'}
        The RF channel, for the log.

    Raises
    ------
    DataFileError
        If a file cannot be read, changed or deleted, or holds no whole row
        or a damaged first row, so its rows cannot be placed in time; every
        file is checked before any is cut, so a refused file changes none.

    Notes
    -----
    Every file is truncated just before its first row at or after
    ``redo_epoch``, and deleted when it has no earlier row. When a damaged
    file's last good row comes before that, every file is cut after that
    row instead (see :func:`cut_epoch`), so the files stay in step. The run
    then goes on one epoch after the newest row left. Running it again
    after an interruption finishes the deletion: a file already cut is left
    as it is. The redo is logged once at INFO, with how many files it cut,
    deleted and left.
    """
    data_files = list(data_files)
    file_checks = [
        check_file(data_file, file_kind) for data_file, file_kind in data_files
    ]
    last_kept_epoch = cut_epoch(file_checks, redo_epoch - _EPOCH)
    cuts = [
        _keep_through(data_file, file_kind, last_kept_epoch)
        for data_file, file_kind in data_files
    ]
    _log.info(
        "redo of channel %s from %s: %d files cut, %d deleted,"
        " %d with no row at or after it",
        channel,
        redo_epoch,
        cuts.count("cut"),
        cuts.count("deleted"),
        cuts.count("kept"),
    )


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
        that is not a directory, or the device fails.

    Notes
    -----
    A directory made here is flushed into ``processed_path``, so the files
    written into it later are found after a crash.
    """
    made_any = False
    for subdirectory in (MEAS_SUBDIRECTORY, DDIFF_SUBDIRECTORY):
        archive = processed_path / subdirectory
        if archive.is_dir():
            continue
        try:
            archive.mkdir()
        except OSError as exc:
            _fail(f"cannot make archive {archive}: {exc}", exc)
        made_any = True
    if made_any:
        _sync_directory(processed_path)


def make_processed_path(processed_path: Path) -> None:
    """Make ``processed_path``, and every directory above it, when missing.

    A first run may be given a ``processed_path`` that is not there yet, and
    the run lock goes straight into it, so it is made before the lock is
    taken.

    Parameters
    ----------
    processed_path : Path
        The directory the run writes under.

    Raises
    ------
    DataFileError
        If a directory cannot be made, or its name is taken by something
        that is not a directory, or the device fails.

    Notes
    -----
    Each directory made here is flushed into the one above it, so the path
    is found after a crash.
    """
    missing_directories = [
        directory
        for directory in (processed_path, *processed_path.parents)
        if not directory.is_dir()
    ]
    for directory in reversed(missing_directories):
        try:
            directory.mkdir(exist_ok=True)
        except OSError as exc:
            _fail(f"cannot make processed directory {directory}: {exc}", exc)
        _sync_directory(directory.parent)


# ---------------------------------------------------------- the write journal


def _write_journal(journal: Path, first_epoch: datetime) -> None:
    """Write the journal of a run's first write, about to start, and flush it.

    Parameters
    ----------
    journal : Path
        The journal, which is not there yet.
    first_epoch : datetime
        The first epoch of the rows the run writes.

    Raises
    ------
    DataFileError
        If the journal cannot be written or the device fails.
    """
    try:
        with journal.open("xb") as open_file:
            open_file.write(f"{first_epoch.isoformat()}\n".encode("ascii"))
            open_file.flush()
            os.fsync(open_file.fileno())
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
        journal_bytes = journal.read_bytes()
    except OSError as exc:
        _fail(f"cannot read journal {journal}: {exc}", exc)
    try:
        first_epoch = datetime.fromisoformat(journal_bytes.decode("ascii").rstrip("\n"))
    except (
        UnicodeDecodeError,
        ValueError,
    ):
        return None
    return (
        first_epoch
        if first_epoch.tzinfo is not None and journal_bytes.endswith(b"\n")
        else None
    )


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
