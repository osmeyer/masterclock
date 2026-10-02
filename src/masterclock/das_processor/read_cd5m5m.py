"""Reading DAS 5 MHz phase measurement files.

The Data Acquisition System (DAS) writes one file per MJD day, named
``cd5m5m_<MJD>.dat``, appending to the current day's file as data arrives.
Each line is one measurement with five whitespace-separated columns:

* MJD of the measurement, five digits, a point and six decimals,
* phase in picoseconds - a whole number from zero up to
  :data:`~masterclock.domain.phase.PHASE_MAX`, since the phase wraps at one period
  of the 5 MHz signal (unwrapping is handled downstream),
* RMS of the measurement in picoseconds,
* switch position - four characters: a reference digit ``0`` to ``9`` (naming
  the reference ``mc<digit>``, for example ``1A01`` was measured against
  ``mc1``), the switch name, and the two-digit port number on that switch,
* the name of the measured clock.

Measurements are taken in ten-minute epochs (144 per day), and all
measurements within an epoch must be processed together: every measurement
carries *interpolated* values - its measurement datetime floored to the
previous ten-minute mark - and :func:`read_blocks` yields one
:class:`DASData` per interpolated epoch.

Lines that deviate from this structure are logged at WARNING level and
skipped (see :func:`read_measurements`). So are the kinds of line that parse
but cannot be believed: one whose MJD falls on a different day from the one
its file is named for, one taken within :data:`EPOCH_EDGE` of the end of its
own epoch, one whose time runs backwards, and one repeating a
reference-clock pair already measured in the same epoch. The log names a
skipped line by the reason it was refused for, so a line that parsed is
never described as malformed. A skipped line never ends a read. A last line
with no newline does: it makes the whole file malformed.

A directory holds the daily files: :func:`read_all_blocks` scans it, keeps
only regular files (or links to them) whose names match
:data:`DATA_FILE_PATTERN`, and yields the ten-minute blocks of each daily
file in chronological order.
"""

import re
from datetime import datetime, timedelta
from itertools import chain, dropwhile, groupby
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    ValidationError,
    model_validator,
)

from masterclock.app.exceptions import describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import MJD_ORIGIN, datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor.epochs import EPOCH_LENGTH, floor_to_ten_minutes
from masterclock.das_processor.exceptions import (
    DataFileError,
    DuplicatePairError,
    LateLineError,
    MalformedLineError,
    OutOfOrderError,
    RefusedLineError,
    WrongDayError,
)
from masterclock.domain.phase import PHASE_MAX
from masterclock.domain.references import REFERENCE_PATTERN, REFERENCE_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

FIRST_DAY: Final[int] = 50_000
"""The earliest MJD day a data file can cover."""

LAST_DAY: Final[int] = 99_999
"""The latest MJD day a data file can cover."""

DATA_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(r"cd5m5m_([5-9][0-9]{4})\.dat")
"""Filename pattern of a daily DAS data file, whose group captures the MJD.

It matches the name of every day from :data:`FIRST_DAY` to :data:`LAST_DAY`
and no other.
"""

DATA_FILE_TEMPLATE: Final[str] = "cd5m5m_{mjd}.dat"
"""Name of the daily DAS data file covering one MJD day."""

_FIELD_COUNT: Final[int] = 5
"""Number of whitespace-separated columns in a measurement line."""

MJD_WIDTH: Final[int] = 12
"""Width of the measurement MJD column."""

MJD_DECIMALS: Final[int] = 6
"""Decimal places of the measurement MJD column, as the DAS itself writes."""

_PLAIN_NUMBERS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "MJD",
        re.compile(
            rf"[0-9]{{{MJD_WIDTH - MJD_DECIMALS - 1}}}\.[0-9]{{{MJD_DECIMALS}}}"
        ),
    ),
    ("phase", re.compile(r"[0-9]+")),
    ("RMS", re.compile(r"[0-9]+")),
)
"""The number columns, in file order, each with the only form it may take.

Digits, as the DAS writes them, and for the MJD exactly its form there:
five digits, a point and six decimals, for every MJD of a data day. A
sign, an exponent, an underscore, a decimal point in a whole number, or an
MJD with other decimals means something other than the DAS wrote the line.
"""


MEASURED_PHASE_WIDTH: Final[int] = 9
"""Width of the phase column."""

RMS_WIDTH: Final[int] = 4
"""Width of the RMS column."""

RMS_MAX: Final[int] = 10**RMS_WIDTH - 1
"""The largest RMS a line may give, ps: the most its column holds."""

SWITCH_PATTERN: Final[str] = r"^[0-9][A-Z][0-9]{2}$"
"""How a switch position is spelled.

The reference digit, the switch name, and the port number on that switch.
Four characters in that order and no other, so the leading digit a reference
is read from is there by the shape of the value rather than by hope.
"""

COLUMN_SEPARATOR: Final[str] = " "
"""What stands between two columns of a line.

A separator of its own rather than padding folded into the width of the
column before it. The RMS is not bounded above, so a value wider than its
column would otherwise run into the phase beside it and the line would stop
being five fields.
"""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

EPOCH_EDGE: Final[timedelta] = timedelta(seconds=10)
"""How near the end of its epoch a measurement may not be taken.

A measurement's values are recorded against the mark its epoch begins at,
so one taken at the very end of an epoch is read back nearly a whole epoch
from where it was taken. The last of an epoch is refused rather than read
that far back.

The bound is closed: a measurement exactly this far from the next mark is
within the edge and is refused, so the two sides of the boundary do not
both claim it.
"""


def _refuse_passed_in(data: object, derived: tuple[str, ...]) -> object:
    """Refuse input to a model that passes in a value the model works out.

    Parameters
    ----------
    data : object
        What the model is being built from. Only a mapping can name a field,
        so anything else is let through for pydantic to judge.
    derived : tuple[str, ...]
        The names of the fields the model works out for itself.

    Returns
    -------
    object
        ``data``, unchanged.

    Raises
    ------
    ValueError
        Naming every one of ``derived`` that ``data`` passes in. A plain
        ``ValueError`` so that pydantic reports it as a validation error.

    Examples
    --------
    >>> _refuse_passed_in({"a": 1}, ("b",))
    {'a': 1}
    >>> _refuse_passed_in({"a": 1, "b": 2}, ("b",))
    Traceback (most recent call last):
    ...
    ValueError: worked out, so may not be passed in: b
    """
    if isinstance(data, dict):
        passed = [name for name in derived if name in data]
        if passed:
            names = ", ".join(passed)
            raise ValueError(f"worked out, so may not be passed in: {names}")
    return data


class DASMeasurement(BaseModel):
    """One validated DAS phase measurement.

    Parameters
    ----------
    measurement_mjd : float
        MJD at which the measurement was taken, on a day from
        :data:`FIRST_DAY` to :data:`LAST_DAY`.
    measured_phase : int
        Phase in picoseconds, from 0 to
        :data:`~masterclock.domain.phase.PHASE_MAX` inclusive; a whole
        :data:`~masterclock.domain.phase.PHASE_PERIOD` would be indistinguishable
        from zero.
    rms : int
        RMS of the measurement in picoseconds, from 0 to :data:`RMS_MAX`.
    switch : str
        Switch position, as :data:`SWITCH_PATTERN` spells it: the reference
        digit, the switch name, and the port number on that switch. Kept
        whole, as the line wrote it.
    clock : str
        Name of the measured clock.

    Attributes
    ----------
    reference : str
        The reference measured against, worked out from the switch.
    measurement_datetime : AwareDatetime
        The measurement instant, worked out from ``measurement_mjd``.
    interpolated_datetime : AwareDatetime
        The start of the measurement's ten-minute epoch, worked out from
        ``measurement_datetime``.
    interpolated_mjd : PositiveFloat
        That same epoch, worked out from ``interpolated_datetime``.

    Raises
    ------
    pydantic.ValidationError
        If a column is invalid, or if any of the four attributes is passed
        in.

    Notes
    -----
    The four attributes follow from the five columns and are worked out when
    the record is built rather than each time they are read. None of them
    may be passed in, so no two values a record holds can disagree.

    Examples
    --------
    >>> measurement = DASMeasurement(
    ...     measurement_mjd=60010.000694,
    ...     measured_phase=12345,
    ...     rms=21,
    ...     switch="3B07",
    ...     clock="clkb",
    ... )
    >>> measurement.reference
    'mc3'
    >>> measurement.interpolated_mjd
    60010.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measurement_mjd: Annotated[float, Field(ge=FIRST_DAY, lt=LAST_DAY + 1)]
    measured_phase: Annotated[int, Field(ge=0, le=PHASE_MAX)]
    rms: Annotated[int, Field(ge=0, le=RMS_MAX)]
    switch: Annotated[str, Field(pattern=SWITCH_PATTERN)]
    clock: Annotated[str, Field(min_length=1)]
    # The four below are worked out in model_post_init. Each has a default
    # only so that a call can leave it out; _refuse_derived makes sure every
    # call does, and model_post_init replaces every default.
    reference: Annotated[str, Field(pattern=REFERENCE_PATTERN)] = ""
    measurement_datetime: AwareDatetime = MJD_ORIGIN
    interpolated_datetime: AwareDatetime = MJD_ORIGIN
    interpolated_mjd: PositiveFloat = 1.0

    _DERIVED: ClassVar[tuple[str, ...]] = (
        "reference",
        "measurement_datetime",
        "interpolated_datetime",
        "interpolated_mjd",
    )
    """The attributes worked out from the columns, which may not be passed in."""

    @model_validator(mode="before")
    @classmethod
    def _refuse_derived(cls, data: object) -> object:
        """Refuse a call that passes in any attribute worked out from the columns.

        Parameters
        ----------
        data : object
            What the record is being built from.

        Returns
        -------
        object
            ``data``, unchanged.

        Raises
        ------
        ValueError
            Naming each of :attr:`_DERIVED` that was passed in.
        """
        return _refuse_passed_in(data, cls._DERIVED)

    def model_post_init(self, _context: object, /) -> None:
        """Work out the four attributes that follow from the columns.

        Parameters
        ----------
        _context : object
            Pydantic's validation context, unused.

        Notes
        -----
        This runs once the columns have been validated, so the switch is a
        switch and the MJD is a number by the time they are read here.
        """
        measured = mjd_to_datetime(self.measurement_mjd)
        epoch = floor_to_ten_minutes(measured)
        object.__setattr__(self, "reference", f"{REFERENCE_PREFIX}{self.switch[0]}")
        object.__setattr__(self, "measurement_datetime", measured)
        object.__setattr__(self, "interpolated_datetime", epoch)
        object.__setattr__(self, "interpolated_mjd", datetime_to_mjd(epoch))

    def __str__(self) -> str:
        """Render the measurement as a data file line, laid out as the DAS writes it.

        Returns
        -------
        str
            The line, without a trailing newline: the columns in file order,
            each separated by one :data:`COLUMN_SEPARATOR` and right-justified
            to its width, except the clock name, which is written as it is.
            The MJD is rounded to :data:`MJD_DECIMALS` places, so a line read
            with more places, or spaced differently, is not given back as it
            was read.

        Examples
        --------
        >>> str(
        ...     DASMeasurement(
        ...         measurement_mjd=60010.000694,
        ...         measured_phase=12345,
        ...         rms=21,
        ...         switch="3B07",
        ...         clock="clkb",
        ...     )
        ... )
        '60010.000694     12345   21 3B07 clkb'
        """
        return COLUMN_SEPARATOR.join(
            (
                f"{self.measurement_mjd:{MJD_WIDTH}.{MJD_DECIMALS}f}",
                f"{self.measured_phase:{MEASURED_PHASE_WIDTH}d}",
                f"{self.rms:{RMS_WIDTH}d}",
                self.switch,
                self.clock,
            )
        )


class DASData(BaseModel):
    """All measurements belonging to one ten-minute epoch.

    Parameters
    ----------
    interpolated_datetime : AwareDatetime
        The ten-minute mark this epoch starts at; the same instant every
        member measurement was referred back to.
    measurements : tuple[DASMeasurement, ...]
        The measurements in this epoch, in file order; never empty.

    Attributes
    ----------
    interpolated_mjd : PositiveFloat
        That same mark, worked out from ``interpolated_datetime``. It may
        not be passed in.

    Raises
    ------
    pydantic.ValidationError
        If a field is invalid, if ``interpolated_mjd`` is passed in, or if a
        measurement belongs to another epoch.

    Notes
    -----
    The mark is the datetime and the MJD is a rendering of it, the same way
    round as in the measurements the block holds, so a block and its own
    members cannot disagree about which of the two is the real value.

    Measurements stay in file order. Nothing here indexes them by pair: an
    epoch is processed whole, so what reads a block reads all of it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    interpolated_datetime: AwareDatetime
    measurements: Annotated[tuple[DASMeasurement, ...], Field(min_length=1)]
    # Worked out in model_post_init, as in a measurement.
    interpolated_mjd: PositiveFloat = 1.0

    _DERIVED: ClassVar[tuple[str, ...]] = ("interpolated_mjd",)
    """The attribute worked out from the mark, which may not be passed in."""

    @model_validator(mode="before")
    @classmethod
    def _refuse_derived(cls, data: object) -> object:
        """Refuse a call that passes in the MJD worked out from the mark.

        Parameters
        ----------
        data : object
            What the block is being built from.

        Returns
        -------
        object
            ``data``, unchanged.

        Raises
        ------
        ValueError
            If ``interpolated_mjd`` was passed in.
        """
        return _refuse_passed_in(data, cls._DERIVED)

    def model_post_init(self, _context: object, /) -> None:
        """Work out the epoch as an MJD.

        Parameters
        ----------
        _context : object
            Pydantic's validation context, unused.
        """
        object.__setattr__(
            self, "interpolated_mjd", datetime_to_mjd(self.interpolated_datetime)
        )

    @model_validator(mode="after")
    def _check_one_epoch(self) -> Self:
        """Refuse a block holding a measurement from another epoch.

        Returns
        -------
        Self
            The block, unchanged.

        Raises
        ------
        ValueError
            Naming the first measurement whose epoch is not the block's.
            A plain ``ValueError`` because pydantic gathers one raised in a
            validator into its own ``ValidationError``; any other type would
            escape the model's validation instead of joining it.

        Notes
        -----
        Grouping a stream is what ordinarily builds a block, and that cannot
        produce a mismatch. This is for every other way one might be built,
        where the epoch reported and the measurements held would otherwise be
        two independent claims.
        """
        for measurement in self.measurements:
            if measurement.interpolated_datetime != self.interpolated_datetime:
                raise ValueError(
                    "a measurement of epoch "
                    f"{measurement.interpolated_datetime.isoformat(sep=' ')} "
                    "does not belong in the block for epoch "
                    f"{self.interpolated_datetime.isoformat(sep=' ')}"
                )
        return self


def parse_line(line: str) -> DASMeasurement:
    """Parse one measurement line.

    Parameters
    ----------
    line : str
        The raw line from a DAS data file.

    Returns
    -------
    DASMeasurement
        The validated measurement.

    Raises
    ------
    MalformedLineError
        If the line is not valid UTF-8 text, does not have exactly five
        columns, has a number column not written in its plain form (see
        :data:`_PLAIN_NUMBERS`), or any column fails validation.

    Notes
    -----
    A data file is read with every byte that is not UTF-8 kept as a lone
    surrogate, so the line it was on reaches here and is refused as
    malformed instead of ending the read of the whole file.
    """
    try:
        line.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise MalformedLineError("not UTF-8 text") from exc
    fields = line.split()
    if len(fields) != _FIELD_COUNT:
        raise MalformedLineError(f"expected {_FIELD_COUNT} fields, found {len(fields)}")
    measurement_mjd, measured_phase, rms, switch, clock = fields
    for (name, form), text in zip(
        _PLAIN_NUMBERS, (measurement_mjd, measured_phase, rms), strict=True
    ):
        if form.fullmatch(text) is None:
            raise MalformedLineError(f"{name} {text!r} is not a plain number")
    try:
        return DASMeasurement.model_validate(
            {
                "measurement_mjd": measurement_mjd,
                "measured_phase": measured_phase,
                "rms": rms,
                "switch": switch,
                "clock": clock,
            }
        )
    except ValidationError as exc:
        raise MalformedLineError(describe_error(exc)) from exc


def _file_mjd(path: Path) -> int:
    """Extract the MJD day from a daily data file's name.

    Parameters
    ----------
    path : Path
        Path to a daily data file named ``cd5m5m_<MJD>.dat``.

    Returns
    -------
    int
        The MJD day the file covers.

    Raises
    ------
    DataFileError
        If the file name does not match :data:`DATA_FILE_PATTERN`, and so
        carries no MJD to read.
    """
    match = DATA_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        message = f"not a daily data file name: {path.name!r}"
        _log.error(message)
        raise DataFileError(message)
    return int(match.group(1))


def data_file_name(mjd: int) -> str:
    """Name the daily data file covering one MJD day.

    The inverse of :func:`_file_mjd`: every name it gives matches
    :data:`DATA_FILE_PATTERN`, and reading the day back from the name gives
    ``mjd``.

    Parameters
    ----------
    mjd : int
        The MJD day the file covers, from :data:`FIRST_DAY` to
        :data:`LAST_DAY`.

    Returns
    -------
    str
        The file's name, without a directory.

    Raises
    ------
    ValueError
        If ``mjd`` is outside :data:`FIRST_DAY` to :data:`LAST_DAY`.

    Examples
    --------
    >>> data_file_name(60010)
    'cd5m5m_60010.dat'
    """
    if not FIRST_DAY <= mjd <= LAST_DAY:
        message = f"MJD day {mjd} is outside {FIRST_DAY} to {LAST_DAY}"
        raise ValueError(message)
    return DATA_FILE_TEMPLATE.format(mjd=mjd)


def _check_day(measurement: DASMeasurement, file_mjd: int) -> None:
    """Refuse a measurement that does not belong to the day it was found in.

    Parameters
    ----------
    measurement : DASMeasurement
        The measurement parsed from the line.
    file_mjd : int
        The MJD day named by the file the line came from.

    Raises
    ------
    WrongDayError
        If the whole-day part of the measurement's MJD is not ``file_mjd``.

    Notes
    -----
    A measurement in the wrong day's file would otherwise be processed as
    though it belonged there, placing it in an epoch that another file also
    covers. Since the files are read in the order their names give, that
    would put measurements out of time order without anything saying so.
    """
    day = int(measurement.measurement_mjd)
    if day != file_mjd:
        raise WrongDayError(
            f"MJD {measurement.measurement_mjd} is not in day {file_mjd}",
        )


def _check_early(measurement: DASMeasurement) -> None:
    """Refuse a measurement taken too near the end of its own epoch.

    Parameters
    ----------
    measurement : DASMeasurement
        The measurement parsed from the line.

    Raises
    ------
    LateLineError
        If the next ten-minute mark is :data:`EPOCH_EDGE` away or less.

    Notes
    -----
    Measured from the measurement's own epoch rather than from the day, so
    every epoch is judged by its own end.
    """
    mark = measurement.interpolated_datetime + EPOCH_LENGTH
    before = mark - measurement.measurement_datetime
    if before <= EPOCH_EDGE:
        raise LateLineError(
            f"measured {before.total_seconds():.3f} s before the next epoch, "
            f"inside the last {EPOCH_EDGE.total_seconds():.0f} s of its own"
        )


def _check_forward(measurement: DASMeasurement, previous_mjd: float | None) -> None:
    """Refuse a measurement whose time runs backwards.

    Parameters
    ----------
    measurement : DASMeasurement
        The measurement parsed from the line.
    previous_mjd : float or None
        The MJD of the last measurement accepted from this file, or ``None``
        when none has been accepted yet.

    Raises
    ------
    OutOfOrderError
        If the measurement is earlier than the one before it.

    Notes
    -----
    Not strictly increasing: the DAS records several clocks at one instant,
    so repeated timestamps are ordinary. Only going backwards is refused,
    since a measurement out of order would be grouped into an epoch that has
    already been yielded and processed.
    """
    if previous_mjd is not None and measurement.measurement_mjd < previous_mjd:
        raise OutOfOrderError(
            f"MJD {measurement.measurement_mjd} is earlier than the "
            f"preceding {previous_mjd}",
        )


def _check_unseen(measurement: DASMeasurement, seen: set[tuple[str, str]]) -> None:
    """Refuse a reference-clock pair already measured in this epoch.

    Parameters
    ----------
    measurement : DASMeasurement
        The measurement parsed from the line.
    seen : set of (str, str)
        The reference-clock pairs already accepted in the current epoch.

    Raises
    ------
    DuplicatePairError
        If this pair has already been measured in this epoch.

    Notes
    -----
    A pair has one phase per epoch. A second reading of the same pair would
    give the epoch two values for one measurement, and nothing downstream
    could say which to believe.
    """
    if (measurement.reference, measurement.clock) in seen:
        raise DuplicatePairError(
            f"{measurement.reference}-{measurement.clock} was already measured "
            "in this epoch",
        )


def _check_terminated(line: str, number: int, path: Path) -> None:
    """Refuse a file whose line has no newline at its end.

    Parameters
    ----------
    line : str
        The line as read, with its newline if it has one.
    number : int
        The line's number in the file, counting from one.
    path : Path
        The file the line came from, for the message.

    Raises
    ------
    DataFileError
        If ``line`` does not end with a newline.

    Notes
    -----
    Only the last line of a file can lack one. The DAS ends every line it
    writes with a newline, so a file whose last line has none is malformed:
    the line may be cut short, and a clock name cut short still reads as a
    clock name.
    """
    if not line.endswith("\n"):
        message = f"malformed data file {path}: line {number} has no newline"
        _log.error(message)
        raise DataFileError(message)


def read_measurements(path: Path) -> Iterator[DASMeasurement]:
    """Yield the valid measurements from a DAS data file, in file order.

    Lines that deviate from the expected structure are logged at WARNING
    level (with their line number and content) and skipped, as are lines
    that parse but belong to another day, were taken within
    :data:`EPOCH_EDGE` of the end of their epoch, run backwards in time, or
    repeat a reference-clock pair already measured in the same epoch. Each is
    named in the log by the reason it was refused for, so only a line that
    would not parse is called malformed. A skipped line is not remembered: it
    sets neither the preceding time nor the pairs seen in the epoch.

    Parameters
    ----------
    path : Path
        Path to a ``cd5m5m_<MJD>.dat`` file.

    Yields
    ------
    DASMeasurement
        Each valid measurement, in file order.

    Raises
    ------
    DataFileError
        If the file cannot be opened or read, or its name carries no MJD.
        As this is a generator, the error is raised on first iteration, not
        at call time. Also if the file's last line has no newline, raised
        when that line is reached, after every measurement before it has
        been yielded.
    """
    file_mjd = _file_mjd(path)
    previous_mjd: float | None = None
    epoch: datetime | None = None
    seen: set[tuple[str, str]] = set()
    try:
        with path.open(encoding="utf-8", errors="surrogateescape") as file:
            for number, line in enumerate(file, start=1):
                _check_terminated(line, number, path)
                try:
                    measurement = parse_line(line)
                    _check_day(measurement, file_mjd)
                    _check_early(measurement)
                    _check_forward(measurement, previous_mjd)
                    if measurement.interpolated_datetime != epoch:
                        epoch = measurement.interpolated_datetime
                        seen = set()
                    _check_unseen(measurement, seen)
                except RefusedLineError as exc:
                    _log.warning(
                        "skipping %s line %d of %s: %r (%s)",
                        exc.kind,
                        number,
                        path,
                        line.rstrip("\n"),
                        exc,
                    )
                    continue
                previous_mjd = measurement.measurement_mjd
                seen.add((measurement.reference, measurement.clock))
                yield measurement
    except OSError as exc:
        message = f"cannot read data file {path}: {exc}"
        _log.error(message)
        raise DataFileError(message) from exc


def iter_blocks(measurements: Iterable[DASMeasurement]) -> Iterator[DASData]:
    """Group a measurement stream into ten-minute blocks, one per epoch.

    Measurements are grouped consecutively on their ``interpolated_datetime``: a
    block is yielded whenever the next measurement falls in a different
    ten-minute epoch. For the time-ordered streams the DAS produces, this
    yields exactly one block per epoch present in the stream.

    Parameters
    ----------
    measurements : Iterable[DASMeasurement]
        A stream of measurements, ordinarily in time order.

    Yields
    ------
    DASData
        The consecutive measurements of each ten-minute epoch, in stream
        order.

    Notes
    -----
    Grouping is by consecutive run, so a stream that went back to an earlier
    epoch would yield that epoch as two blocks rather than one; nothing is
    reordered or merged. No stream read from the DAS files can do that:
    :func:`read_measurements` refuses a measurement earlier than the last it
    accepted, the epoch is that measurement's instant floored and so moves
    with it, and each daily file holds one day and they are read in order.
    The behaviour is stated for a caller grouping a stream of its own.
    """
    for interpolated_datetime, group in groupby(
        measurements, key=lambda measurement: measurement.interpolated_datetime
    ):
        yield DASData(
            interpolated_datetime=interpolated_datetime, measurements=tuple(group)
        )


def read_blocks(path: Path) -> Iterator[DASData]:
    """Read a DAS data file as one ten-minute block at a time.

    Equivalent to ``iter_blocks(read_measurements(path))``: refused lines are
    logged and skipped, and the valid measurements are grouped by their
    ten-minute epoch.

    Parameters
    ----------
    path : Path
        Path to a ``cd5m5m_<MJD>.dat`` file.

    Returns
    -------
    Iterator[DASData]
        The measurements of each ten-minute epoch, in file order.

    Raises
    ------
    DataFileError
        If the file cannot be opened or read. The file is opened by
        :func:`read_measurements`, which is a generator, so the error
        surfaces on first iteration rather than at call time.
    """
    return iter_blocks(read_measurements(path))


def find_data_files(directory: Path) -> tuple[Path, ...]:
    """Find the daily DAS data files in a directory, in chronological order.

    An entry is a data file when its name matches :data:`DATA_FILE_PATTERN`
    (``cd5m5m_<MJD>.dat``) and it is a regular file or a link to one. Every
    other entry is
    logged at DEBUG level and ignored, under the reason it failed: the name
    or the kind of thing it is. The fixed-width MJD in the name makes the
    lexicographic sort chronological.

    Parameters
    ----------
    directory : Path
        The directory holding the daily ``cd5m5m_<MJD>.dat`` files.

    Returns
    -------
    tuple[Path, ...]
        The data file paths, sorted by MJD. Empty (with a WARNING logged) if
        the directory holds no data files.

    Raises
    ------
    DataFileError
        If the directory cannot be listed.
    """
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        message = f"cannot list the DAS directory {directory}: {exc}"
        _log.error(message)
        raise DataFileError(message) from exc
    files: list[Path] = []
    for entry in sorted(entries):
        if not DATA_FILE_PATTERN.fullmatch(entry.name):
            _log.debug("ignoring %s, which is not named as a daily data file", entry)
        elif not entry.is_file():
            _log.debug("ignoring %s, named as a daily data file but not a file", entry)
        else:
            files.append(entry)
    if not files:
        _log.warning("no cd5m5m data files found in %s", directory)
    return tuple(files)


def read_all_blocks(
    directory: Path, start_at_mjd: float | None = None
) -> Iterator[DASData]:
    """Read the ten-minute blocks of every daily data file in a directory.

    The data files are found with :func:`find_data_files` and read in
    chronological order, yielding each file's blocks in turn as if the daily
    files were one continuous measurement stream.

    Parameters
    ----------
    directory : Path
        The directory holding the daily ``cd5m5m_<MJD>.dat`` files.
    start_at_mjd : float, optional
        If given, start at the ten-minute epoch containing this MJD: the
        first block yielded is the first whose ``interpolated_datetime`` is at
        least the ten-minute floor of ``start_at_mjd``. Daily files that end
        before that epoch are skipped entirely (by the MJD in their name),
        without being read.

    Returns
    -------
    Iterator[DASData]
        The measurements of each ten-minute epoch, file by file.

    Raises
    ------
    DataFileError
        At call time if the directory cannot be listed, or during iteration
        if a data file cannot be opened or read.
    ValueError
        At call time if ``start_at_mjd`` is not a number, or is outside the
        years 1 to 9999.
    OverflowError
        At call time if ``start_at_mjd`` is infinite or too large for the
        platform's time functions.
    """
    files = find_data_files(directory)
    if start_at_mjd is None:
        return iter_blocks(
            chain.from_iterable(read_measurements(path) for path in files)
        )
    start_mjd = datetime_to_mjd(floor_to_ten_minutes(mjd_to_datetime(start_at_mjd)))
    kept = tuple(path for path in files if _file_mjd(path) + 1 > start_mjd)
    blocks = iter_blocks(chain.from_iterable(read_measurements(path) for path in kept))
    return dropwhile(lambda block: block.interpolated_mjd < start_mjd, blocks)
