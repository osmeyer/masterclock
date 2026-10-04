"""The errors of das_processor: its data files, and its worker processes.

Every one descends from :class:`~masterclock.app.exceptions.MasterClockError`,
so the one ``except`` clause that catches every error the project raises
catches these too.

:class:`RefusedLineError` is an intermediate base because a caller that skips
a bad line of data does so whatever the reason was.
"""

from typing import ClassVar

from masterclock.app.exceptions import MasterClockError


class DataFileError(MasterClockError):
    """Raised when a data file or directory cannot be used as a run needs.

    Opening, reading, writing, listing, creating and removing alike. Covers
    the raw DAS ``cd5m5m`` files and the directory holding them, and every
    file of the processed archive along with the directories they are
    written into.

    Examples
    --------
    >>> try:
    ...     raise DataFileError("cannot read data file cd5m5m_61098.dat")
    ... except MasterClockError as exc:
    ...     str(exc)
    'cannot read data file cd5m5m_61098.dat'
    """


class RefusedLineError(MasterClockError):
    """Raised when a line of a data file must not be used.

    Every reason is a subclass carrying :attr:`refusal_kind`, the word a log
    describes such a line by, so a reader of the log is told why the line
    was refused rather than being given one word for every reason - which
    tells a reader of a duplicate to go looking for a fault in the file that
    is not there.

    Notes
    -----
    Never raised itself. It is what one ``except`` clause catches when the
    reason does not matter, and it carries no ``refusal_kind`` of its own so that a
    subclass that forgot to name one fails where it is logged rather than
    quietly describing itself as something else.
    """

    refusal_kind: ClassVar[str]
    """The word a log describes a line refused for this reason by."""


class MalformedLineError(RefusedLineError):
    """Raised when a line cannot be read as a record at all.

    The wrong number of columns, or a column that is not the kind of value
    it should be. The only refusal that is a judgement on how the line was
    written.

    Examples
    --------
    >>> try:
    ...     raise MalformedLineError("expected 5 fields, found 3")
    ... except MasterClockError as exc:
    ...     f"{MalformedLineError.refusal_kind}: {exc}"
    'malformed: expected 5 fields, found 3'
    """

    refusal_kind: ClassVar[str] = "malformed"


class InconsistentLineError(RefusedLineError):
    """Raised when a line's columns contradict each other.

    Every column read, and a column that follows from the others does not
    hold what they give. The line is well formed and still cannot be half
    believed.

    Examples
    --------
    >>> try:
    ...     raise InconsistentLineError("interpolated MJD column reads '1'")
    ... except MasterClockError as exc:
    ...     f"{InconsistentLineError.refusal_kind}: {exc}"
    "inconsistent: interpolated MJD column reads '1'"
    """

    refusal_kind: ClassVar[str] = "inconsistent"


class WrongDayError(RefusedLineError):
    """Raised when a measurement falls outside the day its file is named for.

    Examples
    --------
    >>> try:
    ...     raise WrongDayError("MJD 58854.01 is not in day 58853")
    ... except MasterClockError as exc:
    ...     f"{WrongDayError.refusal_kind}: {exc}"
    'wrong-day: MJD 58854.01 is not in day 58853'
    """

    refusal_kind: ClassVar[str] = "wrong-day"


class OutOfOrderError(RefusedLineError):
    """Raised when a measurement is earlier than the one accepted before it.

    Examples
    --------
    >>> try:
    ...     raise OutOfOrderError("MJD 58853.11 is earlier than the preceding")
    ... except MasterClockError as exc:
    ...     f"{OutOfOrderError.refusal_kind}: {exc}"
    'out-of-order: MJD 58853.11 is earlier than the preceding'
    """

    refusal_kind: ClassVar[str] = "out-of-order"


class DuplicatePairError(RefusedLineError):
    """Raised when a reference-clock pair is measured twice in one epoch.

    Examples
    --------
    >>> try:
    ...     raise DuplicatePairError("clka-clkb was already measured")
    ... except MasterClockError as exc:
    ...     f"{DuplicatePairError.refusal_kind}: {exc}"
    'duplicate: clka-clkb was already measured'
    """

    refusal_kind: ClassVar[str] = "duplicate"


class LateLineError(RefusedLineError):
    """Raised when a measurement was taken too near the end of its epoch.

    Its values are recorded against the mark its epoch begins at, so a
    measurement at the very end of one is read back nearly a whole epoch
    from where it was taken.

    Examples
    --------
    >>> try:
    ...     raise LateLineError("measured 3 s before the next epoch")
    ... except MasterClockError as exc:
    ...     f"{LateLineError.refusal_kind}: {exc}"
    'late: measured 3 s before the next epoch'
    """

    refusal_kind: ClassVar[str] = "late"


class WorkerError(MasterClockError):
    """Raised when a worker process fails or stops answering.

    A failure of the project's own kinds in a worker is sent back and
    raised again as it is; this is raised for any other failure there, and
    for a worker that stopped answering.

    Examples
    --------
    >>> try:
    ...     raise WorkerError("worker 2 stopped answering")
    ... except MasterClockError as exc:
    ...     str(exc)
    'worker 2 stopped answering'
    """
