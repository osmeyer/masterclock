"""What one series of the forward estimator holds: its state, settings and rows.

A series is the run of rows of one pair or one triple, one row for each
epoch it is in. A row dormant with no measurement (D and P) is not written,
so a series' file can have gaps. A row of a pair whose clock or reference is
disabled (O) holds the epoch's reading and no state: nothing tracks it. Each
row is the whole state of the series at its epoch: the estimate, the
counters and the reject buffer the next epoch starts from. Across a gap
nothing carries: the series starts cold, as a new one does.

Every type here is a plain frozen dataclass, built without checks: its
values come from code or from data already checked where it entered the
program. A row alone is checked against the rules a row keeps
(:func:`check_row`), when the estimator finishes it and when it is read
back from a file.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final, Literal

from gmpy2 import mpq

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError

type FilterStates = Literal[1, 2, 3]
"""How many states a series' estimator has: phase; rate; drift."""

type Reject = tuple[datetime, float]
"""One entry of a row's reject buffer: an epoch and a value.

While a series is tracked, the value is a rejected innovation. While it is
dormant, it is a measurement kept to acquire from.
"""

type PairKey = tuple[str, str]
"""A pair (a, b): reference a measured against clock or reference b."""

type TripleKey = tuple[str, str, str]
"""A triple (r, s, c): clock c against reference r, through s; r = s when local."""

type SeriesKey = PairKey | TripleKey
"""Either kind of series."""

FLAG_ORDER: Final[str] = "ARXPODSNU"
"""Every flag a row can carry, in the order they are written in a row."""

OUTCOMES: Final[frozenset[str]] = frozenset("ARXPO")
"""The flags of which every row carries exactly one: what the epoch did."""

_NO_STATE: Final[frozenset[str]] = frozenset("DO")
"""The flags of a row with no state: dormant, or disabled."""

MAX_REJECTS: Final[int] = 3
"""How many entries a row's reject buffer holds at most."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


@dataclass(frozen=True, slots=True)
class State:
    """The estimator's state at one epoch: phase, rate and drift.

    Parameters
    ----------
    x : mpq
        Phase, ps. Exact within the epoch; rounded once when it is stored.
    y : float
        Rate, ps/s; 0.0 for a 1-state series.
    d : float, optional
        Drift, ps/s²; 0.0, the default, for a 1- or 2-state series.

    Examples
    --------
    >>> State(x=mpq(123457438, 100), y=0.0123)
    State(x=mpq(61728719,50), y=0.0123, d=0.0)
    """

    x: mpq
    y: float
    d: float = 0.0


@dataclass(frozen=True, slots=True)
class SeriesParams:
    """Everything one series needs from the configuration at one epoch.

    The clock configuration checks every value when it is read, so the
    settings it gives are never checked again.

    Parameters
    ----------
    filter_states : {1, 2, 3}
        How many states the estimator has, fixed for the life of the series.
    M : float or None
        Estimator time constant, epochs, at least 1; ``None`` exactly for a
        1-state series, which passes its measurements through.
    M_sigma : float
        Averaging constant of the innovation scale, epochs, at least 1.
    sigma0 : float
        Innovation scale a cold start begins with, ps; above zero.
    gmax : int
        Gap limit: how many held rows a prediction may run.
    n_break : int
        Counted rejects that make the series dormant, from 3 to ``gmax``.
    rms_max : int or None
        RMS limit of the gate, ps, above zero, for a pair; ``None`` for a
        triple, whose gate has no RMS test.
    disabled : bool, optional
        Whether the series is disabled: a pair one of whose clocks the
        configuration disables; never a triple.
    """

    filter_states: FilterStates
    M: float | None
    M_sigma: float
    sigma0: float
    gmax: int
    n_break: int
    rms_max: int | None
    disabled: bool = False


@dataclass(frozen=True, slots=True)
class Row:
    """The estimator's columns of one row: a series at one epoch.

    A row is built unchecked; :func:`check_row` says whether it keeps the
    rules given here.

    Parameters
    ----------
    interpolated_datetime : datetime
        The epoch start E, with its timezone.
    innovation : float or None
        The measurement less the prediction, ps; ``None`` without a
        measurement or without a prediction.
    x_fs : int or None
        Estimated phase at E, in whole femtoseconds (see
        :data:`~masterclock.domain.phase.FS_PER_PS`); ``None`` when dormant
        or disabled.
    y : float or None
        Estimated rate, ps/s; ``None`` when dormant or disabled, 0.0 for a
        1-state series.
    d : float or None
        Estimated drift, ps/s²; ``None`` when dormant or disabled, 0.0 for
        a 1- or 2-state series.
    innovation_scale : float or None
        The innovation scale, ps; ``None`` when dormant or disabled.
    step_offset : int
        Sum of the phase steps accepted in this segment, ps.
    epochs_in_segment : int
        Rows since the segment started, at least 0; 0 on its first row.
    epochs_since_accept : int
        Rows since the last accepted measurement, not counting dormant rows
        that buffer a measurement, at least 0; 0 on an accepted row.
    consecutive_rejects : int
        Consecutive counted rejects, at least 0.
    rejects : tuple of (datetime, float)
        The reject buffer, oldest first, at most :data:`MAX_REJECTS`
        entries, each value finite.
    filter_states : {1, 2, 3}
        How many states the estimator has.
    time_constant : float or None
        Estimator time constant M of this segment; ``None`` exactly for a
        1-state series.
    scale_time_constant : float
        Averaging constant M_sigma of this segment.
    flags : str
        Letters of :data:`FLAG_ORDER`, in that order: exactly one of A, R,
        X, P and O; D never with A, U never with D or on a 1-state series,
        and O alone.

    Every float is finite. A dormant row (D) or a disabled one (O) holds
    none of ``x_fs``, ``y``, ``d`` and ``innovation_scale``, and every other
    row all four; a P or O row holds no innovation; a 2-state row has no
    drift other than 0.0, and a 1-state row no rate or drift other than 0.0.
    """

    interpolated_datetime: datetime
    innovation: float | None
    x_fs: int | None
    y: float | None
    d: float | None
    innovation_scale: float | None
    step_offset: int
    epochs_in_segment: int
    epochs_since_accept: int
    consecutive_rejects: int
    rejects: tuple[Reject, ...]
    filter_states: FilterStates
    time_constant: float | None
    scale_time_constant: float
    flags: str

    def known_state(self) -> tuple[int, float, float]:
        """Give the row's phase, rate and drift.

        Returns
        -------
        tuple of (int, float, float)
            The phase in whole femtoseconds, the rate and the drift.

        Raises
        ------
        FilterError
            If the row is dormant, and so holds no state.
        """
        if self.x_fs is None or self.y is None or self.d is None:
            message = f"a dormant row of {self.interpolated_datetime} holds no state"
            _log.error(message)
            raise FilterError(message)
        return self.x_fs, self.y, self.d


_STATE_FIELDS: Final[tuple[str, ...]] = ("x_fs", "y", "d", "innovation_scale")
"""The fields a dormant or disabled row leaves empty and every other row fills."""


def check_row(row: Row) -> None:
    """Refuse a row that breaks a rule a row keeps (see :class:`Row`).

    Nothing is logged: the caller says what the row was for. The kinds of
    the values are not checked; a row read from a file has them checked
    where it is read.

    Parameters
    ----------
    row : Row
        The row.

    Raises
    ------
    ValueError
        Naming the first rule the row breaks: a float that is not finite, a
        counter below 0, a reject buffer too long or out of order, flags
        that are unknown, repeated, out of order or not one outcome, D with
        A or U, O with any other flag, a state that does not fit the flags
        or the model, an innovation on a P or O row, or a time constant
        given for a 1-state series or missing for another.

    Examples
    --------
    >>> from datetime import UTC
    >>> row = Row(
    ...     interpolated_datetime=datetime(2025, 9, 23, 6, 0, tzinfo=UTC),
    ...     innovation=None, x_fs=None, y=None, d=None, innovation_scale=None,
    ...     step_offset=0, epochs_in_segment=0, epochs_since_accept=0,
    ...     consecutive_rejects=0, rejects=(), filter_states=1,
    ...     time_constant=None, scale_time_constant=50.0, flags="PD",
    ... )
    >>> check_row(row)
    """
    _check_numbers(row)
    _check_rejects(row.rejects)
    _check_flags(row.flags)
    _check_state(row)


def _check_numbers(row: Row) -> None:
    """Refuse a float that is not finite, or a counter below 0.

    Parameters
    ----------
    row : Row
        The row.

    Raises
    ------
    ValueError
        If a float field is nan or infinite, or a counter is negative.
    """
    for field_name, number in (
        ("innovation", row.innovation),
        ("y", row.y),
        ("d", row.d),
        ("innovation_scale", row.innovation_scale),
        ("time_constant", row.time_constant),
        ("scale_time_constant", row.scale_time_constant),
    ):
        if number is not None and not math.isfinite(number):
            message = f"{field_name} {number} is not finite"
            raise ValueError(message)
    for field_name, count in (
        ("epochs_in_segment", row.epochs_in_segment),
        ("epochs_since_accept", row.epochs_since_accept),
        ("consecutive_rejects", row.consecutive_rejects),
    ):
        if count < 0:
            message = f"{field_name} {count} is below 0"
            raise ValueError(message)


def _check_rejects(reject_buffer: tuple[Reject, ...]) -> None:
    """Refuse a reject buffer too long, out of order or not finite.

    Parameters
    ----------
    reject_buffer : tuple of (datetime, float)
        The buffer.

    Raises
    ------
    ValueError
        If a value is not finite, it holds more than :data:`MAX_REJECTS`
        entries, or its epochs do not rise from first to last.
    """
    for _, reject_value in reject_buffer:
        if not math.isfinite(reject_value):
            message = f"rejects {reject_value} is not finite"
            raise ValueError(message)
    if len(reject_buffer) > MAX_REJECTS:
        message = f"the reject buffer holds at most three entries: {len(reject_buffer)}"
        raise ValueError(message)
    if any(
        later_epoch <= earlier_epoch
        for (earlier_epoch, _), (later_epoch, _) in pairwise(reject_buffer)
    ):
        message = "the reject buffer is held oldest first"
        raise ValueError(message)


def _check_flags(row_flags: str) -> None:
    """Refuse flags that are unknown, repeated, out of order or no outcome.

    Parameters
    ----------
    row_flags : str
        The flags.

    Raises
    ------
    ValueError
        If a letter is not one of :data:`FLAG_ORDER`, appears twice or out
        of order, the flags hold other than exactly one of A, R, X, P and
        O, D stands with A or U, or O does not stand alone.
    """
    ordered_flags = "".join(letter for letter in FLAG_ORDER if letter in row_flags)
    if row_flags != ordered_flags:
        message = (
            f"flags {row_flags!r} are not distinct letters of {FLAG_ORDER} in order"
        )
        raise ValueError(message)
    if len(OUTCOMES & set(row_flags)) != 1:
        message = f"flags {row_flags!r} hold other than exactly one of A, R, X, P and O"
        raise ValueError(message)
    if "O" in row_flags and row_flags != "O":
        message = f"flags {row_flags!r}: a disabled row's O stands alone"
        raise ValueError(message)
    if "D" in row_flags and ("A" in row_flags or "U" in row_flags):
        message = f"flags {row_flags!r}: a dormant row is never accepted or unsettled"
        raise ValueError(message)


def _check_state(row: Row) -> None:
    """Refuse a state that does not fit the row's flags or model.

    Parameters
    ----------
    row : Row
        The row, its flags already checked.

    Raises
    ------
    ValueError
        If a dormant or disabled row holds any part of a state, another row
        lacks one, a row without a measurement or a disabled row holds an
        innovation, or the row breaks a rule of its model (see
        :func:`_check_model`).
    """
    has_no_state = not _NO_STATE.isdisjoint(row.flags)
    empty_fields = [
        field_name for field_name in _STATE_FIELDS if getattr(row, field_name) is None
    ]
    if empty_fields != (list(_STATE_FIELDS) if has_no_state else []):
        message = (
            "a dormant or disabled row has no x_fs, y, d or innovation_scale and"
            f" every other row has all four; flags {row.flags!r}, empty"
            f" {empty_fields}"
        )
        raise ValueError(message)
    if not {"P", "O"}.isdisjoint(row.flags) and row.innovation is not None:
        message = (
            "a row with no measurement (P) or of a disabled series (O) has no"
            " innovation"
        )
        raise ValueError(message)
    _check_model(row)


def _check_model(row: Row) -> None:
    """Refuse a time constant, rate, drift or settling the row's model does not have.

    Parameters
    ----------
    row : Row
        The row.

    Raises
    ------
    ValueError
        If the time constant is given for a 1-state series or missing for
        another, a 2-state row has a drift other than 0.0, a 1-state row a
        rate or drift other than 0.0, or a 1-state row carries U.
    """
    if (row.time_constant is None) != (row.filter_states == 1):
        message = "a time constant is given exactly when filter_states is 2 or 3"
        raise ValueError(message)
    if row.filter_states < 3 and row.d not in {None, 0.0}:
        message = f"d is 0.0 in a {row.filter_states}-state row: {row.d}"
        raise ValueError(message)
    if row.filter_states == 1 and row.y not in {None, 0.0}:
        message = f"y is 0.0 in a 1-state row: {row.y}"
        raise ValueError(message)
    if row.filter_states == 1 and "U" in row.flags:
        message = "U is never carried by a 1-state row"
        raise ValueError(message)
