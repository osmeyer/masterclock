"""The phase of a 5 MHz signal, as the measurements give it, and exact sums of it.

A phase is a whole number of picoseconds. It wraps at one period of the
signal, so a reading can only say where in the period it fell. Decycling
puts back the whole periods a reading lost (:func:`decycle`), by comparing
it with where the estimator expected the phase to be.

Where a phase is combined with a float - a rate times a time, say - the sum
is formed exactly, as a :class:`~fractions.Fraction`, and rounded once,
ties to even. A float holds every whole number only up to 2**53, and a sum
done in floats would round at every step; a Fraction holds every float
exactly, so the sum is exact at any size. A time offset is taken from two
datetimes as a whole number of microseconds, so it is exact too.
"""

import math
from datetime import timedelta
from fractions import Fraction
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError, PhaseError

if TYPE_CHECKING:
    from datetime import datetime

    from masterclock.domain.series import State

PHASE_PERIOD: Final[int] = 200_000
"""One period of a 5 MHz signal, in picoseconds."""

PHASE_MAX: Final[int] = PHASE_PERIOD - 1
"""The largest phase a reading can give, in picoseconds.

A whole period would be indistinguishable from zero.
"""

EPOCH_SECONDS: Final[int] = 600
"""How long one epoch lasts, in seconds: from one ten-minute mark to the next.

A whole number, so a phase moved on by a rate over one epoch stays exact.
"""

_MICROSECOND: Final[timedelta] = timedelta(microseconds=1)
"""The step a datetime counts in."""

_MICROSECONDS_PER_SECOND: Final[int] = 1_000_000
"""Microseconds in one second."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def seconds(later: datetime, earlier: datetime) -> Fraction:
    """Give the time from one datetime to another in seconds, exactly.

    Parameters
    ----------
    later : datetime
        Where the offset ends. Must carry a timezone.
    earlier : datetime
        Where it starts. Must carry a timezone.

    Returns
    -------
    Fraction
        ``later - earlier`` in seconds: a whole number of microseconds over
        one million, negative when ``later`` is the earlier of the two.

    Raises
    ------
    PhaseError
        If either datetime is naive, since it then names no one instant.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> seconds(
    ...     datetime(2025, 9, 23, 6, 2, 17, 203200, tzinfo=UTC),
    ...     datetime(2025, 9, 23, 6, 0, tzinfo=UTC),
    ... )
    Fraction(85752, 625)
    """
    if later.tzinfo is None or earlier.tzinfo is None:
        message = f"cannot take an offset between naive datetimes: {later}, {earlier}"
        _log.error(message)
        raise PhaseError(message)
    return Fraction((later - earlier) // _MICROSECOND, _MICROSECONDS_PER_SECOND)


def exact(value: float) -> Fraction:
    """Give the exact value a float holds, as a fraction.

    Parameters
    ----------
    value : float
        A finite float, such as a rate, a drift or a steering value.

    Returns
    -------
    Fraction
        The binary value the float holds, exactly: 0.1 gives the fraction
        nearest one tenth that a float can hold, not one tenth itself.

    Raises
    ------
    FilterError
        If ``value`` is nan or infinite, which no phase can be summed with.

    Examples
    --------
    >>> exact(0.0123) == 0.0123
    True
    >>> exact(2.5)
    Fraction(5, 2)
    """
    if not math.isfinite(value):
        message = f"value {value} is not finite"
        _log.error(message)
        raise FilterError(message)
    return Fraction(value)


def round_even(value: Fraction | int) -> int:
    """Round an exact value to the nearest whole number, a tie to the even one.

    Parameters
    ----------
    value : Fraction or int
        An exact sum, such as a phase plus a float term.

    Returns
    -------
    int
        The nearest whole number; exactly half way between two, the even
        one. Exact at any size.

    Examples
    --------
    >>> [round_even(Fraction(n, 2)) for n in (5, 7, -5)]
    [2, 4, -2]
    >>> round_even(Fraction(12_345_773_124, 10_000))
    1234577
    """
    return round(value)


class Decycled(BaseModel):
    """A reading with its whole periods put back, referred to its epoch start.

    Parameters
    ----------
    cycle_count : int
        The whole periods added to the reading: n.
    z : int
        The decycled phase at the epoch start, ps: z_E.

    Raises
    ------
    pydantic.ValidationError
        If a value is not an int, or a field is unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cycle_count: int
    z: int


def _check_reading(phi: int, delta: Fraction) -> None:
    """Refuse a reading that is not a phase, or not taken within its epoch.

    Parameters
    ----------
    phi : int
        The reading, ps.
    delta : Fraction
        The measurement time after the epoch start, s.

    Raises
    ------
    PhaseError
        If ``phi`` is outside 0 to :data:`PHASE_MAX`, or ``delta`` is
        outside 0 to :data:`EPOCH_SECONDS`, the end excluded.
    """
    if not 0 <= phi <= PHASE_MAX:
        message = f"reading {phi} is not within one period"
        _log.error(message)
        raise PhaseError(message)
    if not 0 <= delta < EPOCH_SECONDS:
        message = (
            f"measurement time {delta} s after its epoch start is not within its epoch"
        )
        _log.error(message)
        raise PhaseError(message)


def decycle(
    phi: int,
    delta: Fraction,
    w: Fraction,
    prediction: State | None,
    anchor: int | None,
) -> Decycled:
    """Put back a reading's whole periods and refer it to its epoch start.

    With a prediction, the reading is decycled against the phase the
    estimator predicts at the measurement time,
    x̂(t) = x⁻ + y⁻δ + ½d⁻δ² + w, and the motion and steering since the
    epoch start are taken off again. Without one, it is decycled against
    ``anchor``, the last measurement a dormant series buffered, or given no
    whole periods when there is none. Every sum is exact; z_E is rounded
    once, a tie to even.

    Parameters
    ----------
    phi : int
        The reading, ps, from 0 to :data:`PHASE_MAX`.
    delta : Fraction
        The measurement time after the epoch start, s: δ, from 0 to
        :data:`EPOCH_SECONDS`, the end excluded.
    w : Fraction
        Steering applied between the epoch start and the measurement, ps.
    prediction : State or None
        The predicted state at the epoch start, or ``None`` when the series
        has none.
    anchor : int or None
        The last buffered measurement of a series without a prediction, ps,
        or ``None``. Not used when there is a prediction.

    Returns
    -------
    Decycled
        The whole periods added and the decycled phase at the epoch start.

    Raises
    ------
    PhaseError
        If ``phi`` is not a phase within one period, or ``delta`` is not
        within the epoch.

    Examples
    --------
    The worked epoch of the design: the last row held x = 1 234 567 and
    y = 0.0123 ps/s, and the reading 34 579 ps came 137.2032 s after the
    mark.

    >>> from masterclock.domain.series import State
    >>> prediction = State(x=1_234_567 + exact(0.0123) * 600, y=0.0123)
    >>> decycle(34_579, Fraction(1_372_032, 10_000), Fraction(0), prediction, None)
    Decycled(cycle_count=6, z=1234577)
    """
    _check_reading(phi, delta)
    if prediction is None:
        n = 0 if anchor is None else round_even((anchor - phi + w) / PHASE_PERIOD)
        return Decycled(cycle_count=n, z=round_even(phi + n * PHASE_PERIOD - w))
    motion = exact(prediction.y) * delta + exact(prediction.d) * delta * delta / 2
    predicted = prediction.x + motion + w
    n = round_even((predicted - phi) / PHASE_PERIOD)
    return Decycled(cycle_count=n, z=round_even(phi + n * PHASE_PERIOD - motion - w))
