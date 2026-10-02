"""The phase of a 5 MHz signal, as the measurements give it, and exact sums of it.

A phase is a whole number of picoseconds. It wraps at one period of the
signal, so a reading can only say where in the period it fell.

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

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError, PhaseError

if TYPE_CHECKING:
    from datetime import datetime

PHASE_PERIOD: Final[int] = 200_000
"""One period of a 5 MHz signal, in picoseconds."""

PHASE_MAX: Final[int] = PHASE_PERIOD - 1
"""The largest phase a reading can give, in picoseconds.

A whole period would be indistinguishable from zero.
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
