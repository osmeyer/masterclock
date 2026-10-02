"""The ten-minute epochs das_processor steps through, and their text.

A datetime is rounded down or up to a ten-minute mark, and a mark and the
same instant as an MJD are rendered as the text the program's files carry
(see :func:`format_epoch`). Every datetime is first taken to UTC as
:func:`~masterclock.app.timeutil.ensure_utc` takes it, and every datetime
returned is in UTC. Every function here is wrapped in a bounded LRU cache.
"""

from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from masterclock.app.timeutil import ensure_utc
from masterclock.domain.phase import EPOCH_SECONDS

if TYPE_CHECKING:
    from datetime import datetime

    from masterclock.app.timeutil import DatetimeLike

EPOCH_LENGTH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""How long one epoch lasts: from one ten-minute mark to the next."""

_CACHE_SIZE: Final[int] = 1024
"""Bound of each function's LRU cache, in distinct arguments remembered."""


@lru_cache(maxsize=_CACHE_SIZE)
def floor_to_ten_minutes(given_datetime: DatetimeLike) -> datetime:
    """Round a datetime down to a ten-minute mark.

    A datetime already exactly on a ten-minute mark (minute divisible by ten,
    zero seconds and microseconds) is returned unchanged. The input is
    normalized to UTC first (see
    :func:`~masterclock.app.timeutil.ensure_utc`).

    Parameters
    ----------
    given_datetime : datetime or str
        The datetime (naive datetimes are assumed UTC), or an ISO 8601 string.

    Returns
    -------
    datetime
        The most recent ten-minute mark at or before ``given_datetime``, in UTC.

    Raises
    ------
    ValueError
        If ``given_datetime`` is a string that is not valid ISO 8601.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> floor_to_ten_minutes(datetime(2026, 7, 12, 4, 37, 56, tzinfo=UTC))
    datetime.datetime(2026, 7, 12, 4, 30, tzinfo=datetime.timezone.utc)
    >>> floor_to_ten_minutes(datetime(2026, 7, 12, 4, 40, tzinfo=UTC))
    datetime.datetime(2026, 7, 12, 4, 40, tzinfo=datetime.timezone.utc)
    """
    utc_datetime = ensure_utc(given_datetime)
    return utc_datetime.replace(
        minute=utc_datetime.minute - utc_datetime.minute % 10, second=0, microsecond=0
    )


@lru_cache(maxsize=_CACHE_SIZE)
def ceil_to_ten_minutes(given_datetime: DatetimeLike) -> datetime:
    """Round a datetime up to the next ten-minute mark.

    The result is always strictly after the input: a datetime already
    exactly on a ten-minute mark (minute divisible by ten, zero seconds and
    microseconds) returns the *next* mark, ten minutes later. The input is
    normalized to UTC first (see
    :func:`~masterclock.app.timeutil.ensure_utc`).

    Parameters
    ----------
    given_datetime : datetime or str
        The datetime (naive datetimes are assumed UTC), or an ISO 8601 string.

    Returns
    -------
    datetime
        The earliest ten-minute mark strictly after ``given_datetime``, in UTC.

    Raises
    ------
    ValueError
        If ``given_datetime`` is a string that is not valid ISO 8601.
    OverflowError
        If the next mark is after the last instant of year 9999.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> ceil_to_ten_minutes(datetime(2026, 7, 12, 4, 37, 56, tzinfo=UTC))
    datetime.datetime(2026, 7, 12, 4, 40, tzinfo=datetime.timezone.utc)
    >>> ceil_to_ten_minutes(datetime(2026, 7, 12, 4, 40, tzinfo=UTC))
    datetime.datetime(2026, 7, 12, 4, 50, tzinfo=datetime.timezone.utc)
    """
    return floor_to_ten_minutes(given_datetime) + EPOCH_LENGTH


@lru_cache(maxsize=_CACHE_SIZE)
def format_epoch(
    epoch_start: datetime, mjd: float, mjd_width: int, mjd_decimals: int
) -> tuple[str, str]:
    """Render a ten-minute mark, and that same mark as an MJD, as text.

    Parameters
    ----------
    epoch_start : datetime
        The mark (naive datetimes are assumed UTC). It is written in UTC
        whatever its timezone (see
    :func:`~masterclock.app.timeutil.ensure_utc`).
    mjd : float
        The same instant as a Modified Julian Day. Given rather than worked
        out here, so what is rendered is the value its caller holds.
    mjd_width : int
        Width to right-justify the MJD to.
    mjd_decimals : int
        Decimal places to give it. The float is rounded to the nearest
        value with that many places; a float exactly halfway between two
        goes to the one whose last digit is even. A value that is, or
        rounds to, negative zero is written as zero, with no minus sign.

    Returns
    -------
    tuple[str, str]
        The mark to the second (any fraction of a second is dropped, not
        rounded), in UTC with its ``+00:00`` offset, and the MJD
        right-justified to ``mjd_width``. An MJD that needs more characters than
        ``mjd_width`` is given in full, so the second string is then longer.

    Notes
    -----
    Cached because a mark is rendered once for every line that reports it,
    and every line written for one ten-minute epoch carries the same two
    strings.

    The cache treats arguments that compare equal as the same call: two
    datetimes naming the same instant in different timezones, or ``-0.0`` and
    ``0.0``. This does no harm, since each such pair is written as the same
    text.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> format_epoch(datetime(2024, 3, 5, 6, 20, tzinfo=UTC), 60374.263889, 13, 6)
    ('2024-03-05 06:20:00+00:00', ' 60374.263889')
    """
    utc_start = ensure_utc(epoch_start)
    return utc_start.isoformat(
        sep=" ", timespec="seconds"
    ), f"{mjd:z{mjd_width}.{mjd_decimals}f}"
