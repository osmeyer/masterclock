"""Datetime conversion utilities.

Converts an instant in time between the representations below:

* :class:`~datetime.datetime` objects (or ISO 8601 strings),
* Modified Julian Days (MJD, as floats or strings),
* Julian Days (JD, as floats or strings),
* Unix timestamps (seconds since 1970-01-01 00:00:00 UTC, as floats or strings).

Every datetime entering these functions is normalized to UTC: naive datetimes
are assumed to already be in UTC and are made timezone-aware, while aware
datetimes in any other timezone are converted (see :func:`ensure_utc`). All
returned datetimes are timezone-aware in UTC. Every function here is wrapped
in a bounded LRU cache.

The Julian Day is the continuous count of days since noon on 4713-01-01 BCE
(proleptic Julian calendar); the Modified Julian Day is ``JD - 2400000.5``,
so MJD 0 began at midnight on 1858-11-17 UTC and MJD rolls over at
midnight rather than noon. The Unix epoch corresponds to JD 2440587.5 and
MJD 40587.

Every day is taken to be :data:`SECONDS_PER_DAY` long, as in Unix time, so
leap seconds are not counted, and a time written with second 60 is refused
as invalid ISO 8601. MJD and JD are carried as floats, so how finely they
resolve an instant depends on the size of the number: near present-day
values, adjacent floats are under a microsecond apart as an MJD and about
40 microseconds apart as a JD.
"""

from datetime import UTC, datetime
from functools import lru_cache
from typing import Final

type NumberLike = float | str
"""A number given as a float (or int) or as its string representation.

A string is read by :class:`float`, so anything it accepts is taken:
surrounding whitespace, underscores between digits, and ``nan`` and ``inf``.
"""

type DatetimeLike = datetime | str
"""A datetime given as a :class:`~datetime.datetime` or an ISO 8601 string."""

SECONDS_PER_DAY: Final[float] = 86_400.0
"""Number of seconds in one day."""

MJD_AT_UNIX_EPOCH: Final[float] = 40_587.0
"""Modified Julian Day of the Unix epoch, midnight on 1970-01-01 UTC."""

JD_AT_UNIX_EPOCH: Final[float] = 2_440_587.5
"""Julian Day of the Unix epoch, midnight on 1970-01-01 UTC."""

JD_MINUS_MJD: Final[float] = 2_400_000.5
"""Constant offset between Julian Day and Modified Julian Day."""

MJD_ORIGIN: Final[datetime] = datetime(1858, 11, 17, tzinfo=UTC)
"""The instant Modified Julian Day zero names, midnight on 1858-11-17 UTC.

It is meant as the placeholder value of a datetime field that a record fills
in for itself, since such a field still needs a value when it is declared.
"""

_CACHE_SIZE: Final[int] = 1024
"""Bound of each function's LRU cache, in distinct arguments remembered."""


@lru_cache(maxsize=_CACHE_SIZE)
def ensure_utc(value: DatetimeLike) -> datetime:
    """Return the given datetime as a timezone-aware UTC datetime.

    Naive datetimes are assumed to already be in UTC and are tagged as such;
    aware datetimes in any other timezone are converted to UTC.

    Parameters
    ----------
    value : datetime or str
        The datetime, or an ISO 8601 string parseable by
        :meth:`datetime.datetime.fromisoformat`.

    Returns
    -------
    datetime
        The same instant, timezone-aware in UTC.

    Raises
    ------
    ValueError
        If ``value`` is a string that is not valid ISO 8601.

    Examples
    --------
    >>> from datetime import datetime
    >>> ensure_utc(datetime(2026, 7, 12, 3, 0))
    datetime.datetime(2026, 7, 12, 3, 0, tzinfo=datetime.timezone.utc)
    >>> ensure_utc("2026-07-12 05:00:00+02:00")
    datetime.datetime(2026, 7, 12, 3, 0, tzinfo=datetime.timezone.utc)
    """
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


@lru_cache(maxsize=_CACHE_SIZE)
def unix_to_mjd(value: NumberLike) -> float:
    """Convert a Unix timestamp to a Modified Julian Day.

    Parameters
    ----------
    value : float or str
        Seconds since the Unix epoch.

    Returns
    -------
    float
        The equivalent Modified Julian Day.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.

    Examples
    --------
    >>> unix_to_mjd(0.0)
    40587.0
    >>> unix_to_mjd("86400")
    40588.0
    """
    return float(value) / SECONDS_PER_DAY + MJD_AT_UNIX_EPOCH


@lru_cache(maxsize=_CACHE_SIZE)
def mjd_to_unix(value: NumberLike) -> float:
    """Convert a Modified Julian Day to a Unix timestamp.

    Parameters
    ----------
    value : float or str
        The Modified Julian Day.

    Returns
    -------
    float
        Seconds since the Unix epoch.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.
    """
    return (float(value) - MJD_AT_UNIX_EPOCH) * SECONDS_PER_DAY


@lru_cache(maxsize=_CACHE_SIZE)
def unix_to_jd(value: NumberLike) -> float:
    """Convert a Unix timestamp to a Julian Day.

    Parameters
    ----------
    value : float or str
        Seconds since the Unix epoch.

    Returns
    -------
    float
        The equivalent Julian Day.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.
    """
    return float(value) / SECONDS_PER_DAY + JD_AT_UNIX_EPOCH


@lru_cache(maxsize=_CACHE_SIZE)
def jd_to_unix(value: NumberLike) -> float:
    """Convert a Julian Day to a Unix timestamp.

    Parameters
    ----------
    value : float or str
        The Julian Day.

    Returns
    -------
    float
        Seconds since the Unix epoch.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.
    """
    return (float(value) - JD_AT_UNIX_EPOCH) * SECONDS_PER_DAY


@lru_cache(maxsize=_CACHE_SIZE)
def mjd_to_jd(value: NumberLike) -> float:
    """Convert a Modified Julian Day to a Julian Day.

    Parameters
    ----------
    value : float or str
        The Modified Julian Day.

    Returns
    -------
    float
        The equivalent Julian Day (``MJD + 2400000.5``).

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.

    Examples
    --------
    >>> mjd_to_jd(51544.5)
    2451545.0
    """
    return float(value) + JD_MINUS_MJD


@lru_cache(maxsize=_CACHE_SIZE)
def jd_to_mjd(value: NumberLike) -> float:
    """Convert a Julian Day to a Modified Julian Day.

    Parameters
    ----------
    value : float or str
        The Julian Day.

    Returns
    -------
    float
        The equivalent Modified Julian Day (``JD - 2400000.5``).

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float.
    """
    return float(value) - JD_MINUS_MJD


@lru_cache(maxsize=_CACHE_SIZE)
def datetime_to_unix(value: DatetimeLike) -> float:
    """Convert a datetime to a Unix timestamp.

    Parameters
    ----------
    value : datetime or str
        The datetime (naive datetimes are assumed UTC), or an ISO 8601 string.

    Returns
    -------
    float
        Seconds since the Unix epoch.

    Raises
    ------
    ValueError
        If ``value`` is a string that is not valid ISO 8601.
    """
    return ensure_utc(value).timestamp()


@lru_cache(maxsize=_CACHE_SIZE)
def datetime_to_mjd(value: DatetimeLike) -> float:
    """Convert a datetime to a Modified Julian Day.

    Parameters
    ----------
    value : datetime or str
        The datetime (naive datetimes are assumed UTC), or an ISO 8601 string.

    Returns
    -------
    float
        The equivalent Modified Julian Day.

    Raises
    ------
    ValueError
        If ``value`` is a string that is not valid ISO 8601.
    """
    return unix_to_mjd(datetime_to_unix(value))


@lru_cache(maxsize=_CACHE_SIZE)
def datetime_to_jd(value: DatetimeLike) -> float:
    """Convert a datetime to a Julian Day.

    Parameters
    ----------
    value : datetime or str
        The datetime (naive datetimes are assumed UTC), or an ISO 8601 string.

    Returns
    -------
    float
        The equivalent Julian Day.

    Raises
    ------
    ValueError
        If ``value`` is a string that is not valid ISO 8601.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> datetime_to_jd(datetime(2000, 1, 1, 12, tzinfo=UTC))
    2451545.0
    """
    return unix_to_jd(datetime_to_unix(value))


@lru_cache(maxsize=_CACHE_SIZE)
def unix_to_datetime(value: NumberLike) -> datetime:
    """Convert a Unix timestamp to a UTC datetime.

    Parameters
    ----------
    value : float or str
        Seconds since the Unix epoch.

    Returns
    -------
    datetime
        The equivalent timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float, is NaN,
        or names an instant outside years 1 to 9999.
    OverflowError
        If ``value`` is too large in size for the system's time functions,
        infinity included.
    """
    return datetime.fromtimestamp(float(value), tz=UTC)


@lru_cache(maxsize=_CACHE_SIZE)
def mjd_to_datetime(value: NumberLike) -> datetime:
    """Convert a Modified Julian Day to a UTC datetime.

    Parameters
    ----------
    value : float or str
        The Modified Julian Day.

    Returns
    -------
    datetime
        The equivalent timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float, is NaN,
        or names an instant outside years 1 to 9999.
    OverflowError
        If ``value`` is too large in size for the system's time functions,
        infinity included.
    """
    return unix_to_datetime(mjd_to_unix(value))


@lru_cache(maxsize=_CACHE_SIZE)
def jd_to_datetime(value: NumberLike) -> datetime:
    """Convert a Julian Day to a UTC datetime.

    Parameters
    ----------
    value : float or str
        The Julian Day.

    Returns
    -------
    datetime
        The equivalent timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        If ``value`` is a string that cannot be parsed as a float, is NaN,
        or names an instant outside years 1 to 9999.
    OverflowError
        If ``value`` is too large in size for the system's time functions,
        infinity included.
    """
    return unix_to_datetime(jd_to_unix(value))
