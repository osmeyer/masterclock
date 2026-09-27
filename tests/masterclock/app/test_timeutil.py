"""Tests for src/masterclock/app/timeutil.py.

The rules covered: every conversion between datetimes, Unix timestamps,
Modified Julian Days and Julian Days agrees with exact rational arithmetic to
within the spacing of the floats involved; every datetime is taken as UTC,
naive ones included, and every datetime returned is in UTC; and every
function is cached.
"""

import inspect
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from typing import Final
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import timeutil

# The range the property tests draw from: from MJD zero to the end of 2199.
# The tolerances below are worked out from the size of the numbers in it.
FIRST: Final = datetime(1858, 11, 17, tzinfo=UTC)
LAST: Final = datetime(2199, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)
UNIX_FIRST: Final = (FIRST - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
UNIX_LAST: Final = (LAST - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
MJD_LAST: Final = UNIX_LAST / 86_400 + 40_587
JD_FIRST: Final = 2_400_000.5
JD_LAST: Final = MJD_LAST + JD_FIRST
# The same offset as JD_FIRST, held exactly for the rational arithmetic.
JD_OFFSET: Final = Fraction(4_800_001, 2)

MICROSECOND: Final = Fraction(1, 10**6)
DAY_SECONDS: Final = 86_400

# In this range adjacent floats are under a microsecond apart as seconds
# since 1970 or as an MJD, and about 40 microseconds apart as a JD. Each
# conversion rounds a few times, so each tolerance allows a little more than
# the spacing it depends on.
UNIX_TOLERANCE: Final = 2 * MICROSECOND
MJD_TOLERANCE: Final = 2 * MICROSECOND / DAY_SECONDS
JD_TOLERANCE: Final = 25 * MICROSECOND / DAY_SECONDS

NUMBER_FUNCTIONS: Final = (
    timeutil.unix_to_mjd,
    timeutil.mjd_to_unix,
    timeutil.unix_to_jd,
    timeutil.jd_to_unix,
    timeutil.mjd_to_jd,
    timeutil.jd_to_mjd,
    timeutil.unix_to_datetime,
    timeutil.mjd_to_datetime,
    timeutil.jd_to_datetime,
)
DATETIME_FUNCTIONS: Final = (
    timeutil.ensure_utc,
    timeutil.datetime_to_unix,
    timeutil.datetime_to_mjd,
    timeutil.datetime_to_jd,
)
TO_DATETIME: Final = (
    timeutil.unix_to_datetime,
    timeutil.mjd_to_datetime,
    timeutil.jd_to_datetime,
)

instants: Final = st.datetimes(
    min_value=FIRST.replace(tzinfo=None),
    max_value=LAST.replace(tzinfo=None),
    timezones=st.just(UTC),
)
naive_instants: Final = st.datetimes(
    min_value=FIRST.replace(tzinfo=None) + timedelta(days=1),
    max_value=LAST.replace(tzinfo=None) - timedelta(days=1),
)
offsets: Final = st.builds(
    timezone,
    st.integers(min_value=-23 * 60 - 59, max_value=23 * 60 + 59).map(
        lambda minutes: timedelta(minutes=minutes)
    ),
)
# Kept a day inside the range, so no offset can move them outside it.
zoned_instants: Final = st.builds(
    lambda moment, zone: moment.replace(tzinfo=zone), naive_instants, offsets
)
unix_seconds: Final = st.floats(min_value=UNIX_FIRST, max_value=UNIX_LAST)
mjds: Final = st.floats(min_value=0.0, max_value=MJD_LAST)
jds: Final = st.floats(min_value=JD_FIRST, max_value=JD_LAST)


def exact_unix(moment: datetime) -> Fraction:
    """Return the exact seconds from the Unix epoch to ``moment``."""
    since = moment - datetime(1970, 1, 1, tzinfo=UTC)
    return Fraction(since // timedelta(microseconds=1)) * MICROSECOND


def exact_mjd(moment: datetime) -> Fraction:
    """Return the exact Modified Julian Day of ``moment``."""
    return exact_unix(moment) / DAY_SECONDS + 40_587


def within(value: float, exact: Fraction, tolerance: Fraction) -> bool:
    """Say whether ``value`` is no further than ``tolerance`` from ``exact``."""
    return abs(Fraction(value) - exact) <= tolerance


def test_the_constants_agree_with_the_calendar() -> None:
    """Place MJD zero, the Unix epoch and J2000 where the calendar puts them."""
    unix_epoch = datetime(1970, 1, 1, tzinfo=UTC)
    assert timeutil.MJD_ORIGIN.tzinfo is UTC
    assert timeutil.MJD_ORIGIN + timedelta(timeutil.MJD_AT_UNIX_EPOCH) == unix_epoch
    assert timeutil.JD_AT_UNIX_EPOCH - timeutil.JD_MINUS_MJD == (
        timeutil.MJD_AT_UNIX_EPOCH
    )
    assert timedelta(days=1).total_seconds() == timeutil.SECONDS_PER_DAY
    j2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)
    assert timeutil.datetime_to_jd(j2000) == 2_451_545.0
    assert timeutil.datetime_to_mjd(timeutil.MJD_ORIGIN) == 0.0


def test_a_naive_datetime_is_taken_as_utc() -> None:
    """Tag a naive datetime as UTC without moving it."""
    result = timeutil.ensure_utc(datetime(2031, 4, 9, 17, 3, 11, 25))
    assert result == datetime(2031, 4, 9, 17, 3, 11, 25, tzinfo=UTC)
    assert result.tzinfo is UTC


@contextmanager
def local_time(zone: str) -> Iterator[None]:
    """Set the local time zone to ``zone`` inside the block, and start caches empty.

    A test run on a machine whose local time is UTC cannot tell reading a
    naive datetime as UTC from reading it as local time, so the tests that
    depend on the difference set the local zone themselves.
    """
    try:
        # patch.dict puts the environment back as it was, TZ included.
        with mock.patch.dict(os.environ, {"TZ": zone}):
            time.tzset()
            for function in NUMBER_FUNCTIONS + DATETIME_FUNCTIONS:
                function.cache_clear()
            yield
    finally:
        time.tzset()


@pytest.mark.parametrize("zone", ["UTC0", "XST+09:30", "YST-13"])
def test_naive_datetimes_are_utc_whatever_the_local_time(zone: str) -> None:
    """Read a naive datetime as UTC, not local time, in any local zone."""
    naive = datetime(2031, 4, 9, 17, 3, 11, 25)
    aware = naive.replace(tzinfo=UTC)
    with local_time(zone):
        assert timeutil.ensure_utc(naive) == aware
        assert timeutil.ensure_utc(naive.isoformat()) == aware
        assert timeutil.datetime_to_unix(naive) == aware.timestamp()
        assert timeutil.datetime_to_mjd(naive) == timeutil.datetime_to_mjd(aware)
        assert timeutil.datetime_to_jd(naive) == timeutil.datetime_to_jd(aware)
        assert timeutil.unix_to_datetime(aware.timestamp()) == aware


def test_an_aware_datetime_is_converted_to_utc() -> None:
    """Convert an aware datetime in another zone to the same instant in UTC."""
    zone = timezone(timedelta(hours=-7, minutes=-30))
    result = timeutil.ensure_utc(datetime(2031, 4, 9, 17, 3, tzinfo=zone))
    assert result == datetime(2031, 4, 10, 0, 33, tzinfo=UTC)
    assert result.tzinfo is UTC


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2031-04-09T17:03:11", datetime(2031, 4, 9, 17, 3, 11, tzinfo=UTC)),
        ("2031-04-09 17:03:11+01:00", datetime(2031, 4, 9, 16, 3, 11, tzinfo=UTC)),
        ("2031-04-09T17:03:11Z", datetime(2031, 4, 9, 17, 3, 11, tzinfo=UTC)),
        ("2031-04-09", datetime(2031, 4, 9, tzinfo=UTC)),
    ],
)
def test_an_iso_8601_string_is_read_as_a_datetime(
    text: str, expected: datetime
) -> None:
    """Read an ISO 8601 string, naive meaning UTC, into a UTC datetime."""
    result = timeutil.ensure_utc(text)
    assert result == expected
    assert result.tzinfo is UTC


@given(zoned_instants)
def test_ensure_utc_keeps_the_instant(moment: datetime) -> None:
    """Return the same instant as given, in UTC, whatever the zone."""
    result = timeutil.ensure_utc(moment)
    assert result == moment
    assert result.tzinfo is UTC


@pytest.mark.parametrize("function", DATETIME_FUNCTIONS)
@pytest.mark.parametrize(
    "text", ["not a date", "2016-12-31T23:59:60+00:00", "2031-02-30T00:00:00"]
)
def test_a_string_that_is_not_a_datetime_is_refused(
    function: Callable[[str], object], text: str
) -> None:
    """Raise ValueError for bad ISO 8601, a leap second or a day that isn't."""
    with pytest.raises(ValueError, match=r"."):
        function(text)


@pytest.mark.parametrize("function", NUMBER_FUNCTIONS)
def test_a_string_that_is_not_a_number_is_refused(
    function: Callable[[str], object],
) -> None:
    """Raise ValueError for a string float() cannot read."""
    with pytest.raises(ValueError, match="could not convert"):
        function("forty")


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (timeutil.unix_to_mjd, 0.0, 40_587.0),
        (timeutil.unix_to_mjd, 86_400, 40_588.0),
        (timeutil.unix_to_mjd, -43_200.0, 40_586.5),
        (timeutil.mjd_to_unix, 40_588.0, 86_400.0),
        (timeutil.mjd_to_unix, 0.0, -3_506_716_800.0),
        (timeutil.unix_to_jd, 0.0, 2_440_587.5),
        (timeutil.unix_to_jd, 43_200.0, 2_440_588.0),
        (timeutil.jd_to_unix, 2_440_588.0, 43_200.0),
        (timeutil.mjd_to_jd, 51_544.5, 2_451_545.0),
        (timeutil.jd_to_mjd, 2_451_545.0, 51_544.5),
    ],
)
def test_known_values_convert_exactly(
    function: Callable[[float], float], value: float, expected: float
) -> None:
    """Convert values whose answer is known and exact in floats."""
    assert function(value) == expected


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (timeutil.unix_to_mjd, 1_234_567_890.25),
        (timeutil.mjd_to_unix, 61_234.125),
        (timeutil.unix_to_jd, -98_765.5),
        (timeutil.jd_to_unix, 2_461_234.625),
        (timeutil.mjd_to_jd, 61_234.125),
        (timeutil.jd_to_mjd, 2_461_234.625),
        (timeutil.unix_to_datetime, 1_234_567_890.25),
        (timeutil.mjd_to_datetime, 61_234.125),
        (timeutil.jd_to_datetime, 2_461_234.625),
    ],
)
def test_a_number_as_text_gives_what_the_number_gives(
    function: Callable[[float | str], object], value: float
) -> None:
    """Give the same answer for a number and for its text."""
    assert function(repr(value)) == function(value)
    assert function(f"  {value}  ") == function(value)


@given(unix_seconds)
def test_unix_to_mjd_and_jd_agree_with_exact_arithmetic(seconds: float) -> None:
    """Match the exact MJD and JD of a Unix timestamp within float spacing."""
    exact = Fraction(seconds) / DAY_SECONDS + 40_587
    assert within(timeutil.unix_to_mjd(seconds), exact, MJD_TOLERANCE)
    assert within(timeutil.unix_to_jd(seconds), exact + JD_OFFSET, JD_TOLERANCE)


@given(mjds)
def test_mjd_to_unix_and_jd_agree_with_exact_arithmetic(mjd: float) -> None:
    """Match the exact Unix timestamp and JD of an MJD within float spacing."""
    exact = (Fraction(mjd) - 40_587) * DAY_SECONDS
    assert within(timeutil.mjd_to_unix(mjd), exact, UNIX_TOLERANCE)
    assert within(timeutil.mjd_to_jd(mjd), Fraction(mjd) + JD_OFFSET, JD_TOLERANCE)


@given(jds)
def test_jd_to_unix_and_mjd_agree_with_exact_arithmetic(jd: float) -> None:
    """Match the exact Unix timestamp and MJD of a JD within float spacing."""
    mjd = Fraction(jd) - JD_OFFSET
    assert within(timeutil.jd_to_unix(jd), (mjd - 40_587) * DAY_SECONDS, UNIX_TOLERANCE)
    assert within(timeutil.jd_to_mjd(jd), mjd, MJD_TOLERANCE)


@given(instants)
def test_datetime_conversions_agree_with_exact_arithmetic(moment: datetime) -> None:
    """Match the exact Unix timestamp, MJD and JD of a datetime."""
    mjd = exact_mjd(moment)
    assert within(timeutil.datetime_to_unix(moment), exact_unix(moment), UNIX_TOLERANCE)
    assert within(timeutil.datetime_to_mjd(moment), mjd, MJD_TOLERANCE)
    assert within(timeutil.datetime_to_jd(moment), mjd + JD_OFFSET, JD_TOLERANCE)


@given(instants)
def test_a_naive_datetime_converts_as_utc(moment: datetime) -> None:
    """Convert a naive datetime as though it carried UTC."""
    naive = moment.replace(tzinfo=None)
    assert timeutil.datetime_to_unix(naive) == timeutil.datetime_to_unix(moment)
    assert timeutil.datetime_to_mjd(naive) == timeutil.datetime_to_mjd(moment)
    assert timeutil.datetime_to_jd(naive) == timeutil.datetime_to_jd(moment)


@given(zoned_instants)
def test_a_datetime_in_another_zone_converts_as_its_instant(
    moment: datetime,
) -> None:
    """Convert an aware datetime by the instant it names, not its wall clock."""
    assert within(
        timeutil.datetime_to_mjd(moment),
        exact_mjd(moment),
        MJD_TOLERANCE,
    )


@given(instants)
def test_a_datetime_survives_the_round_trip_through_unix_time(
    moment: datetime,
) -> None:
    """Come back to the same microsecond from a datetime by way of Unix time."""
    result = timeutil.unix_to_datetime(timeutil.datetime_to_unix(moment))
    assert result == moment
    assert result.tzinfo is UTC


@given(instants)
def test_a_datetime_survives_the_round_trip_through_mjd_and_jd(
    moment: datetime,
) -> None:
    """Come back to the same instant within float spacing by way of MJD or JD."""
    by_mjd = timeutil.mjd_to_datetime(timeutil.datetime_to_mjd(moment))
    by_jd = timeutil.jd_to_datetime(timeutil.datetime_to_jd(moment))
    assert abs(by_mjd - moment) <= timedelta(microseconds=2)
    assert abs(by_jd - moment) <= timedelta(microseconds=25)
    assert by_mjd.tzinfo is UTC
    assert by_jd.tzinfo is UTC


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (timeutil.unix_to_datetime, 0.0),
        (timeutil.mjd_to_datetime, 40_587.0),
        (timeutil.jd_to_datetime, 2_440_587.5),
    ],
)
def test_the_unix_epoch_converts_to_a_utc_datetime(
    function: Callable[[float], datetime], value: float
) -> None:
    """Give midnight on 1970-01-01, in UTC, for the Unix epoch."""
    result = function(value)
    assert result == datetime(1970, 1, 1, tzinfo=UTC)
    assert result.tzinfo is UTC


@pytest.mark.parametrize(
    ("function", "first", "last"),
    [
        (timeutil.unix_to_datetime, -62_135_596_800.0, 253_402_300_799.0),
        (timeutil.mjd_to_datetime, -678_575.0, 2_973_483.9999),
        (timeutil.jd_to_datetime, 1_721_425.5, 5_373_484.4999),
    ],
)
def test_years_1_to_9999_are_the_range_of_datetimes(
    function: Callable[[float], datetime], first: float, last: float
) -> None:
    """Accept numbers inside years 1 to 9999 and refuse the ones outside."""
    assert function(first) == datetime(1, 1, 1, tzinfo=UTC)
    assert function(last).year == 9999
    with pytest.raises(ValueError, match="not 0"):
        function(first - 1)
    with pytest.raises(ValueError, match="not 10000"):
        function(last + 1)


@pytest.mark.parametrize("function", TO_DATETIME)
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ("nan", ValueError, "NaN"),
        ("inf", OverflowError, "out of range"),
        ("-inf", OverflowError, "out of range"),
    ],
)
def test_a_number_that_names_no_instant_is_refused(
    function: Callable[[str], datetime],
    value: str,
    error: type[Exception],
    message: str,
) -> None:
    """Raise ValueError for NaN and OverflowError for infinity."""
    with pytest.raises(error, match=message):
        function(value)


def test_every_function_is_cached() -> None:
    """Wrap every function the module defines in an LRU cache of the set size."""
    # Names with double underscores are left out: Python itself adds
    # __annotate__ to a module whose names carry annotations.
    functions = {
        name: value
        for name, value in vars(timeutil).items()
        if not name.startswith("__")
        and inspect.isfunction(inspect.unwrap(value))
        and inspect.unwrap(value).__module__ == timeutil.__name__
    }
    assert "ensure_utc" in functions
    assert "jd_to_datetime" in functions
    expected = {"maxsize": timeutil._CACHE_SIZE, "typed": False}
    settings: dict[str, object] = {
        name: getattr(function, "cache_parameters", dict)()
        for name, function in functions.items()
    }
    assert settings == dict.fromkeys(functions, expected)
