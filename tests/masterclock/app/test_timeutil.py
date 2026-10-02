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
RANGE_START: Final = datetime(1858, 11, 17, tzinfo=UTC)
RANGE_END: Final = datetime(2199, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)
UNIX_RANGE_START: Final = (
    RANGE_START - datetime(1970, 1, 1, tzinfo=UTC)
).total_seconds()
UNIX_RANGE_END: Final = (RANGE_END - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
MJD_RANGE_END: Final = UNIX_RANGE_END / 86_400 + 40_587
JD_RANGE_START: Final = 2_400_000.5
JD_RANGE_END: Final = MJD_RANGE_END + JD_RANGE_START
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
TO_DATETIME_FUNCTIONS: Final = (
    timeutil.unix_to_datetime,
    timeutil.mjd_to_datetime,
    timeutil.jd_to_datetime,
)

instants: Final = st.datetimes(
    min_value=RANGE_START.replace(tzinfo=None),
    max_value=RANGE_END.replace(tzinfo=None),
    timezones=st.just(UTC),
)
naive_instants: Final = st.datetimes(
    min_value=RANGE_START.replace(tzinfo=None) + timedelta(days=1),
    max_value=RANGE_END.replace(tzinfo=None) - timedelta(days=1),
)
zone_offsets: Final = st.builds(
    timezone,
    st.integers(min_value=-23 * 60 - 59, max_value=23 * 60 + 59).map(
        lambda minutes: timedelta(minutes=minutes)
    ),
)
# Kept a day inside the range, so no offset can move them outside it.
zoned_instants: Final = st.builds(
    lambda moment, zone: moment.replace(tzinfo=zone), naive_instants, zone_offsets
)
unix_seconds: Final = st.floats(min_value=UNIX_RANGE_START, max_value=UNIX_RANGE_END)
mjds: Final = st.floats(min_value=0.0, max_value=MJD_RANGE_END)
jds: Final = st.floats(min_value=JD_RANGE_START, max_value=JD_RANGE_END)


def exact_unix(instant: datetime) -> Fraction:
    """Return the exact seconds from the Unix epoch to ``instant``."""
    since_unix_epoch = instant - datetime(1970, 1, 1, tzinfo=UTC)
    return Fraction(since_unix_epoch // timedelta(microseconds=1)) * MICROSECOND


def exact_mjd(instant: datetime) -> Fraction:
    """Return the exact Modified Julian Day of ``instant``."""
    return exact_unix(instant) / DAY_SECONDS + 40_587


def within(converted: float, exact_value: Fraction, tolerance: Fraction) -> bool:
    """Say whether ``converted`` is within ``tolerance`` of ``exact_value``."""
    return abs(Fraction(converted) - exact_value) <= tolerance


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
    utc_datetime = timeutil.ensure_utc(datetime(2031, 4, 9, 17, 3, 11, 25))
    assert utc_datetime == datetime(2031, 4, 9, 17, 3, 11, 25, tzinfo=UTC)
    assert utc_datetime.tzinfo is UTC


@contextmanager
def local_time(time_zone: str) -> Iterator[None]:
    """Set the local time zone to ``time_zone`` in the block, and start caches empty.

    A test run on a machine whose local time is UTC cannot tell reading a
    naive datetime as UTC from reading it as local time, so the tests that
    depend on the difference set the local zone themselves.
    """
    try:
        # patch.dict puts the environment back as it was, TZ included.
        with mock.patch.dict(os.environ, {"TZ": time_zone}):
            time.tzset()
            for cached_function in NUMBER_FUNCTIONS + DATETIME_FUNCTIONS:
                cached_function.cache_clear()
            yield
    finally:
        time.tzset()


@pytest.mark.parametrize("time_zone", ["UTC0", "XST+09:30", "YST-13"])
def test_naive_datetimes_are_utc_whatever_the_local_time(time_zone: str) -> None:
    """Read a naive datetime as UTC, not local time, in any local zone."""
    naive_datetime = datetime(2031, 4, 9, 17, 3, 11, 25)
    aware_datetime = naive_datetime.replace(tzinfo=UTC)
    with local_time(time_zone):
        assert timeutil.ensure_utc(naive_datetime) == aware_datetime
        assert timeutil.ensure_utc(naive_datetime.isoformat()) == aware_datetime
        assert timeutil.datetime_to_unix(naive_datetime) == aware_datetime.timestamp()
        assert timeutil.datetime_to_mjd(naive_datetime) == timeutil.datetime_to_mjd(
            aware_datetime
        )
        assert timeutil.datetime_to_jd(naive_datetime) == timeutil.datetime_to_jd(
            aware_datetime
        )
        assert timeutil.unix_to_datetime(aware_datetime.timestamp()) == aware_datetime


def test_an_aware_datetime_is_converted_to_utc() -> None:
    """Convert an aware datetime in another zone to the same instant in UTC."""
    time_zone = timezone(timedelta(hours=-7, minutes=-30))
    utc_datetime = timeutil.ensure_utc(datetime(2031, 4, 9, 17, 3, tzinfo=time_zone))
    assert utc_datetime == datetime(2031, 4, 10, 0, 33, tzinfo=UTC)
    assert utc_datetime.tzinfo is UTC


@pytest.mark.parametrize(
    ("iso_text", "expected_datetime"),
    [
        ("2031-04-09T17:03:11", datetime(2031, 4, 9, 17, 3, 11, tzinfo=UTC)),
        ("2031-04-09 17:03:11+01:00", datetime(2031, 4, 9, 16, 3, 11, tzinfo=UTC)),
        ("2031-04-09T17:03:11Z", datetime(2031, 4, 9, 17, 3, 11, tzinfo=UTC)),
        ("2031-04-09", datetime(2031, 4, 9, tzinfo=UTC)),
    ],
)
def test_an_iso_8601_string_is_read_as_a_datetime(
    iso_text: str, expected_datetime: datetime
) -> None:
    """Read an ISO 8601 string, naive meaning UTC, into a UTC datetime."""
    utc_datetime = timeutil.ensure_utc(iso_text)
    assert utc_datetime == expected_datetime
    assert utc_datetime.tzinfo is UTC


@given(zoned_instants)
def test_ensure_utc_keeps_the_instant(zoned_instant: datetime) -> None:
    """Return the same instant as given, in UTC, whatever the zone."""
    utc_datetime = timeutil.ensure_utc(zoned_instant)
    assert utc_datetime == zoned_instant
    assert utc_datetime.tzinfo is UTC


@pytest.mark.parametrize("conversion", DATETIME_FUNCTIONS)
@pytest.mark.parametrize(
    "bad_text", ["not a date", "2016-12-31T23:59:60+00:00", "2031-02-30T00:00:00"]
)
def test_a_string_that_is_not_a_datetime_is_refused(
    conversion: Callable[[str], object], bad_text: str
) -> None:
    """Raise ValueError for bad ISO 8601, a leap second or a day that isn't."""
    with pytest.raises(ValueError, match=r"."):
        conversion(bad_text)


@pytest.mark.parametrize("conversion", NUMBER_FUNCTIONS)
def test_a_string_that_is_not_a_number_is_refused(
    conversion: Callable[[str], object],
) -> None:
    """Raise ValueError for a string float() cannot read."""
    with pytest.raises(ValueError, match="could not convert"):
        conversion("forty")


@pytest.mark.parametrize(
    ("conversion", "given_number", "expected_number"),
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
    conversion: Callable[[float], float], given_number: float, expected_number: float
) -> None:
    """Convert values whose answer is known and exact in floats."""
    assert conversion(given_number) == expected_number


@pytest.mark.parametrize(
    ("conversion", "given_number"),
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
    conversion: Callable[[float | str], object], given_number: float
) -> None:
    """Give the same answer for a number and for its text."""
    assert conversion(repr(given_number)) == conversion(given_number)
    assert conversion(f"  {given_number}  ") == conversion(given_number)


@given(unix_seconds)
def test_unix_to_mjd_and_jd_agree_with_exact_arithmetic(unix_time: float) -> None:
    """Match the exact MJD and JD of a Unix timestamp within float spacing."""
    exact_mjd_value = Fraction(unix_time) / DAY_SECONDS + 40_587
    assert within(timeutil.unix_to_mjd(unix_time), exact_mjd_value, MJD_TOLERANCE)
    assert within(
        timeutil.unix_to_jd(unix_time), exact_mjd_value + JD_OFFSET, JD_TOLERANCE
    )


@given(mjds)
def test_mjd_to_unix_and_jd_agree_with_exact_arithmetic(mjd: float) -> None:
    """Match the exact Unix timestamp and JD of an MJD within float spacing."""
    exact_unix_value = (Fraction(mjd) - 40_587) * DAY_SECONDS
    assert within(timeutil.mjd_to_unix(mjd), exact_unix_value, UNIX_TOLERANCE)
    assert within(timeutil.mjd_to_jd(mjd), Fraction(mjd) + JD_OFFSET, JD_TOLERANCE)


@given(jds)
def test_jd_to_unix_and_mjd_agree_with_exact_arithmetic(jd: float) -> None:
    """Match the exact Unix timestamp and MJD of a JD within float spacing."""
    exact_mjd_value = Fraction(jd) - JD_OFFSET
    assert within(
        timeutil.jd_to_unix(jd),
        (exact_mjd_value - 40_587) * DAY_SECONDS,
        UNIX_TOLERANCE,
    )
    assert within(timeutil.jd_to_mjd(jd), exact_mjd_value, MJD_TOLERANCE)


@given(instants)
def test_datetime_conversions_agree_with_exact_arithmetic(instant: datetime) -> None:
    """Match the exact Unix timestamp, MJD and JD of a datetime."""
    exact_mjd_value = exact_mjd(instant)
    assert within(
        timeutil.datetime_to_unix(instant), exact_unix(instant), UNIX_TOLERANCE
    )
    assert within(timeutil.datetime_to_mjd(instant), exact_mjd_value, MJD_TOLERANCE)
    assert within(
        timeutil.datetime_to_jd(instant), exact_mjd_value + JD_OFFSET, JD_TOLERANCE
    )


@given(instants)
def test_a_naive_datetime_converts_as_utc(instant: datetime) -> None:
    """Convert a naive datetime as though it carried UTC."""
    naive_datetime = instant.replace(tzinfo=None)
    assert timeutil.datetime_to_unix(naive_datetime) == timeutil.datetime_to_unix(
        instant
    )
    assert timeutil.datetime_to_mjd(naive_datetime) == timeutil.datetime_to_mjd(instant)
    assert timeutil.datetime_to_jd(naive_datetime) == timeutil.datetime_to_jd(instant)


@given(zoned_instants)
def test_a_datetime_in_another_zone_converts_as_its_instant(
    zoned_instant: datetime,
) -> None:
    """Convert an aware datetime by the instant it names, not its wall clock."""
    assert within(
        timeutil.datetime_to_mjd(zoned_instant),
        exact_mjd(zoned_instant),
        MJD_TOLERANCE,
    )


@given(instants)
def test_a_datetime_survives_the_round_trip_through_unix_time(
    instant: datetime,
) -> None:
    """Come back to the same microsecond from a datetime by way of Unix time."""
    round_trip_datetime = timeutil.unix_to_datetime(timeutil.datetime_to_unix(instant))
    assert round_trip_datetime == instant
    assert round_trip_datetime.tzinfo is UTC


@given(instants)
def test_a_datetime_survives_the_round_trip_through_mjd_and_jd(
    instant: datetime,
) -> None:
    """Come back to the same instant within float spacing by way of MJD or JD."""
    by_mjd = timeutil.mjd_to_datetime(timeutil.datetime_to_mjd(instant))
    by_jd = timeutil.jd_to_datetime(timeutil.datetime_to_jd(instant))
    assert abs(by_mjd - instant) <= timedelta(microseconds=2)
    assert abs(by_jd - instant) <= timedelta(microseconds=25)
    assert by_mjd.tzinfo is UTC
    assert by_jd.tzinfo is UTC


@pytest.mark.parametrize(
    ("conversion", "epoch_number"),
    [
        (timeutil.unix_to_datetime, 0.0),
        (timeutil.mjd_to_datetime, 40_587.0),
        (timeutil.jd_to_datetime, 2_440_587.5),
    ],
)
def test_the_unix_epoch_converts_to_a_utc_datetime(
    conversion: Callable[[float], datetime], epoch_number: float
) -> None:
    """Give midnight on 1970-01-01, in UTC, for the Unix epoch."""
    utc_datetime = conversion(epoch_number)
    assert utc_datetime == datetime(1970, 1, 1, tzinfo=UTC)
    assert utc_datetime.tzinfo is UTC


@pytest.mark.parametrize(
    ("conversion", "first_number", "last_number"),
    [
        (timeutil.unix_to_datetime, -62_135_596_800.0, 253_402_300_799.0),
        (timeutil.mjd_to_datetime, -678_575.0, 2_973_483.9999),
        (timeutil.jd_to_datetime, 1_721_425.5, 5_373_484.4999),
    ],
)
def test_years_1_to_9999_are_the_range_of_datetimes(
    conversion: Callable[[float], datetime], first_number: float, last_number: float
) -> None:
    """Accept numbers inside years 1 to 9999 and refuse the ones outside."""
    assert conversion(first_number) == datetime(1, 1, 1, tzinfo=UTC)
    assert conversion(last_number).year == 9999
    with pytest.raises(ValueError, match="not 0"):
        conversion(first_number - 1)
    with pytest.raises(ValueError, match="not 10000"):
        conversion(last_number + 1)


@pytest.mark.parametrize("conversion", TO_DATETIME_FUNCTIONS)
@pytest.mark.parametrize(
    ("number_text", "expected_error", "error_text"),
    [
        ("nan", ValueError, "NaN"),
        ("inf", OverflowError, "out of range"),
        ("-inf", OverflowError, "out of range"),
    ],
)
def test_a_number_that_names_no_instant_is_refused(
    conversion: Callable[[str], datetime],
    number_text: str,
    expected_error: type[Exception],
    error_text: str,
) -> None:
    """Raise ValueError for NaN and OverflowError for infinity."""
    with pytest.raises(expected_error, match=error_text):
        conversion(number_text)


def test_every_function_is_cached() -> None:
    """Wrap every function the module defines in an LRU cache of the set size."""
    # Names with double underscores are left out: Python itself adds
    # __annotate__ to a module whose names carry annotations.
    module_functions = {
        function_name: module_value
        for function_name, module_value in vars(timeutil).items()
        if not function_name.startswith("__")
        and inspect.isfunction(inspect.unwrap(module_value))
        and inspect.unwrap(module_value).__module__ == timeutil.__name__
    }
    assert "ensure_utc" in module_functions
    assert "jd_to_datetime" in module_functions
    expected_cache_settings = {"maxsize": timeutil._CACHE_SIZE, "typed": False}
    cache_settings: dict[str, object] = {
        function_name: getattr(cached_function, "cache_parameters", dict)()
        for function_name, cached_function in module_functions.items()
    }
    assert cache_settings == dict.fromkeys(module_functions, expected_cache_settings)
