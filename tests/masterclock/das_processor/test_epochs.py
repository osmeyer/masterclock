"""Tests for src/masterclock/das_processor/epochs.py.

The rules covered: an epoch lasts ten minutes; ten-minute rounding lands on
a mark in UTC, down or strictly up, whatever the zone of the datetime or the
local time; a mark is
rendered in UTC as fixed text, with its MJD rounded and padded; and every
function is cached.
"""

import inspect
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from typing import Final
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import timeutil
from masterclock.das_processor import epochs

# Kept a day inside the range of datetimes the property tests draw from, so
# no offset can move them outside it.
naive_instants: Final = st.datetimes(
    min_value=datetime(1858, 11, 18),
    max_value=datetime(2199, 12, 30, 23, 59, 59, 999_999),
)
utc_offsets: Final = st.builds(
    timezone,
    st.integers(min_value=-23 * 60 - 59, max_value=23 * 60 + 59).map(
        lambda minutes: timedelta(minutes=minutes)
    ),
)
zoned_instants: Final = st.builds(
    lambda naive_instant, utc_offset: naive_instant.replace(tzinfo=utc_offset),
    naive_instants,
    utc_offsets,
)
ROUNDING_FUNCTIONS: Final = (epochs.floor_to_ten_minutes, epochs.ceil_to_ten_minutes)


@contextmanager
def local_time(local_zone: str) -> Iterator[None]:
    """Set the local time zone to ``local_zone`` in the block, and start caches empty.

    A test run on a machine whose local time is UTC cannot tell reading a
    naive datetime as UTC from reading it as local time, so the test that
    depends on the difference sets the local zone itself.
    """
    try:
        # patch.dict puts the environment back as it was, TZ included.
        with mock.patch.dict(os.environ, {"TZ": local_zone}):
            time.tzset()
            timeutil.ensure_utc.cache_clear()
            for rounding_function in ROUNDING_FUNCTIONS:
                rounding_function.cache_clear()
            yield
    finally:
        time.tzset()


@pytest.mark.parametrize("local_zone", ["UTC0", "XST+09:30", "YST-13"])
def test_a_naive_datetime_is_rounded_as_utc_whatever_the_local_time(
    local_zone: str,
) -> None:
    """Round a naive datetime as UTC, not local time, in any local zone."""
    naive_datetime = datetime(2031, 4, 9, 17, 3, 11, 25)
    with local_time(local_zone):
        assert epochs.floor_to_ten_minutes(naive_datetime) == datetime(
            2031, 4, 9, 17, 0, tzinfo=UTC
        )
        assert epochs.ceil_to_ten_minutes(naive_datetime) == datetime(
            2031, 4, 9, 17, 10, tzinfo=UTC
        )


@pytest.mark.parametrize("rounding_function", ROUNDING_FUNCTIONS)
@pytest.mark.parametrize(
    "bad_text", ["not a date", "2016-12-31T23:59:60+00:00", "2031-02-30T00:00:00"]
)
def test_a_string_that_is_not_a_datetime_is_refused(
    rounding_function: Callable[[str], object], bad_text: str
) -> None:
    """Raise ValueError for bad ISO 8601, a leap second or a day that isn't."""
    with pytest.raises(ValueError, match=r"."):
        rounding_function(bad_text)


@pytest.mark.parametrize(
    ("given_instant", "floor_mark", "ceil_mark"),
    [
        (
            datetime(2031, 4, 9, 17, 3, 11, 25, tzinfo=UTC),
            datetime(2031, 4, 9, 17, 0, tzinfo=UTC),
            datetime(2031, 4, 9, 17, 10, tzinfo=UTC),
        ),
        (
            datetime(2031, 4, 9, 17, 50, tzinfo=UTC),
            datetime(2031, 4, 9, 17, 50, tzinfo=UTC),
            datetime(2031, 4, 9, 18, 0, tzinfo=UTC),
        ),
        (
            datetime(2031, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC),
            datetime(2031, 12, 31, 23, 50, tzinfo=UTC),
            datetime(2032, 1, 1, tzinfo=UTC),
        ),
        (
            datetime(2031, 4, 9, 17, 9, 59, 999_999),
            datetime(2031, 4, 9, 17, 0, tzinfo=UTC),
            datetime(2031, 4, 9, 17, 10, tzinfo=UTC),
        ),
        (
            "2031-04-09T22:50:00+05:45",
            datetime(2031, 4, 9, 17, 0, tzinfo=UTC),
            datetime(2031, 4, 9, 17, 10, tzinfo=UTC),
        ),
    ],
)
def test_rounding_to_ten_minutes_gives_known_marks(
    given_instant: datetime | str, floor_mark: datetime, ceil_mark: datetime
) -> None:
    """Round known datetimes down and up to the expected UTC marks."""
    assert epochs.floor_to_ten_minutes(given_instant) == floor_mark
    assert epochs.ceil_to_ten_minutes(given_instant) == ceil_mark
    assert epochs.floor_to_ten_minutes(given_instant).tzinfo is UTC
    assert epochs.ceil_to_ten_minutes(given_instant).tzinfo is UTC


def is_ten_minute_mark(instant: datetime) -> bool:
    """Say whether ``instant`` is exactly on a ten-minute mark."""
    return instant.minute % 10 == 0 and instant.second == instant.microsecond == 0


@given(zoned_instants)
def test_floor_gives_the_last_mark_at_or_before(instant: datetime) -> None:
    """Give a UTC mark no later than the instant and under ten minutes before."""
    floor_mark = epochs.floor_to_ten_minutes(instant)
    assert floor_mark.tzinfo is UTC
    assert is_ten_minute_mark(floor_mark)
    assert floor_mark <= instant < floor_mark + timedelta(minutes=10)
    assert epochs.floor_to_ten_minutes(floor_mark) == floor_mark


@given(zoned_instants)
def test_ceil_gives_the_first_mark_strictly_after(instant: datetime) -> None:
    """Give a UTC mark after the instant and no more than ten minutes later."""
    ceil_mark = epochs.ceil_to_ten_minutes(instant)
    assert ceil_mark.tzinfo is UTC
    assert is_ten_minute_mark(ceil_mark)
    assert instant < ceil_mark <= instant + timedelta(minutes=10)
    assert ceil_mark == epochs.floor_to_ten_minutes(instant) + timedelta(minutes=10)


def test_an_epoch_lasts_from_one_mark_to_the_next() -> None:
    """Make EPOCH_LENGTH the step between two ten-minute marks."""
    epoch_start = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    assert timedelta(minutes=10) == epochs.EPOCH_LENGTH
    assert epochs.ceil_to_ten_minutes(epoch_start) == epoch_start + epochs.EPOCH_LENGTH


def test_ceil_past_the_last_datetime_is_refused() -> None:
    """Raise OverflowError when the next mark would be after year 9999."""
    with pytest.raises(OverflowError, match=r"."):
        epochs.ceil_to_ten_minutes(datetime(9999, 12, 31, 23, 55, tzinfo=UTC))


def test_format_epoch_renders_a_mark_and_its_mjd() -> None:
    """Write the mark to the second with its offset, and the MJD padded."""
    epoch_start = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    mjd = timeutil.datetime_to_mjd(epoch_start)
    assert epochs.format_epoch(epoch_start, mjd, 14, 6) == (
        "2031-04-09 17:10:00+00:00",
        "  62965.715278",
    )


@pytest.mark.parametrize(
    "other_zone",
    [timezone(timedelta(hours=2)), timezone(timedelta(hours=-9, minutes=-30))],
)
def test_format_epoch_writes_the_mark_in_utc_whatever_its_zone(
    other_zone: timezone,
) -> None:
    """Write a mark given in another zone, or naive, as the same UTC text."""
    epoch_start = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    expected_text = "2031-04-09 17:10:00+00:00"
    epochs.format_epoch.cache_clear()
    # The other zone goes first, so an answer kept from it would show.
    assert (
        epochs.format_epoch(epoch_start.astimezone(other_zone), 1.0, 3, 1)[0]
        == expected_text
    )
    assert epochs.format_epoch(epoch_start, 1.0, 3, 1)[0] == expected_text
    assert (
        epochs.format_epoch(epoch_start.replace(tzinfo=None), 1.0, 3, 1)[0]
        == expected_text
    )


def test_format_epoch_drops_a_fraction_of_a_second() -> None:
    """Drop, not round, any fraction of a second in the mark."""
    epoch_start = datetime(2031, 4, 9, 17, 10, 7, 999_999, tzinfo=UTC)
    assert epochs.format_epoch(epoch_start, 1.0, 3, 1)[0] == "2031-04-09 17:10:07+00:00"


@pytest.mark.parametrize(
    ("mjd", "mjd_width", "mjd_decimals", "expected_text"),
    [
        (62_965.5, 10, 2, "  62965.50"),
        (62_965.5, 7, 0, "  62966"),
        (62_966.5, 7, 0, "  62966"),
        (62_965.25, 9, 1, "  62965.2"),
        (62_965.75, 9, 1, "  62965.8"),
        (62_965.715278, 4, 3, "62965.715"),
    ],
)
def test_format_epoch_rounds_and_pads_the_mjd(
    mjd: float, mjd_width: int, mjd_decimals: int, expected_text: str
) -> None:
    """Round halves to even, right-justify, and never cut a wide MJD short."""
    epoch_start = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    assert (
        epochs.format_epoch(epoch_start, mjd, mjd_width, mjd_decimals)[1]
        == expected_text
    )


@pytest.mark.parametrize("negative_zero", [-0.0, -0.0000001])
def test_format_epoch_writes_negative_zero_without_a_sign(negative_zero: float) -> None:
    """Write an MJD that is or rounds to negative zero as plain zero."""
    epoch_start = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    epochs.format_epoch.cache_clear()
    # The negative zero goes first, so an answer kept from it would show.
    assert epochs.format_epoch(epoch_start, negative_zero, 9, 6)[1] == " 0.000000"
    assert epochs.format_epoch(epoch_start, 0.0, 9, 6)[1] == " 0.000000"


def test_every_function_is_cached() -> None:
    """Wrap every function the module defines in an LRU cache of the set size."""
    # Names with double underscores are left out: Python itself adds
    # __annotate__ to a module whose names carry annotations.
    module_functions = {
        function_name: module_value
        for function_name, module_value in vars(epochs).items()
        if not function_name.startswith("__")
        and inspect.isfunction(inspect.unwrap(module_value))
        and inspect.unwrap(module_value).__module__ == epochs.__name__
    }
    assert "floor_to_ten_minutes" in module_functions
    assert "format_epoch" in module_functions
    expected_cache = {"maxsize": epochs._CACHE_SIZE, "typed": False}
    cache_settings: dict[str, object] = {
        function_name: getattr(cached_function, "cache_parameters", dict)()
        for function_name, cached_function in module_functions.items()
    }
    assert cache_settings == dict.fromkeys(module_functions, expected_cache)
