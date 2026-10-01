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
offsets: Final = st.builds(
    timezone,
    st.integers(min_value=-23 * 60 - 59, max_value=23 * 60 + 59).map(
        lambda minutes: timedelta(minutes=minutes)
    ),
)
zoned_instants: Final = st.builds(
    lambda moment, zone: moment.replace(tzinfo=zone), naive_instants, offsets
)
ROUNDING: Final = (epochs.floor_to_ten_minutes, epochs.ceil_to_ten_minutes)


@contextmanager
def local_time(zone: str) -> Iterator[None]:
    """Set the local time zone to ``zone`` inside the block, and start caches empty.

    A test run on a machine whose local time is UTC cannot tell reading a
    naive datetime as UTC from reading it as local time, so the test that
    depends on the difference sets the local zone itself.
    """
    try:
        # patch.dict puts the environment back as it was, TZ included.
        with mock.patch.dict(os.environ, {"TZ": zone}):
            time.tzset()
            timeutil.ensure_utc.cache_clear()
            for function in ROUNDING:
                function.cache_clear()
            yield
    finally:
        time.tzset()


@pytest.mark.parametrize("zone", ["UTC0", "XST+09:30", "YST-13"])
def test_a_naive_datetime_is_rounded_as_utc_whatever_the_local_time(
    zone: str,
) -> None:
    """Round a naive datetime as UTC, not local time, in any local zone."""
    naive = datetime(2031, 4, 9, 17, 3, 11, 25)
    with local_time(zone):
        assert epochs.floor_to_ten_minutes(naive) == datetime(
            2031, 4, 9, 17, 0, tzinfo=UTC
        )
        assert epochs.ceil_to_ten_minutes(naive) == datetime(
            2031, 4, 9, 17, 10, tzinfo=UTC
        )


@pytest.mark.parametrize("function", ROUNDING)
@pytest.mark.parametrize(
    "text", ["not a date", "2016-12-31T23:59:60+00:00", "2031-02-30T00:00:00"]
)
def test_a_string_that_is_not_a_datetime_is_refused(
    function: Callable[[str], object], text: str
) -> None:
    """Raise ValueError for bad ISO 8601, a leap second or a day that isn't."""
    with pytest.raises(ValueError, match=r"."):
        function(text)


@pytest.mark.parametrize(
    ("moment", "down", "up"),
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
    moment: datetime | str, down: datetime, up: datetime
) -> None:
    """Round known datetimes down and up to the expected UTC marks."""
    assert epochs.floor_to_ten_minutes(moment) == down
    assert epochs.ceil_to_ten_minutes(moment) == up
    assert epochs.floor_to_ten_minutes(moment).tzinfo is UTC
    assert epochs.ceil_to_ten_minutes(moment).tzinfo is UTC


def is_mark(moment: datetime) -> bool:
    """Say whether ``moment`` is exactly on a ten-minute mark."""
    return moment.minute % 10 == 0 and moment.second == moment.microsecond == 0


@given(zoned_instants)
def test_floor_gives_the_last_mark_at_or_before(moment: datetime) -> None:
    """Give a UTC mark no later than the instant and under ten minutes before."""
    mark = epochs.floor_to_ten_minutes(moment)
    assert mark.tzinfo is UTC
    assert is_mark(mark)
    assert mark <= moment < mark + timedelta(minutes=10)
    assert epochs.floor_to_ten_minutes(mark) == mark


@given(zoned_instants)
def test_ceil_gives_the_first_mark_strictly_after(moment: datetime) -> None:
    """Give a UTC mark after the instant and no more than ten minutes later."""
    mark = epochs.ceil_to_ten_minutes(moment)
    assert mark.tzinfo is UTC
    assert is_mark(mark)
    assert moment < mark <= moment + timedelta(minutes=10)
    assert mark == epochs.floor_to_ten_minutes(moment) + timedelta(minutes=10)


def test_an_epoch_lasts_from_one_mark_to_the_next() -> None:
    """Make EPOCH_LENGTH the step between two ten-minute marks."""
    mark = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    assert timedelta(minutes=10) == epochs.EPOCH_LENGTH
    assert epochs.ceil_to_ten_minutes(mark) == mark + epochs.EPOCH_LENGTH


def test_ceil_past_the_last_datetime_is_refused() -> None:
    """Raise OverflowError when the next mark would be after year 9999."""
    with pytest.raises(OverflowError, match=r"."):
        epochs.ceil_to_ten_minutes(datetime(9999, 12, 31, 23, 55, tzinfo=UTC))


def test_format_epoch_renders_a_mark_and_its_mjd() -> None:
    """Write the mark to the second with its offset, and the MJD padded."""
    at = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    mjd = timeutil.datetime_to_mjd(at)
    assert epochs.format_epoch(at, mjd, 14, 6) == (
        "2031-04-09 17:10:00+00:00",
        "  62965.715278",
    )


@pytest.mark.parametrize(
    "zone", [timezone(timedelta(hours=2)), timezone(timedelta(hours=-9, minutes=-30))]
)
def test_format_epoch_writes_the_mark_in_utc_whatever_its_zone(
    zone: timezone,
) -> None:
    """Write a mark given in another zone, or naive, as the same UTC text."""
    at = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    expected = "2031-04-09 17:10:00+00:00"
    epochs.format_epoch.cache_clear()
    # The other zone goes first, so an answer kept from it would show.
    assert epochs.format_epoch(at.astimezone(zone), 1.0, 3, 1)[0] == expected
    assert epochs.format_epoch(at, 1.0, 3, 1)[0] == expected
    assert epochs.format_epoch(at.replace(tzinfo=None), 1.0, 3, 1)[0] == expected


def test_format_epoch_drops_a_fraction_of_a_second() -> None:
    """Drop, not round, any fraction of a second in the mark."""
    at = datetime(2031, 4, 9, 17, 10, 7, 999_999, tzinfo=UTC)
    assert epochs.format_epoch(at, 1.0, 3, 1)[0] == "2031-04-09 17:10:07+00:00"


@pytest.mark.parametrize(
    ("mjd", "width", "decimals", "expected"),
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
    mjd: float, width: int, decimals: int, expected: str
) -> None:
    """Round halves to even, right-justify, and never cut a wide MJD short."""
    at = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    assert epochs.format_epoch(at, mjd, width, decimals)[1] == expected


@pytest.mark.parametrize("zero", [-0.0, -0.0000001])
def test_format_epoch_writes_negative_zero_without_a_sign(zero: float) -> None:
    """Write an MJD that is or rounds to negative zero as plain zero."""
    at = datetime(2031, 4, 9, 17, 10, tzinfo=UTC)
    epochs.format_epoch.cache_clear()
    # The negative zero goes first, so an answer kept from it would show.
    assert epochs.format_epoch(at, zero, 9, 6)[1] == " 0.000000"
    assert epochs.format_epoch(at, 0.0, 9, 6)[1] == " 0.000000"


def test_every_function_is_cached() -> None:
    """Wrap every function the module defines in an LRU cache of the set size."""
    # Names with double underscores are left out: Python itself adds
    # __annotate__ to a module whose names carry annotations.
    functions = {
        name: value
        for name, value in vars(epochs).items()
        if not name.startswith("__")
        and inspect.isfunction(inspect.unwrap(value))
        and inspect.unwrap(value).__module__ == epochs.__name__
    }
    assert "floor_to_ten_minutes" in functions
    assert "format_epoch" in functions
    expected = {"maxsize": epochs._CACHE_SIZE, "typed": False}
    settings: dict[str, object] = {
        name: getattr(function, "cache_parameters", dict)()
        for name, function in functions.items()
    }
    assert settings == dict.fromkeys(functions, expected)
