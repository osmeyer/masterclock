"""Tests for src/masterclock/das_processor/read_steering.py.

The rules covered: a reference's steering file holds one event per line,
an MJD, a phase change and a rate change, in time order; read_steering
gives the events in (after, through], as SteerEvents; a missing file means
the reference has never been steered and gives no events; and a file that
cannot be read, a line that does not parse, a value in another form than
plain decimals or not finite, an MJD outside the data days, a line earlier
than the one before it, and a last line with no newline each raise
DataFileError.

An event lies from the first data day's start to the last day's end, and a
refusal is word for word and logged as raised.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_steering import (
    STEERING_FILE_TEMPLATE,
    read_steering,
)
from masterclock.domain.steering import SteerEvent

STEERING_LINES: Final = (
    "60941.250000 0.0 -0.00012\n"
    "60941.253125 1.5 0.0\n"
    "60941.256944 -2.25 3e-05\n"
    "60941.260000 0.0 +1.0E-6\n"
)
"""An invented steering file: four events on one day."""

WINDOW_START: Final = mjd_to_datetime(60941.25)
"""Exactly the first event's time."""

WINDOW_END: Final = mjd_to_datetime(60941.256944)
"""Exactly the third event's time."""


def steering_directory_with(
    tmp_path: Path, steering_text: str = STEERING_LINES, mc: str = "mc2"
) -> Path:
    """Write ``steering_text`` as ``mc``'s steering file and give its directory."""
    (tmp_path / STEERING_FILE_TEMPLATE.format(mc=mc)).write_text(
        steering_text, encoding="ascii"
    )
    return tmp_path


def test_events_after_and_through_are_given(tmp_path: Path) -> None:
    """Give the events in (after, through], in time order."""
    steer_events = read_steering(
        steering_directory_with(tmp_path), "mc2", WINDOW_START, WINDOW_END
    )
    assert steer_events == (
        SteerEvent(applied_datetime=mjd_to_datetime(60941.253125), dx=1.5, dy=0.0),
        SteerEvent(applied_datetime=WINDOW_END, dx=-2.25, dy=3e-05),
    )


def test_every_event_is_given_over_a_wide_span(tmp_path: Path) -> None:
    """Give all four events for a span that holds them all."""
    window_start = datetime(2025, 9, 23, tzinfo=UTC)
    window_end = datetime(2025, 9, 24, tzinfo=UTC)
    steer_events = read_steering(
        steering_directory_with(tmp_path), "mc2", window_start, window_end
    )
    assert [steer_event.dy for steer_event in steer_events] == [
        -0.00012,
        0.0,
        3e-05,
        1e-06,
    ]


def test_a_missing_file_gives_no_events(tmp_path: Path) -> None:
    """Give nothing for a reference that has never been steered."""
    assert read_steering(tmp_path, "mc4", WINDOW_START, WINDOW_END) == ()


def test_another_reference_s_file_is_not_read(tmp_path: Path) -> None:
    """Read only the named reference's file."""
    assert (
        read_steering(
            steering_directory_with(tmp_path, mc="mc1"), "mc2", WINDOW_START, WINDOW_END
        )
        == ()
    )


def test_events_at_the_same_instant_are_in_order(tmp_path: Path) -> None:
    """Take two events at one instant as in time order."""
    steering_text = "60941.253125 1.0 0.0\n60941.253125 2.0 0.0\n"
    steer_events = read_steering(
        steering_directory_with(tmp_path, steering_text),
        "mc2",
        WINDOW_START,
        WINDOW_END,
    )
    assert [steer_event.dx for steer_event in steer_events] == [1.0, 2.0]


@pytest.mark.parametrize(
    ("steering_text", "message_pattern"),
    [
        ("60941.253125 1.5\n", "line 1"),
        ("60941.253125 1.5 0.0 7\n", "line 1"),
        ("\n", "line 1"),
        ("60941.253125 1.5 nan\n", "line 1"),
        ("60941.253125 inf 0.0\n", "line 1"),
        ("60941.253125 1_5 0.0\n", "line 1"),
        ("60941 1.5 0.0\n", "line 1"),
        ("6.0941253125e4 1.5 0.0\n", "line 1"),
        ("60941.253125 0x1p3 0.0\n", "line 1"),
        ("40000.5 1.5 0.0\n", "line 1"),
        ("60941.253125 1.5 0.0\n60941.25 0.0 0.0\n", "line 2 .* earlier"),
        ("60941.253125 1.5 0.0\n60941.256944 0.0 0.0", "line 2 .* no newline"),
        ("60941.253125 1e400 0.0\n", "line 1"),
    ],
)
def test_a_damaged_file_is_refused(
    tmp_path: Path, steering_text: str, message_pattern: str
) -> None:
    """Raise DataFileError naming the line that cannot be used."""
    with pytest.raises(DataFileError, match=message_pattern):
        read_steering(
            steering_directory_with(tmp_path, steering_text),
            "mc2",
            WINDOW_START,
            WINDOW_END,
        )


def test_a_line_out_of_the_span_is_still_checked(tmp_path: Path) -> None:
    """Refuse a damaged line even when it lies outside the span asked for."""
    steering_text = STEERING_LINES + "60942.0 x 0.0\n"
    with pytest.raises(DataFileError, match="line 5"):
        read_steering(
            steering_directory_with(tmp_path, steering_text),
            "mc2",
            WINDOW_START,
            WINDOW_END,
        )


def test_a_file_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for a steering file that is not a readable file."""
    (tmp_path / STEERING_FILE_TEMPLATE.format(mc="mc2")).mkdir()
    with pytest.raises(DataFileError, match="cannot read"):
        read_steering(tmp_path, "mc2", WINDOW_START, WINDOW_END)


def test_a_file_that_is_not_ascii_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for bytes that are not ASCII."""
    steering_file = tmp_path / STEERING_FILE_TEMPLATE.format(mc="mc2")
    steering_file.write_bytes("60941.253125 1,5 0.0\n".replace(",", "·").encode())
    with pytest.raises(DataFileError, match="cannot read"):
        read_steering(tmp_path, "mc2", WINDOW_START, WINDOW_END)


def test_a_name_that_is_not_a_reference_s_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for the steering of a clock that is no reference."""
    with pytest.raises(DataFileError, match="ox23"):
        read_steering(tmp_path, "ox23", WINDOW_START, WINDOW_END)


@pytest.mark.parametrize(
    ("mjd", "taken"),
    [("50000.000000", True), ("100000.000000", False), ("99999.500000", True)],
)
def test_an_event_lies_on_a_data_day(tmp_path: Path, mjd: str, taken: bool) -> None:
    """Take an event from the first data day's start to the last day's end."""
    steering_directory = steering_directory_with(tmp_path, f"{mjd} 1.5 0.0\n")
    window_start = datetime(1990, 1, 1, tzinfo=UTC)
    window_end = datetime(2300, 1, 1, tzinfo=UTC)
    if taken:
        (steer_event,) = read_steering(
            steering_directory, "mc2", window_start, window_end
        )
        assert steer_event.dx == 1.5
    else:
        with pytest.raises(DataFileError) as refusal:
            read_steering(steering_directory, "mc2", window_start, window_end)
        steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
        assert str(refusal.value) == (
            f"steering file {steering_file}: line 1:"
            f" {mjd!r} is not an MJD on a data day"
        )


def test_a_refusal_is_logged_as_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each DataFileError at ERROR in the words it is raised with."""
    with pytest.raises(DataFileError) as refusal:
        read_steering(
            steering_directory_with(tmp_path, "\n"), "mc2", WINDOW_START, WINDOW_END
        )
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(refusal.value)
    ]
