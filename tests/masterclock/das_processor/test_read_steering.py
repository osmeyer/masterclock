"""Tests for src/masterclock/das_processor/read_steering.py.

The rules covered: a reference's steering file holds one event per line,
an MJD, a phase change and a rate change, in time order; SteeringFiles
gives the events in (after, through], as SteerEvents; a missing file means
the reference has never been steered and gives no events; and a file that
cannot be read, a line that does not parse, a value in another form than
plain decimals or not finite, an MJD outside the data days, a line earlier
than the one before it, and a last line with no newline each raise
DataFileError.

Each line is read and checked once in a run: later reads take only the
lines appended since, numbered and ordered after the lines before them; a
file replaced or shorter than what was read is read again from its start,
one removed gives no events, one that appears is read; and a refused read
keeps nothing of it.

An event lies from the first data day's start to the last day's end, and a
refusal is word for word and logged as raised. Each event read is checked
by a pydantic model with a steering event's fields, in their order: an
instant with its timezone, finite changes, no unknown field.
"""

import dataclasses
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NoReturn

import pytest
from pydantic import ValidationError

from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor import read_steering as read_steering_module
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_steering import (
    STEERING_FILE_TEMPLATE,
    SteerEventFields,
    SteeringFiles,
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


def read_steering(
    steering_path: Path, mc: str, window_start: datetime, window_end: datetime
) -> tuple[SteerEvent, ...]:
    """Read ``mc``'s events in (window_start, window_end] afresh."""
    return SteeringFiles(steering_path).events(mc, window_start, window_end)


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
    """Raise DataFileError for a steering file that is not a regular file."""
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
    with pytest.raises(DataFileError, match="nav23"):
        read_steering(tmp_path, "nav23", WINDOW_START, WINDOW_END)


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


def test_the_event_model_has_an_event_s_fields_in_order() -> None:
    """Give the read model exactly SteerEvent's fields, so an event builds."""
    assert list(SteerEventFields.model_fields) == [
        event_field.name for event_field in dataclasses.fields(SteerEvent)
    ]


@pytest.mark.parametrize(
    ("event_field", "wrong_value"),
    [
        ("dx", float("nan")),
        ("dy", float("inf")),
        ("dx", "1"),
        ("applied_datetime", WINDOW_START.replace(tzinfo=None)),
        ("colour", "red"),
    ],
)
def test_an_event_read_is_checked(event_field: str, wrong_value: object) -> None:
    """Refuse a change not finite or not a number, a naive instant, an unknown field."""
    event_fields = {
        "applied_datetime": WINDOW_START,
        "dx": 0.0,
        "dy": 0.0,
        event_field: wrong_value,
    }
    with pytest.raises(ValidationError, match=event_field):
        SteerEventFields.model_validate(event_fields)


# ------------------------------------------------------ read once in a run


def counted_parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every steering line parsed."""
    parsed_lines: list[str] = []
    real_parse = read_steering_module._parse_steering_line

    def count_parse(line_place: str, line: str) -> SteerEvent:
        """Note the line, then parse it."""
        parsed_lines.append(line)
        return real_parse(line_place, line)

    monkeypatch.setattr(read_steering_module, "_parse_steering_line", count_parse)
    return parsed_lines


def test_each_line_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parse a file's lines once, however many windows are asked for."""
    steering_files = SteeringFiles(steering_directory_with(tmp_path))
    parsed_lines = counted_parses(monkeypatch)
    first_events = steering_files.events("mc2", WINDOW_START, WINDOW_END)
    assert steering_files.events("mc2", WINDOW_START, WINDOW_END) == first_events
    assert (
        len(steering_files.events("mc2", datetime(2025, 9, 23, tzinfo=UTC), WINDOW_END))
        == 3
    )
    assert parsed_lines == STEERING_LINES.splitlines()


def test_appended_lines_are_read_after_the_lines_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read only lines appended since, and give their events with the earlier ones."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    steering_files.events("mc2", WINDOW_START, WINDOW_END)
    parsed_lines = counted_parses(monkeypatch)
    steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
    with steering_file.open("a", encoding="ascii") as steering_stream:
        steering_stream.write("60941.270000 4.0 0.0\n")
    steer_events = steering_files.events("mc2", WINDOW_START, mjd_to_datetime(60941.27))
    assert [steer_event.dx for steer_event in steer_events] == [1.5, -2.25, 0.0, 4.0]
    assert parsed_lines == ["60941.270000 4.0 0.0"]


@pytest.mark.parametrize(
    ("appended_text", "message_pattern"),
    [
        ("60942.0 x 0.0\n", "line 5"),
        ("60941.255000 0.0 0.0\n", "line 5 .* earlier"),
        ("60941.270000 4.0 0.0", "line 5 .* no newline"),
        ("60941.270000 4.0 0.0\n60941.27 1·0 0.0\n", "cannot read"),
    ],
)
def test_an_appended_line_is_checked_as_a_line_of_the_file(
    tmp_path: Path, appended_text: str, message_pattern: str
) -> None:
    """Refuse a damaged appended line, named and ordered after the lines before it."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    steering_files.events("mc2", WINDOW_START, WINDOW_END)
    steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
    with steering_file.open("ab") as steering_stream:
        steering_stream.write(appended_text.encode())
    with pytest.raises(DataFileError, match=message_pattern):
        steering_files.events("mc2", WINDOW_START, WINDOW_END)


def test_a_refused_read_keeps_nothing_of_it(tmp_path: Path) -> None:
    """Keep the events read before a refused read, and read its lines again later."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    wide_start = datetime(2025, 9, 23, tzinfo=UTC)
    wide_end = datetime(2025, 9, 24, tzinfo=UTC)
    events_before = steering_files.events("mc2", wide_start, wide_end)
    steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
    with steering_file.open("a", encoding="ascii") as steering_stream:
        steering_stream.write("60941.270000 4.0 0.0\n60941.280000 5.0")
    with pytest.raises(DataFileError, match=r"line 6 .* no newline"):
        steering_files.events("mc2", wide_start, wide_end)
    with pytest.raises(DataFileError, match=r"line 6 .* no newline"):
        steering_files.events("mc2", wide_start, wide_end)
    with steering_file.open("a", encoding="ascii") as steering_stream:
        steering_stream.write(" 0.0\n")
    steer_events = steering_files.events("mc2", wide_start, wide_end)
    assert steer_events[:4] == events_before
    assert [steer_event.dx for steer_event in steer_events[4:]] == [4.0, 5.0]


def test_a_replaced_or_shorter_file_is_read_again(tmp_path: Path) -> None:
    """Read a file from its start when it is another file, or shorter than read."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    wide_start = datetime(2025, 9, 23, tzinfo=UTC)
    wide_end = datetime(2025, 9, 24, tzinfo=UTC)
    steering_files.events("mc2", wide_start, wide_end)
    steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
    steering_file.write_text("60941.253125 7.0 0.0\n", encoding="ascii")
    assert [
        steer_event.dx
        for steer_event in steering_files.events("mc2", wide_start, wide_end)
    ] == [7.0]
    replacement = tmp_path / "replacement.dat"
    replacement.write_text(
        "60941.250000 8.0 0.0\n60941.253125 9.0 0.0\n", encoding="ascii"
    )
    replacement.replace(steering_file)
    assert [
        steer_event.dx
        for steer_event in steering_files.events("mc2", wide_start, wide_end)
    ] == [8.0, 9.0]


def test_a_file_replaced_by_one_as_long_is_read_again(tmp_path: Path) -> None:
    """Read a replaced file from its start, though it is as long as what was read."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    wide_start = datetime(2025, 9, 23, tzinfo=UTC)
    wide_end = datetime(2025, 9, 24, tzinfo=UTC)
    steering_files.events("mc2", wide_start, wide_end)
    steering_file = steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")
    replacement = tmp_path / "replacement.dat"
    replacement.write_text(STEERING_LINES.replace("1.5", "6.5"), encoding="ascii")
    replacement.replace(steering_file)
    steer_events = steering_files.events("mc2", wide_start, wide_end)
    assert [steer_event.dx for steer_event in steer_events][:2] == [0.0, 6.5]


def test_a_file_removed_gives_no_events_and_one_that_appears_is_read(
    tmp_path: Path,
) -> None:
    """Give nothing once a file is gone, and read a file that appears later."""
    steering_directory = steering_directory_with(tmp_path)
    steering_files = SteeringFiles(steering_directory)
    assert steering_files.events("mc2", WINDOW_START, WINDOW_END)
    (steering_directory / STEERING_FILE_TEMPLATE.format(mc="mc2")).unlink()
    assert steering_files.events("mc2", WINDOW_START, WINDOW_END) == ()
    assert steering_files.events("mc3", WINDOW_START, WINDOW_END) == ()
    steering_directory_with(tmp_path, mc="mc3")
    assert len(steering_files.events("mc3", WINDOW_START, WINDOW_END)) == 2


def test_a_file_that_cannot_be_looked_at_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError when the file's status cannot be read, or it cannot open."""
    steering_directory = steering_directory_with(tmp_path)

    def refuse(*_: object, **__: object) -> NoReturn:
        """Fail as a file system refusing access would."""
        message = "refused"
        raise PermissionError(message)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "stat", refuse)
        with pytest.raises(DataFileError, match=r"cannot read .* refused"):
            read_steering(steering_directory, "mc2", WINDOW_START, WINDOW_END)
    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(DataFileError, match=r"cannot read .* refused"):
        read_steering(steering_directory, "mc2", WINDOW_START, WINDOW_END)


def test_a_steering_file_that_is_not_a_regular_file_is_refused(
    tmp_path: Path,
) -> None:
    """Raise DataFileError for a named pipe, which shows no length to read."""
    os.mkfifo(tmp_path / STEERING_FILE_TEMPLATE.format(mc="mc2"))
    with pytest.raises(DataFileError, match="not a regular file"):
        read_steering(tmp_path, "mc2", WINDOW_START, WINDOW_END)
