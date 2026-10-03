"""Tests for src/masterclock/domain/steering.py.

The rules covered: a steering event holds its changes of phase and rate
and the instant they were applied, and is frozen; a series is steered by every reference
in its effective difference, with the signs of design 7.1 and none for a
self pair; the input over the previous epoch counts the events in
(E - T, E], moved on to E; the steering inside an epoch counts the events
in (E, t], moved on to the measurement time t; every phase term is exact;
and an event inside an epoch, before the measurement, leaves the decycled
phase at E as it was, and enters the next epoch's input in full.

w adds up every event inside the epoch.
"""

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest
from gmpy2 import mpq

from masterclock.domain import phase, steering
from masterclock.domain.series import State

EPOCH_START: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented ten-minute mark, E."""

T: Final = timedelta(seconds=phase.EPOCH_SECONDS)
"""One epoch."""


def steer_event(
    applied_at: datetime, dx: float = 0.0, dy: float = 0.0
) -> steering.SteerEvent:
    """Build a steering event at ``applied_at``."""
    return steering.SteerEvent(applied_datetime=applied_at, dx=dx, dy=dy)


# ------------------------------------------------------------------- events


def test_an_event_holds_its_changes() -> None:
    """Keep the instant and both changes as given."""
    built_event = steer_event(EPOCH_START, 1.5, -0.00012)
    assert (built_event.applied_datetime, built_event.dx, built_event.dy) == (
        EPOCH_START,
        1.5,
        -0.00012,
    )


def test_an_event_is_frozen() -> None:
    """Refuse a change to a built event."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        steer_event(EPOCH_START).dx = 1.0  # type: ignore[misc]


# -------------------------------------------------------------------- signs


@pytest.mark.parametrize(
    ("series_key", "expected_signs"),
    [
        (("mc1", "mc1"), {}),
        (("mc1", "mc2"), {"mc1": 1, "mc2": -1}),
        (("mc2", "mc1"), {"mc2": 1, "mc1": -1}),
        (("mc2", "ox23"), {"mc2": 1}),
        (("mc1", "mc2", "ox23"), {"mc1": 1}),
        (("mc2", "mc2", "ox23"), {"mc2": 1}),
        (("mc2", "mcq"), {"mc2": 1}),
        (("mc12", "mc2"), {"mc2": -1}),
    ],
)
def test_each_series_is_steered_with_the_signs_of_its_difference(
    series_key: tuple[str, ...], expected_signs: dict[str, int]
) -> None:
    """Give the signs of design 7.1 for self, link and clock pairs and triples (U5)."""
    assert steering.signs(series_key) == expected_signs


# ----------------------------------------------------------------------- u


def test_u_moves_each_event_on_to_the_epoch_start() -> None:
    """Sum dx and dy times the time from the event to E, exactly."""
    applied_at = EPOCH_START - timedelta(seconds=100)
    steering_events = {"mc2": (steer_event(applied_at, dx=3.0, dy=0.25),)}
    ux, uy = steering.steer_u(("mc2", "ox23"), EPOCH_START, steering_events)
    assert ux == 3 + mpq(1, 4) * 100
    assert uy == 0.25


def test_u_takes_each_reference_with_its_sign() -> None:
    """Add the first reference's events and take away the second's."""
    steering_events = {
        "mc1": (steer_event(EPOCH_START, dx=5.0, dy=0.5),),
        "mc2": (steer_event(EPOCH_START, dx=2.0, dy=0.125),),
    }
    assert steering.steer_u(("mc1", "mc2"), EPOCH_START, steering_events) == (
        mpq(3),
        0.375,
    )
    assert steering.steer_u(("mc1", "mc1"), EPOCH_START, steering_events) == (
        mpq(0),
        0.0,
    )


def test_u_is_exact() -> None:
    """Hold 0.1 ps as the float it is, not as one tenth."""
    steering_events = {"mc1": (steer_event(EPOCH_START, dx=0.1),)}
    ux, _ = steering.steer_u(("mc1", "c"), EPOCH_START, steering_events)
    assert ux == phase.exact(0.1)
    assert isinstance(ux, mpq)


@pytest.mark.parametrize(
    ("event_offset", "is_counted"),
    [
        (-T, False),
        (-T + timedelta(microseconds=1), True),
        (timedelta(0), True),
        (timedelta(microseconds=1), False),
    ],
)
def test_u_counts_the_events_after_one_epoch_ago_through_the_start(
    event_offset: timedelta, is_counted: bool
) -> None:
    """Leave out an event at E - T or after E; count one just after E - T and at E."""
    steering_events = {"mc1": (steer_event(EPOCH_START + event_offset, dx=7.0),)}
    ux, _ = steering.steer_u(("mc1", "c"), EPOCH_START, steering_events)
    assert ux == (7 if is_counted else 0)


def test_a_reference_with_no_events_adds_nothing() -> None:
    """Give no input for a series whose references were never steered."""
    assert steering.steer_u(("mc1", "mc2"), EPOCH_START, {}) == (mpq(0), 0.0)


# ----------------------------------------------------------------------- w


def test_w_moves_each_event_on_to_the_measurement() -> None:
    """Sum dx and dy times the time from the event to t, exactly."""
    measured_at = EPOCH_START + timedelta(seconds=137, microseconds=203_200)
    applied_at = EPOCH_START + timedelta(seconds=37, microseconds=203_200)
    steering_events = {"mc1": (steer_event(applied_at, dx=-1.0, dy=0.5),)}
    assert steering.steer_w(
        ("mc1", "c"), EPOCH_START, steering_events, measured_at
    ) == mpq(49)


@pytest.mark.parametrize(
    ("event_offset", "is_counted"),
    [
        (timedelta(0), False),
        (timedelta(microseconds=1), True),
        (timedelta(seconds=137), True),
        (timedelta(seconds=137, microseconds=1), False),
    ],
)
def test_w_counts_the_events_after_the_start_through_the_measurement(
    event_offset: timedelta, is_counted: bool
) -> None:
    """Leave out an event at E or after t; count one just after E and at t."""
    measured_at = EPOCH_START + timedelta(seconds=137)
    steering_events = {"mc2": (steer_event(EPOCH_START + event_offset, dx=-4.0),)}
    w = steering.steer_w(("mc1", "mc2"), EPOCH_START, steering_events, measured_at)
    assert w == (4 if is_counted else 0)


# ------------------------------------------------------- an epoch's event


def test_an_event_inside_the_epoch_is_taken_off_and_counted_next_epoch() -> None:
    """Keep z_E at the true phase at E, then predict E + T with the event (U4).

    The pair (mc2, ox23) holds a true phase of 1 000 000 ps at E with no
    rate. mc2 is steered by 40 ps and 0.002 ps/s 100 s after E, and the pair
    is measured 300 s after E.
    """
    series_key = ("mc2", "ox23")
    steered_at = EPOCH_START + timedelta(seconds=100)
    measured_at = EPOCH_START + timedelta(seconds=300)
    steering_events = {"mc2": (steer_event(steered_at, dx=40.0, dy=0.002),)}
    true_at_mark = 1_000_000
    true_at_measurement = true_at_mark + 40 + phase.exact(0.002) * 200
    reading = phase.round_even(true_at_measurement) % phase.PHASE_PERIOD

    w = steering.steer_w(series_key, EPOCH_START, steering_events, measured_at)
    prediction = State(x=mpq(true_at_mark), y=0.0)
    decycled = phase.decycle(reading, mpq(300), w, prediction, None)
    assert decycled.z == true_at_mark

    next_epoch_start = EPOCH_START + T
    ux, uy = steering.steer_u(series_key, next_epoch_start, steering_events)
    true_at_following = true_at_mark + 40 + phase.exact(0.002) * 500
    assert decycled.z + ux == true_at_following
    assert uy == 0.002


def test_w_adds_up_every_event_inside_the_epoch() -> None:
    """Sum the events of the epoch up to t, not keep the last alone."""
    measured_at = EPOCH_START + timedelta(seconds=137)
    steering_events = {
        "mc2": (
            steer_event(EPOCH_START + timedelta(seconds=10), dx=-4.0),
            steer_event(EPOCH_START + timedelta(seconds=20), dx=-1.0),
        )
    }
    assert (
        steering.steer_w(("mc1", "mc2"), EPOCH_START, steering_events, measured_at) == 5
    )
