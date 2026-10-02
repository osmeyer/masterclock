"""Tests for src/masterclock/domain/phase.py.

The rules covered: a phase period is one period of a 5 MHz signal in whole
picoseconds, and the largest phase a reading gives is one short of it; a
time offset between two datetimes is an exact fraction of seconds, and
naive datetimes are refused; a float becomes an exact fraction, and a value
that is not finite is refused; rounding is to the nearest whole number,
a tie going to the even one, exactly at any size; an epoch lasts 600 s; and
a measurement is decycled against the prediction at its own time, or
against the last buffered measurement, or with no cycles added, and
referred back to its epoch start with one exact rounding.
"""

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from masterclock.domain import phase
from masterclock.domain.exceptions import FilterError, PhaseError
from masterclock.domain.series import State

MARK: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented ten-minute mark."""


def test_the_period_is_one_cycle_of_5_mhz_in_picoseconds() -> None:
    """Make the period exactly 1 / 5 MHz, written in picoseconds."""
    assert Fraction(phase.PHASE_PERIOD, 10**12) == Fraction(1, 5_000_000)


def test_the_largest_phase_is_one_short_of_a_period() -> None:
    """Keep a whole period out, since it would read as zero."""
    assert phase.PHASE_MAX == phase.PHASE_PERIOD - 1


# ----------------------------------------------------------------- seconds


def test_an_offset_is_exact_to_the_microsecond() -> None:
    """Give 137.2032 s as exactly 1 372 032 / 10 000, not a float near it."""
    later = MARK + timedelta(seconds=137, microseconds=203_200)
    assert phase.seconds(later, MARK) == Fraction(1_372_032, 10_000)


def test_an_offset_can_be_negative() -> None:
    """Give an earlier first datetime as a negative offset."""
    assert phase.seconds(MARK, MARK + timedelta(microseconds=1)) == Fraction(-1, 10**6)


@pytest.mark.parametrize(
    ("later", "earlier"),
    [
        (MARK.replace(tzinfo=None), MARK),
        (MARK, MARK.replace(tzinfo=None)),
        (MARK.replace(tzinfo=None), MARK.replace(tzinfo=None)),
    ],
)
def test_a_naive_datetime_is_refused(later: datetime, earlier: datetime) -> None:
    """Refuse a datetime without a timezone, which names no one instant."""
    with pytest.raises(PhaseError, match="naive"):
        phase.seconds(later, earlier)


@given(st.integers(min_value=-(10**15), max_value=10**15))
def test_an_offset_is_its_microseconds_over_a_million(microseconds: int) -> None:
    """Give any offset as its whole number of microseconds over 10**6."""
    later = MARK + timedelta(microseconds=microseconds)
    assert phase.seconds(later, MARK) == Fraction(microseconds, 10**6)


# ------------------------------------------------------------------- exact


def test_a_float_becomes_the_fraction_it_holds() -> None:
    """Give 0.1 as the binary value the float holds, not as one tenth."""
    assert phase.exact(0.1) == Fraction(3_602_879_701_896_397, 2**55)
    assert phase.exact(0.1) != Fraction(1, 10)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_not_finite_is_refused(value: float) -> None:
    """Refuse nan and the infinities, which no phase can hold."""
    with pytest.raises(FilterError, match="not finite"):
        phase.exact(value)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_every_finite_float_is_held_exactly(value: float) -> None:
    """Give a fraction equal to the float, numerator and denominator alike."""
    assert phase.exact(value) == Fraction(*value.as_integer_ratio())


# -------------------------------------------------------------- round_even


@pytest.mark.parametrize(
    ("value", "rounded"),
    [
        (Fraction(5, 2), 2),
        (Fraction(7, 2), 4),
        (Fraction(-5, 2), -2),
        (Fraction(-7, 2), -4),
        (Fraction(12_345_773_124, 10_000), 1_234_577),
        (Fraction(2**63 * 2 + 1, 2), 2**63),
        (17, 17),
    ],
)
def test_rounding_is_to_nearest_with_ties_to_even(
    value: Fraction, rounded: int
) -> None:
    """Round 2.5 to 2, 3.5 to 4 and -2.5 to -2, exactly at any size."""
    result = phase.round_even(value)
    assert result == rounded
    assert type(result) is int


@given(st.integers(min_value=-(2**70), max_value=2**70))
def test_every_half_rounds_to_even(whole: int) -> None:
    """Round every n + 1/2 to whichever of n and n + 1 is even."""
    result = phase.round_even(Fraction(2 * whole + 1, 2))
    assert result in {whole, whole + 1}
    assert result % 2 == 0


def reference_round(numerator: int, denominator: int) -> int:
    """Round numerator / denominator by integer arithmetic alone, ties to even."""
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (
        2 * remainder == denominator and quotient % 2 == 1
    ):
        return quotient + 1
    return quotient


@given(
    st.integers(min_value=2**53, max_value=2**63),
    st.floats(min_value=-1e6, max_value=1e6),
    st.sampled_from([1, -1]),
)
def test_a_large_phase_plus_a_float_rounds_as_integers_do(
    whole: int, term: float, sign: int
) -> None:
    """Round a phase beyond 2**53 plus a float term as exact integer sums do."""
    numerator, denominator = term.as_integer_ratio()
    total = sign * whole * denominator + numerator
    assert phase.round_even(sign * whole + phase.exact(term)) == reference_round(
        total, denominator
    )


@given(st.integers(min_value=2**53, max_value=2**63))
def test_a_large_phase_plus_a_half_is_a_tie_to_even(whole: int) -> None:
    """Treat a phase beyond 2**53 plus exactly 0.5 as a tie, not a float near it."""
    assert phase.round_even(whole + phase.exact(0.5)) == whole + whole % 2


# -------------------------------------------------------------- decycling

P: Final = phase.PHASE_PERIOD
"""One period, ps."""


def test_an_epoch_lasts_600_seconds() -> None:
    """Make one epoch ten minutes, in whole seconds."""
    assert phase.EPOCH_SECONDS == 600


def test_the_worked_epoch_decycles_to_six_cycles() -> None:
    """Reproduce Appendix A: n = 6 and z_E = 1 234 577."""
    prediction = State(x=1_234_567 + phase.exact(0.0123) * 600, y=0.0123, d=0.0)
    delta = Fraction(1_372_032, 10_000)
    result = phase.decycle(34_579, delta, Fraction(0), prediction, None)
    assert result == phase.Decycled(cycle_count=6, z=1_234_577)


def motion(prediction: State, delta: Fraction) -> Fraction:
    """Work out y delta + d delta**2 / 2 exactly, as the reference does."""
    y = Fraction(*prediction.y.as_integer_ratio())
    d = Fraction(*prediction.d.as_integer_ratio())
    return y * delta + d * delta * delta / 2


@given(
    truth=st.integers(min_value=-(2**62), max_value=2**62),
    rate=st.floats(min_value=-150.0, max_value=150.0),
    drift=st.floats(min_value=-1e-6, max_value=1e-6),
    microseconds=st.integers(min_value=0, max_value=599_999_999),
    error=st.integers(min_value=-90_000, max_value=90_000),
    steer=st.fractions(min_value=-1000, max_value=1000, max_denominator=10**6),
)
def test_a_measurement_decycles_to_the_truth(
    truth: int,
    rate: float,
    drift: float,
    microseconds: int,
    error: int,
    steer: Fraction,
) -> None:
    """Recover the whole phase with a prediction up to 0.45 P wrong (U3).

    ``truth`` is the whole phase at the measurement time, of any sign and
    far beyond 2**53, and the reading is it modulo one period. The rate is
    large enough that its motion over the measurement offset reaches 0.45 P
    too, so a decycling that left the motion out would miss whole periods.
    """
    delta = Fraction(microseconds, 10**6)
    reading = truth % P
    move = motion(State(x=Fraction(0), y=rate, d=drift), delta)
    prediction = State(x=truth - move - steer + error, y=rate, d=drift)
    result = phase.decycle(reading, delta, steer, prediction, None)
    assert result.cycle_count == (truth - reading) // P
    assert result.z == round(truth - move - steer)


@pytest.mark.parametrize("rate", [0.031, -0.031])
@pytest.mark.parametrize("start", [-3 * P - 5, P - 7, 2**60 + P - 3])
def test_a_ramp_decycles_across_every_wrap(rate: float, start: int) -> None:
    """Follow a ramp through the wrap at 0 and at PHASE_MAX, epoch after epoch.

    Each epoch is predicted from the last epoch's result and a rate in error
    by enough to put the prediction 0.45 P off, so every wrap is crossed with
    the decycling bound nearly used up.
    """
    period = Fraction(phase.EPOCH_SECONDS)
    wrong = rate + 0.45 * P / phase.EPOCH_SECONDS
    previous = start
    for epoch in range(1, 40):
        truth = start + round(
            Fraction(*Fraction(rate).as_integer_ratio()) * period * epoch
        )
        reading = truth % P
        prediction = State(x=previous + phase.exact(wrong) * period, y=wrong, d=0.0)
        result = phase.decycle(reading, Fraction(0), Fraction(0), prediction, None)
        assert result.z == truth, epoch
        previous = result.z


def test_a_value_on_an_exact_half_rounds_to_even() -> None:
    """Round a z_E that falls on n + 1/2 to the even neighbour, once."""
    prediction = State(x=Fraction(100), y=0.5, d=0.0)
    for reading, expected in ((101, 100), (103, 102)):
        result = phase.decycle(reading, Fraction(1), Fraction(0), prediction, None)
        assert result.z == expected


@given(
    whole=st.integers(min_value=2**53, max_value=2**63),
    rate=st.floats(min_value=-0.1, max_value=0.1),
    microseconds=st.integers(min_value=0, max_value=599_999_999),
)
def test_large_phases_decycle_exactly(
    whole: int, rate: float, microseconds: int
) -> None:
    """Match a Fraction reference for phases beyond 2**53 (U27, decycling)."""
    delta = Fraction(microseconds, 10**6)
    move = motion(State(x=Fraction(0), y=rate), delta)
    prediction = State(x=Fraction(whole), y=rate)
    reading = round(whole + move) % P
    result = phase.decycle(reading, delta, Fraction(0), prediction, None)
    n = round((whole + move - reading) / P)
    assert result.cycle_count == n
    assert result.z == round(reading + n * P - move)


@pytest.mark.parametrize(
    ("anchor", "reading", "steer", "cycles"),
    [
        (None, 123, Fraction(0), 0),
        (None, 123, Fraction(50), 0),
        (P + 123, 123, Fraction(0), 1),
        (-P + 100, 100, Fraction(0), -1),
        (5 * P + 10, P - 10, Fraction(0), 4),
        (10, P - 10, Fraction(0), -1),
        (P + 123, 123, Fraction(P), 2),
    ],
)
def test_without_a_prediction_the_anchor_decycles(
    anchor: int | None, reading: int, steer: Fraction, cycles: int
) -> None:
    """Decycle against the last buffered value, or with no cycles at all."""
    result = phase.decycle(reading, Fraction(77), steer, None, anchor)
    assert result.cycle_count == cycles
    assert result.z == round(reading + cycles * P - steer)


@pytest.mark.parametrize("reading", [-1, P])
def test_a_reading_outside_one_period_is_refused(reading: int) -> None:
    """Refuse a reading that is not a phase within one period."""
    with pytest.raises(PhaseError, match="not within one period"):
        phase.decycle(reading, Fraction(0), Fraction(0), None, None)


@pytest.mark.parametrize("delta", [Fraction(-1, 10**6), Fraction(600)])
def test_a_measurement_outside_its_epoch_is_refused(delta: Fraction) -> None:
    """Refuse a measurement time before its epoch start or a whole epoch on."""
    with pytest.raises(PhaseError, match="not within its epoch"):
        phase.decycle(0, delta, Fraction(0), None, None)


def test_a_decycled_measurement_is_frozen_and_strict() -> None:
    """Refuse a change, an unknown field, and a cycle count that is not an int."""
    result = phase.Decycled(cycle_count=1, z=2)
    with pytest.raises(ValidationError, match="frozen"):
        result.z = 3  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs"):
        phase.Decycled.model_validate({"cycle_count": 1, "z": 2, "colour": "red"})
    with pytest.raises(ValidationError, match="cycle_count"):
        phase.Decycled.model_validate({"cycle_count": 1.0, "z": 2})


def test_with_a_prediction_the_anchor_is_not_used() -> None:
    """Decycle against the prediction even when an anchor is given."""
    prediction = State(x=Fraction(3 * P + 10), y=0.0)
    result = phase.decycle(10, Fraction(0), Fraction(0), prediction, -7 * P)
    assert result == phase.Decycled(cycle_count=3, z=3 * P + 10)


def test_a_fast_rate_is_decycled_with_its_motion() -> None:
    """Count the rate's motion over the offset when choosing the periods.

    At 150 ps/s, 500 s after the mark the phase has moved 75 000 ps; with
    the prediction 40 000 ps low as well, leaving the motion out would put
    the reading a period away.
    """
    truth = 7 * P + 12_345
    prediction = State(x=Fraction(truth - 75_000 - 40_000), y=150.0)
    result = phase.decycle(truth % P, Fraction(500), Fraction(0), prediction, None)
    assert result == phase.Decycled(cycle_count=7, z=truth - 75_000)
