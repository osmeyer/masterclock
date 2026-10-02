"""Tests for src/masterclock/domain/phase.py.

The rules covered: a phase period is one period of a 5 MHz signal in whole
picoseconds, and the largest phase a reading gives is one short of it; a
time offset between two datetimes is an exact fraction of seconds, and
naive datetimes are refused; a float becomes an exact fraction, and a value
that is not finite is refused; and rounding is to the nearest whole number,
a tie going to the even one, exactly at any size.
"""

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.domain import phase
from masterclock.domain.exceptions import FilterError, PhaseError

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
