"""Tests for src/masterclock/domain/phase.py.

The rule covered: a phase period is one period of a 5 MHz signal in whole
picoseconds, and the largest phase a reading gives is one short of it.
"""

from fractions import Fraction

from masterclock.domain import phase


def test_the_period_is_one_cycle_of_5_mhz_in_picoseconds() -> None:
    """Make the period exactly 1 / 5 MHz, written in picoseconds."""
    assert Fraction(phase.PHASE_PERIOD, 10**12) == Fraction(1, 5_000_000)


def test_the_largest_phase_is_one_short_of_a_period() -> None:
    """Keep a whole period out, since it would read as zero."""
    assert phase.PHASE_MAX == phase.PHASE_PERIOD - 1
