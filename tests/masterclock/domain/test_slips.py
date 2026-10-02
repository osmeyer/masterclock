"""Tests for src/masterclock/domain/slips.py.

The rules covered: for a clock measured against references r < s, D is the
difference of its two clock pairs' innovations less the link's two-way
innovation, and is flagged when it lies within five combined scales of a
non-zero whole number m of periods; with three or more references the
slipped pair is the one in every flagged D, in no unflagged one, every
flagged D giving it the same correction, and with two it is the one whose
last row is not a settled acceptance; the correction is -m whole periods
for the first pair of a D and +m for the second; an undecided case corrects
nothing and excludes every clock pair in a flagged D; pairs without an
innovation, or excluded by screening, and links not usable both ways are
not used; and each correction or undecided case gives an event.
"""

from fractions import Fraction
from itertools import permutations
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.domain import slips
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import PHASE_PERIOD
from masterclock.domain.series import PairKey

P: Final = PHASE_PERIOD
"""One period, ps."""

THREE: Final = ("mc1", "mc2", "mc3")
"""Three invented references."""


def epoch(
    refs: tuple[str, ...],
    clocks: dict[str, int],
    link_errors: dict[PairKey, int] | None = None,
) -> dict[PairKey, Fraction]:
    """Give the innovations of ``refs``' links and of nav1 against each.

    ``clocks`` gives each reference's (r, nav1) innovation; a link error x
    on (a, b) puts x on (a, b) and -x on (b, a), changing its two-way value.
    """
    innovations = {pair: Fraction(0) for pair in permutations(refs, 2)}
    for (a, b), x in (link_errors or {}).items():
        innovations[(a, b)] += x
        innovations[(b, a)] -= x
    for r, value in clocks.items():
        innovations[(r, "nav1")] = Fraction(value)
    return innovations


def check(
    innovations: dict[PairKey, Fraction],
    refs: tuple[str, ...],
    last_flags: dict[PairKey, str] | None = None,
    excluded: frozenset[PairKey] = frozenset(),
    scales: dict[PairKey, float] | None = None,
) -> slips.Slips:
    """Run the slip check with 1 ps scales and settled rows unless given."""
    flags = dict.fromkeys(innovations, "A") if last_flags is None else last_flags
    return slips.slip_check(
        innovations,
        scales or dict.fromkeys(innovations, 1.0),
        flags,
        frozenset(refs),
        excluded,
    )


# ----------------------------------------------------- three references


@pytest.mark.parametrize(
    ("ref", "shift", "cycles"), [("mc1", P, -1), ("mc2", -P, 1), ("mc3", 2 * P, -2)]
)
def test_a_slip_against_one_of_three_references_is_corrected(
    ref: str, shift: int, cycles: int
) -> None:
    """Correct the one pair in every flagged D by whole periods (U17)."""
    clocks = {r: (shift + 3 if r == ref else 0) for r in THREE}
    result = check(epoch(THREE, clocks), THREE)
    assert result.corrections == {(ref, "nav1"): cycles}
    assert result.excluded == frozenset()
    assert [(e.kind, e.clock, e.pairs, e.cycles) for e in result.events] == [
        ("slip_corrected", "nav1", ((ref, "nav1"),), cycles)
    ]


def test_no_slip_gives_nothing() -> None:
    """Correct and exclude nothing when every D is near zero."""
    result = check(epoch(THREE, dict.fromkeys(THREE, 2)), THREE)
    assert result == slips.Slips(corrections={}, excluded=frozenset(), events=())


def test_a_d_far_from_a_whole_period_is_not_a_slip() -> None:
    """Leave a D more than five combined scales from mP unflagged."""
    clocks = {"mc1": P + 100, "mc2": 0, "mc3": 0}
    assert check(epoch(THREE, clocks), THREE).corrections == {}


def test_the_link_s_two_way_value_is_taken_off() -> None:
    """Flag D = nu(r,c) - nu(s,c) - two-way(r,s), not the bare difference."""
    clocks = {"mc1": P, "mc2": 0, "mc3": 0}
    innovations = epoch(THREE, clocks, {("mc1", "mc2"): P, ("mc1", "mc3"): P})
    assert check(innovations, THREE).corrections == {}


def test_a_pair_in_a_clean_d_is_not_corrected() -> None:
    """Leave undecided a slip whose candidates both sit in an unflagged D."""
    innovations = epoch(THREE, {"mc1": P, "mc2": 0, "mc3": 0}, {("mc1", "mc3"): P})
    result = check(innovations, THREE)
    assert result.corrections == {}
    assert result.excluded == frozenset({("mc1", "nav1"), ("mc2", "nav1")})
    assert [(e.kind, e.pairs, e.cycles) for e in result.events] == [
        ("slip_undecided", (("mc1", "nav1"), ("mc2", "nav1")), 0)
    ]


def test_the_one_common_pair_in_a_clean_d_is_not_corrected() -> None:
    """Leave undecided the one pair of every flagged D when an unflagged D has it."""
    four = (*THREE, "mc4")
    clocks = {"mc1": P, "mc2": 0, "mc3": 0, "mc4": 0}
    innovations = epoch(four, clocks, {("mc1", "mc4"): P})
    result = check(innovations, four)
    assert result.corrections == {}
    assert result.excluded == frozenset(
        {("mc1", "nav1"), ("mc2", "nav1"), ("mc3", "nav1")}
    )


def test_corrections_that_disagree_are_undecided() -> None:
    """Correct nothing when the flagged Ds give the common pair two corrections."""
    clocks = {"mc1": P, "mc2": 0, "mc3": 2 * P}
    innovations = epoch(THREE, clocks, {("mc2", "mc3"): -2 * P})
    result = check(innovations, THREE)
    assert result.corrections == {}
    assert result.excluded == frozenset((r, "nav1") for r in THREE)


def test_no_pair_common_to_every_flagged_d_is_undecided() -> None:
    """Correct nothing when the flagged Ds share no pair."""
    clocks = {"mc1": P, "mc2": 0, "mc3": -P}
    result = check(epoch(THREE, clocks), THREE)
    assert result.corrections == {}
    assert result.excluded == frozenset((r, "nav1") for r in THREE)


# ------------------------------------------------------- two references

TWO: Final = ("mc1", "mc2")
"""Two invented references."""


@pytest.mark.parametrize("weak", ["P", "R", "X", "AU", "new"])
def test_with_two_references_the_weak_pair_is_corrected(weak: str) -> None:
    """Correct the pair whose last row is not a settled acceptance (U17)."""
    innovations = epoch(TWO, {"mc1": 0, "mc2": P})
    flags = dict.fromkeys(innovations, "A")
    if weak == "new":
        del flags[("mc2", "nav1")]
    else:
        flags[("mc2", "nav1")] = weak
    result = check(innovations, TWO, flags)
    assert result.corrections == {("mc2", "nav1"): -1}
    assert result.events[0].kind == "slip_corrected"


def test_the_first_pair_weak_is_corrected_the_other_way() -> None:
    """Give -m to the first pair of a D, +m to the second."""
    innovations = epoch(TWO, {"mc1": P, "mc2": 0})
    flags = {**dict.fromkeys(innovations, "A"), ("mc1", "nav1"): "R"}
    assert check(innovations, TWO, flags).corrections == {("mc1", "nav1"): -1}


@pytest.mark.parametrize("flags", [("A", "A"), ("AU", "P")])
def test_with_two_references_both_or_neither_weak_is_undecided(
    flags: tuple[str, str],
) -> None:
    """Exclude both clock pairs when both or neither are weak."""
    innovations = epoch(TWO, {"mc1": 0, "mc2": P})
    last = {
        **dict.fromkeys(innovations, "A"),
        ("mc1", "nav1"): flags[0],
        ("mc2", "nav1"): flags[1],
    }
    result = check(innovations, TWO, last)
    assert result.corrections == {}
    assert result.excluded == frozenset({("mc1", "nav1"), ("mc2", "nav1")})
    assert [e.kind for e in result.events] == ["slip_undecided"]


# ---------------------------------------------------------------- what is used


def test_a_pair_excluded_by_screening_is_not_used() -> None:
    """Leave out a clock pair that screening excluded."""
    clocks = {"mc1": P, "mc2": 0, "mc3": 0}
    excluded = frozenset({("mc1", "nav1")})
    assert check(epoch(THREE, clocks), THREE, excluded=excluded).corrections == {}


def test_a_link_not_usable_both_ways_gives_no_d() -> None:
    """Form no D across a link with a direction missing or excluded."""
    innovations = epoch(TWO, {"mc1": 0, "mc2": P})
    flags = {**dict.fromkeys(innovations, "A"), ("mc2", "nav1"): "AU"}
    del innovations[("mc2", "mc1")]
    assert check(innovations, TWO, flags).corrections == {}
    innovations = epoch(TWO, {"mc1": 0, "mc2": P})
    excluded = frozenset({("mc1", "mc2")})
    assert check(innovations, TWO, flags, excluded).corrections == {}


def test_only_clock_pairs_are_checked() -> None:
    """Check clocks, never a reference measured as a clock."""
    innovations = epoch(THREE, {})
    innovations[("mc1", "mc2")] += P
    assert check(innovations, THREE) == slips.Slips(
        corrections={}, excluded=frozenset(), events=()
    )


def test_the_tolerance_is_five_combined_scales() -> None:
    """Flag a D within 5 sigma_D of mP and not one just beyond it."""
    sigma_d = (1 + 1 + 0.25 * 2) ** 0.5
    near = {"mc1": 0, "mc2": P + int(5 * sigma_d)}
    far = {"mc1": 0, "mc2": P + int(5 * sigma_d) + 1}
    flags = {**dict.fromkeys(epoch(TWO, near), "A"), ("mc2", "nav1"): "R"}
    assert check(epoch(TWO, near), TWO, flags).corrections == {("mc2", "nav1"): -1}
    assert check(epoch(TWO, far), TWO, flags).corrections == {}


def test_a_pair_needs_a_scale() -> None:
    """Raise FilterError for a pair with an innovation but no scale."""
    innovations = epoch(TWO, {"mc1": 0, "mc2": 0})
    with pytest.raises(FilterError, match="no innovation scale"):
        slips.slip_check(innovations, {}, {}, frozenset(TWO), frozenset())


def test_an_event_names_a_known_kind() -> None:
    """Refuse an event of a kind the slip check does not give."""
    with pytest.raises(ValidationError):
        slips.SlipEvent.model_validate(
            {"kind": "other", "clock": "nav1", "pairs": (), "cycles": 0}
        )
