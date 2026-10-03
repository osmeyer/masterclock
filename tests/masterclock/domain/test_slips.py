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

The D scale combines as the design writes it and a slip lies strictly inside
five of them; a D that cannot be worked out does not stop the rest; and
every undecided clock's pairs are excluded. A reference measured as a
clock by the other references is checked as any clock is.
"""

from fractions import Fraction
from itertools import permutations
from typing import Final

import pytest

from masterclock.domain import slips
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import PHASE_PERIOD
from masterclock.domain.series import PairKey

P: Final = PHASE_PERIOD
"""One period, ps."""

THREE_REFS: Final = ("mc1", "mc2", "mc3")
"""Three invented references."""


def epoch_innovations(
    refs: tuple[str, ...],
    clock_innovations: dict[str, int],
    link_errors: dict[PairKey, int] | None = None,
) -> dict[PairKey, Fraction]:
    """Give the innovations of ``refs``' links and of nav1 against each.

    ``clock_innovations`` gives each reference's (r, nav1) innovation; a link error x
    on (a, b) puts x on (a, b) and -x on (b, a), changing its two-way value.
    """
    innovations = {pair: Fraction(0) for pair in permutations(refs, 2)}
    for (a, b), x in (link_errors or {}).items():
        innovations[(a, b)] += x
        innovations[(b, a)] -= x
    for r, clock_innovation in clock_innovations.items():
        innovations[(r, "nav1")] = Fraction(clock_innovation)
    return innovations


def of_nav1(slip_result: slips.Slips) -> slips.Slips:
    """Keep only what the slip check found for the clock nav1."""
    return slips.Slips(
        corrections={
            pair: cycles
            for pair, cycles in slip_result.corrections.items()
            if pair[1] == "nav1"
        },
        excluded=frozenset(pair for pair in slip_result.excluded if pair[1] == "nav1"),
        events=tuple(
            slip_event
            for slip_event in slip_result.events
            if slip_event.clock == "nav1"
        ),
    )


def run_slip_check(
    innovations: dict[PairKey, Fraction],
    refs: tuple[str, ...],
    last_flags: dict[PairKey, str] | None = None,
    excluded: frozenset[PairKey] = frozenset(),
    scales: dict[PairKey, float] | None = None,
) -> slips.Slips:
    """Run the slip check with 1 ps scales and settled rows unless given."""
    last_row_flags = (
        dict.fromkeys(innovations, "A") if last_flags is None else last_flags
    )
    return slips.slip_check(
        innovations,
        scales or dict.fromkeys(innovations, 1.0),
        last_row_flags,
        frozenset(refs),
        excluded,
    )


# ----------------------------------------------------- three references


@pytest.mark.parametrize(
    ("slipped_ref", "phase_shift", "cycles"),
    [("mc1", P, -1), ("mc2", -P, 1), ("mc3", 2 * P, -2)],
)
def test_a_slip_against_one_of_three_references_is_corrected(
    slipped_ref: str, phase_shift: int, cycles: int
) -> None:
    """Correct the one pair in every flagged D by whole periods (U17)."""
    clock_innovations = {
        r: (phase_shift + 3 if r == slipped_ref else 0) for r in THREE_REFS
    }
    slip_result = run_slip_check(
        epoch_innovations(THREE_REFS, clock_innovations), THREE_REFS
    )
    assert slip_result.corrections == {(slipped_ref, "nav1"): cycles}
    assert slip_result.excluded == frozenset()
    assert [
        (slip_event.finding, slip_event.clock, slip_event.pairs, slip_event.cycles)
        for slip_event in slip_result.events
    ] == [("slip_corrected", "nav1", ((slipped_ref, "nav1"),), cycles)]


def test_no_slip_gives_nothing() -> None:
    """Correct and exclude nothing when every D is near zero."""
    slip_result = run_slip_check(
        epoch_innovations(THREE_REFS, dict.fromkeys(THREE_REFS, 2)), THREE_REFS
    )
    assert slip_result == slips.Slips(corrections={}, excluded=frozenset(), events=())


def test_a_d_far_from_a_whole_period_is_not_a_slip() -> None:
    """Leave a D more than five combined scales from mP unflagged."""
    clock_innovations = {"mc1": P + 100, "mc2": 0, "mc3": 0}
    assert (
        run_slip_check(
            epoch_innovations(THREE_REFS, clock_innovations), THREE_REFS
        ).corrections
        == {}
    )


def test_the_link_s_two_way_value_is_taken_off() -> None:
    """Flag D = nu(r,c) - nu(s,c) - two-way(r,s), not the bare difference."""
    clock_innovations = {"mc1": P, "mc2": 0, "mc3": 0}
    innovations = epoch_innovations(
        THREE_REFS, clock_innovations, {("mc1", "mc2"): P, ("mc1", "mc3"): P}
    )
    assert run_slip_check(innovations, THREE_REFS).corrections == {}


def test_a_pair_in_a_clean_d_is_not_corrected() -> None:
    """Leave undecided a slip whose candidates both sit in an unflagged D."""
    innovations = epoch_innovations(
        THREE_REFS, {"mc1": P, "mc2": 0, "mc3": 0}, {("mc1", "mc3"): P}
    )
    slip_result = of_nav1(run_slip_check(innovations, THREE_REFS))
    assert slip_result.corrections == {}
    assert slip_result.excluded == frozenset({("mc1", "nav1"), ("mc2", "nav1")})
    assert [
        (slip_event.finding, slip_event.pairs, slip_event.cycles)
        for slip_event in slip_result.events
    ] == [("slip_undecided", (("mc1", "nav1"), ("mc2", "nav1")), 0)]


def test_the_one_common_pair_in_a_clean_d_is_not_corrected() -> None:
    """Leave undecided the one pair of every flagged D when an unflagged D has it."""
    four_refs = (*THREE_REFS, "mc4")
    clock_innovations = {"mc1": P, "mc2": 0, "mc3": 0, "mc4": 0}
    innovations = epoch_innovations(four_refs, clock_innovations, {("mc1", "mc4"): P})
    slip_result = of_nav1(run_slip_check(innovations, four_refs))
    assert slip_result.corrections == {}
    assert slip_result.excluded == frozenset(
        {("mc1", "nav1"), ("mc2", "nav1"), ("mc3", "nav1")}
    )


def test_corrections_that_disagree_are_undecided() -> None:
    """Correct nothing when the flagged Ds give the common pair two corrections."""
    clock_innovations = {"mc1": P, "mc2": 0, "mc3": 2 * P}
    innovations = epoch_innovations(
        THREE_REFS, clock_innovations, {("mc2", "mc3"): -2 * P}
    )
    slip_result = of_nav1(run_slip_check(innovations, THREE_REFS))
    assert slip_result.corrections == {}
    assert slip_result.excluded == frozenset((r, "nav1") for r in THREE_REFS)


def test_no_pair_common_to_every_flagged_d_is_undecided() -> None:
    """Correct nothing when the flagged Ds share no pair."""
    clock_innovations = {"mc1": P, "mc2": 0, "mc3": -P}
    slip_result = run_slip_check(
        epoch_innovations(THREE_REFS, clock_innovations), THREE_REFS
    )
    assert slip_result.corrections == {}
    assert slip_result.excluded == frozenset((r, "nav1") for r in THREE_REFS)


# ------------------------------------------------------- two references

TWO_REFS: Final = ("mc1", "mc2")
"""Two invented references."""


@pytest.mark.parametrize("weak_flags", ["P", "R", "X", "AU", "new"])
def test_with_two_references_the_weak_pair_is_corrected(weak_flags: str) -> None:
    """Correct the pair whose last row is not a settled acceptance (U17)."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": 0, "mc2": P})
    last_flags = dict.fromkeys(innovations, "A")
    if weak_flags == "new":
        del last_flags[("mc2", "nav1")]
    else:
        last_flags[("mc2", "nav1")] = weak_flags
    slip_result = run_slip_check(innovations, TWO_REFS, last_flags)
    assert slip_result.corrections == {("mc2", "nav1"): -1}
    assert slip_result.events[0].finding == "slip_corrected"


def test_the_first_pair_weak_is_corrected_the_other_way() -> None:
    """Give -m to the first pair of a D, +m to the second."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": P, "mc2": 0})
    last_flags = {**dict.fromkeys(innovations, "A"), ("mc1", "nav1"): "R"}
    assert run_slip_check(innovations, TWO_REFS, last_flags).corrections == {
        ("mc1", "nav1"): -1
    }


@pytest.mark.parametrize("pair_flags", [("A", "A"), ("AU", "P")])
def test_with_two_references_both_or_neither_weak_is_undecided(
    pair_flags: tuple[str, str],
) -> None:
    """Exclude both clock pairs when both or neither are weak."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": 0, "mc2": P})
    last_flags = {
        **dict.fromkeys(innovations, "A"),
        ("mc1", "nav1"): pair_flags[0],
        ("mc2", "nav1"): pair_flags[1],
    }
    slip_result = run_slip_check(innovations, TWO_REFS, last_flags)
    assert slip_result.corrections == {}
    assert slip_result.excluded == frozenset({("mc1", "nav1"), ("mc2", "nav1")})
    assert [slip_event.finding for slip_event in slip_result.events] == [
        "slip_undecided"
    ]


# ---------------------------------------------------------------- what is used


def test_a_pair_excluded_by_screening_is_not_used() -> None:
    """Leave out a clock pair that screening excluded."""
    clock_innovations = {"mc1": P, "mc2": 0, "mc3": 0}
    excluded_pairs = frozenset({("mc1", "nav1")})
    assert (
        run_slip_check(
            epoch_innovations(THREE_REFS, clock_innovations),
            THREE_REFS,
            excluded=excluded_pairs,
        ).corrections
        == {}
    )


def test_a_link_not_usable_both_ways_gives_no_d() -> None:
    """Form no D across a link with a direction missing or excluded."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": 0, "mc2": P})
    last_flags = {**dict.fromkeys(innovations, "A"), ("mc2", "nav1"): "AU"}
    del innovations[("mc2", "mc1")]
    assert run_slip_check(innovations, TWO_REFS, last_flags).corrections == {}
    innovations = epoch_innovations(TWO_REFS, {"mc1": 0, "mc2": P})
    excluded_pairs = frozenset({("mc1", "mc2")})
    assert (
        run_slip_check(innovations, TWO_REFS, last_flags, excluded_pairs).corrections
        == {}
    )


def test_a_reference_measured_as_a_clock_is_checked_too() -> None:
    """Correct a link that slipped, the reference at its far end checked as a clock."""
    four_refs = (*THREE_REFS, "mc4")
    innovations = epoch_innovations(four_refs, {})
    innovations[("mc1", "mc4")] += P
    slip_result = run_slip_check(innovations, four_refs)
    assert slip_result.corrections == {("mc1", "mc4"): -1}
    assert [
        (slip_event.finding, slip_event.clock) for slip_event in slip_result.events
    ] == [("slip_corrected", "mc4")]


def test_the_tolerance_is_five_combined_scales() -> None:
    """Flag a D within 5 sigma_D of mP and not one just beyond it."""
    sigma_d = (1 + 1 + 0.25 * 2) ** 0.5
    within_tolerance = {"mc1": 0, "mc2": P + int(5 * sigma_d)}
    beyond_tolerance = {"mc1": 0, "mc2": P + int(5 * sigma_d) + 1}
    last_flags = {
        **dict.fromkeys(epoch_innovations(TWO_REFS, within_tolerance), "A"),
        ("mc2", "nav1"): "R",
    }
    assert run_slip_check(
        epoch_innovations(TWO_REFS, within_tolerance), TWO_REFS, last_flags
    ).corrections == {("mc2", "nav1"): -1}
    assert (
        run_slip_check(
            epoch_innovations(TWO_REFS, beyond_tolerance), TWO_REFS, last_flags
        ).corrections
        == {}
    )


def test_a_pair_needs_a_scale() -> None:
    """Raise FilterError for a pair with an innovation but no scale."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": 0, "mc2": 0})
    with pytest.raises(FilterError, match="no innovation scale"):
        slips.slip_check(innovations, {}, {}, frozenset(TWO_REFS), frozenset())


# ------------------------------------------ tolerances and gathering, exactly


EXACT_D_SCALES: Final = {
    ("mc1", "nav1"): 2.0,
    ("mc2", "nav1"): 14.0,
    ("mc1", "mc2"): 6.0,
    ("mc2", "mc1"): 8.0,
}
"""Scales whose combined D scale is exactly 15 ps: 4 + 196 + (36 + 64) / 4 = 225."""

WEAK_FIRST: Final = {
    ("mc1", "nav1"): "R",
    ("mc2", "nav1"): "A",
    ("mc1", "mc2"): "A",
    ("mc2", "mc1"): "A",
}
"""Last flags that name (mc1, nav1) the weak pair of two."""


@pytest.mark.parametrize(("offset_ps", "slipped"), [(74, True), (75, False)])
def test_a_slip_lies_strictly_within_five_d_scales_of_a_period(
    offset_ps: int, slipped: bool
) -> None:
    """Combine the clock pairs' scales and a quarter of the links' (11.1)."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": P + offset_ps, "mc2": 0})
    slip_result = run_slip_check(
        innovations, TWO_REFS, WEAK_FIRST, scales=EXACT_D_SCALES
    )
    assert slip_result.corrections == ({("mc1", "nav1"): -1} if slipped else {})


@pytest.mark.parametrize("offset_ps", [60, 70])
def test_a_slip_well_inside_the_tolerance_is_found(offset_ps: int) -> None:
    """Find a slip at 4 and at 4.7 D scales from a period."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": P + offset_ps, "mc2": 0})
    slip_result = run_slip_check(
        innovations, TWO_REFS, WEAK_FIRST, scales=EXACT_D_SCALES
    )
    assert slip_result.corrections == {("mc1", "nav1"): -1}


def test_a_link_that_gives_no_d_does_not_stop_the_others() -> None:
    """Work out every later D after a pair of references with no usable link."""
    innovations = epoch_innovations(THREE_REFS, {"mc1": 0, "mc2": 0, "mc3": P})
    del innovations[("mc1", "mc2")], innovations[("mc2", "mc1")]
    slip_result = run_slip_check(innovations, THREE_REFS)
    assert slip_result.corrections == {("mc3", "nav1"): -1}


def test_every_undecided_clock_s_pairs_are_excluded() -> None:
    """Exclude the clock pairs of each undecided slip, every clock's together."""
    innovations = epoch_innovations(TWO_REFS, {"mc1": P, "mc2": 0})
    innovations |= {("mc1", "nav2"): Fraction(P), ("mc2", "nav2"): Fraction(0)}
    slip_result = run_slip_check(innovations, TWO_REFS)
    assert slip_result.excluded == frozenset(
        {("mc1", "nav1"), ("mc2", "nav1"), ("mc1", "nav2"), ("mc2", "nav2")}
    )


def test_a_missing_scale_is_logged_as_raised(caplog: pytest.LogCaptureFixture) -> None:
    """Log the FilterError for an innovation without a scale, in its own words."""
    with pytest.raises(FilterError) as raised:
        slips.slip_check(
            {("mc1", "nav1"): Fraction(0)}, {}, {}, frozenset({"mc1"}), frozenset()
        )
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(raised.value)
    ]
