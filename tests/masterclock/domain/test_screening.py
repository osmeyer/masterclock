"""Tests for src/masterclock/domain/screening.py.

The rules covered: the self-measurement of a reference fails when its
innovation is outside the gate, and then excludes only that reference's
pairs that share its shift within three combined scales; a failing self
measurement with nothing shared, and a missing one, exclude nothing, the
missing one with an event; reciprocity fails when a link's two directions'
innovations do not cancel within five combined scales, and then excludes
the one direction a median closure estimate from the other references
shows bad, or both directions when it shows both, neither or there is no
estimate; closure fails for a triangle whose two-way innovations do not sum
to zero within five combined scales, and a link is excluded, both ways,
when it is in every failing triangle and no passing one; pairs without a
measurement and a prediction are neither tested nor excluded, nor are pairs
an earlier test excluded tested again; and every test that fails gives an
event.
"""

from fractions import Fraction
from itertools import permutations
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.domain import screening
from masterclock.domain.exceptions import FilterError
from masterclock.domain.series import PairKey

CLOCKS: Final = ("nav1", "nav2", "nav3", "nav4")
"""Invented clocks, all measured against mc1."""


def links(
    refs: tuple[str, ...], errors: dict[PairKey, float] | None = None
) -> dict[PairKey, Fraction]:
    """Give every link of ``refs`` an innovation of 0, plus a two-way error.

    An error x on (a, b) puts x on (a, b) and -x on (b, a): their sum still
    cancels, so only the two-way value r̄_ab = x changes.
    """
    innovations = {pair: Fraction(0) for pair in permutations(refs, 2)}
    for (a, b), x in (errors or {}).items():
        innovations[(a, b)] += Fraction(x)
        innovations[(b, a)] -= Fraction(x)
    return innovations


def ones(innovations: dict[PairKey, Fraction]) -> dict[PairKey, float]:
    """Give every pair an innovation scale of 1 ps."""
    return dict.fromkeys(innovations, 1.0)


def screen(
    innovations: dict[PairKey, Fraction],
    refs: tuple[str, ...],
    scales: dict[PairKey, float] | None = None,
) -> screening.Screening:
    """Screen with every pair at an innovation scale of 1 ps unless given."""
    return screening.screen_references(
        innovations, scales or ones(innovations), frozenset(refs)
    )


# ----------------------------------------------------------- self-measurement


def self_epoch(shift: int, sharing: int) -> dict[PairKey, Fraction]:
    """Give mc1's self pair ``shift``, and the first ``sharing`` clock pairs too."""
    innovations = {("mc1", "mc1"): Fraction(shift)}
    for i, clock in enumerate(CLOCKS):
        innovations[("mc1", clock)] = Fraction(shift if i < sharing else 0)
    return innovations


def test_a_self_shift_excludes_the_pairs_that_share_it() -> None:
    """Exclude only the pairs of r that show the self pair's shift (U23)."""
    result = screen(self_epoch(100, 2), ("mc1",))
    assert result.excluded == frozenset({("mc1", "nav1"), ("mc1", "nav2")})
    assert [event.kind for event in result.events] == ["self_fail"]
    assert result.events[0].references == ("mc1",)
    assert result.events[0].excluded == (("mc1", "nav1"), ("mc1", "nav2"))


def test_a_self_shift_alone_excludes_nothing() -> None:
    """Exclude nothing when no pair of r shares the shift, with an event (U23)."""
    result = screen(self_epoch(100, 0), ("mc1",))
    assert result.excluded == frozenset()
    assert [(event.kind, event.excluded) for event in result.events] == [
        ("self_fail", ())
    ]


def test_a_missing_self_measurement_excludes_nothing_with_an_event() -> None:
    """Exclude nothing when (r, r) has a prediction but no measurement (U23)."""
    innovations = self_epoch(100, 4)
    del innovations[("mc1", "mc1")]
    scales = {**ones(innovations), ("mc1", "mc1"): 1.0}
    result = screen(innovations, ("mc1",), scales)
    assert result.excluded == frozenset()
    assert [(event.kind, event.references) for event in result.events] == [
        ("self_missing", ("mc1",))
    ]


def test_a_self_pair_without_a_prediction_is_not_tested() -> None:
    """Neither test nor warn for a self pair that has no prediction."""
    innovations = self_epoch(100, 4)
    del innovations[("mc1", "mc1")]
    assert screen(innovations, ("mc1",)) == screening.Screening(
        excluded=frozenset(), events=()
    )


def test_a_self_innovation_inside_the_gate_passes() -> None:
    """Pass a self pair at exactly five scales, compared exactly."""
    assert screen(self_epoch(5, 4), ("mc1",)).excluded == frozenset()
    assert screen(self_epoch(-5, 4), ("mc1",)).excluded == frozenset()


def test_sharing_the_shift_means_within_three_combined_scales() -> None:
    """Share within 3 sqrt(2) ps for scales of 1 ps, not within 5 sqrt(2)."""
    innovations = self_epoch(100, 0)
    innovations[("mc1", "nav1")] = Fraction(100 - 4)
    innovations[("mc1", "nav2")] = Fraction(100 - 6)
    result = screen(innovations, ("mc1",))
    assert result.excluded == frozenset({("mc1", "nav1")})


def test_the_shared_test_uses_each_pair_s_own_scale() -> None:
    """Combine the pair's scale with the self pair's: a wide pair shares more."""
    innovations = self_epoch(100, 0)
    innovations[("mc1", "nav1")] = Fraction(100 - 20)
    scales = {**ones(innovations), ("mc1", "nav1"): 10.0}
    assert screen(innovations, ("mc1",), scales).excluded == frozenset(
        {("mc1", "nav1")}
    )


# --------------------------------------------------------------- reciprocity

REFS: Final = ("mc1", "mc2", "mc3", "mc4")
"""Four invented references."""


def test_a_delay_on_one_direction_excludes_that_direction() -> None:
    """Exclude only the delayed direction of a link (U15)."""
    innovations = links(REFS)
    innovations[("mc1", "mc2")] += 100
    result = screen(innovations, REFS)
    assert result.excluded == frozenset({("mc1", "mc2")})
    assert [(e.kind, e.references, e.excluded) for e in result.events] == [
        ("reciprocity_fail", ("mc1", "mc2"), (("mc1", "mc2"),))
    ]


def test_a_delay_on_the_other_direction_excludes_that_one() -> None:
    """Exclude only (s, r) when it is (s, r) that is delayed (U15)."""
    innovations = links(REFS)
    innovations[("mc2", "mc1")] += 100
    assert screen(innovations, REFS).excluded == frozenset({("mc2", "mc1")})


def test_a_delay_on_both_directions_excludes_both() -> None:
    """Exclude both directions when the estimate shows both bad (U15)."""
    innovations = links(REFS)
    innovations[("mc1", "mc2")] += 100
    innovations[("mc2", "mc1")] += 100
    result = screen(innovations, REFS)
    assert result.excluded == frozenset({("mc1", "mc2"), ("mc2", "mc1")})


def test_a_reciprocity_failure_with_no_estimate_excludes_both() -> None:
    """Exclude both directions when no third reference gives an estimate."""
    innovations = links(("mc1", "mc2"))
    innovations[("mc1", "mc2")] += 100
    result = screen(innovations, ("mc1", "mc2"))
    assert result.excluded == frozenset({("mc1", "mc2"), ("mc2", "mc1")})


def test_a_reciprocity_failure_neither_direction_explains_excludes_both() -> None:
    """Exclude both when each direction lies within the estimate's reach."""
    innovations = links(REFS)
    innovations[("mc1", "mc2")] += 4
    innovations[("mc2", "mc1")] += 4
    scales = {**ones(innovations), ("mc1", "mc2"): 0.5, ("mc2", "mc1"): 0.5}
    result = screen(innovations, REFS, scales)
    assert result.excluded == frozenset({("mc1", "mc2"), ("mc2", "mc1")})


def test_innovations_that_cancel_pass_reciprocity() -> None:
    """Pass a link whose two innovations sum to within five combined scales."""
    innovations = links(("mc1", "mc2"))
    innovations[("mc1", "mc2")] += 7
    assert screen(innovations, ("mc1", "mc2")).excluded == frozenset()


def test_the_closure_estimate_is_the_median() -> None:
    """Keep one bad estimate from moving the estimate: the median, not the mean."""
    refs = ("mc1", "mc2", "mc3", "mc4", "mc5")
    innovations = links(refs, {("mc2", "mc5"): 3_000})
    innovations[("mc1", "mc2")] += 100
    result = screen(innovations, refs)
    assert ("mc1", "mc2") in result.excluded
    assert ("mc2", "mc1") not in result.excluded


def test_the_estimate_carries_the_link_s_own_two_way_value() -> None:
    """Estimate the link from the triangles' closure, so a true offset is kept."""
    innovations = links(REFS, {("mc2", "mc3"): -50, ("mc2", "mc4"): -50})
    innovations[("mc1", "mc2")] += 50 + 100
    innovations[("mc2", "mc1")] -= 50
    result = screen(innovations, REFS)
    assert result.excluded == frozenset({("mc1", "mc2")})


def test_a_third_reference_missing_a_link_gives_no_estimate() -> None:
    """Take estimates only from references with both links usable."""
    innovations = links(REFS)
    innovations[("mc1", "mc2")] += 100
    del innovations[("mc4", "mc1")]
    result = screen(innovations, REFS)
    assert result.excluded == frozenset({("mc1", "mc2")})


def test_reciprocity_with_no_usable_third_reference_excludes_both() -> None:
    """Exclude both directions when every third reference lacks a link."""
    innovations = links(("mc1", "mc2", "mc3"))
    innovations[("mc1", "mc2")] += 100
    del innovations[("mc3", "mc1")]
    result = screen(innovations, ("mc1", "mc2", "mc3"))
    assert result.excluded == frozenset({("mc1", "mc2"), ("mc2", "mc1")})


# ------------------------------------------------------------------- closure


def test_an_error_in_one_link_excludes_only_that_link() -> None:
    """Exclude the one link in both failing triangles of four references (U16)."""
    result = screen(links(REFS, {("mc1", "mc2"): 100}), REFS)
    assert result.excluded == frozenset({("mc1", "mc2"), ("mc2", "mc1")})
    assert [(e.kind, e.references) for e in result.events] == [
        ("closure_fail", ("mc1", "mc2"))
    ]


def test_one_failing_triangle_of_three_references_excludes_all_three_links() -> None:
    """Exclude every link of the one triangle when it fails."""
    refs = ("mc1", "mc2", "mc3")
    result = screen(links(refs, {("mc2", "mc3"): 100}), refs)
    assert result.excluded == frozenset(permutations(refs, 2))
    assert [e.references for e in result.events] == [
        ("mc1", "mc2"),
        ("mc1", "mc3"),
        ("mc2", "mc3"),
    ]


def test_a_link_in_a_passing_triangle_is_kept() -> None:
    """Keep the links of a failing triangle that each sit in a passing one."""
    errors = {("mc1", "mc4"): -4.0, ("mc3", "mc4"): -8.0, ("mc2", "mc3"): 12.0}
    innovations = links(REFS, errors)
    result = screen(innovations, REFS)
    assert result.excluded == frozenset()
    assert result.events == ()


def test_closure_passes_within_five_combined_scales() -> None:
    """Pass a triangle whose sum is within five combined scales."""
    refs = ("mc1", "mc2", "mc3")
    assert screen(links(refs, {("mc2", "mc3"): 6}), refs).excluded == frozenset()


def test_closure_skips_links_an_earlier_test_excluded() -> None:
    """Evaluate no triangle through a direction reciprocity excluded."""
    refs = ("mc1", "mc2", "mc3")
    innovations = links(refs)
    innovations[("mc1", "mc2")] += 100
    result = screen(innovations, refs)
    assert result.excluded == frozenset({("mc1", "mc2")})
    assert [e.kind for e in result.events] == ["reciprocity_fail"]


# -------------------------------------------------------------------- general


def test_pairs_without_a_measurement_are_neither_tested_nor_excluded() -> None:
    """Screen only pairs given an innovation: one missing direction skips a link."""
    innovations = links(REFS, {("mc1", "mc2"): 100})
    del innovations[("mc2", "mc1")]
    assert screen(innovations, REFS) == screening.Screening(
        excluded=frozenset(), events=()
    )


def test_a_quiet_epoch_excludes_nothing() -> None:
    """Give no exclusions and no events when every test passes."""
    innovations = {**links(REFS), **self_epoch(0, 4)}
    assert screen(innovations, REFS) == screening.Screening(
        excluded=frozenset(), events=()
    )


def test_an_innovation_needs_a_scale() -> None:
    """Raise FilterError for a pair given an innovation but no scale."""
    innovations = links(("mc1", "mc2"))
    scales = {("mc1", "mc2"): 1.0}
    with pytest.raises(FilterError, match="no innovation scale"):
        screening.screen_references(innovations, scales, frozenset({"mc1", "mc2"}))


def test_an_event_names_a_known_kind() -> None:
    """Refuse an event of a kind screening does not give."""
    with pytest.raises(ValidationError):
        screening.ScreeningEvent.model_validate(
            {"kind": "other", "references": ("mc1",), "excluded": ()}
        )


def test_a_triangle_with_a_link_not_measured_both_ways_is_not_tested() -> None:
    """Skip a triangle whose third link is missing a direction."""
    refs = ("mc1", "mc2", "mc3")
    innovations = links(refs, {("mc2", "mc3"): 100})
    del innovations[("mc2", "mc1")]
    assert screen(innovations, refs) == screening.Screening(
        excluded=frozenset(), events=()
    )
