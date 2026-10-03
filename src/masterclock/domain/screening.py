"""Screening of the reference measurements before any pair is filtered.

Three tests run in order, each on the innovations of one epoch: the
measurement less the prediction, so constant hardware delays drop out.

* Self-measurement: a reference measured against itself carries the same
  signal on both inputs, so a jump in it is a shift in that reference's
  measurement system. The pairs of the reference that share the shift are
  excluded.
* Reciprocity: the two directions of a link carry opposite phases, so their
  innovations cancel. When they do not, the direction that a closure
  estimate from the other references shows bad is excluded, or both.
* Closure: the two-way innovations around a triangle of references sum to
  zero. A link in every failing triangle and in no passing one is excluded,
  both ways.

A pair is tested only when it has both a measurement and a prediction, and
not after an earlier test excluded it. Nothing is logged here: what the
tests found is returned as events, for the program to log.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from statistics import median
from typing import Final, Literal

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError
from masterclock.domain.filter import K_OUT, within_gate
from masterclock.domain.series import PairKey

K_SHARED: Final[float] = 3.0
"""How many combined scales from a self pair's shift a pair may lie and share it."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

type _TwoWay = tuple[float, float]
"""A link's two-way innovation and its scale."""


@dataclass(frozen=True, slots=True)
class ScreeningEvent:
    """One thing screening found at an epoch, for the program to log.

    Parameters
    ----------
    finding : {'self_missing', 'self_fail', 'reciprocity_fail', 'closure_fail'}
        What was found: a self pair with a prediction but no measurement, a
        self-measurement outside the gate, a link whose directions do not
        cancel, or a link excluded by closure.
    references : tuple of str
        The reference of a self test, or the two of a link.
    excluded : tuple of (str, str)
        The pairs the finding excluded, sorted; empty when none.
    """

    finding: Literal["self_missing", "self_fail", "reciprocity_fail", "closure_fail"]
    references: tuple[str, ...]
    excluded: tuple[PairKey, ...]


@dataclass(frozen=True, slots=True)
class Screening:
    """What screening decided at an epoch.

    Parameters
    ----------
    excluded : frozenset of (str, str)
        The pairs excluded at the epoch (design 9.5).
    events : tuple of ScreeningEvent
        What the tests found, in the order they ran.
    """

    excluded: frozenset[PairKey]
    events: tuple[ScreeningEvent, ...]


class _EpochInnovations:
    """One epoch's innovations and scales, and the exclusions made so far."""

    def __init__(
        self, innovations: Mapping[PairKey, Fraction], scales: Mapping[PairKey, float]
    ) -> None:
        """Hold the epoch's values.

        Parameters
        ----------
        innovations : Mapping of (str, str) to Fraction
            The innovation of every pair with a measurement and a prediction.
        scales : Mapping of (str, str) to float
            The innovation scale of every pair with a prediction.
        """
        self.innovations = innovations
        self.scales = scales
        self.excluded: set[PairKey] = set()
        self.events: list[ScreeningEvent] = []

    def usable(self, pair: PairKey) -> bool:
        """Tell whether a pair can be tested: measured, predicted, not excluded.

        Parameters
        ----------
        pair : (str, str)
            The pair.

        Returns
        -------
        bool
            Whether the pair has an innovation and is not yet excluded.
        """
        return pair in self.innovations and pair not in self.excluded

    def nu(self, pair: PairKey) -> float:
        """Give a pair's innovation as a float, for the statistics.

        Parameters
        ----------
        pair : (str, str)
            A pair with an innovation.

        Returns
        -------
        float
            The innovation, ps.
        """
        return float(self.innovations[pair])

    def two_way(self, a: str, b: str) -> _TwoWay | None:
        """Give a link's two-way innovation and its scale.

        Parameters
        ----------
        a, b : str
            The link's references, in the order the value is taken.

        Returns
        -------
        tuple of (float, float) or None
            (nu(a, b) - nu(b, a)) / 2 and half the two scales combined;
            ``None`` unless both directions are usable.
        """
        if not (self.usable((a, b)) and self.usable((b, a))):
            return None
        two_way_value = 0.5 * (self.nu((a, b)) - self.nu((b, a)))
        return two_way_value, 0.5 * math.hypot(self.scales[(a, b)], self.scales[(b, a)])

    def exclude(
        self,
        finding: Literal["self_fail", "reciprocity_fail", "closure_fail"],
        references: tuple[str, ...],
        excluded_pairs: set[PairKey],
    ) -> None:
        """Exclude pairs and record why.

        Parameters
        ----------
        finding : str
            The test that failed.
        references : tuple of str
            The reference or link it failed for.
        excluded_pairs : set of (str, str)
            The pairs to exclude.
        """
        self.excluded |= excluded_pairs
        self.events.append(
            ScreeningEvent(
                finding=finding,
                references=references,
                excluded=tuple(sorted(excluded_pairs)),
            )
        )


def screen_references(
    innovations: Mapping[PairKey, Fraction],
    scales: Mapping[PairKey, float],
    refs: frozenset[str],
) -> Screening:
    """Screen one epoch's reference measurements (design 10).

    Parameters
    ----------
    innovations : Mapping of (str, str) to Fraction
        The innovation, ps, of every pair that has both a measurement and a
        prediction at the epoch.
    scales : Mapping of (str, str) to float
        The innovation scale, ps, of every pair that has a prediction; a
        self pair here but not in ``innovations`` is a missing
        self-measurement.
    refs : frozenset of str
        The references of the epoch.

    Returns
    -------
    Screening
        The pairs excluded and the events, from the self-measurement,
        reciprocity and closure tests in that order.

    Raises
    ------
    FilterError
        If a pair has an innovation but no scale.

    Examples
    --------
    >>> innovations = {("mc1", "mc1"): Fraction(100), ("mc1", "nav1"): Fraction(99)}
    >>> scales = dict.fromkeys(innovations, 3.0)
    >>> sorted(screen_references(innovations, scales, frozenset({"mc1"})).excluded)
    [('mc1', 'nav1')]
    """
    unscaled_pairs = sorted(set(innovations) - set(scales))
    if unscaled_pairs:
        message = f"pairs with an innovation have no innovation scale: {unscaled_pairs}"
        _log.error(message)
        raise FilterError(message)
    epoch_innovations = _EpochInnovations(innovations, scales)
    sorted_refs = tuple(sorted(refs))
    for r in sorted_refs:
        _self_test(epoch_innovations, r)
    for r, s in combinations(sorted_refs, 2):
        _reciprocity(epoch_innovations, sorted_refs, r, s)
    _closure(epoch_innovations, sorted_refs)
    return Screening(
        excluded=frozenset(epoch_innovations.excluded),
        events=tuple(epoch_innovations.events),
    )


def _self_test(epoch_innovations: _EpochInnovations, r: str) -> None:
    """Exclude the pairs of r that share a shift in its self pair (design 10.1).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    r : str
        The reference.
    """
    self_pair = (r, r)
    if self_pair not in epoch_innovations.scales:
        return
    if self_pair not in epoch_innovations.innovations:
        epoch_innovations.events.append(
            ScreeningEvent(finding="self_missing", references=(r,), excluded=())
        )
        return
    if within_gate(
        epoch_innovations.innovations[self_pair], epoch_innovations.scales[self_pair]
    ):
        return
    self_shift = epoch_innovations.nu(self_pair)
    shared_pairs = {
        pair
        for pair in epoch_innovations.innovations
        if pair[0] == r
        and pair[1] != r
        and abs(epoch_innovations.nu(pair) - self_shift)
        <= K_SHARED
        * math.hypot(
            epoch_innovations.scales[pair], epoch_innovations.scales[self_pair]
        )
    }
    epoch_innovations.exclude("self_fail", (r,), shared_pairs)


def _reciprocity(
    epoch_innovations: _EpochInnovations, refs: tuple[str, ...], r: str, s: str
) -> None:
    """Exclude the bad direction of a link whose directions do not cancel (design 10.2).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    refs : tuple of str
        Every reference of the epoch, sorted.
    r, s : str
        The link's references, r before s.
    """
    forward, back = (r, s), (s, r)
    if not (epoch_innovations.usable(forward) and epoch_innovations.usable(back)):
        return
    rho = epoch_innovations.nu(forward) + epoch_innovations.nu(back)
    if abs(rho) <= K_OUT * math.hypot(
        epoch_innovations.scales[forward], epoch_innovations.scales[back]
    ):
        return
    closure_estimates = _closure_estimates(epoch_innovations, refs, r, s)
    epoch_innovations.exclude(
        "reciprocity_fail",
        (r, s),
        _bad_directions(epoch_innovations, r, s, closure_estimates),
    )


def _closure_estimates(
    epoch_innovations: _EpochInnovations, refs: tuple[str, ...], r: str, s: str
) -> list[_TwoWay]:
    """Estimate a link's two-way innovation from the other references (design 10.2).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    refs : tuple of str
        Every reference of the epoch, sorted.
    r, s : str
        The link's references.

    Returns
    -------
    list of (float, float)
        For each third reference t with usable links s-t and t-r, the
        estimate -(two-way(s, t) + two-way(t, r)) and its scale.
    """
    closure_estimates = []
    for t in refs:
        if t in {r, s}:
            continue
        st, tr = epoch_innovations.two_way(s, t), epoch_innovations.two_way(t, r)
        if st is not None and tr is not None:
            closure_estimates.append((-(st[0] + tr[0]), math.hypot(st[1], tr[1])))
    return closure_estimates


def _bad_directions(
    epoch_innovations: _EpochInnovations,
    r: str,
    s: str,
    closure_estimates: list[_TwoWay],
) -> set[PairKey]:
    """Name the direction of a failing link to exclude (design 10.2).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    r, s : str
        The link's references, r before s.
    closure_estimates : list of (float, float)
        The closure estimates of the link (see :func:`_closure_estimates`).

    Returns
    -------
    set of (str, str)
        The one direction that lies outside five combined scales of the
        median estimate; both directions when both or neither do, or there
        is no estimate.
    """
    forward, back = (r, s), (s, r)
    if not closure_estimates:
        return {forward, back}
    median_estimate = median(
        direction_value for direction_value, _ in closure_estimates
    )
    median_spread = median(estimate_scale for _, estimate_scale in closure_estimates)
    outlying = {
        pair
        for pair, direction_value in (
            (forward, epoch_innovations.nu(forward)),
            (back, -epoch_innovations.nu(back)),
        )
        if abs(direction_value - median_estimate)
        > K_OUT * math.hypot(epoch_innovations.scales[pair], median_spread)
    }
    return outlying if len(outlying) == 1 else {forward, back}


def _closure(epoch_innovations: _EpochInnovations, refs: tuple[str, ...]) -> None:
    """Exclude the links that only failing triangles hold (design 10.3).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    refs : tuple of str
        Every reference of the epoch, sorted.
    """
    failing_triangles, passing_triangles = _triangles(epoch_innovations, refs)
    if not failing_triangles:
        return
    for a, b in combinations(refs, 2):
        link_refs = {a, b}
        if all(link_refs <= triangle for triangle in failing_triangles) and not any(
            link_refs <= triangle for triangle in passing_triangles
        ):
            epoch_innovations.exclude("closure_fail", (a, b), {(a, b), (b, a)})


def _triangles(
    epoch_innovations: _EpochInnovations, refs: tuple[str, ...]
) -> tuple[list[frozenset[str]], list[frozenset[str]]]:
    """Test every triangle of references whose links are all usable (design 10.3).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values and exclusions.
    refs : tuple of str
        Every reference of the epoch, sorted.

    Returns
    -------
    tuple of (list of frozenset of str, list of frozenset of str)
        The failing triangles, whose two-way innovations sum to more than
        five combined scales either way, and the passing ones.
    """
    failing_triangles: list[frozenset[str]] = []
    passing_triangles: list[frozenset[str]] = []
    for r, s, t in combinations(refs, 3):
        triangle_legs = [
            epoch_innovations.two_way(r, s),
            epoch_innovations.two_way(s, t),
            epoch_innovations.two_way(t, r),
        ]
        known_legs = [leg for leg in triangle_legs if leg is not None]
        if len(known_legs) < len(triangle_legs):
            continue
        leg_sum = sum(leg_value for leg_value, _ in known_legs)
        leg_scale = math.sqrt(sum(leg_spread**2 for _, leg_spread in known_legs))
        triangle = frozenset({r, s, t})
        (
            failing_triangles if abs(leg_sum) > K_OUT * leg_scale else passing_triangles
        ).append(triangle)
    return failing_triangles, passing_triangles
