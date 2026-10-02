"""The cross-reference cycle-slip check: a wrong whole period, found and put right.

A clock measured against two or more references shows a wrong cycle count
as a mismatch of a whole number of periods between its pairs' innovations.
Every clock measured is checked, a reference measured as a clock by the
other references included, so a slip on a link is found as on any pair.
For clock c and references r < s, D = nu(r, c) - nu(s, c) - two-way(r, s)
is near zero; a D near a non-zero whole number m of periods, within five
combined scales, is flagged. The pair that slipped is then found:

* with three or more references, the one pair in every flagged D and in
  no unflagged one, every flagged D giving it the same correction;
* with two, the one of the two whose last row is not a settled acceptance.

Its cycle count is corrected by -m periods when it is the first pair of a
D and by +m when it is the second. When no pair can be named, every clock
pair in a flagged D is excluded for the epoch instead. The check runs after
screening and before the pairs are filtered. Nothing is logged here: what
it found is returned as events.
"""

import math
from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError
from masterclock.domain.filter import K_OUT
from masterclock.domain.phase import PHASE_PERIOD
from masterclock.domain.series import PairKey

SETTLED_NOT: Final[frozenset[str]] = frozenset("PRXU")
"""Flags that make a last row other than a settled acceptance."""

_MANY: Final[int] = 3
"""From how many references a slip is found by the flagged Ds alone."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

type _D = tuple[str, str, int]
"""A D: its references r and s, and its whole number of periods m."""


class SlipEvent(BaseModel):
    """One thing the slip check found at an epoch, for the program to log.

    Parameters
    ----------
    finding : {'slip_corrected', 'slip_undecided'}
        A slip corrected, or one no pair could be named for.
    clock : str
        The clock.
    pairs : tuple of (str, str)
        The corrected pair, or every clock pair excluded, sorted.
    cycles : int
        The correction in whole periods; 0 when undecided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    finding: Literal["slip_corrected", "slip_undecided"]
    clock: str
    pairs: tuple[PairKey, ...]
    cycles: int


class Slips(BaseModel):
    """What the slip check decided at an epoch.

    Parameters
    ----------
    corrections : dict of (str, str) to int
        The whole periods to add to each slipped pair's cycle count; its z
        moves by that many periods and its row carries S.
    excluded : frozenset of (str, str)
        The clock pairs excluded because a slip could not be placed, beside
        those screening excluded.
    events : tuple of SlipEvent
        What was found, clock by clock in sorted order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    corrections: dict[PairKey, int]
    excluded: frozenset[PairKey]
    events: tuple[SlipEvent, ...]


def slip_check(
    innovations: Mapping[PairKey, Fraction],
    scales: Mapping[PairKey, float],
    last_flags: Mapping[PairKey, str],
    refs: frozenset[str],
    excluded: frozenset[PairKey],
) -> Slips:
    """Find and place cycle slips in one epoch's pairs, by their clocks (design 11).

    Parameters
    ----------
    innovations : Mapping of (str, str) to Fraction
        The innovation, ps, of every pair with a measurement and a
        prediction.
    scales : Mapping of (str, str) to float
        The innovation scale, ps, of every pair with a prediction.
    last_flags : Mapping of (str, str) to str
        The flags of each pair's last row; a pair missing here is new.
    refs : frozenset of str
        The references of the epoch.
    excluded : frozenset of (str, str)
        The pairs screening excluded, which are not used.

    Returns
    -------
    Slips
        The corrections, the extra exclusions and the events.

    Raises
    ------
    FilterError
        If a pair has an innovation but no scale.

    Examples
    --------
    >>> from itertools import permutations
    >>> refs = ("mc1", "mc2", "mc3")
    >>> innovations = {pair: Fraction(0) for pair in permutations(refs, 2)}
    >>> innovations |= {("mc1", "nav1"): Fraction(200_000),
    ...                 ("mc2", "nav1"): Fraction(0), ("mc3", "nav1"): Fraction(0)}
    >>> scales = dict.fromkeys(innovations, 1.0)
    >>> flags = dict.fromkeys(innovations, "A")
    >>> slip_check(innovations, scales, flags, frozenset(refs), frozenset()).corrections
    {('mc1', 'nav1'): -1}
    """
    unscaled_pairs = sorted(set(innovations) - set(scales))
    if unscaled_pairs:
        message = f"pairs with an innovation have no innovation scale: {unscaled_pairs}"
        _log.error(message)
        raise FilterError(message)
    epoch_innovations = _EpochInnovations(innovations, scales, excluded)
    sorted_refs = tuple(sorted(refs))
    measured_clocks = sorted({pair[1] for pair in innovations})
    clock_events = [
        _check_clock(epoch_innovations, sorted_refs, c, last_flags)
        for c in measured_clocks
    ]
    return _decided(
        tuple(clock_event for clock_event in clock_events if clock_event is not None)
    )


def _decided(clock_events: tuple[SlipEvent, ...]) -> Slips:
    """Gather what the clocks' events decided.

    Parameters
    ----------
    clock_events : tuple of SlipEvent
        Every clock's event, in clock order.

    Returns
    -------
    Slips
        The corrections of the corrected slips, the clock pairs of the
        undecided ones to exclude, and the events.
    """
    corrections: dict[PairKey, int] = {}
    undecided_pairs: set[PairKey] = set()
    for clock_event in clock_events:
        if clock_event.finding == "slip_corrected":
            corrections[clock_event.pairs[0]] = clock_event.cycles
        else:
            undecided_pairs |= set(clock_event.pairs)
    return Slips(
        corrections=corrections,
        excluded=frozenset(undecided_pairs),
        events=clock_events,
    )


def _check_clock(
    epoch_innovations: _EpochInnovations,
    refs: tuple[str, ...],
    c: str,
    last_flags: Mapping[PairKey, str],
) -> SlipEvent | None:
    """Check one clock's pairs for a slip.

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values.
    refs : tuple of str
        Every reference of the epoch, sorted.
    c : str
        The clock.
    last_flags : Mapping of (str, str) to str
        The flags of each pair's last row; missing for a new pair.

    Returns
    -------
    SlipEvent or None
        The slip corrected, or undecided with every clock pair of a flagged
        D to exclude; ``None`` when no D is flagged.
    """
    usable_refs = [r for r in refs if epoch_innovations.usable((r, c))]
    flagged_ds, clean_ds = _ds(epoch_innovations, usable_refs, c)
    if not flagged_ds:
        return None
    if len(usable_refs) >= _MANY:
        slipped = _attribute_many(flagged_ds, clean_ds)
    else:
        slipped = _attribute_two(c, flagged_ds[0], last_flags)
    if slipped is None:
        excluded_pairs = tuple(
            sorted({(x, c) for r, s, _ in flagged_ds for x in (r, s)})
        )
        return SlipEvent(
            finding="slip_undecided", clock=c, pairs=excluded_pairs, cycles=0
        )
    q, k = slipped
    return SlipEvent(finding="slip_corrected", clock=c, pairs=((q, c),), cycles=k)


class _EpochInnovations:
    """One epoch's innovations and scales, and the pairs screening excluded."""

    def __init__(
        self,
        innovations: Mapping[PairKey, Fraction],
        scales: Mapping[PairKey, float],
        excluded: frozenset[PairKey],
    ) -> None:
        """Hold the epoch's values.

        Parameters
        ----------
        innovations : Mapping of (str, str) to Fraction
            The innovations.
        scales : Mapping of (str, str) to float
            The innovation scales.
        excluded : frozenset of (str, str)
            The pairs screening excluded.
        """
        self.innovations = innovations
        self.scales = scales
        self.excluded = excluded

    def usable(self, pair: PairKey) -> bool:
        """Tell whether a pair has an innovation and was not excluded.

        Parameters
        ----------
        pair : (str, str)
            The pair.

        Returns
        -------
        bool
            Whether the pair can be used.
        """
        return pair in self.innovations and pair not in self.excluded

    def nu(self, pair: PairKey) -> float:
        """Give a pair's innovation as a float, for the statistic.

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


def _ds(
    epoch_innovations: _EpochInnovations, usable_refs: list[str], c: str
) -> tuple[list[_D], list[_D]]:
    """Work out D for every two references a clock is measured against (design 11.1).

    Parameters
    ----------
    epoch_innovations : _EpochInnovations
        The epoch's values.
    usable_refs : list of str
        The references whose pair with ``c`` is usable, sorted.
    c : str
        The clock.

    Returns
    -------
    tuple of (list of (str, str, int), list of (str, str, int))
        The flagged Ds and the unflagged ones, each as (r, s, m), for every
        r < s whose link is usable both ways.
    """
    flagged_ds: list[_D] = []
    clean_ds: list[_D] = []
    for r, s in combinations(usable_refs, 2):
        if not (epoch_innovations.usable((r, s)) and epoch_innovations.usable((s, r))):
            continue
        two_way = 0.5 * (epoch_innovations.nu((r, s)) - epoch_innovations.nu((s, r)))
        d = epoch_innovations.nu((r, c)) - epoch_innovations.nu((s, c)) - two_way
        sigma = math.sqrt(
            epoch_innovations.scales[(r, c)] ** 2
            + epoch_innovations.scales[(s, c)] ** 2
            + 0.25
            * (
                epoch_innovations.scales[(r, s)] ** 2
                + epoch_innovations.scales[(s, r)] ** 2
            )
        )
        m = round(d / PHASE_PERIOD)
        slipped = m != 0 and abs(d - m * PHASE_PERIOD) < K_OUT * sigma
        (flagged_ds if slipped else clean_ds).append((r, s, m))
    return flagged_ds, clean_ds


def _attribute_many(flagged_ds: list[_D], clean_ds: list[_D]) -> tuple[str, int] | None:
    """Name a slipped pair from three or more references (design 11.2).

    Parameters
    ----------
    flagged, clean : list of (str, str, int)
        The flagged and unflagged Ds of one clock.

    Returns
    -------
    tuple of (str, int) or None
        The one reference in every flagged D and in no unflagged one, and
        its correction, -m when it is a D's first reference and +m when it
        is the second. ``None`` when there is no such reference, or the
        flagged Ds give it different corrections.
    """
    common_refs = set.intersection(*({r, s} for r, s, _ in flagged_ds))
    common_refs -= {x for r, s, _ in clean_ds for x in (r, s)}
    if len(common_refs) != 1:
        return None
    q = common_refs.pop()
    ks = {-m if q == r else m for r, _, m in flagged_ds}
    return (q, ks.pop()) if len(ks) == 1 else None


def _attribute_two(
    c: str, flagged_d: _D, last_flags: Mapping[PairKey, str]
) -> tuple[str, int] | None:
    """Name a slipped pair from two references (design 11.2).

    Parameters
    ----------
    c : str
        The clock.
    flagged_d : (str, str, int)
        The one flagged D.
    last_flags : Mapping of (str, str) to str
        The flags of each pair's last row; missing for a new pair.

    Returns
    -------
    tuple of (str, int) or None
        The reference of the one pair whose last row is not a settled
        acceptance (new, or flagged P, R, X or U), and its correction.
        ``None`` when both or neither are.
    """
    r, s, m = flagged_d
    unsettled_refs = [
        x
        for x in (r, s)
        if (x, c) not in last_flags or SETTLED_NOT & set(last_flags[(x, c)])
    ]
    if len(unsettled_refs) != 1:
        return None
    q = unsettled_refs[0]
    return q, -m if q == r else m
