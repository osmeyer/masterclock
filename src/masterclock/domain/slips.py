"""The cross-reference cycle-slip check: a wrong whole period, found and put right.

A clock measured against two or more references shows a wrong cycle count
as a mismatch of a whole number of periods between its pairs' innovations.
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
    kind : {'slip_corrected', 'slip_undecided'}
        A slip corrected, or one no pair could be named for.
    clock : str
        The clock.
    pairs : tuple of (str, str)
        The corrected pair, or every clock pair excluded, sorted.
    cycles : int
        The correction in whole periods; 0 when undecided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["slip_corrected", "slip_undecided"]
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
    """Find and place cycle slips in one epoch's clock pairs (design 11).

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
    >>> innovations |= {("mc1", "ox1"): Fraction(200_000),
    ...                 ("mc2", "ox1"): Fraction(0), ("mc3", "ox1"): Fraction(0)}
    >>> scales = dict.fromkeys(innovations, 1.0)
    >>> flags = dict.fromkeys(innovations, "A")
    >>> slip_check(innovations, scales, flags, frozenset(refs), frozenset()).corrections
    {('mc1', 'ox1'): -1}
    """
    missing = sorted(set(innovations) - set(scales))
    if missing:
        message = f"pairs with an innovation have no innovation scale: {missing}"
        _log.error(message)
        raise FilterError(message)
    epoch = _Epoch(innovations, scales, excluded)
    ordered = tuple(sorted(refs))
    clocks = sorted({pair[1] for pair in innovations if pair[1] not in refs})
    found = [_check_clock(epoch, ordered, c, last_flags) for c in clocks]
    return _decided(tuple(event for event in found if event is not None))


def _decided(events: tuple[SlipEvent, ...]) -> Slips:
    """Gather what the clocks' events decided.

    Parameters
    ----------
    events : tuple of SlipEvent
        Every clock's event, in clock order.

    Returns
    -------
    Slips
        The corrections of the corrected slips, the clock pairs of the
        undecided ones to exclude, and the events.
    """
    corrections: dict[PairKey, int] = {}
    extra: set[PairKey] = set()
    for event in events:
        if event.kind == "slip_corrected":
            corrections[event.pairs[0]] = event.cycles
        else:
            extra |= set(event.pairs)
    return Slips(corrections=corrections, excluded=frozenset(extra), events=events)


def _check_clock(
    epoch: _Epoch, refs: tuple[str, ...], c: str, last_flags: Mapping[PairKey, str]
) -> SlipEvent | None:
    """Check one clock's pairs for a slip.

    Parameters
    ----------
    epoch : _Epoch
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
    against = [r for r in refs if epoch.usable((r, c))]
    flagged, clean = _ds(epoch, against, c)
    if not flagged:
        return None
    if len(against) >= _MANY:
        named = _attribute_many(flagged, clean)
    else:
        named = _attribute_two(c, flagged[0], last_flags)
    if named is None:
        pairs = tuple(sorted({(x, c) for r, s, _ in flagged for x in (r, s)}))
        return SlipEvent(kind="slip_undecided", clock=c, pairs=pairs, cycles=0)
    q, k = named
    return SlipEvent(kind="slip_corrected", clock=c, pairs=((q, c),), cycles=k)


class _Epoch:
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


def _ds(epoch: _Epoch, against: list[str], c: str) -> tuple[list[_D], list[_D]]:
    """Work out D for every two references a clock is measured against (design 11.1).

    Parameters
    ----------
    epoch : _Epoch
        The epoch's values.
    against : list of str
        The references whose pair with ``c`` is usable, sorted.
    c : str
        The clock.

    Returns
    -------
    tuple of (list of (str, str, int), list of (str, str, int))
        The flagged Ds and the unflagged ones, each as (r, s, m), for every
        r < s whose link is usable both ways.
    """
    flagged: list[_D] = []
    clean: list[_D] = []
    for r, s in combinations(against, 2):
        if not (epoch.usable((r, s)) and epoch.usable((s, r))):
            continue
        two_way = 0.5 * (epoch.nu((r, s)) - epoch.nu((s, r)))
        d = epoch.nu((r, c)) - epoch.nu((s, c)) - two_way
        sigma = math.sqrt(
            epoch.scales[(r, c)] ** 2
            + epoch.scales[(s, c)] ** 2
            + 0.25 * (epoch.scales[(r, s)] ** 2 + epoch.scales[(s, r)] ** 2)
        )
        m = round(d / PHASE_PERIOD)
        slipped = m != 0 and abs(d - m * PHASE_PERIOD) < K_OUT * sigma
        (flagged if slipped else clean).append((r, s, m))
    return flagged, clean


def _attribute_many(flagged: list[_D], clean: list[_D]) -> tuple[str, int] | None:
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
    common = set.intersection(*({r, s} for r, s, _ in flagged))
    common -= {x for r, s, _ in clean for x in (r, s)}
    if len(common) != 1:
        return None
    q = common.pop()
    ks = {-m if q == r else m for r, _, m in flagged}
    return (q, ks.pop()) if len(ks) == 1 else None


def _attribute_two(
    c: str, flagged: _D, last_flags: Mapping[PairKey, str]
) -> tuple[str, int] | None:
    """Name a slipped pair from two references (design 11.2).

    Parameters
    ----------
    c : str
        The clock.
    flagged : (str, str, int)
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
    r, s, m = flagged
    weak = [
        x
        for x in (r, s)
        if (x, c) not in last_flags or SETTLED_NOT & set(last_flags[(x, c)])
    ]
    if len(weak) != 1:
        return None
    q = weak[0]
    return q, -m if q == r else m
