"""Find the stretches when a clock is not running properly (design 15.4).

A DAS channel whose clock is off or disconnected still gives readings, but
their phase is spread over the whole period; a clock reduced to its quartz
crystal still gives a signal, but its phase lurches by tens of nanoseconds
from one epoch to the next. Either way the one-epoch changes are tens of
nanoseconds apart, where a running clock's are far closer. This script
reads the measurement files of a das_processor run and, for every clock
that is not a reference, judges each epoch against every reference that
measures the clock, from the raw measured phase. A stretch is reported only
when every reference that can judge it agrees, stretches closer together
than the shortest one reported are joined, and only those lasting at least
that long are kept. Each stretch is printed with the MJD from which the
clock is to be disabled and the MJD at which it is to be enabled again,
ready to be looked over and put into the clock configuration. Run it as::

    uv run --frozen python scripts/no_signal.py RUN --rf a --skip ox

Each stretch is marked by what the references show inside it: readings
that disagree as much as independent ones would mean no signal; readings
that agree mean a signal that lurches. A clock that runs far off
frequency, or whose phase jumps, still runs properly between its jumps and
is not reported: only the spread of the changes is judged, never their
size.
"""

import argparse
import math
import statistics
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final, NamedTuple

from characterize import (
    EPOCHS_PER_DAY,
    MAD_TO_SIGMA,
    MEAS_SLICES,
    NO_SIGNAL_PS,
    OUTLIER_SPREADS,
    epoch_number,
)
from masterclock.das_processor.channels import RF_CHOICES, RfChannel
from masterclock.das_processor.config import MEAS_SUBDIRECTORY
from masterclock.das_processor.files import EMPTY, MEAS_HEADER_LINES
from masterclock.das_processor.registry import series_file, series_key_of
from masterclock.domain.phase import EPOCH_SECONDS, PHASE_PERIOD
from masterclock.domain.references import is_reference

HALF_WINDOW: Final[int] = 12
"""The epochs on either side of an epoch whose changes judge it."""

MIN_WINDOW_CHANGES: Final[int] = 9
"""The fewest changes a window needs to judge its epoch."""

MIN_HOURS: Final[float] = 12.0
"""The shortest stretch reported, and the shortest time between two that keeps
them apart, hours, unless the command line says otherwise."""

DISAGREEING_PS: Final[float] = 20_000.0
"""The typical spread of the references' changes at an epoch above which a
stretch holds no signal, ps: independent readings over the whole period
spread about 31 ns, and over twelve hours never below 22 ns, while readings
of one signal spread far less."""

MIN_COMPARED: Final[int] = 3
"""The fewest references with a change at an epoch to compare them there."""

EPOCHS_PER_HOUR: Final[int] = 3600 // EPOCH_SECONDS
"""Epochs in an hour."""

NO_SIGNAL: Final[str] = "no_signal"
"""The kind of a stretch whose references disagree as independent readings do."""

QUARTZ: Final[str] = "quartz"
"""The kind of a stretch whose references see one signal that lurches."""

type Phases = dict[int, int]
"""Raw measured phases, ps, by epoch."""

type Changes = dict[int, int]
"""One-epoch phase changes, ps, wrapped into the half period either way, by
the epoch each ends at."""

type Verdicts = dict[int, bool]
"""Whether each judged epoch is unlike a running clock, by epoch."""


def read_phases(data_file: Path) -> Phases:
    """Read the raw measured phase of every row with a reading.

    Parameters
    ----------
    data_file : Path
        A measurement file.

    Returns
    -------
    Phases
        The measured phase of every row that has one, by epoch.
    """
    phases = {}
    with data_file.open(encoding="ascii") as open_file:
        for line_number, row_line in enumerate(open_file):
            if line_number < MEAS_HEADER_LINES:
                continue
            phase_text = row_line[MEAS_SLICES["measured_phase"]].strip()
            if phase_text != EMPTY:
                mjd_text = row_line[MEAS_SLICES["interpolated_mjd"]].strip()
                phases[epoch_number(mjd_text)] = int(phase_text)
    return phases


def wrapped_change(change: int) -> int:
    """Wrap a phase change into the half period either way.

    Parameters
    ----------
    change : int
        A difference of two raw phases, ps.

    Returns
    -------
    int
        The change less the whole periods that bring it into
        (-PHASE_PERIOD / 2, PHASE_PERIOD / 2].

    Examples
    --------
    >>> wrapped_change(30_000), wrapped_change(190_000), wrapped_change(-100_000)
    (30000, -10000, 100000)
    """
    half = PHASE_PERIOD // 2
    return -((half - change) % PHASE_PERIOD) + half


def one_epoch_changes(phases: Phases) -> Changes:
    """Give every wrapped change between readings one epoch apart.

    Parameters
    ----------
    phases : Phases
        Raw phases by epoch.

    Returns
    -------
    Changes
        For every epoch whose previous epoch has a reading, the wrapped
        change from it.

    Examples
    --------
    >>> one_epoch_changes({1: 0, 2: 199_000, 4: 5, 5: 10})
    {2: -1000, 5: 5}
    """
    return {
        epoch: wrapped_change(phase - phases[epoch - 1])
        for epoch, phase in phases.items()
        if epoch - 1 in phases
    }


def spread(changes: Sequence[int]) -> tuple[float, float]:
    """Give the median of some changes and their median absolute departure from it.

    Parameters
    ----------
    changes : sequence of int
        At least one change.

    Returns
    -------
    (float, float)
        The median, and the median of the absolute differences from it.

    Examples
    --------
    >>> spread([1, 2, 10])
    (2, 1)
    """
    median_change = statistics.median(changes)
    return median_change, statistics.median(abs(c - median_change) for c in changes)


def judge(changes: Changes) -> Verdicts:
    """Judge each epoch from the changes within HALF_WINDOW epochs of it.

    Parameters
    ----------
    changes : Changes
        One pair's changes.

    Returns
    -------
    Verdicts
        For every epoch with a change whose window holds at least
        :data:`MIN_WINDOW_CHANGES` changes, whether they typically depart
        from their median by more than :data:`NO_SIGNAL_PS`.

    Examples
    --------
    >>> steady = {epoch: 500 for epoch in range(1, 40)}
    >>> random_phase = {40 + k: (79_193 * k * k) % 200_000 - 100_000 for k in range(40)}
    >>> verdicts = judge({**steady, **random_phase})
    >>> verdicts[10], verdicts[70]
    (False, True)
    """
    epochs = sorted(changes)
    verdicts = {}
    low = high = 0
    for epoch in epochs:
        while epochs[low] < epoch - HALF_WINDOW:
            low += 1
        while high < len(epochs) and epochs[high] <= epoch + HALF_WINDOW:
            high += 1
        if high - low >= MIN_WINDOW_CHANGES:
            window = [changes[e] for e in epochs[low:high]]
            verdicts[epoch] = spread(window)[1] > NO_SIGNAL_PS
    return verdicts


def agreed(per_reference: Sequence[Verdicts]) -> Verdicts:
    """Combine the references' verdicts: unlike a clock only when every one says so.

    Parameters
    ----------
    per_reference : sequence of Verdicts
        Each reference's verdicts on the clock.

    Returns
    -------
    Verdicts
        For every epoch at least one reference judged, in epoch order, true
        exactly when every reference that judged it found it unlike a
        running clock.

    Examples
    --------
    >>> agreed([{1: True, 2: True, 3: True}, {2: False, 3: True}])
    {1: True, 2: False, 3: True}
    """
    combined: Verdicts = {}
    for verdicts in per_reference:
        for epoch, bad in verdicts.items():
            combined[epoch] = combined.get(epoch, True) and bad
    return dict(sorted(combined.items()))


class Span(NamedTuple):
    """Epochs from a start to an end.

    Parameters
    ----------
    start : int
        The first epoch in it.
    end : int
        The epoch after the last in it.
    open_ended : bool
        Whether it reaches the last epoch judged, and so has no real end.
    """

    start: int
    end: int
    open_ended: bool


def runs(verdicts: Verdicts) -> list[Span]:
    """Find the runs of epochs judged unlike a running clock.

    Parameters
    ----------
    verdicts : Verdicts
        Verdicts by epoch, in epoch order.

    Returns
    -------
    list of Span
        Each run of true verdicts, from its first epoch to the epoch after
        its last; a run ends only at an epoch judged like a clock, so epochs
        no one judged, a gap in the readings among them, do not break it. A
        run that reaches the last epoch judged is open-ended.

    Examples
    --------
    >>> runs({1: False, 2: True, 3: True, 7: True, 8: False, 9: True})
    [Span(start=2, end=8, open_ended=False), Span(start=9, end=10, open_ended=True)]
    """
    found = []
    start: int | None = None
    last = 0
    for epoch, bad in verdicts.items():
        if bad:
            if start is None:
                start = epoch
            last = epoch
        elif start is not None:
            found.append(Span(start, last + 1, open_ended=False))
            start = None
    if start is not None:
        found.append(Span(start, last + 1, open_ended=True))
    return found


def joined(spans: Sequence[Span], min_epochs: int) -> list[Span]:
    """Join spans that overlap or lie fewer than ``min_epochs`` epochs apart.

    Parameters
    ----------
    spans : sequence of Span
        Spans in any order.
    min_epochs : int
        The fewest epochs between two spans that keeps them apart.

    Returns
    -------
    list of Span
        The spans in start order, each joined with every later one that
        starts fewer than ``min_epochs`` epochs after its end; a joined span
        is open-ended when any part of it is.

    Examples
    --------
    >>> joined([Span(20, 30, False), Span(1, 10, False), Span(12, 15, True)], 3)
    [Span(start=1, end=15, open_ended=True), Span(start=20, end=30, open_ended=False)]
    """
    result: list[Span] = []
    for span in sorted(spans):
        if result and span.start - result[-1].end < min_epochs:
            last = result[-1]
            result[-1] = Span(
                last.start, max(last.end, span.end), last.open_ended or span.open_ended
            )
        else:
            result.append(span)
    return result


def _departs(change: int, good: Sequence[int]) -> bool:
    """Tell whether a change strays from a clock's own changes.

    Parameters
    ----------
    change : int
        The change, ps.
    good : sequence of int
        Changes of the clock running properly, at least one.

    Returns
    -------
    bool
        Whether the change differs from their median by more than
        :data:`OUTLIER_SPREADS` robust spreads of them.

    Examples
    --------
    >>> _departs(500, [100, 110, 120]), _departs(115, [100, 110, 120])
    (True, False)
    """
    median_change, departure = spread(good)
    return abs(change - median_change) > OUTLIER_SPREADS * MAD_TO_SIGMA * departure


def first_bad(changes: Changes, run_start: int) -> int:
    """Find the epoch at which the clock stops running properly, near a run's start.

    The verdicts of a window are blurred by half its width, so the epoch
    is found again from the changes alone: the first one from HALF_WINDOW
    epochs before the run that strays from the clock's own changes in the
    window before that.

    Parameters
    ----------
    changes : Changes
        One pair's changes.
    run_start : int
        The first epoch of a run of bad verdicts.

    Returns
    -------
    int
        That epoch; ``run_start`` when the window before has fewer than
        :data:`MIN_WINDOW_CHANGES` changes, or no change strays.
    """
    edge = run_start - HALF_WINDOW
    good = [changes[e] for e in range(edge - 2 * HALF_WINDOW, edge) if e in changes]
    if len(good) < MIN_WINDOW_CHANGES:
        return run_start
    for epoch in range(edge, run_start + HALF_WINDOW + 1):
        if epoch in changes and _departs(changes[epoch], good):
            return epoch
    return run_start


def first_good(changes: Changes, run_end: int) -> int:
    """Find the epoch at which the clock runs properly again, near a run's end.

    The epoch sought is the one the last straying change ends at: the last
    change, up to HALF_WINDOW epochs after the run, that strays from the
    clock's own changes in the window after that. With no signal, that is
    the change into the first proper reading, which still starts from a
    bad one; when the clock lurched, it is the last lurch, and the clock
    runs properly from the reading it ends on.

    Parameters
    ----------
    changes : Changes
        One pair's changes.
    run_end : int
        The epoch after the last of a run of bad verdicts.

    Returns
    -------
    int
        That epoch; ``run_end`` when the window after has fewer than
        :data:`MIN_WINDOW_CHANGES` changes, or no change strays.
    """
    edge = run_end - 1 + HALF_WINDOW
    good = [
        changes[e] for e in range(edge + 1, edge + 2 * HALF_WINDOW + 1) if e in changes
    ]
    if len(good) < MIN_WINDOW_CHANGES:
        return run_end
    for epoch in range(edge, run_end - 1 - HALF_WINDOW - 1, -1):
        if epoch in changes and _departs(changes[epoch], good):
            return epoch
    return run_end


def reference_spread(per_reference: Sequence[Changes], span: Span) -> float | None:
    """Give how far the references' changes typically lie apart in a span.

    Parameters
    ----------
    per_reference : sequence of Changes
        Each reference's changes of the clock.
    span : Span
        The span.

    Returns
    -------
    float or None
        The median, over the span's epochs with at least
        :data:`MIN_COMPARED` changes, of their median absolute departure
        from their median, ps; ``None`` when no epoch has that many. Each
        change is first taken relative to the first, wrapped, so changes
        either side of the half period that lie close together stay close.

    Examples
    --------
    >>> one_signal = [{1: 100, 2: 50}, {1: 110, 2: 40}, {1: 90, 2: 60}]
    >>> reference_spread(one_signal, Span(1, 3, False))
    10.0
    >>> reference_spread(one_signal[:2], Span(1, 3, False)) is None
    True
    """
    epoch_spreads = []
    for epoch in range(span.start, span.end):
        at_epoch = [changes[epoch] for changes in per_reference if epoch in changes]
        if len(at_epoch) >= MIN_COMPARED:
            relative = [wrapped_change(c - at_epoch[0]) for c in at_epoch]
            epoch_spreads.append(spread(relative)[1])
    return statistics.median(epoch_spreads) if epoch_spreads else None


class BadStretch(NamedTuple):
    """One stretch when a clock is not running properly.

    Parameters
    ----------
    clock : str
        The clock.
    disabled_from : int
        The epoch of the first reading unlike a running clock: the clock is
        to be disabled from it.
    enabled_at : int or None
        The epoch of the first proper reading again: the clock is to be
        enabled at it; ``None`` when that never comes.
    kind : str or None
        :data:`NO_SIGNAL` when the references disagree by more than
        :data:`DISAGREEING_PS`, :data:`QUARTZ` when they agree better;
        ``None`` when they cannot be compared.
    references : tuple of str
        The references whose readings judged the clock.
    """

    clock: str
    disabled_from: int
    enabled_at: int | None
    kind: str | None
    references: tuple[str, ...]


def stretches(
    clock: str,
    per_reference: dict[str, Changes],
    min_epochs: int,
) -> list[BadStretch]:
    """Find a clock's stretches unlike a running clock that every reference agrees on.

    Parameters
    ----------
    clock : str
        The clock.
    per_reference : dict of str to Changes
        Each reference's changes of the clock.
    min_epochs : int
        The fewest epochs a stretch must cover, and the fewest between two
        that keeps them apart.

    Returns
    -------
    list of BadStretch
        The runs of agreed verdicts, joined when fewer than ``min_epochs``
        apart; their ends found again from every reference's changes (the
        earliest start and the latest end any gives) and joined again the
        same way; then each covering at least ``min_epochs`` epochs, an
        open-ended one up to the epoch after the last judged.
    """
    every_changes = list(per_reference.values())
    verdicts = agreed([judge(changes) for changes in every_changes])
    refined = [
        Span(
            min(first_bad(changes, run.start) for changes in every_changes),
            max(first_good(changes, run.end) for changes in every_changes),
            run.open_ended,
        )
        for run in joined(runs(verdicts), min_epochs)
    ]
    found = []
    for span in joined(refined, min_epochs):
        if span.end - span.start < min_epochs:
            continue
        typical = reference_spread(every_changes, span)
        kind = (
            None
            if typical is None
            else NO_SIGNAL
            if typical > DISAGREEING_PS
            else QUARTZ
        )
        found.append(
            BadStretch(
                clock,
                span.start,
                None if span.open_ended else span.end,
                kind,
                tuple(per_reference),
            )
        )
    return found


def clock_stretches(
    processed_path: Path,
    channel: RfChannel,
    clock: str,
    references: Sequence[str],
    min_epochs: int,
) -> list[BadStretch]:
    """Read one clock's pair files and find its stretches.

    Parameters
    ----------
    processed_path : Path
        The run's processed_path.
    channel : {'a', 'b'}
        The RF channel.
    clock : str
        The clock.
    references : sequence of str
        The references with a pair file for it.
    min_epochs : int
        The fewest epochs a stretch must cover.

    Returns
    -------
    list of BadStretch
        See :func:`stretches`.
    """
    per_reference = {
        reference: one_epoch_changes(
            read_phases(series_file(processed_path, channel, (reference, clock)))
        )
        for reference in references
    }
    return stretches(clock, per_reference, min_epochs)


def measured_clocks(
    processed_path: Path, channel: RfChannel, skipped: tuple[str, ...]
) -> dict[str, list[str]]:
    """Find each clock looked at, with the references measuring it.

    Parameters
    ----------
    processed_path : Path
        The run's processed_path.
    channel : {'a', 'b'}
        The RF channel.
    skipped : tuple of str
        Name prefixes of the clocks left alone.

    Returns
    -------
    dict of str to list of str
        Each clock with a pair file (r, c), r a reference, c neither a
        reference nor named with a prefix in ``skipped``, with those
        references, sorted.
    """
    found: dict[str, list[str]] = {}
    for data_file in sorted((processed_path / MEAS_SUBDIRECTORY).iterdir()):
        series_key = series_key_of(data_file.name, channel)
        if series_key is None or len(series_key) != 2:
            continue
        reference, clock = series_key
        if (
            is_reference(reference)
            and not is_reference(clock)
            and not clock.startswith(skipped)
        ):
            found.setdefault(clock, []).append(reference)
    return found


def effective_mjd(epoch: int) -> str:
    """Write an epoch's start as an MJD a clock-configuration entry can use.

    Parameters
    ----------
    epoch : int
        The epoch number.

    Returns
    -------
    str
        The epoch start's MJD to six decimals, rounded down, so the first
        epoch at or after it is this epoch.

    Examples
    --------
    >>> effective_mjd(8775540), effective_mjd(8775542)
    ('60941.250000', '60941.263888')
    """
    whole_days, epochs_into_day = divmod(epoch, EPOCHS_PER_DAY)
    micro_days = epochs_into_day * 1_000_000 // EPOCHS_PER_DAY
    return f"{whole_days}.{micro_days:06d}"


def format_stretch(stretch: BadStretch) -> str:
    """Write one stretch as one line of the report.

    Parameters
    ----------
    stretch : BadStretch
        The stretch.

    Returns
    -------
    str
        The clock, the disable MJD, the enable MJD, the hours between them,
        the kind and the references, separated by spaces; '-' for what is
        missing.

    Examples
    --------
    >>> ended = BadStretch("hm1", 8775540, 8775612, QUARTZ, ("mc1", "mc2"))
    >>> print(format_stretch(ended))
    hm1 60941.250000 60941.750000 12.0 quartz mc1,mc2
    >>> print(format_stretch(BadStretch("hm1", 8775540, None, None, ("mc1",))))
    hm1 60941.250000 - - - mc1
    """
    if stretch.enabled_at is None:
        enable, hours = "-", "-"
    else:
        enable = effective_mjd(stretch.enabled_at)
        hours = f"{(stretch.enabled_at - stretch.disabled_from) / EPOCHS_PER_HOUR:.1f}"
    return " ".join(
        [
            stretch.clock,
            effective_mjd(stretch.disabled_from),
            enable,
            hours,
            stretch.kind or "-",
            ",".join(stretch.references),
        ]
    )


REPORT_HEADER: Final[str] = "clock disabled_mjd enabled_mjd hours kind references"
"""The report's first line: what each field holds."""

type ClockJob = tuple[Path, RfChannel, str, tuple[str, ...], int]
"""The arguments of :func:`clock_stretches` for one clock."""


def _find_one(job: ClockJob) -> list[BadStretch]:
    """Find one clock's stretches, from one argument, for a pool of processes.

    Parameters
    ----------
    job : (Path, str, str, tuple of str, int)
        The arguments of :func:`clock_stretches`.

    Returns
    -------
    list of BadStretch
        The clock's stretches.
    """
    return clock_stretches(*job)


def main(argv: Sequence[str] | None = None) -> int:
    """Find every clock's stretches and print the report.

    Parameters
    ----------
    argv : sequence of str or None, optional
        The command-line arguments; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("processed_path", type=Path)
    parser.add_argument("--rf", choices=RF_CHOICES, required=True)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="name prefixes of the clocks left alone",
    )
    parser.add_argument("--jobs", type=int, default=1, help="clocks looked at at once")
    parser.add_argument(
        "--min-hours",
        type=float,
        default=MIN_HOURS,
        help="the shortest stretch, and the shortest time that keeps two apart",
    )
    cli_options = parser.parse_args(argv)
    min_epochs = math.ceil(cli_options.min_hours * EPOCHS_PER_HOUR)
    clocks = measured_clocks(
        cli_options.processed_path, cli_options.rf, tuple(cli_options.skip)
    )
    jobs = [
        (
            cli_options.processed_path,
            cli_options.rf,
            clock,
            tuple(references),
            min_epochs,
        )
        for clock, references in sorted(clocks.items())
    ]
    print(REPORT_HEADER)
    found: Iterator[list[BadStretch]]
    if cli_options.jobs == 1:
        found = map(_find_one, jobs)
        for clock_found in found:
            for stretch in clock_found:
                print(format_stretch(stretch), flush=True)
        return 0
    with ProcessPoolExecutor(max_workers=cli_options.jobs) as pool:
        for clock_found in pool.map(_find_one, jobs):
            for stretch in clock_found:
                print(format_stretch(stretch), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
