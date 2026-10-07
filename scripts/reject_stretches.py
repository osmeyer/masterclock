"""Find the stretches when a clock is rejected again and again (design 15.5).

A clock whose frequency runs away for a few days, or whose readings turn
noisy, is rejected by das_processor at epoch after epoch, and its series
start afresh and go dormant over and over. This script reads the log of a
das_processor run and, for every clock that is not a reference, finds the
stretches where its rejections come thick: rejections closer together
than the shortest stretch reported are joined, and a stretch is kept when
it lasts at least that long and a large enough share of its epochs hold a
rejection. A stretch is then joined with each of the clock's disabled
stretches in the clock configuration that lies within the same distance,
so a fault that has already been disabled in pieces comes out as one
stretch. Each is printed with the MJD from which the clock is to be
disabled and the MJD at which it is to be enabled again, ready to be
looked over and put into the clock configuration in place of the disabled
stretches it takes in. Run it as::

    uv run --frozen python scripts/reject_stretches.py LOG --rf a --clock-config FILE

When many clocks are rejected at the same epochs, the fault is a
reference's, not theirs: the rejections at such an epoch are left out of
every clock's stretches, and the epochs are listed on their own, with the
reference most of their rejections go through.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, NamedTuple

from characterize import EPOCHS_PER_DAY
from masterclock.das_processor.channels import RF_CHOICES, RfChannel
from masterclock.das_processor.clock_config import ClockConfig, read_clock_config
from masterclock.domain.phase import EPOCH_SECONDS
from masterclock.domain.references import is_reference
from no_signal import EPOCHS_PER_HOUR, Span, effective_mjd, joined

MIN_HOURS: Final[float] = 12.0
"""The shortest stretch reported, and the shortest time between two that keeps
them apart, hours, unless the command line says otherwise."""

MIN_SHARE: Final[float] = 0.3
"""The smallest share of a stretch's epochs that must hold a rejection, unless
the command line says otherwise: below it the rejections are scattered, as
those of a clock running properly are, rather than those of a fault."""

SHARED_CLOCKS: Final[int] = 4
"""The fewest clocks rejected at one epoch that make the epoch's rejections a
reference's fault, unless the command line says otherwise."""

UNIX_EPOCH_MJD: Final[int] = 40_587
"""The MJD of 1970-01-01, where Unix time starts."""

REJECTED_AT: Final[str] = " rejected at "
"""What a log line about a rejected reading says between its series and epoch."""


class Rejection(NamedTuple):
    """One rejected reading, from the log.

    Parameters
    ----------
    epoch : int
        The epoch number: the epoch start's MJD times the epochs in a day.
    clock : str
        The clock the series measures, the last name in its key.
    references : tuple of str
        The other names in the series' key: the references it goes through.
    """

    epoch: int
    clock: str
    references: tuple[str, ...]


def epoch_of(epoch_text: str) -> int:
    """Give the epoch number of an epoch start as the log writes it.

    Parameters
    ----------
    epoch_text : str
        The epoch start, ISO 8601 with its UTC offset.

    Returns
    -------
    int
        The epoch number.

    Examples
    --------
    >>> epoch_of("2025-09-28 06:00:00+00:00") == round(60946.25 * EPOCHS_PER_DAY)
    True
    """
    unix_seconds = int(datetime.fromisoformat(epoch_text).timestamp())
    return unix_seconds // EPOCH_SECONDS + UNIX_EPOCH_MJD * EPOCHS_PER_DAY


def parse_rejection(log_line: str, channel: RfChannel) -> Rejection | None:
    """Read a rejection from a log line.

    Parameters
    ----------
    log_line : str
        One line of the log.
    channel : {'a', 'b'}
        The RF channel looked at.

    Returns
    -------
    Rejection or None
        The rejection the line reports; ``None`` when it reports none, or
        one of another channel.

    Examples
    --------
    >>> line = "... | run: das_a.mc2.hm1 rejected at 2025-09-28 06:00:00+00:00: x"
    >>> parse_rejection(line, "a")[1:]
    ('hm1', ('mc2',))
    >>> parse_rejection(line, "b") is None
    True
    """
    before, found, after = log_line.partition(REJECTED_AT)
    if not found:
        return None
    channel_name, *series_key = before.rsplit(" ", 1)[-1].split(".")
    if channel_name != f"das_{channel}":
        return None
    epoch_text = after.split(": ", 1)[0]
    return Rejection(epoch_of(epoch_text), series_key[-1], tuple(series_key[:-1]))


def read_rejections(log_file: Path, channel: RfChannel) -> list[Rejection]:
    """Read every rejection of a clock that is not a reference from a log.

    Parameters
    ----------
    log_file : Path
        The run's log.
    channel : {'a', 'b'}
        The RF channel looked at.

    Returns
    -------
    list of Rejection
        The rejections, in the log's order.
    """
    found = []
    with log_file.open(encoding="utf-8") as open_file:
        for log_line in open_file:
            rejection = parse_rejection(log_line, channel)
            if rejection is not None and not is_reference(rejection.clock):
                found.append(rejection)
    return found


def shared_epochs(rejections: Iterable[Rejection], shared_clocks: int) -> set[int]:
    """Find the epochs with so many clocks rejected that the fault is a reference's.

    Parameters
    ----------
    rejections : iterable of Rejection
        The rejections.
    shared_clocks : int
        The fewest clocks that make an epoch's rejections shared.

    Returns
    -------
    set of int
        The epochs at which at least ``shared_clocks`` clocks are rejected.

    Examples
    --------
    >>> seen = [Rejection(1, c, ("mc1",)) for c in "abc"] + [Rejection(2, "a", ())]
    >>> shared_epochs(seen, 3), shared_epochs(seen, 4)
    ({1}, set())
    """
    clocks_at: dict[int, set[str]] = {}
    for rejection in rejections:
        clocks_at.setdefault(rejection.epoch, set()).add(rejection.clock)
    return {
        epoch for epoch, clocks in clocks_at.items() if len(clocks) >= shared_clocks
    }


class Burst(NamedTuple):
    """A stretch of thick rejections of one clock.

    Parameters
    ----------
    span : Span
        From its first rejected epoch to the epoch after its last.
    rejections : int
        The rejections in it.
    rejected_epochs : int
        The epochs in it with a rejection.
    """

    span: Span
    rejections: int
    rejected_epochs: int


def bursts(counts: Counter[int], min_epochs: int, min_share: float) -> list[Burst]:
    """Find the stretches of one clock's thick rejections.

    Parameters
    ----------
    counts : Counter of int
        The clock's rejections at each epoch, shared epochs left out.
    min_epochs : int
        The fewest epochs a stretch must cover, and the fewest between two
        that keeps them apart.
    min_share : float
        The smallest share of a stretch's epochs that must hold a rejection.

    Returns
    -------
    list of Burst
        The rejected epochs joined when fewer than ``min_epochs`` apart,
        each stretch covering at least ``min_epochs`` epochs with at least
        ``min_share`` of them rejected, in time order.

    Examples
    --------
    >>> thick = Counter({e: 2 for e in range(10, 20, 2)}) + Counter({40: 1})
    >>> [(*b.span[:2], b.rejections, b.rejected_epochs) for b in bursts(thick, 5, 0.5)]
    [(10, 19, 10, 5)]
    >>> bursts(thick, 5, 0.6), bursts(thick, 10, 0.5)
    ([], [])
    """
    found = []
    single_epochs = [Span(epoch, epoch + 1, open_ended=False) for epoch in counts]
    for span in joined(single_epochs, min_epochs):
        inside = [epoch for epoch in counts if span.start <= epoch < span.end]
        length = span.end - span.start
        if length >= min_epochs and len(inside) >= min_share * length:
            found.append(Burst(span, sum(counts[e] for e in inside), len(inside)))
    return found


def disabled_spans(clock_config: ClockConfig, clock: str, data_end: int) -> list[Span]:
    """Give the stretches a clock is disabled for in the clock configuration.

    Parameters
    ----------
    clock_config : ClockConfig
        The clock configuration.
    clock : str
        The clock.
    data_end : int
        The epoch after the last looked at: the end of a stretch the clock
        is never enabled again after.

    Returns
    -------
    list of Span
        From each epoch the clock is disabled from to the epoch it is
        enabled again at, in time order; one never enabled again is
        open-ended. None for a clock the configuration does not name.
    """
    found = []
    disabled_from: int | None = None
    entries = sorted(
        clock_config.clocks.get(clock, ()),
        key=lambda entry: (entry.effective_mjd is not None, entry.effective_mjd or 0.0),
    )
    for entry in entries:
        disabled = entry.overrides().get("disabled")
        epoch = round((entry.effective_mjd or 0.0) * EPOCHS_PER_DAY)
        if disabled is True and disabled_from is None:
            disabled_from = epoch
        elif disabled is False and disabled_from is not None:
            found.append(Span(disabled_from, epoch, open_ended=False))
            disabled_from = None
    if disabled_from is not None:
        found.append(
            Span(disabled_from, max(data_end, disabled_from + 1), open_ended=True)
        )
    return found


class RejectStretch(NamedTuple):
    """One stretch a clock is to be disabled for.

    Parameters
    ----------
    clock : str
        The clock.
    span : Span
        The stretch.
    rejections : int
        The rejections in its bursts.
    rejected_epochs : int
        The epochs in its bursts with a rejection.
    replaced : int
        The disabled stretches of the clock configuration it takes in.
    """

    clock: str
    span: Span
    rejections: int
    rejected_epochs: int
    replaced: int


def reject_stretches(
    clock: str,
    clock_bursts: Sequence[Burst],
    disabled: Sequence[Span],
    min_epochs: int,
) -> list[RejectStretch]:
    """Join a clock's bursts with its disabled stretches.

    Parameters
    ----------
    clock : str
        The clock.
    clock_bursts : sequence of Burst
        Its bursts.
    disabled : sequence of Span
        Its disabled stretches.
    min_epochs : int
        The fewest epochs between two stretches that keeps them apart.

    Returns
    -------
    list of RejectStretch
        The bursts and disabled stretches joined when fewer than
        ``min_epochs`` apart, each that takes in at least one burst, in
        time order.

    Examples
    --------
    >>> one = Burst(Span(10, 20, False), 30, 10)
    >>> disabled = [Span(22, 30, False), Span(90, 99, True)]
    >>> found = reject_stretches("hm1", [one], disabled, 5)
    >>> [(s.span, s.replaced) for s in found]
    [(Span(start=10, end=30, open_ended=False), 1)]
    """
    found = []
    for span in joined([*(b.span for b in clock_bursts), *disabled], min_epochs):
        inside = [b for b in clock_bursts if span.start <= b.span.start < span.end]
        if inside:
            found.append(
                RejectStretch(
                    clock,
                    span,
                    sum(b.rejections for b in inside),
                    sum(b.rejected_epochs for b in inside),
                    sum(span.start <= d.start < span.end for d in disabled),
                )
            )
    return found


class SharedStretch(NamedTuple):
    """A stretch of epochs whose rejections are a reference's fault.

    Parameters
    ----------
    span : Span
        From its first shared epoch to the epoch after its last.
    most_clocks : int
        The most clocks rejected at one of its epochs.
    reference : str
        The reference the most of its rejections go through.
    through_reference : int
        The rejections that go through it.
    rejections : int
        All its rejections.
    """

    span: Span
    most_clocks: int
    reference: str
    through_reference: int
    rejections: int


def shared_stretches(
    rejections: Sequence[Rejection], shared: set[int], min_epochs: int
) -> list[SharedStretch]:
    """Join the shared epochs into stretches and name the reference behind each.

    Parameters
    ----------
    rejections : sequence of Rejection
        Every rejection.
    shared : set of int
        The shared epochs.
    min_epochs : int
        The fewest epochs between two stretches that keeps them apart.

    Returns
    -------
    list of SharedStretch
        The shared epochs joined when fewer than ``min_epochs`` apart, in
        time order, each with the reference named most often by its
        rejections.

    Examples
    --------
    >>> seen = [Rejection(5, c, ("mc1", "mc3")) for c in "ab"]
    >>> seen += [Rejection(6, "c", ("mc3",))]
    >>> shared_stretches(seen, {5, 6}, 3)[0][1:]
    (2, 'mc3', 3, 3)
    """
    found = []
    single_epochs = [Span(epoch, epoch + 1, open_ended=False) for epoch in shared]
    for span in joined(single_epochs, min_epochs):
        inside = [r for r in rejections if span.start <= r.epoch < span.end]
        clocks_at: dict[int, set[str]] = {}
        through: Counter[str] = Counter()
        for rejection in inside:
            clocks_at.setdefault(rejection.epoch, set()).add(rejection.clock)
            through.update(set(rejection.references))
        reference, through_reference = through.most_common(1)[0]
        most_clocks = max(len(clocks) for clocks in clocks_at.values())
        found.append(
            SharedStretch(span, most_clocks, reference, through_reference, len(inside))
        )
    return found


def _hours(span: Span) -> str:
    """Write the hours a span lasts, one decimal.

    Parameters
    ----------
    span : Span
        The span.

    Returns
    -------
    str
        Its hours.

    Examples
    --------
    >>> _hours(Span(0, 9, False))
    '1.5'
    """
    return f"{(span.end - span.start) / EPOCHS_PER_HOUR:.1f}"


REPORT_HEADER: Final[str] = (
    "clock disabled_mjd enabled_mjd hours rejections epochs_with_rejection"
    " disabled_stretches_taken_in"
)
"""The first line of the clocks' report: what each field holds."""

SHARED_HEADER: Final[str] = (
    "shared_from_mjd shared_until_mjd hours most_clocks_at_one_epoch reference"
    " rejections_through_reference rejections"
)
"""The first line of the shared stretches' report: what each field holds."""


def format_stretch(stretch: RejectStretch) -> str:
    """Write one clock's stretch as one line of the report.

    Parameters
    ----------
    stretch : RejectStretch
        The stretch.

    Returns
    -------
    str
        The fields of :data:`REPORT_HEADER`, separated by spaces; '-' for
        the enable MJD and hours of a stretch never enabled again.

    Examples
    --------
    >>> ended = RejectStretch("hm1", Span(8775540, 8775612, False), 90, 60, 2)
    >>> print(format_stretch(ended))
    hm1 60941.250000 60941.750000 12.0 90 60 2
    >>> never = ended._replace(span=Span(8775540, 8775612, True), replaced=0)
    >>> print(format_stretch(never))
    hm1 60941.250000 - - 90 60 0
    """
    span = stretch.span
    enable, hours = (
        ("-", "-") if span.open_ended else (effective_mjd(span.end), _hours(span))
    )
    return " ".join(
        [
            stretch.clock,
            effective_mjd(span.start),
            enable,
            hours,
            str(stretch.rejections),
            str(stretch.rejected_epochs),
            str(stretch.replaced),
        ]
    )


def format_shared(stretch: SharedStretch) -> str:
    """Write one shared stretch as one line of the report.

    Parameters
    ----------
    stretch : SharedStretch
        The stretch.

    Returns
    -------
    str
        The fields of :data:`SHARED_HEADER`, separated by spaces.

    Examples
    --------
    >>> shared = SharedStretch(Span(8775540, 8775546, False), 40, "mc3", 90, 120)
    >>> print(format_shared(shared))
    60941.250000 60941.291666 1.0 40 mc3 90 120
    """
    return " ".join(
        [
            effective_mjd(stretch.span.start),
            effective_mjd(stretch.span.end),
            _hours(stretch.span),
            str(stretch.most_clocks),
            stretch.reference,
            str(stretch.through_reference),
            str(stretch.rejections),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Find every clock's stretches and the shared ones, and print the report.

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
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--rf", choices=RF_CHOICES, required=True)
    parser.add_argument(
        "--clock-config",
        type=Path,
        required=True,
        help="the clock configuration whose disabled stretches are joined in",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=MIN_HOURS,
        help="the shortest stretch, and the shortest time that keeps two apart",
    )
    parser.add_argument(
        "--min-share",
        type=float,
        default=MIN_SHARE,
        help="the smallest share of a stretch's epochs holding a rejection",
    )
    parser.add_argument(
        "--shared-clocks",
        type=int,
        default=SHARED_CLOCKS,
        help="the fewest clocks rejected at one epoch that make it a reference's fault",
    )
    cli_options = parser.parse_args(argv)
    min_epochs = round(cli_options.min_hours * EPOCHS_PER_HOUR)
    clock_config = read_clock_config(cli_options.clock_config)
    rejections = read_rejections(cli_options.log_file, cli_options.rf)
    shared = shared_epochs(rejections, cli_options.shared_clocks)
    data_end = max((r.epoch for r in rejections), default=0) + 1
    counts: dict[str, Counter[int]] = {}
    for rejection in rejections:
        if rejection.epoch not in shared:
            counts.setdefault(rejection.clock, Counter())[rejection.epoch] += 1
    print(REPORT_HEADER)
    for clock, clock_counts in sorted(counts.items()):
        found = reject_stretches(
            clock,
            bursts(clock_counts, min_epochs, cli_options.min_share),
            disabled_spans(clock_config, clock, data_end),
            min_epochs,
        )
        for stretch in found:
            print(format_stretch(stretch))
    print(SHARED_HEADER)
    for shared_stretch in shared_stretches(rejections, shared, min_epochs):
        print(format_shared(shared_stretch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
