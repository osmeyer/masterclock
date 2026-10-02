"""Time das_processor on an invented deployment, against the 600 s of an epoch.

The script builds a deployment of invented DAS files and clock configuration
in a new folder: R references, each measured against itself and every other
reference, and C clocks, each measured against one reference, taken in turn.
That gives R * R + C pair files and R * C triple files, one triple for each
clock through each reference. No reference steers.

It then runs das_processor as the scheduler does, one process per epoch,
for the first epochs of a day whose DAS file is already whole, so each run
reads the whole day's file; and, on a copy of the same data, one process for
the whole day. Give it a folder that does not exist yet or is empty::

    uv run --frozen python scripts/epoch_timing.py FOLDER --references 3 --clocks 20

It prints how long each run took and what share of an epoch's 600 s that
is. A run that fails is reported with its error output and exit status 1.
"""

import argparse
import statistics
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain.phase import PHASE_PERIOD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DAY: Final = datetime(2025, 9, 23, tzinfo=UTC)
"""The invented day the data cover, from its first epoch."""

EPOCH: Final = timedelta(minutes=10)
"""One epoch."""

EPOCH_SECONDS: Final = 600.0
"""The time between two runs of the scheduler, which a run must stay within."""

EPOCHS_PER_DAY: Final = 144
"""How many epochs a day holds."""

SPAN: Final = 580
"""The seconds of each epoch the invented measurements are spread over.

They start one second after the epoch begins and end well before the last
ten seconds, where the DAS reader refuses a measurement.
"""

MAX_REFERENCES: Final = 10
"""How many references a switch position can name: one digit."""

COMMAND: Final = (sys.executable, "-m", "masterclock.das_processor")
"""The program timed."""

CLOCK_TYPES: Final = (
    "rejects_before_restart: 4\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 10.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  mc: {filter_states: 1, scale_time_constant: 5.0,"
    " initial_innovation_scale: 3.0, gap_limit: 8}\n"
    "clocks:\n"
)
"""The invented clock configuration, before its list of clocks."""


def pairs_of(references: int, clocks: int) -> list[tuple[str, str]]:
    """Give every pair measured, references first, each clock after its reference's.

    Parameters
    ----------
    references : int
        How many references, from 1 to :data:`MAX_REFERENCES`.
    clocks : int
        How many other clocks.

    Returns
    -------
    list of (str, str)
        Each (reference, clock): ``mc<i>`` against every reference, then
        ``hm<n>`` against reference ``n mod references``.
    """
    names = [f"mc{i}" for i in range(references)]
    pairs = [(r, s) for r in names for s in names]
    pairs += [(names[n % references], f"hm{n:04d}") for n in range(clocks)]
    return pairs


def build(folder: Path, references: int, clocks: int, epochs: int) -> list[str]:
    """Write a deployment's input files in a new folder, and give its arguments.

    Parameters
    ----------
    folder : Path
        Where to write it; made here.
    references : int
        How many references.
    clocks : int
        How many other clocks.
    epochs : int
        How many epochs of data, from the start of :data:`DAY`.

    Returns
    -------
    list of str
        The das_processor arguments for the deployment, with logging off.
    """
    for name in ("das", "steering", "processed"):
        (folder / name).mkdir(parents=True)
    pairs = pairs_of(references, clocks)
    entries = sorted({clock for _, clock in pairs})
    (folder / "clock_config.yaml").write_text(
        CLOCK_TYPES
        + "".join(
            f"  {name}: [{{type: {'mc' if name.startswith('mc') else 'maser'}}}]\n"
            for name in entries
        ),
        encoding="utf-8",
    )
    days: dict[int, list[str]] = {}
    for index in range(epochs):
        mark = DAY + index * EPOCH
        for slot, (reference, clock) in enumerate(pairs):
            offset = timedelta(seconds=1 + SPAN * slot / len(pairs))
            seconds = int((mark + offset - DAY).total_seconds())
            line = DASMeasurement(
                measurement_mjd=round(datetime_to_mjd(mark + offset), 6),
                measured_phase=(1_000 * slot + (slot % 7 - 3) * seconds // 100)
                % PHASE_PERIOD,
                rms=3,
                switch=f"{reference[-1]}A{slot % 100:02d}",
                clock=clock,
            )
            days.setdefault(int(datetime_to_mjd(mark)), []).append(f"{line}\n")
    for day, lines in days.items():
        (folder / "das" / f"cd5m5m_{day}.dat").write_text("".join(lines))
    return [
        "--rf", "a",
        "--cd5m5m-path", str(folder / "das"),
        "--steering-path", str(folder / "steering"),
        "--processed-path", str(folder / "processed"),
        "--clock-config-file", str(folder / "clock_config.yaml"),
        "--start-from-mjd", f"{datetime_to_mjd(DAY):.6f}",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
    ]  # fmt: skip


class RunFailedError(Exception):
    """A timed run of das_processor exited with a failure."""


def timed(arguments: Sequence[str]) -> float:
    """Run das_processor once and give how long it took.

    Parameters
    ----------
    arguments : Sequence of str
        Its arguments.

    Returns
    -------
    float
        The wall-clock time of the run, process start-up included, in s.

    Raises
    ------
    RunFailedError
        If it exits with a failure; the message holds its error output.
    """
    began = time.perf_counter()
    # The command is das_processor itself, with arguments this script made.
    done = subprocess.run(  # noqa: S603  # nosec B603
        [*COMMAND, *arguments], capture_output=True, text=True, check=False
    )
    took = time.perf_counter() - began
    if done.returncode != 0:
        message = f"exit status {done.returncode}: {done.stderr.strip()}"
        raise RunFailedError(message)
    return took


def share(seconds: float) -> str:
    """Give a time with its share of an epoch.

    Parameters
    ----------
    seconds : float
        The time, in s.

    Returns
    -------
    str
        For example ``"0.512 s, 0.085% of 600 s"``.
    """
    return f"{seconds:.3f} s, {100 * seconds / EPOCH_SECONDS:.3f}% of 600 s"


def report(options: argparse.Namespace) -> list[str]:
    """Build both deployments, time their runs, and give the lines to print.

    Parameters
    ----------
    options : argparse.Namespace
        The parsed command line.

    Returns
    -------
    list of str
        What the deployment holds and how long each run took.

    Raises
    ------
    RunFailedError
        If a run fails.
    """
    references, clocks, epochs = options.references, options.clocks, options.epochs
    stepped = build(options.folder / "stepped", references, clocks, epochs)
    batch = build(options.folder / "batch", references, clocks, epochs)
    lines = [
        f"deployment: {references} references, {clocks} clocks, {epochs} epochs:"
        f" {references * references + clocks} pair files,"
        f" {references * clocks} triple files"
    ]
    runs = [timed([*stepped, "--steps", "1"]) for _ in range(options.runs)]
    lines.append(f"run of one epoch, first, which creates the files: {share(runs[0])}")
    if len(runs) > 1:
        rest = runs[1:]
        lines.append(f"run of one epoch, median of the next {len(rest)}:"
                     f" {share(statistics.median(rest))}")  # fmt: skip
        lines.append(f"run of one epoch, longest of the next {len(rest)}:"
                     f" {share(max(rest))}")  # fmt: skip
    whole = timed(batch)
    lines.append(f"batch run of {epochs} epochs: {whole:.3f} s")
    lines.append(f"batch run, per epoch: {share(whole / epochs)}")
    return lines


def _new_folder(text: str) -> Path:
    """Convert a command-line argument to a folder that is missing or empty."""
    path = Path(text)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        message = f"not a new or empty folder: {text}"
        raise argparse.ArgumentTypeError(message)
    return path


def _count(low: int, high: int | None = None) -> Callable[[str], int]:
    """Give an argument type for a whole number from ``low`` to ``high``."""

    def check(text: str) -> int:
        """Convert the argument, refusing a number out of range."""
        try:
            number = int(text)
        except ValueError:
            number = low - 1
        if number < low or (high is not None and number > high):
            limit = f"from {low}" + ("" if high is None else f" to {high}")
            message = f"not a whole number {limit}: {text}"
            raise argparse.ArgumentTypeError(message)
        return number

    return check


def main(argv: Sequence[str] | None = None) -> int:
    """Time das_processor on an invented deployment and give the exit status.

    Parameters
    ----------
    argv : Sequence of str or None, optional
        The arguments, without the program name; ``None`` for
        :data:`sys.argv`.

    Returns
    -------
    int
        0 when every run finished, 1 when one failed.
    """
    parser = argparse.ArgumentParser(
        description="Time das_processor on an invented deployment."
    )
    parser.add_argument("folder", type=_new_folder, help="a new or empty folder")
    parser.add_argument("--references", type=_count(1, MAX_REFERENCES), default=3)
    parser.add_argument("--clocks", type=_count(0), default=20)
    parser.add_argument(
        "--epochs",
        type=_count(1, EPOCHS_PER_DAY),
        default=EPOCHS_PER_DAY,
        help="epochs of data, from the start of the day (default: the whole day)",
    )
    parser.add_argument(
        "--runs", type=_count(1), default=6, help="runs of one epoch to time"
    )
    options = parser.parse_args(argv)
    if options.runs > options.epochs:
        parser.error("--runs may not be more than --epochs")
    try:
        lines = report(options)
    except RunFailedError as exc:
        print(f"epoch_timing: a run failed: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
