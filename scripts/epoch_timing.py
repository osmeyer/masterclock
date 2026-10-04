"""Time das_processor on an invented deployment, against the 600 s of an epoch.

The script builds a deployment of invented DAS files and clock configuration
in a new folder: R references, each measured against itself and every other
reference, and C clocks, each measured against one reference, taken in turn.
That gives R * R + C pair files and R * (R * R + C) triple files, one
triple for each pair through each reference. No reference steers.

It then runs das_processor as the scheduler does, one process per epoch,
for the first epochs of a day whose DAS file is already whole, so each run
reads the whole day's file; and, on a copy of the same data, one process for
every epoch of the data. Give it a folder that does not exist yet or is empty::

    uv run --frozen python scripts/epoch_timing.py FOLDER --references 3 --clocks 20

It prints how long the first one-epoch run took and the median and longest
of the others, and the batch run's time in total and per epoch; each time
but the batch total also as a share of an epoch's 600 s. A run that fails
is reported with its error output and exit status 1.
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
from masterclock.das_processor.read_cd5m5m import SKIPPED_REFERENCES, DASMeasurement
from masterclock.domain.phase import PHASE_PERIOD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DATA_START: Final = datetime(2025, 9, 23, tzinfo=UTC)
"""The invented day the data cover, from its first epoch."""

EPOCH_LENGTH: Final = timedelta(minutes=10)
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

REFERENCE_NAMES: Final = tuple(
    reference_name
    for reference_name in (f"mc{digit}" for digit in range(10))
    if reference_name not in SKIPPED_REFERENCES
)
"""The references a switch position can name, one digit, that das_processor reads."""

MAX_REFERENCES: Final = len(REFERENCE_NAMES)
"""How many references a deployment can have."""

DAS_PROCESSOR_COMMAND: Final = (sys.executable, "-m", "masterclock.das_processor")
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


def measured_pairs_for(references: int, clocks: int) -> list[tuple[str, str]]:
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
    reference_names = list(REFERENCE_NAMES[:references])
    measured_pairs = [(r, s) for r in reference_names for s in reference_names]
    measured_pairs += [
        (reference_names[clock_index % references], f"hm{clock_index:04d}")
        for clock_index in range(clocks)
    ]
    return measured_pairs


def build_deployment(
    folder: Path, references: int, clocks: int, epochs: int
) -> list[str]:
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
        How many epochs of data, from the start of :data:`DATA_START`.

    Returns
    -------
    list of str
        The das_processor arguments for the deployment, with logging off.
    """
    for subfolder in ("das", "steering", "processed"):
        (folder / subfolder).mkdir(parents=True)
    measured_pairs = measured_pairs_for(references, clocks)
    clock_names = sorted({clock for _, clock in measured_pairs})
    (folder / "clock_config.yaml").write_text(
        CLOCK_TYPES
        + "".join(
            f"  {clock}: [{{type: {'mc' if clock.startswith('mc') else 'maser'},"
            " location: 1}]\n"
            for clock in clock_names
        ),
        encoding="utf-8",
    )
    day_lines: dict[int, list[str]] = {}
    for epoch_index in range(epochs):
        epoch_start = DATA_START + epoch_index * EPOCH_LENGTH
        for pair_slot, (reference, clock) in enumerate(measured_pairs):
            measurement_offset = timedelta(
                seconds=1 + SPAN * pair_slot / len(measured_pairs)
            )
            seconds_into_day = int(
                (epoch_start + measurement_offset - DATA_START).total_seconds()
            )
            das_measurement = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(epoch_start + measurement_offset), 6
                ),
                measured_phase=(
                    1_000 * pair_slot + (pair_slot % 7 - 3) * seconds_into_day // 100
                )
                % PHASE_PERIOD,
                rms=3,
                switch=f"{reference[-1]}A{pair_slot % 100:02d}",
                clock=clock,
            )
            day_lines.setdefault(int(datetime_to_mjd(epoch_start)), []).append(
                f"{das_measurement}\n"
            )
    for mjd_day, day_file_lines in day_lines.items():
        (folder / "das" / f"cd5m5m_{mjd_day}.dat").write_text("".join(day_file_lines))
    return [
        "--rf", "a",
        "--cd5m5m-path", str(folder / "das"),
        "--steering-path", str(folder / "steering"),
        "--processed-path", str(folder / "processed"),
        "--clock-config-file", str(folder / "clock_config.yaml"),
        "--start-from-mjd", f"{datetime_to_mjd(DATA_START):.6f}",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
    ]  # fmt: skip


class RunFailedError(Exception):
    """A timed run of das_processor exited with a failure."""


def timed_run(das_arguments: Sequence[str]) -> float:
    """Run das_processor once and give how long it took.

    Parameters
    ----------
    das_arguments : Sequence of str
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
    started = time.perf_counter()
    # The command is das_processor itself, with arguments this script made.
    finished_run = subprocess.run(  # noqa: S603  # nosec B603
        [*DAS_PROCESSOR_COMMAND, *das_arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    run_seconds = time.perf_counter() - started
    if finished_run.returncode != 0:
        failure = (
            f"exit status {finished_run.returncode}: {finished_run.stderr.strip()}"
        )
        raise RunFailedError(failure)
    return run_seconds


def with_epoch_share(run_seconds: float) -> str:
    """Give a time with its share of an epoch.

    Parameters
    ----------
    run_seconds : float
        The time, in s.

    Returns
    -------
    str
        For example ``"0.512 s, 0.085% of 600 s"``.
    """
    return f"{run_seconds:.3f} s, {100 * run_seconds / EPOCH_SECONDS:.3f}% of 600 s"


def timing_report(cli_options: argparse.Namespace) -> list[str]:
    """Build both deployments, time their runs, and give the lines to print.

    Parameters
    ----------
    cli_options : argparse.Namespace
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
    references, clocks, epochs = (
        cli_options.references,
        cli_options.clocks,
        cli_options.epochs,
    )
    stepped_arguments = build_deployment(
        cli_options.folder / "stepped", references, clocks, epochs
    )
    batch_arguments = build_deployment(
        cli_options.folder / "batch", references, clocks, epochs
    )
    report_lines = [
        f"deployment: {references} references, {clocks} clocks, {epochs} epochs:"
        f" {references * references + clocks} pair files,"
        f" {references * (references * references + clocks)} triple files"
    ]
    one_epoch_times = [
        timed_run([*stepped_arguments, "--steps", "1"]) for _ in range(cli_options.runs)
    ]
    report_lines.append(
        "run of one epoch, first, which creates the files:"
        f" {with_epoch_share(one_epoch_times[0])}"
    )
    if len(one_epoch_times) > 1:
        later_times = one_epoch_times[1:]
        report_lines.append(
            f"run of one epoch, median of the next {len(later_times)}:"
            f" {with_epoch_share(statistics.median(later_times))}"
        )
        report_lines.append(
            f"run of one epoch, longest of the next {len(later_times)}:"
            f" {with_epoch_share(max(later_times))}"
        )
    batch_seconds = timed_run(batch_arguments)
    report_lines.append(f"batch run of {epochs} epochs: {batch_seconds:.3f} s")
    report_lines.append(
        f"batch run, per epoch: {with_epoch_share(batch_seconds / epochs)}"
    )
    return report_lines


def _new_folder(cli_argument: str) -> Path:
    """Convert a command-line argument to a folder that is missing or empty."""
    new_folder = Path(cli_argument)
    if new_folder.exists() and (not new_folder.is_dir() or any(new_folder.iterdir())):
        refusal = f"not a new or empty folder: {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return new_folder


def _count(lowest: int, highest: int | None = None) -> Callable[[str], int]:
    """Give an argument type for a whole number from ``lowest`` to ``highest``."""

    def read_count(cli_argument: str) -> int:
        """Convert the argument, refusing a number out of range."""
        try:
            count = int(cli_argument)
        except ValueError:
            count = lowest - 1
        if count < lowest or (highest is not None and count > highest):
            range_text = f"from {lowest}" + (
                "" if highest is None else f" to {highest}"
            )
            refusal = f"not a whole number {range_text}: {cli_argument}"
            raise argparse.ArgumentTypeError(refusal)
        return count

    return read_count


def main(argv: Sequence[str] | None = None) -> int:
    """Time das_processor on an invented deployment and give the exit status.

    Parameters
    ----------
    argv : Sequence of str or None, optional
        The arguments, without the program name; ``None`` for
        ``sys.argv[1:]``.

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
    cli_options = parser.parse_args(argv)
    if cli_options.runs > cli_options.epochs:
        parser.error("--runs may not be more than --epochs")
    try:
        report_lines = timing_report(cli_options)
    except RunFailedError as exc:
        print(f"epoch_timing: a run failed: {exc}", file=sys.stderr)
        return 1
    for report_line in report_lines:
        print(report_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
