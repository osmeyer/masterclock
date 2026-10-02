"""Tests that das_processor's data files depend only on its inputs (I5).

The rules covered: a run of every epoch in one batch and a run restarted
after every epoch give byte-identical data files, for random small
deployments of invented DAS and steering files, with gaps, steering and
several kinds of clock; two runs over the same input give byte-identical
files; and so do runs made one epoch per process, as the scheduler makes
them, whatever each process's hash seed.
"""

import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import run
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.das_processor.read_steering import STEERING_FILE_TEMPLATE
from masterclock.domain.phase import PHASE_PERIOD

T: Final = timedelta(minutes=10)
"""One epoch."""

DAY: Final = datetime(2025, 9, 23, tzinfo=UTC)
"""The day the invented deployments start on."""

CLOCKS: Final = (
    "rejects_before_restart: 4\n"
    "rms_limit: {default: 50}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 10.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  cesium: {filter_states: 2, time_constant: 5.0, scale_time_constant: 5.0,"
    " initial_innovation_scale: 4.0, gap_limit: 8}\n"
    "  mc: {filter_states: 1, scale_time_constant: 5.0,"
    " initial_innovation_scale: 3.0, gap_limit: 8}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  mc2: [{type: mc}]\n"
    "  mc3: [{type: mc}]\n"
    "  hm1: [{type: maser}]\n"
    "  cs1: [{type: cesium}]\n"
)
"""An invented clock configuration for every clock the deployments use."""

type Deployment = dict[str, object]
"""An invented deployment: its references, clocks, epochs and readings."""


@st.composite
def deployments(draw: st.DrawFn) -> Deployment:
    """Draw a small invented deployment."""
    refs = [f"mc{i}" for i in range(1, draw(st.integers(1, 3)) + 1)]
    clocks = {
        name: draw(st.sampled_from(refs))
        for name in ("hm1", "cs1")[: draw(st.integers(0, 2))]
    }
    start = DAY + timedelta(hours=23) + T * draw(st.integers(0, 5))
    epochs = draw(st.integers(2, 9))
    pairs = [(r, s) for r in refs for s in refs]
    pairs += [(local, clock) for clock, local in clocks.items()]
    phases = {pair: draw(st.integers(0, 10**6)) for pair in pairs}
    rates = {pair: draw(st.integers(-40, 40)) for pair in pairs}
    missing = draw(
        st.sets(st.tuples(st.integers(1, epochs - 1), st.sampled_from(pairs)))
    )
    steering = {
        mc: sorted(
            draw(
                st.lists(
                    st.tuples(st.integers(0, epochs * 600 - 1), st.integers(-5, 5)),
                    max_size=2,
                )
            )
        )
        for mc in refs
    }
    return {
        "start": start,
        "epochs": epochs,
        "pairs": pairs,
        "phases": phases,
        "rates": rates,
        "missing": missing,
        "steering": steering,
    }


def written(directory: Path, deployment: Deployment) -> AppConfig:
    """Write a deployment's input files in ``directory``, and give its settings."""
    for name in ("das", "steering", "processed"):
        (directory / name).mkdir(parents=True)
    (directory / "clock_config.yaml").write_text(CLOCKS, encoding="utf-8")
    start: datetime = deployment["start"]  # type: ignore[assignment]
    epochs: int = deployment["epochs"]  # type: ignore[assignment]
    pairs: list[tuple[str, str]] = deployment["pairs"]  # type: ignore[assignment]
    phases: dict[tuple[str, str], int] = deployment["phases"]  # type: ignore[assignment]
    rates: dict[tuple[str, str], int] = deployment["rates"]  # type: ignore[assignment]
    missing: set[tuple[int, tuple[str, str]]] = deployment["missing"]  # type: ignore[assignment]
    days: dict[int, list[str]] = {}
    for index in range(epochs):
        mark = start + index * T
        for slot, pair in enumerate(pairs):
            if (index, pair) in missing:
                continue
            reference, clock = pair
            offset = 20 + 10 * slot
            phase = phases[pair] + rates[pair] * (index * 600 + offset) // 100
            raw = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(mark + timedelta(seconds=offset)), 6
                ),
                measured_phase=phase % PHASE_PERIOD,
                rms=3,
                switch=f"{reference[-1]}A{slot:02d}",
                clock=clock,
            )
            days.setdefault(int(datetime_to_mjd(mark)), []).append(f"{raw}\n")
    for day, lines in days.items():
        (directory / "das" / f"cd5m5m_{day}.dat").write_text(
            "".join(lines), encoding="ascii"
        )
    steering: dict[str, list[tuple[int, int]]] = deployment["steering"]  # type: ignore[assignment]
    for mc, events in steering.items():
        text = "".join(
            f"{datetime_to_mjd(start + timedelta(seconds=at)):.6f} {dx}.0 0.0\n"
            for at, dx in events
        )
        (directory / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)).write_text(text)
    return AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": directory / "das",
                "steering_path": directory / "steering",
            },
            "processed": {
                "processed_path": directory / "processed",
                "redo_from_mjd": None,
                "start_from_mjd": datetime_to_mjd(start),
                "clock_config_file": directory / "clock_config.yaml",
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )


def archive(config: AppConfig) -> dict[str, bytes]:
    """Give every data file's bytes, by its path under the processed directory."""
    root = config.processed.processed_path
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("das_a.*.dat"))
    }


def batch_and_stepped(
    deployment: Deployment, root: Path
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Run a deployment in one batch and one epoch per run, and give both archives."""
    batch = written(root / "batch", deployment)
    run.run(
        batch,
        read_clock_config(batch.processed.clock_config_file),
        None,
        ShutdownHandler(),
    )
    stepped = written(root / "stepped", deployment)
    clocks = read_clock_config(stepped.processed.clock_config_file)
    epochs: int = deployment["epochs"]  # type: ignore[assignment]
    for _ in range(epochs + 1):
        run.run(stepped, clocks, 1, ShutdownHandler())
    return archive(batch), archive(stepped)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(deployments())
def test_a_batch_and_a_stepped_run_write_the_same_files(deployment: Deployment) -> None:
    """Give byte-identical data files run as a batch or one epoch at a time (I5)."""
    with tempfile.TemporaryDirectory() as directory:
        batch, stepped = batch_and_stepped(deployment, Path(directory))
    assert batch
    assert batch == stepped


FIXED: Final[Deployment] = {
    "start": DAY + timedelta(hours=23, minutes=20),
    "epochs": 6,
    "pairs": [
        ("mc1", "mc1"),
        ("mc1", "mc2"),
        ("mc2", "mc1"),
        ("mc2", "mc2"),
        ("mc1", "hm1"),
    ],
    "phases": {
        ("mc1", "mc1"): 100,
        ("mc1", "mc2"): 5_000,
        ("mc2", "mc1"): 195_000,
        ("mc2", "mc2"): 200,
        ("mc1", "hm1"): 70_000,
    },
    "rates": {
        ("mc1", "mc1"): 0,
        ("mc1", "mc2"): 1,
        ("mc2", "mc1"): -1,
        ("mc2", "mc2"): 0,
        ("mc1", "hm1"): 25,
    },
    "missing": {(3, ("mc1", "hm1"))},
    "steering": {"mc1": [(700, 2)], "mc2": []},
}
"""One invented deployment across midnight, with a gap and a steering event."""


def test_two_runs_over_the_same_input_write_the_same_files(tmp_path: Path) -> None:
    """Give byte-identical data files from two runs over the same input (U22)."""
    first = written(tmp_path / "first", FIXED)
    second = written(tmp_path / "second", FIXED)
    for config in (first, second):
        run.run(
            config,
            read_clock_config(config.processed.clock_config_file),
            None,
            ShutdownHandler(),
        )
    assert archive(first)
    assert archive(first) == archive(second)


def test_one_epoch_per_process_writes_the_same_files(tmp_path: Path) -> None:
    """Give the batch's files from one process per epoch, any hash seed (U22)."""
    batch = written(tmp_path / "batch", FIXED)
    run.run(
        batch,
        read_clock_config(batch.processed.clock_config_file),
        None,
        ShutdownHandler(),
    )
    stepped = written(tmp_path / "stepped", FIXED)
    command = [
        sys.executable,
        "-m",
        "masterclock.das_processor",
        "--rf", "a",
        "--cd5m5m-path", str(stepped.das.cd5m5m_path),
        "--steering-path", str(stepped.das.steering_path),
        "--processed-path", str(stepped.processed.processed_path),
        "--clock-config-file", str(stepped.processed.clock_config_file),
        "--start-from-mjd", f"{stepped.processed.start_from_mjd:.6f}",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
        "--steps", "1",
    ]  # fmt: skip
    for seed in range(7):
        environment = {**os.environ, "PYTHONHASHSEED": str(seed)}
        # The command is the program itself with settings made in this test.
        done = subprocess.run(command, env=environment, check=False)  # noqa: S603  # nosec B603
        assert done.returncode == 0
    assert archive(batch) == archive(stepped)
