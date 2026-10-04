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

DEPLOYMENT_DAY: Final = datetime(2025, 9, 23, tzinfo=UTC)
"""The day the invented deployments start on."""

CLOCK_CONFIG_YAML: Final = (
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
    "  mc1: [{type: mc, location: 1}]\n"
    "  mc2: [{type: mc, location: 1}]\n"
    "  mc3: [{type: mc, location: 1}]\n"
    "  hm1: [{type: maser, location: 1}]\n"
    "  cs1: [{type: cesium, location: 1}]\n"
)
"""An invented clock configuration for every clock the deployments use."""

type Deployment = dict[str, object]
"""An invented deployment: its references, clocks, epochs and readings."""


@st.composite
def invented_deployments(draw: st.DrawFn) -> Deployment:
    """Draw a small invented deployment."""
    refs = [f"mc{ref_number}" for ref_number in range(1, draw(st.integers(1, 3)) + 1)]
    local_reference_of = {
        clock_name: draw(st.sampled_from(refs))
        for clock_name in ("hm1", "cs1")[: draw(st.integers(0, 2))]
    }
    first_epoch_start = (
        DEPLOYMENT_DAY + timedelta(hours=23) + T * draw(st.integers(0, 5))
    )
    epoch_count = draw(st.integers(2, 9))
    pairs = [(r, s) for r in refs for s in refs]
    pairs += [(local, clock) for clock, local in local_reference_of.items()]
    first_phases = {pair: draw(st.integers(0, 10**6)) for pair in pairs}
    rates_ps_per_100_s = {pair: draw(st.integers(-40, 40)) for pair in pairs}
    missing_readings = draw(
        st.sets(st.tuples(st.integers(1, epoch_count - 1), st.sampled_from(pairs)))
    )
    steering = {
        mc: sorted(
            draw(
                st.lists(
                    st.tuples(
                        st.integers(0, epoch_count * 600 - 1), st.integers(-5, 5)
                    ),
                    max_size=2,
                )
            )
        )
        for mc in refs
    }
    return {
        "first_epoch_start": first_epoch_start,
        "epoch_count": epoch_count,
        "pairs": pairs,
        "first_phases": first_phases,
        "rates_ps_per_100_s": rates_ps_per_100_s,
        "missing_readings": missing_readings,
        "steering": steering,
    }


def write_deployment(deployment_directory: Path, deployment: Deployment) -> AppConfig:
    """Write a deployment's input files in ``deployment_directory``; give its config."""
    for directory_name in ("das", "steering", "processed"):
        (deployment_directory / directory_name).mkdir(parents=True)
    (deployment_directory / "clock_config.yaml").write_text(
        CLOCK_CONFIG_YAML, encoding="utf-8"
    )
    first_epoch_start: datetime = deployment["first_epoch_start"]  # type: ignore[assignment]
    epoch_count: int = deployment["epoch_count"]  # type: ignore[assignment]
    pairs: list[tuple[str, str]] = deployment["pairs"]  # type: ignore[assignment]
    first_phases: dict[tuple[str, str], int] = deployment["first_phases"]  # type: ignore[assignment]
    rates_ps_per_100_s: dict[tuple[str, str], int] = deployment["rates_ps_per_100_s"]  # type: ignore[assignment]
    missing_readings: set[tuple[int, tuple[str, str]]] = deployment["missing_readings"]  # type: ignore[assignment]
    das_lines_by_day: dict[int, list[str]] = {}
    for epoch_index in range(epoch_count):
        epoch_start = first_epoch_start + epoch_index * T
        for pair_index, pair in enumerate(pairs):
            if (epoch_index, pair) in missing_readings:
                continue
            reference, clock = pair
            seconds_into_epoch = 20 + 10 * pair_index
            unwrapped_phase = (
                first_phases[pair]
                + rates_ps_per_100_s[pair]
                * (epoch_index * 600 + seconds_into_epoch)
                // 100
            )
            das_measurement = DASMeasurement(
                measurement_mjd=round(
                    datetime_to_mjd(
                        epoch_start + timedelta(seconds=seconds_into_epoch)
                    ),
                    6,
                ),
                measured_phase=unwrapped_phase % PHASE_PERIOD,
                rms=3,
                switch=f"{reference[-1]}A{pair_index:02d}",
                clock=clock,
            )
            das_lines_by_day.setdefault(int(datetime_to_mjd(epoch_start)), []).append(
                f"{das_measurement}\n"
            )
    for data_day, das_lines in das_lines_by_day.items():
        (deployment_directory / "das" / f"cd5m5m_{data_day}.dat").write_text(
            "".join(das_lines), encoding="ascii"
        )
    steering: dict[str, list[tuple[int, int]]] = deployment["steering"]  # type: ignore[assignment]
    for mc, steer_events in steering.items():
        steering_lines = []
        for seconds_after_start, dx in steer_events:
            applied_at = first_epoch_start + timedelta(seconds=seconds_after_start)
            steering_lines.append(f"{datetime_to_mjd(applied_at):.6f} {dx}.0 0.0\n")
        steering_text = "".join(steering_lines)
        (
            deployment_directory / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)
        ).write_text(steering_text)
    return AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": deployment_directory / "das",
                "steering_path": deployment_directory / "steering",
            },
            "processed": {
                "processed_path": deployment_directory / "processed",
                "start_from_mjd": datetime_to_mjd(first_epoch_start),
                "clock_config_file": deployment_directory / "clock_config.yaml",
                "num_workers": None,
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )


def archived_files(config: AppConfig) -> dict[str, bytes]:
    """Give every data file's bytes, by its path under the processed directory."""
    processed_root = config.processed.processed_path
    return {
        str(data_file.relative_to(processed_root)): data_file.read_bytes()
        for data_file in sorted(processed_root.rglob("das_a.*.dat"))
    }


def batch_and_stepped(
    deployment: Deployment, scratch_directory: Path
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Run a deployment in one batch and one epoch per run, and give both archives."""
    batch_config = write_deployment(scratch_directory / "batch", deployment)
    run.run(
        batch_config,
        read_clock_config(batch_config.processed.clock_config_file),
        None,
        ShutdownHandler(),
    )
    stepped_config = write_deployment(scratch_directory / "stepped", deployment)
    clock_config = read_clock_config(stepped_config.processed.clock_config_file)
    epoch_count: int = deployment["epoch_count"]  # type: ignore[assignment]
    for _ in range(epoch_count + 1):
        run.run(stepped_config, clock_config, 1, ShutdownHandler())
    return archived_files(batch_config), archived_files(stepped_config)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(invented_deployments())
def test_a_batch_and_a_stepped_run_write_the_same_files(deployment: Deployment) -> None:
    """Give byte-identical data files run as a batch or one epoch at a time (I5)."""
    with tempfile.TemporaryDirectory() as scratch_directory:
        batch_files, stepped_files = batch_and_stepped(
            deployment, Path(scratch_directory)
        )
    assert batch_files
    assert batch_files == stepped_files


FIXED_DEPLOYMENT: Final[Deployment] = {
    "first_epoch_start": DEPLOYMENT_DAY + timedelta(hours=23, minutes=20),
    "epoch_count": 6,
    "pairs": [
        ("mc1", "mc1"),
        ("mc1", "mc2"),
        ("mc2", "mc1"),
        ("mc2", "mc2"),
        ("mc1", "hm1"),
    ],
    "first_phases": {
        ("mc1", "mc1"): 100,
        ("mc1", "mc2"): 5_000,
        ("mc2", "mc1"): 195_000,
        ("mc2", "mc2"): 200,
        ("mc1", "hm1"): 70_000,
    },
    "rates_ps_per_100_s": {
        ("mc1", "mc1"): 0,
        ("mc1", "mc2"): 1,
        ("mc2", "mc1"): -1,
        ("mc2", "mc2"): 0,
        ("mc1", "hm1"): 25,
    },
    "missing_readings": {(3, ("mc1", "hm1"))},
    "steering": {"mc1": [(700, 2)], "mc2": []},
}
"""One invented deployment across midnight, with a gap and a steering event."""


def test_two_runs_over_the_same_input_write_the_same_files(tmp_path: Path) -> None:
    """Give byte-identical data files from two runs over the same input (U22)."""
    first_config = write_deployment(tmp_path / "first", FIXED_DEPLOYMENT)
    second_config = write_deployment(tmp_path / "second", FIXED_DEPLOYMENT)
    for config in (first_config, second_config):
        run.run(
            config,
            read_clock_config(config.processed.clock_config_file),
            None,
            ShutdownHandler(),
        )
    assert archived_files(first_config)
    assert archived_files(first_config) == archived_files(second_config)


def test_one_epoch_per_process_writes_the_same_files(tmp_path: Path) -> None:
    """Give the batch's files from one process per epoch, any hash seed (U22)."""
    batch_config = write_deployment(tmp_path / "batch", FIXED_DEPLOYMENT)
    run.run(
        batch_config,
        read_clock_config(batch_config.processed.clock_config_file),
        None,
        ShutdownHandler(),
    )
    stepped_config = write_deployment(tmp_path / "stepped", FIXED_DEPLOYMENT)
    stepped_command = [
        sys.executable,
        "-m",
        "masterclock.das_processor",
        "--rf", "a",
        "--cd5m5m-path", str(stepped_config.das.cd5m5m_path),
        "--steering-path", str(stepped_config.das.steering_path),
        "--processed-path", str(stepped_config.processed.processed_path),
        "--clock-config-file", str(stepped_config.processed.clock_config_file),
        "--start-from-mjd", f"{stepped_config.processed.start_from_mjd:.6f}",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
        "--steps", "1",
    ]  # fmt: skip
    for hash_seed in range(7):
        seeded_env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
        # The command is the program itself with settings made in this test.
        stepped_run = subprocess.run(stepped_command, env=seeded_env, check=False)  # noqa: S603  # nosec B603
        assert stepped_run.returncode == 0
    assert archived_files(batch_config) == archived_files(stepped_config)
