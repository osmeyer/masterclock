"""Tests for src/masterclock/das_processor/run.py.

The rules covered: building an epoch resolves everything it needs: its
references from its block; every pair and triple, the existing ones kept;
the steering of every reference that steers a series, read over
(E - T, E + T]; and each series' settings, a pair taking its second
clock's entry and RMS limit and a triple its clock c's entry; an epoch
with no block has no references and the existing series only; and the
epoch is checked to hold settings for exactly its series.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.app.exceptions import ConfigError
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor import run
from masterclock.das_processor.clock_config import ClockConfig, read_clock_config
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.read_cd5m5m import DASData, DASMeasurement
from masterclock.das_processor.read_steering import STEERING_FILE_TEMPLATE
from masterclock.das_processor.registry import Existing

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented epoch start."""

T: Final = timedelta(minutes=10)
"""One epoch."""

NONE: Final = Existing(pairs=frozenset(), triples=frozenset())
"""No series yet."""

CLOCKS: Final = (
    "rejects_before_restart: 6\n"
    "rms_limit: {default: 50, pairs: {mc2.ox23: 80}}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 40}\n"
    "  mc: {filter_states: 1, scale_time_constant: 30.0,"
    " initial_innovation_scale: 2.0, gap_limit: 40}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  mc2: [{type: mc}]\n"
    "  ox23: [{type: maser}]\n"
)
"""An invented clock configuration: two references and a maser."""


def deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Make an invented deployment's directories and give its configuration."""
    for name in ("das", "steering", "processed"):
        (tmp_path / name).mkdir()
    clocks = tmp_path / "clock_config.yaml"
    clocks.write_text(CLOCKS, encoding="utf-8")
    config = AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": tmp_path / "das",
                "steering_path": tmp_path / "steering",
            },
            "processed": {
                "processed_path": tmp_path / "processed",
                "redo_from_mjd": None,
                "start_from_mjd": None,
                "clock_config_file": clocks,
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )
    return config, read_clock_config(clocks)


def block(pairs: list[tuple[str, str]]) -> DASData:
    """Give a block measuring each (reference, clock) pair once."""
    start = datetime_to_mjd(E)
    return DASData(
        interpolated_datetime=E,
        measurements=tuple(
            DASMeasurement(
                measurement_mjd=round(start + (i + 1) * 2e-6, 6),
                measured_phase=1000,
                rms=3,
                switch=f"{reference[-1]}A{i:02d}",
                clock=clock,
            )
            for i, (reference, clock) in enumerate(pairs)
        ),
    )


MEASURED: Final = [
    ("mc1", "mc1"),
    ("mc2", "mc2"),
    ("mc1", "mc2"),
    ("mc2", "mc1"),
    ("mc2", "ox23"),
]
"""Two references measured against each other, and a maser against one."""


def test_an_epoch_holds_its_references_and_series(tmp_path: Path) -> None:
    """Give the block's references, pairs and triples, sorted (3.1, 3.4)."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    assert epoch.interpolated_datetime == E
    assert epoch.refs == frozenset({"mc1", "mc2"})
    assert epoch.pairs == tuple(sorted(MEASURED))
    assert epoch.triples == (("mc1", "mc2", "ox23"), ("mc2", "mc2", "ox23"))


def test_a_series_takes_the_entry_of_its_clock_side(tmp_path: Path) -> None:
    """Give a pair its second clock's entry and a triple its clock c's (8.1)."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    pair = epoch.params[("mc2", "ox23")]
    assert (pair.model, pair.M, pair.rms_max) == (3, 100.0, 80)
    link = epoch.params[("mc1", "mc2")]
    assert (link.model, link.M, link.sigma0, link.rms_max) == (1, None, 2.0, 50)
    triple = epoch.params[("mc1", "mc2", "ox23")]
    assert (triple.model, triple.M, triple.rms_max) == (3, 100.0, None)
    assert set(epoch.params) == set(epoch.pairs) | set(epoch.triples)


def test_steering_is_read_for_every_reference_over_the_epoch_either_side(
    tmp_path: Path,
) -> None:
    """Read each steering reference's events in (E - T, E + T] (I4)."""
    config, clocks = deployment(tmp_path)
    inside = [E - T + timedelta(seconds=60), E + T - timedelta(seconds=30)]
    outside = [E - T - timedelta(seconds=30), E + T + timedelta(seconds=60)]
    times = sorted(inside + outside)
    lines = "".join(f"{datetime_to_mjd(t):.6f} 1.0 0.0\n" for t in times)
    for mc in ("mc1", "mc2"):
        (tmp_path / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)).write_text(lines)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    assert sorted(epoch.steering) == ["mc1", "mc2"]
    expected = [mjd_to_datetime(float(f"{datetime_to_mjd(t):.6f}")) for t in inside]
    for events in epoch.steering.values():
        assert [event.applied_datetime for event in events] == expected


def test_an_epoch_with_no_block_has_the_existing_series_only(tmp_path: Path) -> None:
    """Give no references and every existing series when the DAS measured nothing."""
    config, clocks = deployment(tmp_path)
    existing = Existing(
        pairs=frozenset({("mc2", "ox23"), ("mc2", "mc2")}),
        triples=frozenset({("mc2", "mc2", "ox23")}),
    )
    epoch = run.build_epoch(E, None, existing, config, clocks)
    assert (epoch.block, epoch.refs) == (None, frozenset())
    assert epoch.pairs == (("mc2", "mc2"), ("mc2", "ox23"))
    assert epoch.triples == (("mc2", "mc2", "ox23"),)
    assert sorted(epoch.steering) == ["mc2"]


def test_a_clock_with_no_entry_stops_the_epoch(tmp_path: Path) -> None:
    """Raise ConfigError for a measured clock the clock configuration lacks."""
    config, clocks = deployment(tmp_path)
    with pytest.raises(ConfigError, match="hm9"):
        run.build_epoch(E, block([*MEASURED, ("mc2", "hm9")]), NONE, config, clocks)


def test_an_epoch_holds_settings_for_exactly_its_series(tmp_path: Path) -> None:
    """Refuse an epoch whose settings miss a series or name another."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    values = dict(epoch)
    values["params"] = {k: v for k, v in epoch.params.items() if k != ("mc1", "mc1")}
    with pytest.raises(ValidationError, match="settings"):
        run.Epoch.model_validate(values)


def test_an_epoch_s_block_is_of_its_epoch(tmp_path: Path) -> None:
    """Refuse an epoch holding another epoch's block."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    values = {**dict(epoch), "interpolated_datetime": E + T}
    with pytest.raises(ValidationError, match="block"):
        run.Epoch.model_validate(values)
