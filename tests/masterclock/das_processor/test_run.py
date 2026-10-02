"""Tests for src/masterclock/das_processor/run.py.

The rules covered: building an epoch resolves everything it needs: its
references from its block; every pair and triple, the existing ones kept;
the steering of every reference that steers a series, read over
(E - T, E + T]; and each series' settings, a pair taking its second
clock's entry and RMS limit and a triple its clock c's entry; an epoch
with no block has no references and the existing series only; and the
epoch is checked to hold settings for exactly its series.

The pairs of an epoch are predicted, decycled against the prediction or the
anchor with the steering inside the epoch taken off, screened, checked for
slips, corrected before filtering, and filtered with what screening and the
slip check excluded. The triples are built from the pairs' accepted
measurements of the same epoch, the local triple given its self pair for
both links, marked cold when a pair cold-started, and filtered.

The run's events are logged at the design's levels: each epoch with its
counts at INFO; steps, cold starts, dormancy, configuration changes and
corrected slips at INFO; rejects, screening failures, a missing self pair
and undecided slips at WARNING; each series' outcome at
DEBUG and its prediction and update at TRACE.

The next epoch is one after the oldest epoch every file is good through,
every file rolled back to it, or one before the epoch a write journal
names, the journal then deleted, and a file whose first row is damaged,
refused without a journal, deleted with one, to be made again; with no
file, the epoch containing start_from_mjd.

Each series takes the settings in force at its epoch; a link not accepted
does not make its triple cold; the TRACE lines and steps with a settings
change are logged as they are; each series' last row is read from its
file; a remote
triple goes on when its reference is missing; a run with no data and no
files does nothing; and a gap at the start of a run is predicted.
"""

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.app.exceptions import ConfigError
from masterclock.app.log import TRACE
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor import files, registry, run
from masterclock.das_processor.clock_config import ClockConfig, read_clock_config
from masterclock.das_processor.config import JOURNAL_FILE_TEMPLATE, AppConfig
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import (
    DASData,
    DASMeasurement,
    read_all_blocks,
)
from masterclock.das_processor.read_steering import STEERING_FILE_TEMPLATE
from masterclock.das_processor.registry import Existing
from masterclock.domain.double_difference import Component, double_difference
from masterclock.domain.phase import PHASE_PERIOD
from masterclock.domain.series import Row, SeriesKey, TripleKey

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented epoch start."""

T: Final = timedelta(minutes=10)
"""One epoch."""

NONE: Final = Existing(pairs=frozenset(), triples=frozenset())
"""No series yet."""

CLOCKS: Final = (
    "rejects_before_restart: 6\n"
    "rms_limit: {default: 50, pairs: {mc2.nav23: 80}}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 40}\n"
    "  mc: {filter_states: 1, scale_time_constant: 30.0,"
    " initial_innovation_scale: 2.0, gap_limit: 40}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  mc2: [{type: mc}]\n"
    "  mc3: [{type: mc}]\n"
    "  nav23: [{type: maser}]\n"
)
"""An invented clock configuration: three references and a maser."""


def deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Make an invented deployment's directories and give its configuration."""
    for name in ("das", "steering", "processed"):
        (tmp_path / name).mkdir(parents=True)
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
    ("mc2", "nav23"),
]
"""Two references measured against each other, and a maser against one."""


def test_an_epoch_holds_its_references_and_series(tmp_path: Path) -> None:
    """Give the block's references, pairs and triples, sorted (3.1, 3.4)."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    assert epoch.interpolated_datetime == E
    assert epoch.refs == frozenset({"mc1", "mc2"})
    assert epoch.pairs == tuple(sorted(MEASURED))
    assert epoch.triples == tuple(
        sorted((r, s, c) for r in ("mc1", "mc2") for s, c in MEASURED)
    )


def test_a_series_takes_the_entry_of_its_clock_side(tmp_path: Path) -> None:
    """Give a pair its second clock's entry and a triple its clock c's (8.1)."""
    config, clocks = deployment(tmp_path)
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, clocks)
    pair = epoch.params[("mc2", "nav23")]
    assert (pair.model, pair.M, pair.rms_max) == (3, 100.0, 80)
    link = epoch.params[("mc1", "mc2")]
    assert (link.model, link.M, link.sigma0, link.rms_max) == (1, None, 2.0, 50)
    triple = epoch.params[("mc1", "mc2", "nav23")]
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
        pairs=frozenset({("mc2", "nav23"), ("mc2", "mc2")}),
        triples=frozenset({("mc2", "mc2", "nav23")}),
    )
    epoch = run.build_epoch(E, None, existing, config, clocks)
    assert (epoch.block, epoch.refs) == (None, frozenset())
    assert epoch.pairs == (("mc2", "mc2"), ("mc2", "nav23"))
    assert epoch.triples == (("mc2", "mc2", "nav23"),)
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


# ------------------------------------------------------------ pairs of an epoch

PREVIOUS: Final = E - T
"""The epoch before E."""


def last_row(**changes: object) -> Row:
    """Give a settled 1-state reference row at the epoch before E, ``changes`` made."""
    values: dict[str, object] = {
        "interpolated_datetime": PREVIOUS,
        "innovation": None,
        "x_fs": 0,
        "y": 0.0,
        "d": 0.0,
        "innovation_scale": 2.0,
        "segment": 1,
        "step_offset": 0,
        "epochs_in_segment": 900,
        "epochs_since_accept": 0,
        "consecutive_rejects": 0,
        "rejects": (),
        "filter_states": 1,
        "time_constant": None,
        "scale_time_constant": 30.0,
        "flags": "A",
    }
    values.update(changes)
    return Row.model_validate(values)


WORKED_LAST: Final = last_row(
    x_fs=1_234_567_000,
    y=0.0123,
    innovation_scale=3.0,
    segment=4,
    epochs_in_segment=811,
    filter_states=3,
    time_constant=100.0,
    scale_time_constant=50.0,
)
"""Appendix A's last row of (mc2, nav23)."""


def measured(reference: str, clock: str, phase: int, offset_us: int) -> DASMeasurement:
    """Give a measurement of the pair ``offset_us`` microdays after E."""
    return DASMeasurement(
        measurement_mjd=round(datetime_to_mjd(E) + offset_us * 1e-6, 6),
        measured_phase=phase,
        rms=3,
        switch=f"{reference[-1]}A01",
        clock=clock,
    )


REFERENCE_LAST: Final[dict[SeriesKey, Row]] = {
    ("mc1", "mc1"): last_row(x_fs=1_000_000),
    ("mc2", "mc2"): last_row(x_fs=2_000_000),
    ("mc1", "mc2"): last_row(x_fs=5_000_000),
    ("mc2", "mc1"): last_row(x_fs=-5_000_000),
}
"""The references' last rows: their self and link pairs, 1-state, settled."""

REFERENCE_MEASURED: Final = [
    measured("mc1", "mc1", 1000, 10),
    measured("mc2", "mc2", 2000, 20),
    measured("mc1", "mc2", 5000, 30),
    measured("mc2", "mc1", PHASE_PERIOD - 5000, 40),
]
"""Each self and link pair measured where its last row says it is."""


def epoch_of(
    measurements: list[DASMeasurement],
    last: dict[SeriesKey, Row],
    tmp_path: Path,
    steering: dict[str, str] | None = None,
) -> run.Epoch:
    """Build E from ``measurements``, with ``last``'s series existing.

    ``steering`` gives the text of each reference's steering file.
    """
    config, clocks = deployment(tmp_path)
    for mc, text in (steering or {}).items():
        path = tmp_path / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)
        path.write_text(text, encoding="ascii")
    data = DASData(interpolated_datetime=E, measurements=tuple(measurements))
    existing = Existing(
        pairs=frozenset((k[0], k[1]) for k in last if len(k) == 2),
        triples=frozenset((k[0], k[1], k[-1]) for k in last if len(k) == 3),
    )
    return run.build_epoch(E, data, existing, config, clocks)


def test_the_worked_epoch_s_pairs_are_processed_end_to_end(tmp_path: Path) -> None:
    """Give Appendix A's row, and every reference pair accepted at its value (D3)."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    raw = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    epoch = epoch_of([*REFERENCE_MEASURED, raw], last, tmp_path)
    done = run.process_pairs(epoch, last)
    worked = done.results[("mc2", "nav23")].row
    assert (worked.flags, worked.x_fs, worked.y, worked.d) == (
        "A",
        1_234_574_457,
        0.01230129052352643,
        7.169515400974333e-12,
    )
    assert done.measurements[("mc2", "nav23")].z == 1_234_577
    for key, row in REFERENCE_LAST.items():
        result = done.results[(key[0], key[1])].row
        assert (result.flags, result.x_fs) == ("A", row.x_fs), key
    assert done.screening.events == ()
    assert done.slips.events == ()


def test_a_slip_correction_is_made_before_filtering(tmp_path: Path) -> None:
    """Correct the weak pair's cycle count first, so its row holds S and the truth."""
    jump = PHASE_PERIOD // 2 + 2
    last = {
        **REFERENCE_LAST,
        ("mc1", "nav23"): last_row(
            x_fs=1_000_000_000,
            filter_states=3,
            time_constant=100.0,
            scale_time_constant=50.0,
            innovation_scale=3.0,
            epochs_in_segment=10,
            flags="AU",
        ),
        ("mc2", "nav23"): last_row(
            x_fs=2_000_000_000,
            filter_states=3,
            time_constant=100.0,
            scale_time_constant=50.0,
            innovation_scale=3.0,
        ),
    }
    truth = {("mc1", "nav23"): 1_000_000 + jump, ("mc2", "nav23"): 2_000_000 + jump - 4}
    clocks = [
        measured("mc1", "nav23", truth[("mc1", "nav23")] % PHASE_PERIOD, 50),
        measured("mc2", "nav23", truth[("mc2", "nav23")] % PHASE_PERIOD, 60),
    ]
    epoch = epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert done.slips.corrections == {("mc1", "nav23"): 1}
    assert done.measurements[("mc1", "nav23")].z == truth[("mc1", "nav23")]
    assert done.measurements[("mc1", "nav23")].slip is True
    corrected = done.results[("mc1", "nav23")].row
    assert "S" in corrected.flags
    assert corrected.innovation == float(jump)


def test_a_reference_missing_from_the_block_leaves_its_pairs_predicted(
    tmp_path: Path,
) -> None:
    """Give mc1's pairs predicted rows and screen without it (review focus 4)."""
    last = dict(REFERENCE_LAST)
    epoch = epoch_of([REFERENCE_MEASURED[1]], last, tmp_path)
    assert epoch.refs == frozenset({"mc2"})
    done = run.process_pairs(epoch, last)
    for key in (("mc1", "mc1"), ("mc1", "mc2"), ("mc2", "mc1")):
        assert done.results[key].row.flags == "P"
    assert done.results[("mc2", "mc2")].row.flags == "A"
    assert done.screening.events == ()


def test_a_new_pair_starts_acquiring(tmp_path: Path) -> None:
    """Give a pair with no last row a dormant row with its measurement buffered."""
    epoch = epoch_of(REFERENCE_MEASURED, {}, tmp_path)
    done = run.process_pairs(epoch, {})
    for key in REFERENCE_LAST:
        row = done.results[(key[0], key[1])].row
        assert (row.flags, len(row.rejects), row.segment) == ("RD", 1, 0)
    assert done.predictions[("mc1", "mc1")] is None


def test_screening_excludes_and_the_filter_holds(tmp_path: Path) -> None:
    """Hold as X a pair that shares a self pair's shift inside its gate (9.5, 10.1)."""
    maser = last_row(
        x_fs=2_000_000,
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        innovation_scale=40.0,
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): maser}
    shifted = [
        measured("mc1", "mc1", 1000, 10),
        measured("mc2", "mc2", 2100, 20),
        measured("mc1", "mc2", 5000, 30),
        measured("mc2", "mc1", PHASE_PERIOD - 5000, 40),
        measured("mc2", "nav23", 2008, 50),
    ]
    epoch = epoch_of(shifted, last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert [event.kind for event in done.screening.events] == ["self_fail"]
    assert done.screening.excluded == frozenset({("mc2", "nav23")})
    assert done.results[("mc2", "nav23")].row.flags == "X"
    assert done.results[("mc2", "mc2")].row.flags == "R"


def test_an_epoch_with_no_block_predicts_every_pair(tmp_path: Path) -> None:
    """Give every existing pair a predicted row when the DAS measured nothing (6.2)."""
    config, clocks = deployment(tmp_path)
    existing = Existing(
        pairs=frozenset(k for k in REFERENCE_LAST if len(k) == 2), triples=frozenset()
    )
    epoch = run.build_epoch(E, None, existing, config, clocks)
    done = run.process_pairs(epoch, REFERENCE_LAST)
    assert {result.row.flags for result in done.results.values()} == {"P"}
    assert done.measurements == {}


def test_an_undecided_slip_excludes_both_clock_pairs(tmp_path: Path) -> None:
    """Hold as X the clock pairs of an undecided slip, inside their gates (11.2)."""
    jump = PHASE_PERIOD // 2 + 2
    wide = {
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 25_000.0,
    }
    last = {
        **REFERENCE_LAST,
        ("mc1", "nav23"): last_row(x_fs=1_000_000_000, **wide),
        ("mc2", "nav23"): last_row(x_fs=2_000_000_000, **wide),
    }
    clocks = [
        measured("mc1", "nav23", (1_000_000 + jump) % PHASE_PERIOD, 50),
        measured("mc2", "nav23", (2_000_000 + jump - 4) % PHASE_PERIOD, 60),
    ]
    epoch = epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert [event.kind for event in done.slips.events] == ["slip_undecided"]
    assert done.results[("mc1", "nav23")].row.flags == "X"
    assert done.results[("mc2", "nav23")].row.flags == "X"


def test_a_dormant_pair_is_decycled_against_its_anchor(tmp_path: Path) -> None:
    """Decycle a pair with no prediction against its last buffered measurement (7.5)."""
    dormant = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        rejects=((PREVIOUS, 201_000.0),),
    )
    last = {**REFERENCE_LAST, ("mc1", "mc1"): dormant}
    epoch = epoch_of(REFERENCE_MEASURED, last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert done.measurements[("mc1", "mc1")].z == 201_000


def test_steering_inside_the_epoch_is_taken_off(tmp_path: Path) -> None:
    """Refer a measurement to E with the steering since E taken off (7.1)."""
    last = dict(REFERENCE_LAST)
    event = f"{datetime_to_mjd(E) + 2e-6:.6f} 10.0 0.0\n"
    moved = [
        *REFERENCE_MEASURED[:2],
        measured("mc1", "mc2", 5010, 30),
        REFERENCE_MEASURED[3],
    ]
    epoch = epoch_of(moved, last, tmp_path, {"mc1": event})
    done = run.process_pairs(epoch, last)
    assert done.measurements[("mc1", "mc2")].z == 5000
    assert done.results[("mc1", "mc2")].row.flags == "A"


# ---------------------------------------------------------- triples of an epoch

WORKED_RAW: Final = DASMeasurement(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    switch="2B07",
    clock="nav23",
)
"""Appendix A's raw row."""


def triple_last(**changes: object) -> Row:
    """Give a tracked 3-state triple row at the epoch before E, ``changes`` made."""
    values: dict[str, object] = {
        "x_fs": 1_239_570_000,
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 4.0,
    }
    values.update(changes)
    return last_row(**values)


def test_triples_are_built_from_the_pairs_measurements(tmp_path: Path) -> None:
    """Take dd from the pairs' accepted z of the epoch, not their estimates (12)."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    done = run.process_triples(epoch, last, pairs)
    remote = done.measurements[("mc1", "mc2", "nav23")]
    assert (remote.z, remote.components_used) == (1_234_577 + 5_000, "111")
    assert remote.double_difference_sigma == math.sqrt(9 + 0.25 * (9 + 9))
    local = done.measurements[("mc2", "mc2", "nav23")]
    assert (local.z, local.double_difference_sigma) == (1_234_577, 3.0)
    assert {r.row.flags for r in done.results.values()} == {"RD"}


def test_a_tracked_triple_is_filtered_on_its_double_difference(tmp_path: Path) -> None:
    """Accept a triple's dd against its own prediction."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): WORKED_LAST,
        ("mc1", "mc2", "nav23"): triple_last(x_fs=1_239_577_000),
    }
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    row = done.results[("mc1", "mc2", "nav23")].row
    assert (row.flags, row.innovation) == ("A", 0.0)
    assert row.x_fs == 1_239_577_000


def test_a_missing_link_direction_uses_the_predicted_round_trip(tmp_path: Path) -> None:
    """Give 110 when (s, r) was not measured, through the links' predictions (12.2)."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    epoch = epoch_of([*REFERENCE_MEASURED[:3], WORKED_RAW], last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    remote = done.measurements[("mc1", "mc2", "nav23")]
    assert (remote.z, remote.components_used) == (1_234_577 + 5_000, "110")


def test_a_component_cold_start_makes_the_triple_dormant(tmp_path: Path) -> None:
    """Restart a triple whose clock pair cold-started this epoch (12.6)."""
    acquiring = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS - T, 1_234_577.0), (PREVIOUS, 1_234_577.0)),
    )
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): acquiring,
        ("mc2", "mc2", "nav23"): triple_last(x_fs=1_234_577_000),
    }
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    assert pairs.results[("mc2", "nav23")].cold is True
    done = run.process_triples(epoch, last, pairs)
    assert done.measurements[("mc2", "mc2", "nav23")].cold is True
    row = done.results[("mc2", "mc2", "nav23")].row
    assert (row.flags, row.rejects) == ("RD", ((E, 1_234_579.0),))


def test_a_triple_without_its_clock_pair_holds(tmp_path: Path) -> None:
    """Give a triple a predicted row when its clock pair has no measurement."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): WORKED_LAST,
        ("mc2", "mc2", "nav23"): triple_last(x_fs=1_234_577_000),
    }
    epoch = epoch_of(REFERENCE_MEASURED, last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    assert done.results[("mc2", "mc2", "nav23")].row.flags == "P"
    assert ("mc2", "mc2", "nav23") not in done.measurements


def test_the_local_triple_is_checked_every_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass the self pair for both links of a local triple, so the check runs (12.4)."""
    seen: list[tuple[object, object]] = []

    def spy(triple: TripleKey, sc: Component, rs: Component, sr: Component) -> object:
        """Record the links a local triple is given."""
        if triple == ("mc2", "mc2", "nav23"):
            seen.append((rs, sr))
        return double_difference(triple, sc, rs, sr)

    monkeypatch.setattr(run, "double_difference", spy)
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    run.process_triples(epoch, last, run.process_pairs(epoch, last))
    assert len(seen) == 1
    rs, sr = seen[0]
    assert rs == sr
    assert rs.z == 2000  # type: ignore[attr-defined]


def test_a_rejected_pair_gives_its_triple_no_value(tmp_path: Path) -> None:
    """Leave a triple held when its clock pair was measured but rejected."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): WORKED_LAST,
        ("mc2", "mc2", "nav23"): triple_last(x_fs=1_234_577_000),
    }
    outlier = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34779,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    epoch = epoch_of([*REFERENCE_MEASURED, outlier], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    assert pairs.results[("mc2", "nav23")].row.flags == "R"
    done = run.process_triples(epoch, last, pairs)
    assert ("mc2", "mc2", "nav23") not in done.measurements
    assert done.results[("mc2", "mc2", "nav23")].row.flags == "P"


# ---------------------------------------------------------------- the epoch loop

LATE: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch of the runs below: four epochs before midnight."""


def das_files(tmp_path: Path, marks: list[datetime]) -> None:
    """Write DAS daily files measuring mc1 against itself and nav23 at ``marks``."""
    days: dict[int, list[str]] = {}
    for mark in marks:
        start = datetime_to_mjd(mark)
        for offset, (clock, phase) in enumerate((("mc1", 1000), ("nav23", 50_000))):
            raw = DASMeasurement(
                measurement_mjd=round(start + (offset + 1) * 2e-5, 6),
                measured_phase=phase,
                rms=3,
                switch="1A01" if clock == "mc1" else "1A02",
                clock=clock,
            )
            days.setdefault(int(start), []).append(f"{raw}\n")
    for day, lines in days.items():
        (tmp_path / "das" / f"cd5m5m_{day}.dat").write_text(
            "".join(lines), encoding="ascii"
        )


def loop_deployment(
    tmp_path: Path, start: datetime = LATE
) -> tuple[AppConfig, ClockConfig]:
    """Give a deployment whose first epoch is ``start``."""
    config, clocks = deployment(tmp_path)
    processed = config.processed.model_copy(
        update={"start_from_mjd": datetime_to_mjd(start)}
    )
    return config.model_copy(update={"processed": processed}), clocks


def rows_of(config: AppConfig, key: SeriesKey) -> list[Row]:
    """Read every row of a series' file."""
    path = registry.series_file(config.processed.processed_path, "a", key)
    kind: files.FileKind = "meas" if len(key) == 2 else "ddiff"
    size = files.WIDTHS[kind] + 1
    data = path.read_bytes()[files.HEADER_LINES[kind] * size :]
    lines = [data[i : i + size - 1].decode() for i in range(0, len(data), size)]
    if kind == "meas":
        return [files.parse_meas_row(line).row for line in lines]
    return [files.parse_ddiff_row(line).row for line in lines]


SERIES: Final[tuple[SeriesKey, ...]] = (
    ("mc1", "mc1"),
    ("mc1", "nav23"),
    ("mc1", "mc1", "mc1"),
    ("mc1", "mc1", "nav23"),
)
"""The series of the loop's deployment."""


def recorded_writes(monkeypatch: pytest.MonkeyPatch) -> list[datetime | None]:
    """Record, at every write, the newest epoch the buffer holds text for."""
    marks: list[datetime | None] = []
    real = files.write_buffer

    def spy(buffer: files.DayBuffer) -> None:
        """Record the newest buffered epoch, then write."""
        newest = (
            [buffer.last[key].interpolated_datetime for key in buffer.last]
            if buffer.texts
            else []
        )
        marks.append(max(newest, default=None))
        real(buffer)

    monkeypatch.setattr(run, "write_buffer", spy)
    return marks


def test_a_day_is_written_after_its_last_epoch_and_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write after 23:50 UTC and when the data end (5.8)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE + i * T for i in range(6)])
    writes = recorded_writes(monkeypatch)
    run.run(config, clocks, None, ShutdownHandler())
    assert writes == [LATE + 3 * T, LATE + 5 * T]
    for key in SERIES:
        assert [row.interpolated_datetime for row in rows_of(config, key)] == [
            LATE + i * T for i in range(6)
        ]


def test_a_day_without_a_block_at_23_50_is_still_written_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write after 23:50 although the DAS measured nothing then (review focus 2)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE + i * T for i in (0, 1, 2, 4, 5)])
    writes = recorded_writes(monkeypatch)
    run.run(config, clocks, None, ShutdownHandler())
    assert writes == [LATE + 3 * T, LATE + 5 * T]
    assert rows_of(config, ("mc1", "mc1"))[3].flags == "P"


def test_a_gap_gives_predicted_rows_and_the_run_stops_at_the_end_of_the_data(
    tmp_path: Path,
) -> None:
    """Give every series a row for a gap epoch, and nothing after the data (6.2)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE, LATE + T, LATE + 4 * T])
    run.run(config, clocks, None, ShutdownHandler())
    self_rows = rows_of(config, ("mc1", "mc1"))
    assert [row.interpolated_datetime for row in self_rows] == [
        LATE + i * T for i in range(5)
    ]
    assert [row.flags for row in self_rows] == ["RD", "RD", "PD", "PD", "RD"]


def test_steps_stop_the_run_after_that_many_epochs(tmp_path: Path) -> None:
    """Process --steps N epochs and stop, writing the buffer (6.1)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE + i * T for i in range(6)])
    run.run(config, clocks, 2, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 2
    run.run(config, clocks, 1, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 3


def test_a_shutdown_stops_between_epochs_after_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish the epoch, write the buffer and stop when a shutdown is asked (6.1)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE + i * T for i in range(6)])
    shutdown = ShutdownHandler()
    real = run.process_epoch

    def asking(*args: object) -> run.EpochDone:
        """Process the epoch, asking for a shutdown during the second."""
        done = real(*args)  # type: ignore[arg-type]
        if done.epoch.interpolated_datetime == LATE + T:
            shutdown.request_shutdown()
        return done

    monkeypatch.setattr(run, "process_epoch", asking)
    run.run(config, clocks, None, shutdown)
    assert len(rows_of(config, ("mc1", "mc1"))) == 2
    assert len(rows_of(config, ("mc1", "mc1", "nav23"))) == 2


def test_a_run_restarted_after_every_epoch_writes_the_same_files(
    tmp_path: Path,
) -> None:
    """Give byte-identical files whether run as a batch or one epoch at a time (I5)."""
    batch, clocks = loop_deployment(tmp_path / "batch")
    das_files(tmp_path / "batch", [LATE + i * T for i in range(6)])
    run.run(batch, clocks, None, ShutdownHandler())
    stepped, clocks = loop_deployment(tmp_path / "stepped")
    das_files(tmp_path / "stepped", [LATE + i * T for i in range(6)])
    for _ in range(8):
        run.run(stepped, clocks, 1, ShutdownHandler())
    for key in SERIES:
        one = registry.series_file(batch.processed.processed_path, "a", key)
        other = registry.series_file(stepped.processed.processed_path, "a", key)
        assert one.read_bytes() == other.read_bytes(), key


def test_a_damaged_line_found_at_the_start_redoes_every_file_from_its_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Roll every file back before a damaged line, then redo them alike (6.7, U26)."""
    clean, clocks = loop_deployment(tmp_path / "clean")
    das_files(tmp_path / "clean", [LATE + i * T for i in range(6)])
    run.run(clean, clocks, None, ShutdownHandler())
    damaged, clocks = loop_deployment(tmp_path / "damaged")
    das_files(tmp_path / "damaged", [LATE + i * T for i in range(6)])
    run.run(damaged, clocks, None, ShutdownHandler())
    path = registry.series_file(damaged.processed.processed_path, "a", ("mc1", "nav23"))
    data = bytearray(path.read_bytes())
    size = files.MEAS_WIDTH + 1
    data[(files.MEAS_HEADER_LINES + 2) * size + 3] = ord("x")
    path.write_bytes(bytes(data[: -size // 2]))
    assert run.next_epoch(damaged) == LATE + 2 * T
    assert {r.levelname for r in caplog.records} == {"WARNING"}
    run.run(damaged, clocks, None, ShutdownHandler())
    for key in SERIES:
        one = registry.series_file(clean.processed.processed_path, "a", key)
        other = registry.series_file(damaged.processed.processed_path, "a", key)
        assert one.read_bytes() == other.read_bytes(), key


def test_a_journal_found_at_the_start_rolls_every_file_back_before_its_epoch(
    tmp_path: Path,
) -> None:
    """Roll every file back to before a stopped write's first epoch, then redo (6.7)."""
    clean, clocks = loop_deployment(tmp_path / "clean")
    das_files(tmp_path / "clean", [LATE + i * T for i in range(6)])
    run.run(clean, clocks, None, ShutdownHandler())
    stopped, clocks = loop_deployment(tmp_path / "stopped")
    das_files(tmp_path / "stopped", [LATE + i * T for i in range(6)])
    run.run(stopped, clocks, None, ShutdownHandler())
    journal = stopped.processed.processed_path / JOURNAL_FILE_TEMPLATE.format(rf="a")
    journal.write_text(f"{(LATE + 3 * T).isoformat()}\n", encoding="ascii")
    assert run.next_epoch(stopped) == LATE + 3 * T
    assert not journal.exists()
    series = run.data_series(stopped)
    assert len(series) == len(SERIES)
    for path, kind, key in series:
        assert files.good_through(path, kind) == LATE + 2 * T, key
    run.run(stopped, clocks, None, ShutdownHandler())
    for key in SERIES:
        one = registry.series_file(clean.processed.processed_path, "a", key)
        other = registry.series_file(stopped.processed.processed_path, "a", key)
        assert one.read_bytes() == other.read_bytes(), key


def test_with_no_files_the_run_starts_at_start_from_mjd(tmp_path: Path) -> None:
    """Start at the epoch containing start_from_mjd when no file exists (6.7)."""
    config, _ = loop_deployment(tmp_path, LATE + 2 * T + timedelta(seconds=90))
    assert run.next_epoch(config) == LATE + 2 * T


def test_a_block_before_the_next_epoch_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip a block earlier than the epoch the run is at, rather than loop on it."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE - T, LATE, LATE + T])
    real = read_all_blocks

    def from_earlier(directory: Path, start_at_mjd: float | None = None) -> object:
        """Read from one epoch before the start."""
        del start_at_mjd
        return real(directory, datetime_to_mjd(LATE - T))

    monkeypatch.setattr(run, "read_all_blocks", from_earlier)
    run.run(config, clocks, None, ShutdownHandler())
    rows = rows_of(config, ("mc1", "mc1"))
    assert [row.interpolated_datetime for row in rows] == [LATE, LATE + T]


def test_an_epoch_that_fails_adds_none_of_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the day buffer as it was when an epoch fails part way (5.8 step 1)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE + i * T for i in range(3)])
    buffer = files.DayBuffer("a")
    blocks = list(read_all_blocks(tmp_path / "das", datetime_to_mjd(LATE)))
    files.ensure_archives(config.processed.processed_path)
    run.process_epoch(LATE, blocks[0], buffer, config, clocks)
    before = dict(buffer.texts), dict(buffer.last)
    real = files.DayBuffer.add
    calls: list[SeriesKey] = []

    def failing(
        self: files.DayBuffer,
        path: Path,
        key: SeriesKey,
        record: files.MeasRecord | files.DdiffRecord,
    ) -> None:
        """Add the first row, then fail on the second, as a bad value would."""
        calls.append(key)
        if len(calls) == 2:
            message = "injected"
            raise DataFileError(message)
        real(self, path, key, record)

    monkeypatch.setattr(files.DayBuffer, "add", failing)
    with pytest.raises(DataFileError, match="injected"):
        run.process_epoch(LATE + T, blocks[1], buffer, config, clocks)
    assert (dict(buffer.texts), dict(buffer.last)) == before


# --------------------------------------------------------------- log events

RUN_LOGGER: Final = "masterclock.das_processor.run"
"""The logger the run's events go to."""


def logged(
    caplog: pytest.LogCaptureFixture,
    epoch: run.Epoch,
    last: dict[SeriesKey, Row],
) -> list[tuple[str, str]]:
    """Process ``epoch`` and give the run's log records as (level, message)."""
    pairs = run.process_pairs(epoch, last)
    triples = run.process_triples(epoch, last, pairs)
    done = run.EpochDone(epoch=epoch, pairs=pairs, triples=triples)
    caplog.clear()
    with caplog.at_level(TRACE, logger=RUN_LOGGER):
        run.log_epoch(done, last, "a")
    return [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == RUN_LOGGER
    ]


def at(records: list[tuple[str, str]], level: str) -> list[str]:
    """Give the messages logged at ``level``."""
    return [message for name, message in records if name == level]


def test_an_epoch_is_logged_with_its_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each epoch at INFO with its series, accepted and held (16.2)."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path), last
    )
    assert at(records, "INFO") == [
        "epoch 2025-09-23 06:00:00+00:00: 5 pairs, 10 triples, 5 accepted, 10 held"
    ]


def test_each_series_outcome_and_update_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each series' flags at DEBUG and its prediction and update at TRACE."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path), last
    )
    debug = at(records, "DEBUG")
    assert len(debug) == 5 + 10
    assert "das_a.mc2.nav23: A" in debug
    assert "das_a.mc1.mc2.nav23: RD" in debug
    trace = at(records, "TRACE")
    assert len(trace) == 5 + 10
    assert any(
        line.startswith("das_a.mc2.nav23: prediction 1234574.38")
        and "x 1234574.457" in line
        for line in trace
    )


def test_a_reject_is_logged_at_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a counted reject with its innovation, scale and count (16.2)."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    outlier = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34779,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, outlier], last, tmp_path), last
    )
    assert at(records, "WARNING") == [
        "das_a.mc2.nav23 rejected: innovation 202.6 ps, scale 3.0 ps, 1 consecutive"
    ]


def test_a_phase_step_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a phase step with its size and the step offset (16.2)."""
    stepping = WORKED_LAST.model_copy(
        update={
            "flags": "R",
            "consecutive_rejects": 2,
            "rejects": ((PREVIOUS - T, 150.0), (PREVIOUS, 150.0)),
            "epochs_since_accept": 2,
        }
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): stepping}
    moved = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 150 - 3,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, moved], last, tmp_path), last
    )
    assert "das_a.mc2.nav23 phase step of 150 ps; step offset 150 ps" in at(
        records, "INFO"
    )


def test_a_cold_start_and_dormancy_are_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a series that cold-starts and one that goes dormant (16.2)."""
    acquiring = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS - T, 1_234_579.0), (PREVIOUS, 1_234_579.0)),
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): acquiring}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    info = at(logged(caplog, epoch, last), "INFO")
    assert "das_a.mc2.nav23 cold start: segment 2" in info
    stopping = {
        **REFERENCE_LAST,
        ("mc1", "mc1"): last_row(x_fs=1_000_000, epochs_since_accept=40),
    }
    epoch = epoch_of(REFERENCE_MEASURED[1:], stopping, tmp_path / "second")
    info = at(logged(caplog, epoch, stopping), "INFO")
    assert "das_a.mc1.mc1 dormant" in info


def test_a_configuration_change_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a warm start for new time constants (16.2)."""
    older = WORKED_LAST.model_copy(update={"time_constant": 80.0})
    last = {**REFERENCE_LAST, ("mc2", "nav23"): older}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    info = at(logged(caplog, epoch, last), "INFO")
    assert (
        "das_a.mc2.nav23 configuration change: M 80.0 to 100.0, M_sigma 50.0 to 50.0"
        in info
    )


def test_a_frequency_step_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a frequency step and the segment it starts (16.2)."""
    ramping = WORKED_LAST.model_copy(
        update={
            "flags": "R",
            "consecutive_rejects": 2,
            "rejects": ((PREVIOUS - T, 30.0), (PREVIOUS, 60.0)),
            "epochs_since_accept": 2,
        }
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): ramping}
    moved = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 90 - 3,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    info = at(
        logged(caplog, epoch_of([*REFERENCE_MEASURED, moved], last, tmp_path), last),
        "INFO",
    )
    assert "das_a.mc2.nav23 frequency step: segment 5" in info


def test_screening_and_slip_events_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log screening failures at WARNING and a corrected slip at INFO (16.2)."""
    maser = last_row(
        x_fs=2_000_000,
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        innovation_scale=40.0,
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): maser}
    shifted = [
        measured("mc1", "mc1", 1000, 10),
        measured("mc2", "mc2", 2100, 20),
        measured("mc1", "mc2", 5000, 30),
        measured("mc2", "mc1", PHASE_PERIOD - 5000, 40),
        measured("mc2", "nav23", 2008, 50),
    ]
    warnings = at(logged(caplog, epoch_of(shifted, last, tmp_path), last), "WARNING")
    assert "self-measurement of mc2 failed: excluded das_a.mc2.nav23" in warnings


def test_a_missing_self_measurement_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a self pair with a prediction but no measurement at WARNING (10.1)."""
    last = dict(REFERENCE_LAST)
    epoch = epoch_of(REFERENCE_MEASURED[1:], last, tmp_path)
    assert "self-measurement of mc1 missing" in at(
        logged(caplog, epoch, last), "WARNING"
    )


def test_a_reciprocity_and_a_closure_failure_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the link directions reciprocity excludes, and links closure excludes."""
    last = dict(REFERENCE_LAST)
    bad = [
        *REFERENCE_MEASURED[:2],
        measured("mc1", "mc2", 5100, 30),
        REFERENCE_MEASURED[3],
    ]
    warnings = at(logged(caplog, epoch_of(bad, last, tmp_path), last), "WARNING")
    assert (
        "reciprocity of mc1-mc2 failed: excluded das_a.mc1.mc2, das_a.mc2.mc1"
        in warnings
    )


def test_slips_corrected_and_undecided_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a corrected slip at INFO, as the slip check found it."""
    jump = PHASE_PERIOD // 2 + 2
    clock = {
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 3.0,
    }
    last = {
        **REFERENCE_LAST,
        ("mc1", "nav23"): last_row(
            x_fs=1_000_000_000, epochs_in_segment=10, flags="AU", **clock
        ),
        ("mc2", "nav23"): last_row(x_fs=2_000_000_000, **clock),
    }
    clocks = [
        measured("mc1", "nav23", (1_000_000 + jump) % PHASE_PERIOD, 50),
        measured("mc2", "nav23", (2_000_000 + jump - 4) % PHASE_PERIOD, 60),
    ]
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path), last
    )
    assert "das_a.mc1.nav23 slip corrected: +1 cycles" in at(records, "INFO")
    wide = {**clock, "innovation_scale": 25_000.0}
    last[("mc1", "nav23")] = last_row(x_fs=1_000_000_000, **wide)
    last[("mc2", "nav23")] = last_row(x_fs=2_000_000_000, **wide)
    epoch = epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path / "second")
    warnings = at(logged(caplog, epoch, last), "WARNING")
    assert (
        "slip of clock nav23 undecided: excluded das_a.mc1.nav23, das_a.mc2.nav23"
        in warnings
    )


def test_a_closure_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each link closure excludes, at WARNING (10.3)."""
    refs = ("mc1", "mc2", "mc3")
    values = {"mc1": 1000, "mc2": 2000, "mc3": 3000}
    last: dict[SeriesKey, Row] = {}
    raws = []
    for index, (a, b) in enumerate((a, b) for a in refs for b in refs):
        x = values[a] if a == b else 1000 * (int(a[-1]) - int(b[-1]))
        error = (
            50 if (a, b) == ("mc2", "mc3") else -50 if (a, b) == ("mc3", "mc2") else 0
        )
        last[(a, b)] = last_row(x_fs=x * 1000)
        raws.append(measured(a, b, (x + error) % PHASE_PERIOD, 10 * (index + 1)))
    warnings = at(logged(caplog, epoch_of(raws, last, tmp_path), last), "WARNING")
    assert (
        "closure of link mc2-mc3 failed: excluded das_a.mc2.mc3, das_a.mc3.mc2"
        in warnings
    )


def test_a_first_run_starts_at_the_first_data(tmp_path: Path) -> None:
    """Start at the first block when no series exists yet, so each step progresses."""
    config, clocks = loop_deployment(tmp_path, LATE - 3 * T)
    das_files(tmp_path, [LATE, LATE + T])
    run.run(config, clocks, 1, ShutdownHandler())
    assert [row.interpolated_datetime for row in rows_of(config, ("mc1", "mc1"))] == [
        LATE
    ]
    run.run(config, clocks, 1, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 2


def test_a_link_not_accepted_does_not_make_the_triple_cold(tmp_path: Path) -> None:
    """Take a link rejected for its RMS by its prediction, not marked cold (12.6)."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): WORKED_LAST,
        ("mc1", "mc2", "nav23"): triple_last(x_fs=1_239_577_000),
    }
    link = REFERENCE_MEASURED[3]
    noisy = DASMeasurement.model_validate(
        {
            "measurement_mjd": link.measurement_mjd,
            "measured_phase": link.measured_phase,
            "rms": 99,
            "switch": link.switch,
            "clock": link.clock,
        }
    )
    epoch = epoch_of([*REFERENCE_MEASURED[:3], noisy, WORKED_RAW], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    assert "A" not in pairs.results[("mc2", "mc1")].row.flags
    done = run.process_triples(epoch, last, pairs)
    remote = done.measurements[("mc1", "mc2", "nav23")]
    assert (remote.components_used, remote.cold) == ("110", False)
    assert done.results[("mc1", "mc2", "nav23")].row.flags == "A"


def test_each_series_takes_the_settings_in_force_at_its_epoch(tmp_path: Path) -> None:
    """Give the clock entry that took effect at or before E, not a later one (8.1)."""
    config, _ = deployment(tmp_path)
    dated = CLOCKS.replace(
        "  nav23: [{type: maser}]\n",
        "  nav23: [{type: maser},"
        f" {{effective_mjd: {datetime_to_mjd(E)}, time_constant: 150.0}},"
        f" {{effective_mjd: {datetime_to_mjd(E + T)}, time_constant: 200.0}}]\n",
    )
    path = tmp_path / "dated.yaml"
    path.write_text(dated, encoding="utf-8")
    epoch = run.build_epoch(E, block(MEASURED), NONE, config, read_clock_config(path))
    assert epoch.params[("mc2", "nav23")].M == 150.0


def test_the_prediction_and_update_are_logged_in_full(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log at TRACE each series' prediction, innovation, x, y and d, as they are."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "nav23"): WORKED_LAST,
        ("mc1", "mc2", "nav23"): triple_last(x_fs=1_239_577_000),
    }
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    trace = at(logged(caplog, epoch, last), "TRACE")
    assert (
        "das_a.mc2.nav23: prediction 1234574.38, innovation 2.62, x 1234574.457,"
        " y 0.01230129052352643, d 7.169515400974333e-12"
    ) in trace
    assert (
        "das_a.mc1.mc2.nav23: prediction 1239577.0, innovation 0.0, x 1239577.000,"
        " y 0.0, d 0.0"
    ) in trace


def test_a_phase_step_logs_the_step_alone_and_the_new_offset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the step as the change in offset, and the offset it reaches (16.2)."""
    stepping = WORKED_LAST.model_copy(
        update={
            "flags": "R",
            "consecutive_rejects": 2,
            "rejects": ((PREVIOUS - T, 150.0), (PREVIOUS, 150.0)),
            "epochs_since_accept": 2,
            "step_offset": 40,
        }
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): stepping}
    moved = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 150 - 3,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    records = logged(
        caplog, epoch_of([*REFERENCE_MEASURED, moved], last, tmp_path), last
    )
    assert "das_a.mc2.nav23 phase step of 150 ps; step offset 190 ps" in at(
        records, "INFO"
    )


def test_a_frequency_step_with_a_configuration_change_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log both when new settings and a frequency step come in one epoch (16.2)."""
    ramping = WORKED_LAST.model_copy(
        update={
            "flags": "R",
            "consecutive_rejects": 2,
            "rejects": ((PREVIOUS - T, 30.0), (PREVIOUS, 60.0)),
            "epochs_since_accept": 2,
            "time_constant": 80.0,
        }
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): ramping}
    moved = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 90 - 3,
        rms=3,
        switch="2B07",
        clock="nav23",
    )
    info = at(
        logged(caplog, epoch_of([*REFERENCE_MEASURED, moved], last, tmp_path), last),
        "INFO",
    )
    assert any(m.startswith("das_a.mc2.nav23 configuration change") for m in info)
    assert "das_a.mc2.nav23 frequency step: segment 6" in info


def test_a_cold_start_logs_no_step(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a cold start from dormancy as a cold start, never as a step (16.2)."""
    acquiring = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS - T, 1_234_579.0), (PREVIOUS, 1_234_579.0)),
    )
    last = {**REFERENCE_LAST, ("mc2", "nav23"): acquiring}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    info = at(logged(caplog, epoch, last), "INFO")
    mine = [m for m in info if m.startswith("das_a.mc2.nav23 ")]
    assert mine == ["das_a.mc2.nav23 cold start: segment 2"]


def test_an_epoch_s_log_names_the_series_of_its_channel(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Name each series in the log by the run's RF channel."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE])
    (first,) = read_all_blocks(tmp_path / "das", datetime_to_mjd(LATE))
    files.ensure_archives(config.processed.processed_path)
    with caplog.at_level(logging.DEBUG, logger=RUN_LOGGER):
        run.process_epoch(LATE, first, files.DayBuffer("a"), config, clocks)
    debug = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert "das_a.mc1.mc1: RD" in debug


def test_each_series_last_row_is_read_from_its_file(tmp_path: Path) -> None:
    """Give each series' last row, as its file holds it, for every series."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE, LATE + T])
    run.run(config, clocks, None, ShutdownHandler())
    last = run.read_last_state(config)
    assert sorted(last) == sorted(SERIES)
    for key in SERIES:
        assert last[key] == rows_of(config, key)[-1], key


def das_lines(tmp_path: Path, epochs: list[list[tuple[str, str, int]]]) -> None:
    """Write one DAS day file: each epoch's (reference, clock, phase) from LATE."""
    lines = []
    for index, measured_here in enumerate(epochs):
        start = datetime_to_mjd(LATE + index * T)
        for slot, (reference, clock, phase) in enumerate(measured_here):
            raw = DASMeasurement(
                measurement_mjd=round(start + (slot + 1) * 2e-5, 6),
                measured_phase=phase,
                rms=3,
                switch=f"{reference[-1]}A{slot:02d}",
                clock=clock,
            )
            lines.append(f"{raw}\n")
    day = int(datetime_to_mjd(LATE))
    (tmp_path / "das" / f"cd5m5m_{day}.dat").write_text("".join(lines))


def test_a_remote_triple_goes_on_when_its_reference_is_missing(
    tmp_path: Path,
) -> None:
    """Give an existing triple a row at an epoch its reference r was not measured."""
    config, clocks = loop_deployment(tmp_path)
    both = [
        ("mc1", "mc1", 1000),
        ("mc1", "mc2", 5000),
        ("mc2", "mc1", PHASE_PERIOD - 5000),
        ("mc2", "mc2", 2000),
        ("mc2", "nav23", 50_000),
    ]
    das_lines(tmp_path, [both, both, [both[3], both[4]]])
    run.run(config, clocks, None, ShutdownHandler())
    rows = rows_of(config, ("mc1", "mc2", "nav23"))
    assert [row.interpolated_datetime for row in rows] == [LATE, LATE + T, LATE + 2 * T]


def test_a_run_with_no_data_and_no_series_does_nothing(tmp_path: Path) -> None:
    """Finish without a row when there is neither data nor a file yet."""
    config, clocks = loop_deployment(tmp_path)
    run.run(config, clocks, None, ShutdownHandler())
    assert run.data_series(config) == []


def test_a_gap_at_the_start_of_a_run_is_predicted_not_skipped(tmp_path: Path) -> None:
    """Give the epoch after the files' end a row, though the DAS skipped it (6.2)."""
    config, clocks = loop_deployment(tmp_path)
    das_files(tmp_path, [LATE, LATE + 2 * T])
    run.run(config, clocks, 1, ShutdownHandler())
    run.run(config, clocks, 1, ShutdownHandler())
    rows = rows_of(config, ("mc1", "mc1"))
    assert [row.interpolated_datetime for row in rows] == [LATE, LATE + T]
    assert "P" in rows[1].flags


def test_a_clock_measured_with_an_rms_of_zero_gives_its_triple_a_row(
    tmp_path: Path,
) -> None:
    """Measure a local triple whose clock pair's rms is 0, with a sigma of 0."""
    last = {**REFERENCE_LAST, ("mc2", "nav23"): WORKED_LAST}
    still = DASMeasurement.model_validate(
        {
            "measurement_mjd": WORKED_RAW.measurement_mjd,
            "measured_phase": WORKED_RAW.measured_phase,
            "rms": 0,
            "switch": WORKED_RAW.switch,
            "clock": WORKED_RAW.clock,
        }
    )
    epoch = epoch_of([*REFERENCE_MEASURED, still], last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    assert done.measurements[("mc2", "mc2", "nav23")].double_difference_sigma == 0.0


def test_a_new_file_left_without_its_rows_by_a_stopped_write_is_made_again(
    tmp_path: Path,
) -> None:
    """Delete a file whose rows a crash never wrote, when the journal shows why."""
    clean, clocks = loop_deployment(tmp_path / "clean")
    das_files(tmp_path / "clean", [LATE + i * T for i in range(6)])
    run.run(clean, clocks, None, ShutdownHandler())
    stopped, clocks = loop_deployment(tmp_path / "stopped")
    das_files(tmp_path / "stopped", [LATE + i * T for i in range(6)])
    run.run(stopped, clocks, 2, ShutdownHandler())
    processed = stopped.processed.processed_path
    path = registry.series_file(processed, "a", ("mc1", "nav23"))
    path.write_bytes(
        path.read_bytes()[: files.MEAS_HEADER_LINES * (files.MEAS_WIDTH + 1)]
        + b"\0" * (files.MEAS_WIDTH + 1) * 2
    )
    with pytest.raises(DataFileError, match="damaged first row"):
        run.next_epoch(stopped)
    journal = processed / JOURNAL_FILE_TEMPLATE.format(rf="a")
    journal.write_text(f"{LATE.isoformat()}\n", encoding="ascii")
    assert run.next_epoch(stopped) == LATE
    assert not path.exists()
    run.run(stopped, clocks, None, ShutdownHandler())
    for key in SERIES:
        one = registry.series_file(clean.processed.processed_path, "a", key)
        other = registry.series_file(processed, "a", key)
        assert one.read_bytes() == other.read_bytes(), key
