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
"""

import math
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
"""Appendix A's last row of (mc2, ox23)."""


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
    last = {**REFERENCE_LAST, ("mc2", "ox23"): WORKED_LAST}
    raw = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    epoch = epoch_of([*REFERENCE_MEASURED, raw], last, tmp_path)
    done = run.process_pairs(epoch, last)
    worked = done.results[("mc2", "ox23")].row
    assert (worked.flags, worked.x_fs, worked.y, worked.d) == (
        "A",
        1_234_574_457,
        0.01230129052352643,
        7.169515400974333e-12,
    )
    assert done.measurements[("mc2", "ox23")].z == 1_234_577
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
        ("mc1", "ox23"): last_row(
            x_fs=1_000_000_000,
            filter_states=3,
            time_constant=100.0,
            scale_time_constant=50.0,
            innovation_scale=3.0,
            epochs_in_segment=10,
            flags="AU",
        ),
        ("mc2", "ox23"): last_row(
            x_fs=2_000_000_000,
            filter_states=3,
            time_constant=100.0,
            scale_time_constant=50.0,
            innovation_scale=3.0,
        ),
    }
    truth = {("mc1", "ox23"): 1_000_000 + jump, ("mc2", "ox23"): 2_000_000 + jump - 4}
    clocks = [
        measured("mc1", "ox23", truth[("mc1", "ox23")] % PHASE_PERIOD, 50),
        measured("mc2", "ox23", truth[("mc2", "ox23")] % PHASE_PERIOD, 60),
    ]
    epoch = epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert done.slips.corrections == {("mc1", "ox23"): 1}
    assert done.measurements[("mc1", "ox23")].z == truth[("mc1", "ox23")]
    assert done.measurements[("mc1", "ox23")].slip is True
    corrected = done.results[("mc1", "ox23")].row
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
    last = {**REFERENCE_LAST, ("mc2", "ox23"): maser}
    shifted = [
        measured("mc1", "mc1", 1000, 10),
        measured("mc2", "mc2", 2100, 20),
        measured("mc1", "mc2", 5000, 30),
        measured("mc2", "mc1", PHASE_PERIOD - 5000, 40),
        measured("mc2", "ox23", 2008, 50),
    ]
    epoch = epoch_of(shifted, last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert [event.kind for event in done.screening.events] == ["self_fail"]
    assert done.screening.excluded == frozenset({("mc2", "ox23")})
    assert done.results[("mc2", "ox23")].row.flags == "X"
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
        ("mc1", "ox23"): last_row(x_fs=1_000_000_000, **wide),
        ("mc2", "ox23"): last_row(x_fs=2_000_000_000, **wide),
    }
    clocks = [
        measured("mc1", "ox23", (1_000_000 + jump) % PHASE_PERIOD, 50),
        measured("mc2", "ox23", (2_000_000 + jump - 4) % PHASE_PERIOD, 60),
    ]
    epoch = epoch_of([*REFERENCE_MEASURED, *clocks], last, tmp_path)
    done = run.process_pairs(epoch, last)
    assert [event.kind for event in done.slips.events] == ["slip_undecided"]
    assert done.results[("mc1", "ox23")].row.flags == "X"
    assert done.results[("mc2", "ox23")].row.flags == "X"


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
    clock="ox23",
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
    last = {**REFERENCE_LAST, ("mc2", "ox23"): WORKED_LAST}
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    done = run.process_triples(epoch, last, pairs)
    remote = done.measurements[("mc1", "mc2", "ox23")]
    assert (remote.z, remote.components_used) == (1_234_577 + 5_000, "111")
    assert remote.double_difference_sigma == math.sqrt(9 + 0.25 * (9 + 9))
    local = done.measurements[("mc2", "mc2", "ox23")]
    assert (local.z, local.double_difference_sigma) == (1_234_577, 3.0)
    assert {r.row.flags for r in done.results.values()} == {"RD"}


def test_a_tracked_triple_is_filtered_on_its_double_difference(tmp_path: Path) -> None:
    """Accept a triple's dd against its own prediction."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "ox23"): WORKED_LAST,
        ("mc1", "mc2", "ox23"): triple_last(x_fs=1_239_577_000),
    }
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    row = done.results[("mc1", "mc2", "ox23")].row
    assert (row.flags, row.innovation) == ("A", 0.0)
    assert row.x_fs == 1_239_577_000


def test_a_missing_link_direction_uses_the_predicted_round_trip(tmp_path: Path) -> None:
    """Give 110 when (s, r) was not measured, through the links' predictions (12.2)."""
    last = {**REFERENCE_LAST, ("mc2", "ox23"): WORKED_LAST}
    epoch = epoch_of([*REFERENCE_MEASURED[:3], WORKED_RAW], last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    remote = done.measurements[("mc1", "mc2", "ox23")]
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
        ("mc2", "ox23"): acquiring,
        ("mc2", "mc2", "ox23"): triple_last(x_fs=1_234_577_000),
    }
    epoch = epoch_of([*REFERENCE_MEASURED, WORKED_RAW], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    assert pairs.results[("mc2", "ox23")].cold is True
    done = run.process_triples(epoch, last, pairs)
    assert done.measurements[("mc2", "mc2", "ox23")].cold is True
    row = done.results[("mc2", "mc2", "ox23")].row
    assert (row.flags, row.rejects) == ("RD", ((E, 1_234_579.0),))


def test_a_triple_without_its_clock_pair_holds(tmp_path: Path) -> None:
    """Give a triple a predicted row when its clock pair has no measurement."""
    last = {
        **REFERENCE_LAST,
        ("mc2", "ox23"): WORKED_LAST,
        ("mc2", "mc2", "ox23"): triple_last(x_fs=1_234_577_000),
    }
    epoch = epoch_of(REFERENCE_MEASURED, last, tmp_path)
    done = run.process_triples(epoch, last, run.process_pairs(epoch, last))
    assert done.results[("mc2", "mc2", "ox23")].row.flags == "P"
    assert done.measurements == {}


def test_the_local_triple_is_checked_every_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass the self pair for both links of a local triple, so the check runs (12.4)."""
    seen: list[tuple[object, object]] = []

    def spy(triple: TripleKey, sc: Component, rs: Component, sr: Component) -> object:
        """Record the links a local triple is given."""
        if triple[0] == triple[1]:
            seen.append((rs, sr))
        return double_difference(triple, sc, rs, sr)

    monkeypatch.setattr(run, "double_difference", spy)
    last = {**REFERENCE_LAST, ("mc2", "ox23"): WORKED_LAST}
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
        ("mc2", "ox23"): WORKED_LAST,
        ("mc2", "mc2", "ox23"): triple_last(x_fs=1_234_577_000),
    }
    outlier = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34779,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    epoch = epoch_of([*REFERENCE_MEASURED, outlier], last, tmp_path)
    pairs = run.process_pairs(epoch, last)
    assert pairs.results[("mc2", "ox23")].row.flags == "R"
    done = run.process_triples(epoch, last, pairs)
    assert ("mc2", "mc2", "ox23") not in done.measurements
    assert done.results[("mc2", "mc2", "ox23")].row.flags == "P"
