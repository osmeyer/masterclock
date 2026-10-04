"""Tests for src/masterclock/das_processor/run.py.

The rules covered: building an epoch resolves everything it needs: its
references from its block; every pair, the existing ones kept, and every
triple, existing or new, whose s and c are in one building at the epoch;
the steering of every reference that steers a series, read over
(E - T, E + T]; and each series' settings, a pair taking its second
clock's entry and RMS limit and a triple its clock c's entry, kept from
the last epoch of the run while its series and every clock's settings are
unchanged, and every clock's location, kept while no clock's entry
changes; the steering files are read once in a run, each line parsed
once; an epoch with no block has no references and the existing series
only; and the epoch is checked to hold settings for exactly its
series.

The pairs of an epoch are predicted, decycled against the prediction or the
anchor with the steering inside the epoch taken off, screened, checked for
slips, corrected before filtering, and filtered with what screening and the
slip check excluded. The triples are built from the pairs' accepted
measurements of the same epoch, the local triple given its self pair for
both links, marked cold when a pair cold-started, and filtered.

The run's events are logged at the design's levels: each epoch with its
counts of rows written at INFO; steps, cold starts, dormancy, a series that
stops, configuration changes and corrected slips at INFO; rejects,
screening failures, a missing self pair and undecided slips at WARNING;
each series' outcome at DEBUG and its prediction and update at TRACE.

The next epoch is one after the newest epoch any file is good through: a
damaged file is cut back to its last good row and the rest left; after a
write that stopped part way, every file is cut back to before the epoch its
journal names, the journal then deleted, and a file whose first row is
damaged is refused without a journal and deleted with one, to be made
again; with no file, the epoch containing start_from_mjd. A roll-back is
logged once at WARNING, with its reason, what it did and the newest row
left, and nothing is logged when nothing was cut. A series whose newest row
is not of the epoch before starts cold, in the segment after its newest
row's, as a triple does when its clock comes back to its building, run in
one go or epoch by epoch.

Each series takes the settings in force at its epoch; a link not accepted
does not make its triple cold; the TRACE lines and steps with a settings
change are logged as they are; each series' last row is read from its
file; a remote
triple goes on when its reference is missing; a run with no data and no
files does nothing; and a gap at the start of a run is predicted.

A series dormant with no measurement writes no row: one whose measurements
stop writes predicted rows up to its gap limit and then none, and on its
return starts again in its next segment, in one go or epoch by epoch; and
an epoch that writes no row is not counted as a step, so a run of one step
at a time gets past a gap no series writes.
"""

import dataclasses
import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.exceptions import ConfigError
from masterclock.app.log import TRACE
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor import files, read_steering, registry, run
from masterclock.das_processor.clock_config import ClockConfig, read_clock_config
from masterclock.das_processor.config import JOURNAL_FILE_TEMPLATE, AppConfig
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import (
    DASData,
    DASMeasurement,
    read_all_blocks,
)
from masterclock.das_processor.read_steering import STEERING_FILE_TEMPLATE
from masterclock.das_processor.registry import ExistingSeries
from masterclock.domain.double_difference import Component, double_difference
from masterclock.domain.phase import PHASE_PERIOD
from masterclock.domain.series import Row, SeriesKey, TripleKey, check_row
from masterclock.domain.steering import steer_u

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented epoch start."""

T: Final = timedelta(minutes=10)
"""One epoch."""

NO_SERIES: Final = ExistingSeries(pairs=frozenset(), triples=frozenset())
"""No series yet."""

CLOCK_CONFIG_YAML: Final = (
    "rejects_before_restart: 6\n"
    "rms_limit: {default: 50, pairs: {mc2.ox23: 80}}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 40}\n"
    "  mc: {filter_states: 1, scale_time_constant: 30.0,"
    " initial_innovation_scale: 2.0, gap_limit: 40}\n"
    "clocks:\n"
    "  mc1: [{type: mc, location: 1}]\n"
    "  mc2: [{type: mc, location: 1}]\n"
    "  mc3: [{type: mc, location: 1}]\n"
    "  ox23: [{type: maser, location: 1}]\n"
)
"""An invented clock configuration: three references and a maser."""


def buffered_lines(day_buffer: files.DayBuffer) -> dict[Path, list[str]]:
    """Copy each file's buffered lines, so a later change to the buffer shows."""
    return {
        data_file: list(file_lines)
        for data_file, file_lines in day_buffer.file_lines.items()
    }


def make_deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Make an invented deployment's directories and give its configuration."""
    for directory_name in ("das", "steering", "processed"):
        (tmp_path / directory_name).mkdir(parents=True)
    clock_config_file = tmp_path / "clock_config.yaml"
    clock_config_file.write_text(CLOCK_CONFIG_YAML, encoding="utf-8")
    config = AppConfig.model_validate(
        {
            "das": {
                "rf": "a",
                "cd5m5m_path": tmp_path / "das",
                "steering_path": tmp_path / "steering",
            },
            "processed": {
                "processed_path": tmp_path / "processed",
                "start_from_mjd": None,
                "clock_config_file": clock_config_file,
                "num_workers": None,
            },
            "logging": {"log_file": None, "log_level": None, "backup_count": None},
        }
    )
    return config, read_clock_config(clock_config_file)


def das_block_of(measured_pairs: list[tuple[str, str]]) -> DASData:
    """Give a block measuring each (reference, clock) pair once."""
    epoch_start_mjd = datetime_to_mjd(E)
    return DASData(
        interpolated_datetime=E,
        measurements=tuple(
            DASMeasurement(
                measurement_mjd=round(
                    epoch_start_mjd + (measurement_index + 1) * 2e-6, 6
                ),
                measured_phase=1000,
                rms=3,
                switch=f"{reference[-1]}A{measurement_index:02d}",
                clock=clock,
            )
            for measurement_index, (reference, clock) in enumerate(measured_pairs)
        ),
    )


MEASURED_PAIRS: Final = [
    ("mc1", "mc1"),
    ("mc2", "mc2"),
    ("mc1", "mc2"),
    ("mc2", "mc1"),
    ("mc2", "ox23"),
]
"""Two references measured against each other, and a maser against one."""


def test_an_epoch_holds_its_references_and_series(tmp_path: Path) -> None:
    """Give the block's references, pairs and triples, sorted (3.1, 3.4)."""
    config, clock_config = make_deployment(tmp_path)
    epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS), NO_SERIES, config, clock_config
    )
    assert epoch.interpolated_datetime == E
    assert epoch.refs == frozenset({"mc1", "mc2"})
    assert epoch.pairs == tuple(sorted(MEASURED_PAIRS))
    assert epoch.triples == tuple(
        sorted((r, s, c) for r in ("mc1", "mc2") for s, c in MEASURED_PAIRS)
    )


def test_a_series_takes_the_entry_of_its_clock_side(tmp_path: Path) -> None:
    """Give a pair its second clock's entry and a triple its clock c's (8.1)."""
    config, clock_config = make_deployment(tmp_path)
    epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS), NO_SERIES, config, clock_config
    )
    pair_params = epoch.series_params[("mc2", "ox23")]
    assert (pair_params.filter_states, pair_params.M, pair_params.rms_max) == (
        3,
        100.0,
        80,
    )
    link_params = epoch.series_params[("mc1", "mc2")]
    assert (
        link_params.filter_states,
        link_params.M,
        link_params.sigma0,
        link_params.rms_max,
    ) == (1, None, 2.0, 50)
    triple_params = epoch.series_params[("mc1", "mc2", "ox23")]
    assert (triple_params.filter_states, triple_params.M, triple_params.rms_max) == (
        3,
        100.0,
        None,
    )
    assert set(epoch.series_params) == set(epoch.pairs) | set(epoch.triples)


def test_steering_is_read_for_every_reference_over_the_epoch_either_side(
    tmp_path: Path,
) -> None:
    """Read each steering reference's events in (E - T, E + T] (I4)."""
    config, clock_config = make_deployment(tmp_path)
    inside_times = [E - T + timedelta(seconds=60), E + T - timedelta(seconds=30)]
    outside_times = [E - T - timedelta(seconds=30), E + T + timedelta(seconds=60)]
    event_times = sorted(inside_times + outside_times)
    steering_text = "".join(
        f"{datetime_to_mjd(event_time):.6f} 1.0 0.0\n" for event_time in event_times
    )
    for mc in ("mc1", "mc2"):
        (tmp_path / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)).write_text(
            steering_text
        )
    epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS), NO_SERIES, config, clock_config
    )
    assert sorted(epoch.steering) == ["mc1", "mc2"]
    expected_times = [
        mjd_to_datetime(float(f"{datetime_to_mjd(event_time):.6f}"))
        for event_time in inside_times
    ]
    for steer_events in epoch.steering.values():
        assert [
            steer_event.applied_datetime for steer_event in steer_events
        ] == expected_times


def test_an_epoch_with_no_block_has_the_existing_series_only(tmp_path: Path) -> None:
    """Give no references and every existing series when the DAS measured nothing."""
    config, clock_config = make_deployment(tmp_path)
    earlier_series = ExistingSeries(
        pairs=frozenset({("mc2", "ox23"), ("mc2", "mc2")}),
        triples=frozenset({("mc2", "mc2", "ox23")}),
    )
    epoch = run.build_epoch(E, None, earlier_series, config, clock_config)
    assert (epoch.das_block, epoch.refs) == (None, frozenset())
    assert epoch.pairs == (("mc2", "mc2"), ("mc2", "ox23"))
    assert epoch.triples == (("mc2", "mc2", "ox23"),)
    assert sorted(epoch.steering) == ["mc2"]


def test_a_clock_with_no_entry_stops_the_epoch(tmp_path: Path) -> None:
    """Raise ConfigError for a measured clock the clock configuration lacks."""
    config, clock_config = make_deployment(tmp_path)
    with pytest.raises(ConfigError, match="hm9"):
        run.build_epoch(
            E,
            das_block_of([*MEASURED_PAIRS, ("mc2", "hm9")]),
            NO_SERIES,
            config,
            clock_config,
        )


def test_an_epoch_holds_settings_for_exactly_its_series(tmp_path: Path) -> None:
    """Refuse an epoch whose settings miss a series or name another."""
    config, clock_config = make_deployment(tmp_path)
    epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS), NO_SERIES, config, clock_config
    )
    fewer_params = {
        series_key: series_params
        for series_key, series_params in epoch.series_params.items()
        if series_key != ("mc1", "mc1")
    }
    with pytest.raises(ValueError, match="settings"):
        dataclasses.replace(epoch, series_params=fewer_params)


def test_an_epoch_s_block_is_of_its_epoch(tmp_path: Path) -> None:
    """Refuse an epoch holding another epoch's block."""
    config, clock_config = make_deployment(tmp_path)
    epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS), NO_SERIES, config, clock_config
    )
    with pytest.raises(ValueError, match="block"):
        dataclasses.replace(epoch, interpolated_datetime=E + T)


# ------------------------------------------------------------ pairs of an epoch

PREVIOUS_EPOCH: Final = E - T
"""The epoch before E."""


def last_row(**changed_fields: object) -> Row:
    """Give a settled 1-state reference row before E, with ``changed_fields`` made."""
    row_fields: dict[str, object] = {
        "interpolated_datetime": PREVIOUS_EPOCH,
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
    row_fields.update(changed_fields)
    row = Row(**row_fields)  # type: ignore[arg-type]
    check_row(row)
    return row


WORKED_LAST_ROW: Final = last_row(
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


def das_measurement_of(
    reference: str, clock: str, measured_phase: int, offset_us: int
) -> DASMeasurement:
    """Give a measurement of the pair ``offset_us`` microdays after E."""
    return DASMeasurement(
        measurement_mjd=round(datetime_to_mjd(E) + offset_us * 1e-6, 6),
        measured_phase=measured_phase,
        rms=3,
        switch=f"{reference[-1]}A01",
        clock=clock,
    )


REFERENCE_LAST_ROWS: Final[dict[SeriesKey, Row]] = {
    ("mc1", "mc1"): last_row(x_fs=1_000_000),
    ("mc2", "mc2"): last_row(x_fs=2_000_000),
    ("mc1", "mc2"): last_row(x_fs=5_000_000),
    ("mc2", "mc1"): last_row(x_fs=-5_000_000),
}
"""The references' last rows: their self and link pairs, 1-state, settled."""

REFERENCE_MEASUREMENTS: Final = [
    das_measurement_of("mc1", "mc1", 1000, 10),
    das_measurement_of("mc2", "mc2", 2000, 20),
    das_measurement_of("mc1", "mc2", 5000, 30),
    das_measurement_of("mc2", "mc1", PHASE_PERIOD - 5000, 40),
]
"""Each self and link pair measured where its last row says it is."""


def epoch_of(
    measurements: list[DASMeasurement],
    last_rows: dict[SeriesKey, Row],
    tmp_path: Path,
    steering: dict[str, str] | None = None,
) -> run.Epoch:
    """Build E from ``measurements``, with ``last_rows``'s series existing.

    ``steering`` gives the text of each reference's steering file.
    """
    config, clock_config = make_deployment(tmp_path)
    for mc, steering_text in (steering or {}).items():
        steering_file = tmp_path / "steering" / STEERING_FILE_TEMPLATE.format(mc=mc)
        steering_file.write_text(steering_text, encoding="ascii")
    das_block = DASData(interpolated_datetime=E, measurements=tuple(measurements))
    earlier_series = ExistingSeries(
        pairs=frozenset(
            (series_key[0], series_key[1])
            for series_key in last_rows
            if len(series_key) == 2
        ),
        triples=frozenset(
            (series_key[0], series_key[1], series_key[-1])
            for series_key in last_rows
            if len(series_key) == 3
        ),
    )
    return run.build_epoch(E, das_block, earlier_series, config, clock_config)


def test_the_worked_epoch_s_pairs_are_processed_end_to_end(tmp_path: Path) -> None:
    """Give Appendix A's row, and every reference pair accepted at its value (D3)."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    worked_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    epoch = epoch_of([*REFERENCE_MEASUREMENTS, worked_measurement], last_rows, tmp_path)
    pair_step = run.process_pairs(epoch, last_rows)
    worked_row = pair_step.step_results[("mc2", "ox23")].row
    assert (worked_row.flags, worked_row.x_fs, worked_row.y, worked_row.d) == (
        "A",
        1_234_574_457,
        0.01230129052352643,
        7.169515400974333e-12,
    )
    assert pair_step.measurements[("mc2", "ox23")].z == 1_234_577
    for series_key, row in REFERENCE_LAST_ROWS.items():
        result_row = pair_step.step_results[(series_key[0], series_key[1])].row
        assert (result_row.flags, result_row.x_fs) == ("A", row.x_fs), series_key
    assert pair_step.screening.events == ()
    assert pair_step.slips.events == ()


def test_a_slip_correction_is_made_before_filtering(tmp_path: Path) -> None:
    """Correct the weak pair's cycle count first, so its row holds S and the truth."""
    cycle_jump = PHASE_PERIOD // 2 + 2
    last_rows = {
        **REFERENCE_LAST_ROWS,
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
    true_phases = {
        ("mc1", "ox23"): 1_000_000 + cycle_jump,
        ("mc2", "ox23"): 2_000_000 + cycle_jump - 4,
    }
    clock_measurements = [
        das_measurement_of(
            "mc1", "ox23", true_phases[("mc1", "ox23")] % PHASE_PERIOD, 50
        ),
        das_measurement_of(
            "mc2", "ox23", true_phases[("mc2", "ox23")] % PHASE_PERIOD, 60
        ),
    ]
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, *clock_measurements], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    assert pair_step.slips.corrections == {("mc1", "ox23"): 1}
    assert pair_step.measurements[("mc1", "ox23")].z == true_phases[("mc1", "ox23")]
    assert pair_step.measurements[("mc1", "ox23")].slip is True
    corrected_row = pair_step.step_results[("mc1", "ox23")].row
    assert "S" in corrected_row.flags
    assert corrected_row.innovation == float(cycle_jump)


def test_a_reference_missing_from_the_block_leaves_its_pairs_predicted(
    tmp_path: Path,
) -> None:
    """Give mc1's pairs predicted rows and screen without it (review focus 4)."""
    last_rows = dict(REFERENCE_LAST_ROWS)
    epoch = epoch_of([REFERENCE_MEASUREMENTS[1]], last_rows, tmp_path)
    assert epoch.refs == frozenset({"mc2"})
    pair_step = run.process_pairs(epoch, last_rows)
    for series_key in (("mc1", "mc1"), ("mc1", "mc2"), ("mc2", "mc1")):
        assert pair_step.step_results[series_key].row.flags == "P"
    assert pair_step.step_results[("mc2", "mc2")].row.flags == "A"
    assert pair_step.screening.events == ()


def test_a_new_pair_starts_acquiring(tmp_path: Path) -> None:
    """Give a pair with no last row a dormant row with its measurement buffered."""
    epoch = epoch_of(REFERENCE_MEASUREMENTS, {}, tmp_path)
    pair_step = run.process_pairs(epoch, {})
    for series_key in REFERENCE_LAST_ROWS:
        row = pair_step.step_results[(series_key[0], series_key[1])].row
        assert (row.flags, len(row.rejects), row.segment) == ("RD", 1, 0)
    assert pair_step.predictions[("mc1", "mc1")] is None


def test_screening_excludes_and_the_filter_holds(tmp_path: Path) -> None:
    """Hold as X a pair that shares a self pair's shift inside its gate (9.5, 10.1)."""
    maser_last_row = last_row(
        x_fs=2_000_000,
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        innovation_scale=40.0,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): maser_last_row}
    shifted_measurements = [
        das_measurement_of("mc1", "mc1", 1000, 10),
        das_measurement_of("mc2", "mc2", 2100, 20),
        das_measurement_of("mc1", "mc2", 5000, 30),
        das_measurement_of("mc2", "mc1", PHASE_PERIOD - 5000, 40),
        das_measurement_of("mc2", "ox23", 2008, 50),
    ]
    epoch = epoch_of(shifted_measurements, last_rows, tmp_path)
    pair_step = run.process_pairs(epoch, last_rows)
    assert [
        screening_event.finding for screening_event in pair_step.screening.events
    ] == ["self_fail"]
    assert pair_step.screening.excluded == frozenset({("mc2", "ox23")})
    assert pair_step.step_results[("mc2", "ox23")].row.flags == "X"
    assert pair_step.step_results[("mc2", "mc2")].row.flags == "R"


def test_an_epoch_with_no_block_predicts_every_pair(tmp_path: Path) -> None:
    """Give every existing pair a predicted row when the DAS measured nothing (6.2)."""
    config, clock_config = make_deployment(tmp_path)
    earlier_series = ExistingSeries(
        pairs=frozenset(
            series_key for series_key in REFERENCE_LAST_ROWS if len(series_key) == 2
        ),
        triples=frozenset(),
    )
    epoch = run.build_epoch(E, None, earlier_series, config, clock_config)
    pair_step = run.process_pairs(epoch, REFERENCE_LAST_ROWS)
    assert {result_row.row.flags for result_row in pair_step.step_results.values()} == {
        "P"
    }
    assert pair_step.measurements == {}


def test_an_undecided_slip_excludes_both_clock_pairs(tmp_path: Path) -> None:
    """Hold as X the clock pairs of an undecided slip, inside their gates (11.2)."""
    cycle_jump = PHASE_PERIOD // 2 + 2
    wide_scale_fields = {
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 25_000.0,
    }
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc1", "ox23"): last_row(x_fs=1_000_000_000, **wide_scale_fields),
        ("mc2", "ox23"): last_row(x_fs=2_000_000_000, **wide_scale_fields),
    }
    clock_measurements = [
        das_measurement_of("mc1", "ox23", (1_000_000 + cycle_jump) % PHASE_PERIOD, 50),
        das_measurement_of(
            "mc2", "ox23", (2_000_000 + cycle_jump - 4) % PHASE_PERIOD, 60
        ),
    ]
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, *clock_measurements], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    assert [slip_event.finding for slip_event in pair_step.slips.events] == [
        "slip_undecided"
    ]
    assert pair_step.step_results[("mc1", "ox23")].row.flags == "X"
    assert pair_step.step_results[("mc2", "ox23")].row.flags == "X"


def test_a_dormant_pair_is_decycled_against_its_anchor(tmp_path: Path) -> None:
    """Decycle a pair with no prediction against its last buffered measurement (7.5)."""
    dormant_row = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        rejects=((PREVIOUS_EPOCH, 201_000.0),),
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc1", "mc1"): dormant_row}
    epoch = epoch_of(REFERENCE_MEASUREMENTS, last_rows, tmp_path)
    pair_step = run.process_pairs(epoch, last_rows)
    assert pair_step.measurements[("mc1", "mc1")].z == 201_000


def test_steering_inside_the_epoch_is_taken_off(tmp_path: Path) -> None:
    """Refer a measurement to E with the steering since E taken off (7.1)."""
    last_rows = dict(REFERENCE_LAST_ROWS)
    steering_line = f"{datetime_to_mjd(E) + 2e-6:.6f} 10.0 0.0\n"
    moved_measurements = [
        *REFERENCE_MEASUREMENTS[:2],
        das_measurement_of("mc1", "mc2", 5010, 30),
        REFERENCE_MEASUREMENTS[3],
    ]
    epoch = epoch_of(moved_measurements, last_rows, tmp_path, {"mc1": steering_line})
    pair_step = run.process_pairs(epoch, last_rows)
    assert pair_step.measurements[("mc1", "mc2")].z == 5000
    assert pair_step.step_results[("mc1", "mc2")].row.flags == "A"


# ---------------------------------------------------------- triples of an epoch

WORKED_DAS_MEASUREMENT: Final = DASMeasurement(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    switch="2B07",
    clock="ox23",
)
"""Appendix A's raw row."""


def triple_last_row(**changed_fields: object) -> Row:
    """Give a tracked 3-state triple row before E, with ``changed_fields`` made."""
    row_fields: dict[str, object] = {
        "x_fs": 1_239_570_000,
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 4.0,
    }
    row_fields.update(changed_fields)
    return last_row(**row_fields)


def test_triples_are_built_from_the_pairs_measurements(tmp_path: Path) -> None:
    """Take dd from the pairs' accepted z of the epoch, not their estimates (12)."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    remote_measurement = triple_step.measurements[("mc1", "mc2", "ox23")]
    assert (remote_measurement.z, remote_measurement.components_used) == (
        1_234_577 + 5_000,
        "111",
    )
    assert remote_measurement.double_difference_sigma == math.sqrt(9 + 0.25 * (9 + 9))
    local_measurement = triple_step.measurements[("mc2", "mc2", "ox23")]
    assert (local_measurement.z, local_measurement.double_difference_sigma) == (
        1_234_577,
        3.0,
    )
    assert {
        log_record.row.flags for log_record in triple_step.step_results.values()
    } == {"RD"}


def test_a_tracked_triple_is_filtered_on_its_double_difference(tmp_path: Path) -> None:
    """Accept a triple's dd against its own prediction."""
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): WORKED_LAST_ROW,
        ("mc1", "mc2", "ox23"): triple_last_row(x_fs=1_239_577_000),
    }
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    triple_step = run.process_triples(
        epoch, last_rows, run.process_pairs(epoch, last_rows)
    )
    row = triple_step.step_results[("mc1", "mc2", "ox23")].row
    assert (row.flags, row.innovation) == ("A", 0.0)
    assert row.x_fs == 1_239_577_000


def test_a_missing_link_direction_uses_the_predicted_round_trip(tmp_path: Path) -> None:
    """Give 110 when (s, r) was not measured, through the links' predictions (12.2)."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS[:3], WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    triple_step = run.process_triples(
        epoch, last_rows, run.process_pairs(epoch, last_rows)
    )
    remote_measurement = triple_step.measurements[("mc1", "mc2", "ox23")]
    assert (remote_measurement.z, remote_measurement.components_used) == (
        1_234_577 + 5_000,
        "110",
    )


def test_a_component_cold_start_makes_the_triple_dormant(tmp_path: Path) -> None:
    """Restart a triple whose clock pair cold-started this epoch (12.6)."""
    acquiring_row = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS_EPOCH - T, 1_234_577.0), (PREVIOUS_EPOCH, 1_234_577.0)),
    )
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): acquiring_row,
        ("mc2", "mc2", "ox23"): triple_last_row(x_fs=1_234_577_000),
    }
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    assert pair_step.step_results[("mc2", "ox23")].cold_started is True
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    assert triple_step.measurements[("mc2", "mc2", "ox23")].pair_cold_started is True
    row = triple_step.step_results[("mc2", "mc2", "ox23")].row
    assert (row.flags, row.rejects) == ("RD", ((E, 1_234_579.0),))


def test_a_triple_without_its_clock_pair_holds(tmp_path: Path) -> None:
    """Give a triple a predicted row when its clock pair has no measurement."""
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): WORKED_LAST_ROW,
        ("mc2", "mc2", "ox23"): triple_last_row(x_fs=1_234_577_000),
    }
    epoch = epoch_of(REFERENCE_MEASUREMENTS, last_rows, tmp_path)
    triple_step = run.process_triples(
        epoch, last_rows, run.process_pairs(epoch, last_rows)
    )
    assert triple_step.step_results[("mc2", "mc2", "ox23")].row.flags == "P"
    assert ("mc2", "mc2", "ox23") not in triple_step.measurements


def test_the_local_triple_is_checked_every_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass the self pair for both links of a local triple, so the check runs (12.4)."""
    links_seen: list[tuple[object, object]] = []

    def record_links(
        triple: TripleKey, sc: Component, rs: Component, sr: Component
    ) -> object:
        """Record the links a local triple is given."""
        if triple == ("mc2", "mc2", "ox23"):
            links_seen.append((rs, sr))
        return double_difference(triple, sc, rs, sr)

    monkeypatch.setattr(run, "double_difference", record_links)
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    run.process_triples(epoch, last_rows, run.process_pairs(epoch, last_rows))
    assert len(links_seen) == 1
    rs, sr = links_seen[0]
    assert rs == sr
    assert rs.z == 2000  # type: ignore[attr-defined]


def test_a_rejected_pair_gives_its_triple_no_value(tmp_path: Path) -> None:
    """Leave a triple held when its clock pair was measured but rejected."""
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): WORKED_LAST_ROW,
        ("mc2", "mc2", "ox23"): triple_last_row(x_fs=1_234_577_000),
    }
    outlier_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34779,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, outlier_measurement], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    assert pair_step.step_results[("mc2", "ox23")].row.flags == "R"
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    assert ("mc2", "mc2", "ox23") not in triple_step.measurements
    assert triple_step.step_results[("mc2", "mc2", "ox23")].row.flags == "P"


# ---------------------------------------------------------------- the epoch loop

LATE_START: Final = datetime(2025, 9, 23, 23, 20, tzinfo=UTC)
"""The first epoch of the runs below: four epochs before midnight."""


def write_das_files(tmp_path: Path, epoch_starts: list[datetime]) -> None:
    """Write DAS daily files measuring mc1 and ox23 against mc1 at ``epoch_starts``."""
    das_lines_by_day: dict[int, list[str]] = {}
    for epoch_start in epoch_starts:
        epoch_start_mjd = datetime_to_mjd(epoch_start)
        for clock_index, (clock_name, measured_phase) in enumerate(
            (("mc1", 1000), ("ox23", 50_000))
        ):
            das_measurement = DASMeasurement(
                measurement_mjd=round(epoch_start_mjd + (clock_index + 1) * 2e-5, 6),
                measured_phase=measured_phase,
                rms=3,
                switch="1A01" if clock_name == "mc1" else "1A02",
                clock=clock_name,
            )
            das_lines_by_day.setdefault(int(epoch_start_mjd), []).append(
                f"{das_measurement}\n"
            )
    for data_day, das_lines in das_lines_by_day.items():
        (tmp_path / "das" / f"cd5m5m_{data_day}.dat").write_text(
            "".join(das_lines), encoding="ascii"
        )


def make_loop_deployment(
    tmp_path: Path, first_epoch_start: datetime = LATE_START
) -> tuple[AppConfig, ClockConfig]:
    """Give a deployment whose first epoch is ``first_epoch_start``."""
    config, clock_config = make_deployment(tmp_path)
    processed_config = config.processed.model_copy(
        update={"start_from_mjd": datetime_to_mjd(first_epoch_start)}
    )
    return config.model_copy(update={"processed": processed_config}), clock_config


def rows_of(config: AppConfig, series_key: SeriesKey) -> list[Row]:
    """Read every row of a series' file."""
    series_file_path = registry.series_file(
        config.processed.processed_path, "a", series_key
    )
    file_kind: files.FileKind = "meas" if len(series_key) == 2 else "ddiff"
    line_size = files.WIDTHS[file_kind] + 1
    row_bytes = series_file_path.read_bytes()[
        files.HEADER_LINES[file_kind] * line_size :
    ]
    row_lines = [
        row_bytes[line_start : line_start + line_size - 1].decode()
        for line_start in range(0, len(row_bytes), line_size)
    ]
    if file_kind == "meas":
        return [files.parse_meas_row(row_line).row for row_line in row_lines]
    return [files.parse_ddiff_row(row_line).row for row_line in row_lines]


LOOP_SERIES: Final[tuple[SeriesKey, ...]] = (
    ("mc1", "mc1"),
    ("mc1", "ox23"),
    ("mc1", "mc1", "mc1"),
    ("mc1", "mc1", "ox23"),
)
"""The series of the loop's deployment."""

LOOP_FIRST_ROWS: Final[dict[SeriesKey, int]] = {
    ("mc1", "mc1"): 0,
    ("mc1", "ox23"): 0,
    ("mc1", "mc1", "mc1"): 2,
    ("mc1", "mc1", "ox23"): 2,
}
"""The epoch of each loop series' first row, counted from the first epoch: a
triple's is its first measurement, at its pairs' cold start."""


def epoch_indexes(series_rows: list[Row], first_epoch_start: datetime) -> list[int]:
    """Give each row's epoch as a count of epochs from the first."""
    return [(row.interpolated_datetime - first_epoch_start) // T for row in series_rows]


def recorded_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, datetime | None]]:
    """Record, at every write, which write it is and the newest epoch buffered."""
    newest_epochs: list[tuple[str, datetime | None]] = []

    def recording(
        write_name: str, real_write: Callable[[files.DayBuffer], None]
    ) -> Callable[[files.DayBuffer], None]:
        """Give ``real_write`` recording the newest buffered epoch first."""

        def record_and_write(day_buffer: files.DayBuffer) -> None:
            """Record the newest buffered epoch, then write."""
            buffered_epochs = (
                [
                    day_buffer.last_rows[series_key].interpolated_datetime
                    for series_key in day_buffer.last_rows
                ]
                if day_buffer.file_lines
                else []
            )
            newest_epochs.append((write_name, max(buffered_epochs, default=None)))
            real_write(day_buffer)

        return record_and_write

    monkeypatch.setattr(
        run, "write_buffer", recording("write_buffer", files.write_buffer)
    )
    monkeypatch.setattr(run, "write_final", recording("write_final", files.write_final))
    return newest_epochs


DAY_THEN_FINAL_WRITES: Final = [
    ("write_buffer", LATE_START + 3 * T),
    ("write_final", LATE_START + 5 * T),
]
"""The writes of a run from three epochs before midnight to two after it."""


def test_a_day_is_written_after_its_last_epoch_and_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write after 23:50 UTC and when the data end (5.8)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    write_epochs = recorded_writes(monkeypatch)
    run.run(config, clock_config, None, ShutdownHandler())
    assert write_epochs == DAY_THEN_FINAL_WRITES
    for series_key in LOOP_SERIES:
        assert epoch_indexes(rows_of(config, series_key), LATE_START) == list(
            range(LOOP_FIRST_ROWS[series_key], 6)
        ), series_key


def test_a_day_without_a_block_at_23_50_is_still_written_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write after 23:50 although the DAS measured nothing then (review focus 2)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in (0, 1, 2, 4, 5)])
    write_epochs = recorded_writes(monkeypatch)
    run.run(config, clock_config, None, ShutdownHandler())
    assert write_epochs == DAY_THEN_FINAL_WRITES
    assert rows_of(config, ("mc1", "mc1"))[3].flags == "P"


def test_a_gap_gives_predicted_rows_and_the_run_stops_at_the_end_of_the_data(
    tmp_path: Path,
) -> None:
    """Give a predicted row for a gap epoch, and none after the data (6.2)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in (0, 1, 2, 3, 6)])
    run.run(config, clock_config, None, ShutdownHandler())
    self_rows = rows_of(config, ("mc1", "mc1"))
    assert epoch_indexes(self_rows, LATE_START) == list(range(7))
    assert [row.flags for row in self_rows] == ["RD", "RD", "AN", "A", "P", "P", "A"]


def test_steps_stop_the_run_after_that_many_epochs(tmp_path: Path) -> None:
    """Process --steps N epochs and stop, writing the buffer (6.1)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    run.run(config, clock_config, 2, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 2
    run.run(config, clock_config, 1, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 3


def test_a_shutdown_stops_between_epochs_after_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish the epoch, write the buffer and stop when a shutdown is asked (6.1)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    shutdown = ShutdownHandler()
    real_process_epoch = run.process_epoch

    def ask_for_shutdown(*args: object) -> run.EpochDone:
        """Process the epoch, asking for a shutdown during the second."""
        epoch_done = real_process_epoch(*args)  # type: ignore[arg-type]
        if epoch_done.epoch.interpolated_datetime == LATE_START + T:
            shutdown.request_shutdown()
        return epoch_done

    monkeypatch.setattr(run, "process_epoch", ask_for_shutdown)
    run.run(config, clock_config, None, shutdown)
    assert len(rows_of(config, ("mc1", "mc1"))) == 2
    assert len(rows_of(config, ("mc1", "ox23"))) == 2


def test_a_run_restarted_after_every_epoch_writes_the_same_files(
    tmp_path: Path,
) -> None:
    """Give byte-identical files whether run as a batch or one epoch at a time (I5)."""
    batch_config, clock_config = make_loop_deployment(tmp_path / "batch")
    write_das_files(tmp_path / "batch", [LATE_START + i * T for i in range(6)])
    run.run(batch_config, clock_config, None, ShutdownHandler())
    stepped_config, clock_config = make_loop_deployment(tmp_path / "stepped")
    write_das_files(tmp_path / "stepped", [LATE_START + i * T for i in range(6)])
    for _ in range(8):
        run.run(stepped_config, clock_config, 1, ShutdownHandler())
    for series_key in LOOP_SERIES:
        first_series_file = registry.series_file(
            batch_config.processed.processed_path, "a", series_key
        )
        second_series_file = registry.series_file(
            stepped_config.processed.processed_path, "a", series_key
        )
        assert first_series_file.read_bytes() == second_series_file.read_bytes(), (
            series_key
        )


def test_a_damaged_line_found_at_the_start_cuts_only_its_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Cut a damaged file before its damaged line, and go on after the newest row."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    run.run(config, clock_config, None, ShutdownHandler())
    data_file = registry.series_file(
        config.processed.processed_path, "a", ("mc1", "ox23")
    )
    file_bytes = bytearray(data_file.read_bytes())
    line_size = files.MEAS_WIDTH + 1
    file_bytes[(files.MEAS_HEADER_LINES + 2) * line_size + 3] = ord("x")
    data_file.write_bytes(bytes(file_bytes[: -line_size // 2]))
    caplog.clear()
    assert run.next_epoch(config) == LATE_START + 6 * T
    log_entries = [
        (log_record.levelname, log_record.getMessage()) for log_record in caplog.records
    ]
    assert [level_name for level_name, _ in log_entries] == ["ERROR", "WARNING"]
    assert log_entries[0][1].startswith(
        f"data file {data_file} is damaged after its row of {LATE_START + T}: "
    )
    assert log_entries[1][1] == (
        "cut back the files of channel a after damaged files, each logged at ERROR:"
        f" 1 files cut, 0 deleted, {len(LOOP_SERIES) - 1} left; the newest row is"
        f" now of {LATE_START + 5 * T}"
    )
    for checked_file, file_kind, series_key in run.data_series(config):
        assert files.good_through(checked_file, file_kind) == (
            LATE_START + (T if checked_file == data_file else 5 * T)
        ), series_key


def test_a_journal_found_at_the_start_rolls_every_file_back_before_its_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Roll every file back to before a stopped write's first epoch, then redo (6.7)."""
    clean_config, clock_config = make_loop_deployment(tmp_path / "clean")
    write_das_files(tmp_path / "clean", [LATE_START + i * T for i in range(6)])
    run.run(clean_config, clock_config, None, ShutdownHandler())
    stopped_config, clock_config = make_loop_deployment(tmp_path / "stopped")
    write_das_files(tmp_path / "stopped", [LATE_START + i * T for i in range(6)])
    run.run(stopped_config, clock_config, None, ShutdownHandler())
    journal = stopped_config.processed.processed_path / JOURNAL_FILE_TEMPLATE.format(
        rf="a"
    )
    journal.write_text(f"{(LATE_START + 3 * T).isoformat()}\n", encoding="ascii")
    caplog.clear()
    assert run.next_epoch(stopped_config) == LATE_START + 3 * T
    assert [log_record.getMessage() for log_record in caplog.records] == [
        "cut back the files of channel a after a write that stopped part way:"
        f" {len(LOOP_SERIES)} files cut, 0 deleted, 0 left; the newest row is now of"
        f" {LATE_START + 2 * T}"
    ]
    caplog.clear()
    assert run.next_epoch(stopped_config) == LATE_START + 3 * T
    assert not caplog.records
    assert not journal.exists()
    series_files = run.data_series(stopped_config)
    assert len(series_files) == len(LOOP_SERIES)
    for data_file, file_kind, series_key in series_files:
        assert files.good_through(data_file, file_kind) == LATE_START + 2 * T, (
            series_key
        )
    run.run(stopped_config, clock_config, None, ShutdownHandler())
    for series_key in LOOP_SERIES:
        first_series_file = registry.series_file(
            clean_config.processed.processed_path, "a", series_key
        )
        second_series_file = registry.series_file(
            stopped_config.processed.processed_path, "a", series_key
        )
        assert first_series_file.read_bytes() == second_series_file.read_bytes(), (
            series_key
        )


def test_with_no_files_the_run_starts_at_start_from_mjd(tmp_path: Path) -> None:
    """Start at the epoch containing start_from_mjd when no file exists (6.7)."""
    config, _ = make_loop_deployment(
        tmp_path, LATE_START + 2 * T + timedelta(seconds=90)
    )
    assert run.next_epoch(config) == LATE_START + 2 * T


def test_a_block_before_the_next_epoch_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip a block earlier than the epoch the run is at, rather than loop on it."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START - T, LATE_START, LATE_START + T])
    real_read_all_blocks = read_all_blocks

    def read_from_earlier(
        das_directory: Path, start_at_mjd: float | None = None
    ) -> object:
        """Read from one epoch before the start."""
        del start_at_mjd
        return real_read_all_blocks(das_directory, datetime_to_mjd(LATE_START - T))

    monkeypatch.setattr(run, "read_all_blocks", read_from_earlier)
    run.run(config, clock_config, None, ShutdownHandler())
    series_rows = rows_of(config, ("mc1", "mc1"))
    assert [row.interpolated_datetime for row in series_rows] == [
        LATE_START,
        LATE_START + T,
    ]


def test_an_epoch_that_fails_adds_none_of_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the day buffer as it was when an epoch fails part way (5.8 step 1)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(3)])
    day_buffer = files.DayBuffer("a")
    das_blocks = list(read_all_blocks(tmp_path / "das", datetime_to_mjd(LATE_START)))
    files.ensure_archives(config.processed.processed_path)
    run.process_epoch(LATE_START, das_blocks[0], day_buffer, config, clock_config)
    buffer_before = buffered_lines(day_buffer), dict(day_buffer.last_rows)
    real_add = files.DayBuffer.add
    added_series: list[SeriesKey] = []

    def fail_on_second_add(
        self: files.DayBuffer,
        data_file: Path,
        series_key: SeriesKey,
        file_record: files.MeasRecord | files.DdiffRecord,
    ) -> None:
        """Add the first row, then fail on the second, as a bad value would."""
        added_series.append(series_key)
        if len(added_series) == 2:
            message = "injected"
            raise DataFileError(message)
        real_add(self, data_file, series_key, file_record)

    monkeypatch.setattr(files.DayBuffer, "add", fail_on_second_add)
    with pytest.raises(DataFileError, match="injected"):
        run.process_epoch(
            LATE_START + T, das_blocks[1], day_buffer, config, clock_config
        )
    assert (buffered_lines(day_buffer), dict(day_buffer.last_rows)) == buffer_before


# --------------------------------------------------------------- log events

RUN_LOGGER: Final = "masterclock.das_processor.run"
"""The logger the run's events go to."""


def logged_events(
    caplog: pytest.LogCaptureFixture,
    epoch: run.Epoch,
    last_rows: dict[SeriesKey, Row],
) -> list[tuple[str, str]]:
    """Process ``epoch`` and give the run's log records as (level, message)."""
    pair_step = run.process_pairs(epoch, last_rows)
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    epoch_done = run.EpochDone(
        epoch=epoch, pair_step=pair_step, triple_step=triple_step
    )
    caplog.clear()
    with caplog.at_level(TRACE, logger=RUN_LOGGER):
        run.log_epoch(epoch_done, last_rows, "a")
    return [
        (log_record.levelname, log_record.getMessage())
        for log_record in caplog.records
        if log_record.name == RUN_LOGGER
    ]


def messages_at(log_entries: list[tuple[str, str]], level_name: str) -> list[str]:
    """Give the messages logged at ``level_name``."""
    return [
        entry_message
        for entry_level, entry_message in log_entries
        if entry_level == level_name
    ]


def test_an_epoch_is_logged_with_its_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each epoch at INFO with its series, accepted and held (16.2)."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    log_entries = logged_events(
        caplog,
        epoch_of(
            [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
        ),
        last_rows,
    )
    assert messages_at(log_entries, "INFO") == [
        "epoch 2025-09-23 06:00:00+00:00: 5 pairs, 10 triples, 5 accepted, 10 held"
    ]


def test_each_series_outcome_and_update_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each series' flags at DEBUG and its prediction and update at TRACE."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    log_entries = logged_events(
        caplog,
        epoch_of(
            [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
        ),
        last_rows,
    )
    debug_messages = messages_at(log_entries, "DEBUG")
    assert len(debug_messages) == 5 + 10
    assert "das_a.mc2.ox23: A" in debug_messages
    assert "das_a.mc1.mc2.ox23: RD" in debug_messages
    trace_messages = messages_at(log_entries, "TRACE")
    assert len(trace_messages) == 5 + 10
    assert any(
        trace_line.startswith("das_a.mc2.ox23: prediction 1234574.38")
        and "x 1234574.457" in trace_line
        for trace_line in trace_messages
    )


def test_a_reject_is_logged_at_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a counted reject with its innovation, scale and count (16.2)."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    outlier_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34779,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    log_entries = logged_events(
        caplog,
        epoch_of([*REFERENCE_MEASUREMENTS, outlier_measurement], last_rows, tmp_path),
        last_rows,
    )
    assert messages_at(log_entries, "WARNING") == [
        "das_a.mc2.ox23 rejected: innovation 202.6 ps, scale 3.0 ps, 1 consecutive"
    ]


def test_a_phase_step_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a phase step with its size and the step offset (16.2)."""
    stepping_row = dataclasses.replace(
        WORKED_LAST_ROW,
        flags="R",
        consecutive_rejects=2,
        rejects=((PREVIOUS_EPOCH - T, 150.0), (PREVIOUS_EPOCH, 150.0)),
        epochs_since_accept=2,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): stepping_row}
    moved_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 150 - 3,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    log_entries = logged_events(
        caplog,
        epoch_of([*REFERENCE_MEASUREMENTS, moved_measurement], last_rows, tmp_path),
        last_rows,
    )
    assert "das_a.mc2.ox23 phase step of 150 ps; step offset 150 ps" in messages_at(
        log_entries, "INFO"
    )


def test_a_cold_start_and_dormancy_are_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a series that cold-starts and one that goes dormant (16.2)."""
    acquiring_row = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS_EPOCH - T, 1_234_579.0), (PREVIOUS_EPOCH, 1_234_579.0)),
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): acquiring_row}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    info_messages = messages_at(logged_events(caplog, epoch, last_rows), "INFO")
    assert "das_a.mc2.ox23 cold start: segment 2" in info_messages
    rejecting_last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc1", "mc1"): last_row(x_fs=1_000_000, epochs_since_accept=40),
    }
    epoch = epoch_of(
        [das_measurement_of("mc1", "mc1", 1500, 10), *REFERENCE_MEASUREMENTS[1:]],
        rejecting_last_rows,
        tmp_path / "second",
    )
    info_messages = messages_at(
        logged_events(caplog, epoch, rejecting_last_rows), "INFO"
    )
    assert "das_a.mc1.mc1 dormant" in info_messages


def test_a_configuration_change_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a warm start for new time constants (16.2)."""
    older_row = dataclasses.replace(WORKED_LAST_ROW, time_constant=80.0)
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): older_row}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    info_messages = messages_at(logged_events(caplog, epoch, last_rows), "INFO")
    assert (
        "das_a.mc2.ox23 configuration change: M 80.0 to 100.0, M_sigma 50.0 to 50.0"
        in info_messages
    )


def test_a_frequency_step_is_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a frequency step and the segment it starts (16.2)."""
    ramping_row = dataclasses.replace(
        WORKED_LAST_ROW,
        flags="R",
        consecutive_rejects=2,
        rejects=((PREVIOUS_EPOCH - T, 30.0), (PREVIOUS_EPOCH, 60.0)),
        epochs_since_accept=2,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): ramping_row}
    moved_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 90 - 3,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    info_messages = messages_at(
        logged_events(
            caplog,
            epoch_of([*REFERENCE_MEASUREMENTS, moved_measurement], last_rows, tmp_path),
            last_rows,
        ),
        "INFO",
    )
    assert "das_a.mc2.ox23 frequency step: segment 5" in info_messages


def test_screening_and_slip_events_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log screening failures at WARNING and a corrected slip at INFO (16.2)."""
    maser_last_row = last_row(
        x_fs=2_000_000,
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        innovation_scale=40.0,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): maser_last_row}
    shifted_measurements = [
        das_measurement_of("mc1", "mc1", 1000, 10),
        das_measurement_of("mc2", "mc2", 2100, 20),
        das_measurement_of("mc1", "mc2", 5000, 30),
        das_measurement_of("mc2", "mc1", PHASE_PERIOD - 5000, 40),
        das_measurement_of("mc2", "ox23", 2008, 50),
    ]
    warning_messages = messages_at(
        logged_events(
            caplog, epoch_of(shifted_measurements, last_rows, tmp_path), last_rows
        ),
        "WARNING",
    )
    assert "self-measurement of mc2 failed: excluded das_a.mc2.ox23" in warning_messages


def test_a_missing_self_measurement_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a self pair with a prediction but no measurement at WARNING (10.1)."""
    last_rows = dict(REFERENCE_LAST_ROWS)
    epoch = epoch_of(REFERENCE_MEASUREMENTS[1:], last_rows, tmp_path)
    assert "self-measurement of mc1 missing" in messages_at(
        logged_events(caplog, epoch, last_rows), "WARNING"
    )


def test_a_reciprocity_and_a_closure_failure_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the link directions reciprocity excludes, and links closure excludes."""
    last_rows = dict(REFERENCE_LAST_ROWS)
    bad_link_measurements = [
        *REFERENCE_MEASUREMENTS[:2],
        das_measurement_of("mc1", "mc2", 5100, 30),
        REFERENCE_MEASUREMENTS[3],
    ]
    warning_messages = messages_at(
        logged_events(
            caplog, epoch_of(bad_link_measurements, last_rows, tmp_path), last_rows
        ),
        "WARNING",
    )
    assert (
        "reciprocity of mc1-mc2 failed: excluded das_a.mc1.mc2, das_a.mc2.mc1"
        in warning_messages
    )


def test_slips_corrected_and_undecided_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a corrected slip at INFO, as the slip check found it."""
    cycle_jump = PHASE_PERIOD // 2 + 2
    maser_fields = {
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "innovation_scale": 3.0,
    }
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc1", "ox23"): last_row(
            x_fs=1_000_000_000, epochs_in_segment=10, flags="AU", **maser_fields
        ),
        ("mc2", "ox23"): last_row(x_fs=2_000_000_000, **maser_fields),
    }
    clock_measurements = [
        das_measurement_of("mc1", "ox23", (1_000_000 + cycle_jump) % PHASE_PERIOD, 50),
        das_measurement_of(
            "mc2", "ox23", (2_000_000 + cycle_jump - 4) % PHASE_PERIOD, 60
        ),
    ]
    log_entries = logged_events(
        caplog,
        epoch_of([*REFERENCE_MEASUREMENTS, *clock_measurements], last_rows, tmp_path),
        last_rows,
    )
    assert "das_a.mc1.ox23 slip corrected: +1 cycles" in messages_at(
        log_entries, "INFO"
    )
    wide_scale_fields = {**maser_fields, "innovation_scale": 25_000.0}
    last_rows[("mc1", "ox23")] = last_row(x_fs=1_000_000_000, **wide_scale_fields)
    last_rows[("mc2", "ox23")] = last_row(x_fs=2_000_000_000, **wide_scale_fields)
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, *clock_measurements], last_rows, tmp_path / "second"
    )
    warning_messages = messages_at(logged_events(caplog, epoch, last_rows), "WARNING")
    assert (
        "slip of clock ox23 undecided: excluded das_a.mc1.ox23, das_a.mc2.ox23"
        in warning_messages
    )


def test_a_closure_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each link closure excludes, at WARNING (10.3)."""
    refs = ("mc1", "mc2", "mc3")
    self_phases = {"mc1": 1000, "mc2": 2000, "mc3": 3000}
    last_rows: dict[SeriesKey, Row] = {}
    das_measurements = []
    for measurement_index, (a, b) in enumerate((a, b) for a in refs for b in refs):
        x = self_phases[a] if a == b else 1000 * (int(a[-1]) - int(b[-1]))
        injected_error = (
            50 if (a, b) == ("mc2", "mc3") else -50 if (a, b) == ("mc3", "mc2") else 0
        )
        last_rows[(a, b)] = last_row(x_fs=x * 1000)
        das_measurements.append(
            das_measurement_of(
                a, b, (x + injected_error) % PHASE_PERIOD, 10 * (measurement_index + 1)
            )
        )
    warning_messages = messages_at(
        logged_events(
            caplog, epoch_of(das_measurements, last_rows, tmp_path), last_rows
        ),
        "WARNING",
    )
    assert (
        "closure of link mc2-mc3 failed: excluded das_a.mc2.mc3, das_a.mc3.mc2"
        in warning_messages
    )


def test_a_first_run_starts_at_the_first_data(tmp_path: Path) -> None:
    """Start at the first block when no series exists yet, so each step progresses."""
    config, clock_config = make_loop_deployment(tmp_path, LATE_START - 3 * T)
    write_das_files(tmp_path, [LATE_START, LATE_START + T])
    run.run(config, clock_config, 1, ShutdownHandler())
    assert [row.interpolated_datetime for row in rows_of(config, ("mc1", "mc1"))] == [
        LATE_START
    ]
    run.run(config, clock_config, 1, ShutdownHandler())
    assert len(rows_of(config, ("mc1", "mc1"))) == 2


def test_a_link_not_accepted_does_not_make_the_triple_cold(tmp_path: Path) -> None:
    """Take a link rejected for its RMS by its prediction, not marked cold (12.6)."""
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): WORKED_LAST_ROW,
        ("mc1", "mc2", "ox23"): triple_last_row(x_fs=1_239_577_000),
    }
    link_measurement = REFERENCE_MEASUREMENTS[3]
    noisy_link = DASMeasurement.model_validate(
        {
            "measurement_mjd": link_measurement.measurement_mjd,
            "measured_phase": link_measurement.measured_phase,
            "rms": 99,
            "switch": link_measurement.switch,
            "clock": link_measurement.clock,
        }
    )
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS[:3], noisy_link, WORKED_DAS_MEASUREMENT],
        last_rows,
        tmp_path,
    )
    pair_step = run.process_pairs(epoch, last_rows)
    assert "A" not in pair_step.step_results[("mc2", "mc1")].row.flags
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    remote_measurement = triple_step.measurements[("mc1", "mc2", "ox23")]
    assert (
        remote_measurement.components_used,
        remote_measurement.pair_cold_started,
    ) == ("110", False)
    assert triple_step.step_results[("mc1", "mc2", "ox23")].row.flags == "A"


def test_each_series_takes_the_settings_in_force_at_its_epoch(tmp_path: Path) -> None:
    """Give the clock entry that took effect at or before E, not a later one (8.1)."""
    config, _ = make_deployment(tmp_path)
    dated_clock_config = CLOCK_CONFIG_YAML.replace(
        "  ox23: [{type: maser, location: 1}]\n",
        "  ox23: [{type: maser, location: 1},"
        f" {{effective_mjd: {datetime_to_mjd(E)}, time_constant: 150.0}},"
        f" {{effective_mjd: {datetime_to_mjd(E + T)}, time_constant: 200.0}}]\n",
    )
    dated_file = tmp_path / "dated.yaml"
    dated_file.write_text(dated_clock_config, encoding="utf-8")
    epoch = run.build_epoch(
        E,
        das_block_of(MEASURED_PAIRS),
        NO_SERIES,
        config,
        read_clock_config(dated_file),
    )
    assert epoch.series_params[("mc2", "ox23")].M == 150.0


def test_the_prediction_and_update_are_logged_in_full(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log at TRACE each series' prediction, innovation, x, y and d, as they are."""
    last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc2", "ox23"): WORKED_LAST_ROW,
        ("mc1", "mc2", "ox23"): triple_last_row(x_fs=1_239_577_000),
    }
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    trace_messages = messages_at(logged_events(caplog, epoch, last_rows), "TRACE")
    assert (
        "das_a.mc2.ox23: prediction 1234574.38, innovation 2.62, x 1234574.457,"
        " y 0.01230129052352643, d 7.169515400974333e-12"
    ) in trace_messages
    assert (
        "das_a.mc1.mc2.ox23: prediction 1239577.0, innovation 0.0, x 1239577.000,"
        " y 0.0, d 0.0"
    ) in trace_messages


def test_a_phase_step_logs_the_step_alone_and_the_new_offset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the step as the change in offset, and the offset it reaches (16.2)."""
    stepping_row = dataclasses.replace(
        WORKED_LAST_ROW,
        flags="R",
        consecutive_rejects=2,
        rejects=((PREVIOUS_EPOCH - T, 150.0), (PREVIOUS_EPOCH, 150.0)),
        epochs_since_accept=2,
        step_offset=40,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): stepping_row}
    moved_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 150 - 3,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    log_entries = logged_events(
        caplog,
        epoch_of([*REFERENCE_MEASUREMENTS, moved_measurement], last_rows, tmp_path),
        last_rows,
    )
    assert "das_a.mc2.ox23 phase step of 150 ps; step offset 190 ps" in messages_at(
        log_entries, "INFO"
    )


def test_a_frequency_step_with_a_configuration_change_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log both when new settings and a frequency step come in one epoch (16.2)."""
    ramping_row = dataclasses.replace(
        WORKED_LAST_ROW,
        flags="R",
        consecutive_rejects=2,
        rejects=((PREVIOUS_EPOCH - T, 30.0), (PREVIOUS_EPOCH, 60.0)),
        epochs_since_accept=2,
        time_constant=80.0,
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): ramping_row}
    moved_measurement = DASMeasurement(
        measurement_mjd=60941.251588,
        measured_phase=34579 + 90 - 3,
        rms=3,
        switch="2B07",
        clock="ox23",
    )
    info_messages = messages_at(
        logged_events(
            caplog,
            epoch_of([*REFERENCE_MEASUREMENTS, moved_measurement], last_rows, tmp_path),
            last_rows,
        ),
        "INFO",
    )
    assert any(
        message.startswith("das_a.mc2.ox23 configuration change")
        for message in info_messages
    )
    assert "das_a.mc2.ox23 frequency step: segment 6" in info_messages


def test_a_cold_start_logs_no_step(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a cold start from dormancy as a cold start, never as a step (16.2)."""
    acquiring_row = last_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        filter_states=3,
        time_constant=100.0,
        scale_time_constant=50.0,
        rejects=((PREVIOUS_EPOCH - T, 1_234_579.0), (PREVIOUS_EPOCH, 1_234_579.0)),
    )
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): acquiring_row}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    info_messages = messages_at(logged_events(caplog, epoch, last_rows), "INFO")
    ox23_messages = [
        message for message in info_messages if message.startswith("das_a.mc2.ox23 ")
    ]
    assert ox23_messages == ["das_a.mc2.ox23 cold start: segment 2"]


def test_an_epoch_s_log_names_the_series_of_its_channel(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Name each series in the log by the run's RF channel."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START])
    (first_block,) = read_all_blocks(tmp_path / "das", datetime_to_mjd(LATE_START))
    files.ensure_archives(config.processed.processed_path)
    with caplog.at_level(logging.DEBUG, logger=RUN_LOGGER):
        run.process_epoch(
            LATE_START, first_block, files.DayBuffer("a"), config, clock_config
        )
    debug_messages = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "DEBUG"
    ]
    assert "das_a.mc1.mc1: RD" in debug_messages


def test_each_series_last_row_is_read_from_its_file(tmp_path: Path) -> None:
    """Give each series' last row, as its file holds it, for every series."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(3)])
    run.run(config, clock_config, None, ShutdownHandler())
    last_rows = run.read_last_state(config)
    assert sorted(last_rows) == sorted(LOOP_SERIES)
    for series_key in LOOP_SERIES:
        assert last_rows[series_key] == rows_of(config, series_key)[-1], series_key


def write_das_day(
    tmp_path: Path,
    epoch_readings: list[list[tuple[str, str, int]]],
    first_epoch_start: datetime = LATE_START,
) -> None:
    """Write day files of each epoch's (reference, clock, phase), from the first."""
    das_lines_by_day: dict[int, list[str]] = {}
    for epoch_index, epoch_measurements in enumerate(epoch_readings):
        epoch_start_mjd = datetime_to_mjd(first_epoch_start + epoch_index * T)
        for reading_index, (reference, clock_name, measured_phase) in enumerate(
            epoch_measurements
        ):
            das_measurement = DASMeasurement(
                measurement_mjd=round(epoch_start_mjd + (reading_index + 1) * 2e-5, 6),
                measured_phase=measured_phase,
                rms=3,
                switch=f"{reference[-1]}A{reading_index:02d}",
                clock=clock_name,
            )
            das_lines_by_day.setdefault(int(epoch_start_mjd), []).append(
                f"{das_measurement}\n"
            )
    for data_day, das_lines in das_lines_by_day.items():
        (tmp_path / "das" / f"cd5m5m_{data_day}.dat").write_text("".join(das_lines))


def test_a_remote_triple_goes_on_when_its_reference_is_missing(
    tmp_path: Path,
) -> None:
    """Give an existing triple a row at an epoch its reference r was not measured."""
    config, clock_config = make_loop_deployment(tmp_path)
    both_references = [
        ("mc1", "mc1", 1000),
        ("mc1", "mc2", 5000),
        ("mc2", "mc1", PHASE_PERIOD - 5000),
        ("mc2", "mc2", 2000),
        ("mc2", "ox23", 50_000),
    ]
    write_das_day(
        tmp_path,
        [both_references] * 5 + [[both_references[3], both_references[4]]],
    )
    run.run(config, clock_config, None, ShutdownHandler())
    series_rows = rows_of(config, ("mc1", "mc2", "ox23"))
    assert epoch_indexes(series_rows, LATE_START) == [2, 3, 4, 5]
    assert "A" in series_rows[-2].flags
    assert "P" in series_rows[-1].flags


def test_a_run_with_no_data_and_no_series_does_nothing(tmp_path: Path) -> None:
    """Finish without a row when there is neither data nor a file yet."""
    config, clock_config = make_loop_deployment(tmp_path)
    run.run(config, clock_config, None, ShutdownHandler())
    assert run.data_series(config) == []


def test_a_gap_at_the_start_of_a_run_is_predicted_not_skipped(tmp_path: Path) -> None:
    """Give the epoch after the files' end a row, though the DAS skipped it (6.2)."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in (0, 1, 2, 4)])
    run.run(config, clock_config, 3, ShutdownHandler())
    run.run(config, clock_config, 1, ShutdownHandler())
    series_rows = rows_of(config, ("mc1", "mc1"))
    assert epoch_indexes(series_rows, LATE_START) == [0, 1, 2, 3]
    assert "P" in series_rows[3].flags


def test_a_clock_measured_with_an_rms_of_zero_gives_its_triple_a_row(
    tmp_path: Path,
) -> None:
    """Measure a local triple whose clock pair's rms is 0, with a sigma of 0."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    zero_rms_measurement = DASMeasurement.model_validate(
        {
            "measurement_mjd": WORKED_DAS_MEASUREMENT.measurement_mjd,
            "measured_phase": WORKED_DAS_MEASUREMENT.measured_phase,
            "rms": 0,
            "switch": WORKED_DAS_MEASUREMENT.switch,
            "clock": WORKED_DAS_MEASUREMENT.clock,
        }
    )
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, zero_rms_measurement], last_rows, tmp_path
    )
    triple_step = run.process_triples(
        epoch, last_rows, run.process_pairs(epoch, last_rows)
    )
    assert (
        triple_step.measurements[("mc2", "mc2", "ox23")].double_difference_sigma == 0.0
    )


def test_a_new_file_left_without_its_rows_by_a_stopped_write_is_made_again(
    tmp_path: Path,
) -> None:
    """Delete a file whose rows a crash never wrote, when the journal shows why."""
    clean_config, clock_config = make_loop_deployment(tmp_path / "clean")
    write_das_files(tmp_path / "clean", [LATE_START + i * T for i in range(6)])
    run.run(clean_config, clock_config, None, ShutdownHandler())
    stopped_config, clock_config = make_loop_deployment(tmp_path / "stopped")
    write_das_files(tmp_path / "stopped", [LATE_START + i * T for i in range(6)])
    run.run(stopped_config, clock_config, 2, ShutdownHandler())
    processed_path = stopped_config.processed.processed_path
    data_file = registry.series_file(processed_path, "a", ("mc1", "ox23"))
    data_file.write_bytes(
        data_file.read_bytes()[: files.MEAS_HEADER_LINES * (files.MEAS_WIDTH + 1)]
        + b"\0" * (files.MEAS_WIDTH + 1) * 2
    )
    with pytest.raises(DataFileError, match="damaged first row"):
        run.next_epoch(stopped_config)
    journal = processed_path / JOURNAL_FILE_TEMPLATE.format(rf="a")
    journal.write_text(f"{LATE_START.isoformat()}\n", encoding="ascii")
    assert run.next_epoch(stopped_config) == LATE_START
    assert not data_file.exists()
    run.run(stopped_config, clock_config, None, ShutdownHandler())
    for series_key in LOOP_SERIES:
        first_series_file = registry.series_file(
            clean_config.processed.processed_path, "a", series_key
        )
        second_series_file = registry.series_file(processed_path, "a", series_key)
        assert first_series_file.read_bytes() == second_series_file.read_bytes(), (
            series_key
        )


def test_files_ending_apart_are_left_as_they_are(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Leave a file that ends early, log nothing, and go on after the newest row."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(3)])
    run.run(config, clock_config, None, ShutdownHandler())
    data_file = registry.series_file(
        config.processed.processed_path, "a", ("mc1", "ox23")
    )
    data_file.write_bytes(data_file.read_bytes()[: -(files.MEAS_WIDTH + 1)])
    shorter_bytes = data_file.read_bytes()
    caplog.clear()
    assert run.next_epoch(config) == LATE_START + 3 * T
    assert caplog.records == []
    assert data_file.read_bytes() == shorter_bytes


# ------------------------------------------------- work an epoch need not do


def worked_epoch_done(
    tmp_path: Path,
) -> tuple[run.EpochDone, dict[SeriesKey, Row]]:
    """Process the worked epoch, and give what it did and its last rows."""
    last_rows = {**REFERENCE_LAST_ROWS, ("mc2", "ox23"): WORKED_LAST_ROW}
    epoch = epoch_of(
        [*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], last_rows, tmp_path
    )
    pair_step = run.process_pairs(epoch, last_rows)
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    epoch_done = run.EpochDone(
        epoch=epoch, pair_step=pair_step, triple_step=triple_step
    )
    return epoch_done, last_rows


def test_nothing_is_worked_out_for_a_level_not_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Look at no series without WARNING, and make no TRACE line without TRACE."""
    epoch_done, last_rows = worked_epoch_done(tmp_path)
    looked_at: list[str] = []
    monkeypatch.setattr(
        run, "_log_series", lambda series_label, *_: looked_at.append(series_label)
    )
    monkeypatch.setattr(run, "_log_trace", lambda *_: looked_at.append("TRACE"))
    with caplog.at_level(logging.ERROR, logger=RUN_LOGGER):
        run.log_epoch(epoch_done, last_rows, "a")
    assert (looked_at, caplog.records) == ([], [])
    with caplog.at_level(logging.DEBUG, logger=RUN_LOGGER):
        run.log_epoch(epoch_done, last_rows, "a")
    assert len(looked_at) == 5 + 10
    assert "TRACE" not in looked_at


def test_no_steering_is_worked_out_when_no_event_falls_near(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give each series the zero steer_u gives, without asking it, with no events."""
    epoch_done, last_rows = worked_epoch_done(tmp_path)
    epoch = epoch_done.epoch
    assert not run._steered(epoch)
    series_keys: list[SeriesKey] = [*epoch.pairs, *epoch.triples]
    for series_key in series_keys:
        assert run._steering_input(series_key, E, epoch.steering) == steer_u(
            series_key, E, epoch.steering
        )

    asked: list[str] = []
    monkeypatch.setattr(run, "steer_u", lambda *_: asked.append("steer_u"))
    monkeypatch.setattr(run, "steer_w", lambda *_: asked.append("steer_w"))
    pair_step = run.process_pairs(epoch, last_rows)
    triple_step = run.process_triples(epoch, last_rows, pair_step)
    assert asked == []
    assert pair_step == epoch_done.pair_step
    assert triple_step == epoch_done.triple_step


def test_each_pair_s_part_in_the_triples_is_worked_out_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work out each pair's part once an epoch, however many triples it is in."""
    epoch_done, last_rows = worked_epoch_done(tmp_path)
    real_component = run._component
    worked_out: list[tuple[str, str]] = []

    def count_component(pair_step: run.PairStep, pair: tuple[str, str]) -> Component:
        """Note the pair, and work out its part as the run does."""
        worked_out.append(pair)
        return real_component(pair_step, pair)

    monkeypatch.setattr(run, "_component", count_component)
    triple_step = run.process_triples(epoch_done.epoch, last_rows, epoch_done.pair_step)
    assert sorted(worked_out) == sorted(epoch_done.epoch.pairs)
    assert triple_step == epoch_done.triple_step


def test_a_pair_the_epoch_does_not_hold_takes_part_unaccepted(tmp_path: Path) -> None:
    """Give a pair missing from the epoch the part _component would give it."""
    epoch_done, _ = worked_epoch_done(tmp_path)
    assert run._component(epoch_done.pair_step, ("mc9", "ox99")) == (run._NO_COMPONENT)


DATED_CLOCK_CONFIG_YAML: Final = CLOCK_CONFIG_YAML.replace(
    "  ox23: [{type: maser, location: 1}]\n",
    "  ox23: [{type: maser, location: 1},"
    f" {{effective_mjd: {datetime_to_mjd(E + T)}, time_constant: 150.0}}]\n",
)
"""The clock configuration, with ox23's time constant changing at E + T."""


def dated_deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Give a deployment whose ox23 takes a new time constant at E + T."""
    config, _ = make_deployment(tmp_path)
    dated_file = tmp_path / "dated.yaml"
    dated_file.write_text(DATED_CLOCK_CONFIG_YAML, encoding="utf-8")
    return config, read_clock_config(dated_file)


def counted_settings(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """Record the mark of every epoch whose settings are worked out afresh."""
    worked_out: list[datetime] = []
    real_params_for_series = ClockConfig.params_for_series

    def count_params_for_series(
        self: ClockConfig, series_keys: list[SeriesKey], epoch_start: datetime
    ) -> object:
        """Note the mark, then work the settings out."""
        worked_out.append(epoch_start)
        return real_params_for_series(self, series_keys, epoch_start)

    monkeypatch.setattr(ClockConfig, "params_for_series", count_params_for_series)
    return worked_out


KEPT_SERIES: Final = ExistingSeries(
    pairs=frozenset(MEASURED_PAIRS),
    triples=frozenset({("mc1", "mc2", "ox23"), ("mc2", "mc2", "ox23")}),
)
"""Series that exist before the epochs whose settings are kept."""


def test_the_last_epoch_s_settings_are_kept_while_they_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a copy of the last epoch's settings, as they would be worked out."""
    config, clock_config = dated_deployment(tmp_path)
    first_epoch = run.build_epoch(E + T, None, KEPT_SERIES, config, clock_config)
    worked_out = counted_settings(monkeypatch)
    kept_epoch = run.build_epoch(
        E + 2 * T, None, KEPT_SERIES, config, clock_config, first_epoch
    )
    assert worked_out == []
    assert kept_epoch.series_params == clock_config.params_for_series(
        [*kept_epoch.pairs, *kept_epoch.triples], E + 2 * T
    )
    assert kept_epoch.series_params is not first_epoch.series_params


def test_settings_are_worked_out_again_when_a_clock_s_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work out an epoch's settings when an entry takes effect since the last."""
    config, clock_config = dated_deployment(tmp_path)
    das_block = das_block_of(MEASURED_PAIRS)
    first_epoch = run.build_epoch(E, das_block, NO_SERIES, config, clock_config)
    assert first_epoch.series_params[("mc2", "ox23")].M == 100.0
    earlier_series = ExistingSeries(
        pairs=frozenset(first_epoch.pairs), triples=frozenset(first_epoch.triples)
    )
    worked_out = counted_settings(monkeypatch)
    changed_epoch = run.build_epoch(
        E + T, None, earlier_series, config, clock_config, first_epoch
    )
    assert (changed_epoch.pairs, changed_epoch.triples) == (
        first_epoch.pairs,
        first_epoch.triples,
    )
    assert worked_out == [E + T]
    assert changed_epoch.series_params[("mc2", "ox23")].M == 150.0


def test_settings_are_worked_out_again_for_other_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work out an epoch's settings when its pairs or triples are not the last's."""
    config, clock_config = make_deployment(tmp_path)
    first_epoch = run.build_epoch(
        E, das_block_of(MEASURED_PAIRS[:4]), NO_SERIES, config, clock_config
    )
    earlier_series = ExistingSeries(
        pairs=frozenset(first_epoch.pairs), triples=frozenset(first_epoch.triples)
    )
    worked_out = counted_settings(monkeypatch)
    grown_epoch = run.build_epoch(
        E + T,
        None,
        ExistingSeries(
            pairs=earlier_series.pairs | {("mc2", "ox23")},
            triples=earlier_series.triples | {("mc2", "mc2", "ox23")},
        ),
        config,
        clock_config,
        first_epoch,
    )
    assert worked_out == [E + T]
    assert grown_epoch.series_params[("mc2", "ox23")].M == 100.0


def test_settings_are_kept_only_from_an_earlier_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work out an epoch's settings when the epoch given is not earlier."""
    config, clock_config = dated_deployment(tmp_path)
    later_epoch = run.build_epoch(E + T, None, KEPT_SERIES, config, clock_config)
    assert later_epoch.series_params[("mc2", "ox23")].M == 150.0
    worked_out = counted_settings(monkeypatch)
    epochs = [
        run.build_epoch(
            epoch_start, None, KEPT_SERIES, config, clock_config, later_epoch
        )
        for epoch_start in (E, E + T)
    ]
    assert worked_out == [E, E + T]
    assert epochs[0].series_params[("mc2", "ox23")].M == 100.0


def test_a_run_keeps_each_epoch_s_settings_for_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work settings out at a run's first epoch and at a change, and keep them else."""
    config, _ = make_loop_deployment(tmp_path)
    change_mjd = datetime_to_mjd(LATE_START + 2 * T)
    dated_file = tmp_path / "dated.yaml"
    dated_file.write_text(
        CLOCK_CONFIG_YAML.replace(
            "  ox23: [{type: maser, location: 1}]\n",
            f"  ox23: [{{type: maser, location: 1}}, {{effective_mjd: {change_mjd},"
            " time_constant: 150.0}]\n",
        ),
        encoding="utf-8",
    )
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    worked_out = counted_settings(monkeypatch)
    time_constants: list[float | None] = []
    real_process_epoch = run.process_epoch

    def record_time_constant(*args: object) -> run.EpochDone:
        """Process the epoch, noting the time constant ox23's pair takes."""
        epoch_done = real_process_epoch(*args)  # type: ignore[arg-type]
        time_constants.append(epoch_done.epoch.series_params[("mc1", "ox23")].M)
        return epoch_done

    monkeypatch.setattr(run, "process_epoch", record_time_constant)
    run.run(config, read_clock_config(dated_file), None, ShutdownHandler())
    assert time_constants == [100.0, 100.0, 150.0, 150.0, 150.0, 150.0]
    assert worked_out == [LATE_START, LATE_START + 2 * T]


def test_a_run_reads_each_steering_line_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parse each line of a steering file once in a run, not once an epoch."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START + i * T for i in range(6)])
    steering_lines = [
        f"{datetime_to_mjd(LATE_START + i * T + timedelta(seconds=30)):.6f} 0.0 0.0\n"
        for i in range(6)
    ]
    (tmp_path / "steering" / STEERING_FILE_TEMPLATE.format(mc="mc1")).write_text(
        "".join(steering_lines), encoding="ascii"
    )
    parsed_lines: list[str] = []
    real_parse = read_steering._parse_steering_line

    def count_parse(line_place: str, line: str) -> object:
        """Note the line, then parse it."""
        parsed_lines.append(line)
        return real_parse(line_place, line)

    monkeypatch.setattr(read_steering, "_parse_steering_line", count_parse)
    run.run(config, clock_config, None, ShutdownHandler())
    assert parsed_lines == [steering_line[:-1] for steering_line in steering_lines]


def test_a_series_with_no_row_for_the_epoch_before_starts_again() -> None:
    """Give rows of the epoch before as last rows, and older series their segments."""
    current_row = last_row()
    older_row = last_row(interpolated_datetime=PREVIOUS_EPOCH - 3 * T, segment=6)
    last_rows, last_segments = run.rows_before(
        {("mc1", "mc1"): current_row, ("mc2", "mc2"): older_row}, E
    )
    assert last_rows == {("mc1", "mc1"): current_row}
    assert last_segments == {("mc2", "mc2"): 6}


def test_a_pair_and_a_triple_starting_again_start_in_their_next_segments(
    tmp_path: Path,
) -> None:
    """Start a pair and a triple with no last row dormant in the segment after."""
    epoch = epoch_of([*REFERENCE_MEASUREMENTS, WORKED_DAS_MEASUREMENT], {}, tmp_path)
    last_segments: dict[SeriesKey, int] = {
        ("mc2", "ox23"): 5,
        ("mc1", "mc2", "ox23"): 3,
    }
    pair_step = run.process_pairs(epoch, {}, last_segments)
    triple_step = run.process_triples(epoch, {}, pair_step, last_segments)
    pair_row = pair_step.step_results[("mc2", "ox23")].row
    triple_row = triple_step.step_results[("mc1", "mc2", "ox23")].row
    assert (pair_row.flags, pair_row.segment) == ("RD", 6)
    assert (triple_row.flags, triple_row.segment) == ("PD", 4)
    assert pair_step.step_results[("mc1", "mc1")].row.segment == 0


# ------------------------------------------------------------------ buildings


def test_a_triple_is_left_out_while_its_clock_is_in_another_building(
    tmp_path: Path,
) -> None:
    """Hold no triple whose clock c is not in the building of its s (3.4)."""
    config, _ = make_deployment(tmp_path)
    moved_file = tmp_path / "moved.yaml"
    moved_file.write_text(
        CLOCK_CONFIG_YAML.replace(
            "  ox23: [{type: maser, location: 1}]",
            "  ox23: [{type: maser, location: 2}]",
        ),
        encoding="utf-8",
    )
    epoch = run.build_epoch(
        E,
        das_block_of(MEASURED_PAIRS),
        NO_SERIES,
        config,
        read_clock_config(moved_file),
    )
    assert epoch.locations == {"mc1": 1, "mc2": 1, "mc3": 1, "ox23": 2}
    assert epoch.pairs == tuple(sorted(MEASURED_PAIRS))
    assert epoch.triples == tuple(
        sorted(
            (r, s, c) for r in ("mc1", "mc2") for s, c in MEASURED_PAIRS if c != "ox23"
        )
    )


MOVING_CLOCK_CONFIG_YAML: Final = CLOCK_CONFIG_YAML.replace(
    "  ox23: [{type: maser, location: 1}]\n",
    "  ox23:\n"
    "    - {type: maser, location: 1}\n"
    f"    - {{effective_mjd: {datetime_to_mjd(LATE_START + 5 * T)}, location: 2}}\n"
    f"    - {{effective_mjd: {datetime_to_mjd(LATE_START + 7 * T)}, location: 1}}\n",
)
"""The clock configuration, with ox23 in building 2 for two epochs from the sixth."""


def moving_deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Give a loop deployment of ten epochs whose ox23 moves out and back."""
    config, _ = make_loop_deployment(tmp_path)
    moving_file = tmp_path / "moving.yaml"
    moving_file.write_text(MOVING_CLOCK_CONFIG_YAML, encoding="utf-8")
    write_das_files(tmp_path, [LATE_START + i * T for i in range(10)])
    return config, read_clock_config(moving_file)


def test_locations_are_kept_until_a_clock_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work locations out at the first epoch and at each move, and keep them else."""
    config, clock_config = moving_deployment(tmp_path)
    worked_out: list[datetime] = []
    real_locations_at = ClockConfig.locations_at

    def count_locations_at(
        self: ClockConfig, epoch_start: datetime
    ) -> dict[str, int | None]:
        """Note the mark, then work the locations out."""
        worked_out.append(epoch_start)
        return real_locations_at(self, epoch_start)

    monkeypatch.setattr(ClockConfig, "locations_at", count_locations_at)
    loop_series = ExistingSeries(
        pairs=frozenset({("mc1", "mc1"), ("mc1", "ox23")}),
        triples=frozenset({("mc1", "mc1", "mc1"), ("mc1", "mc1", "ox23")}),
    )
    epochs: list[run.Epoch] = []
    last_epoch = None
    for epoch_index in range(9):
        last_epoch = run.build_epoch(
            LATE_START + epoch_index * T,
            None,
            loop_series,
            config,
            clock_config,
            last_epoch,
        )
        epochs.append(last_epoch)
    ox23_locations = [1] * 5 + [2] * 2 + [1] * 2
    assert [epoch.locations["ox23"] for epoch in epochs] == ox23_locations
    assert [("mc1", "mc1", "ox23") in epoch.triples for epoch in epochs] == [
        ox23_location == 1 for ox23_location in ox23_locations
    ]
    assert worked_out == [LATE_START, LATE_START + 5 * T, LATE_START + 7 * T]
    assert epochs[1].locations is not epochs[0].locations


def test_a_triple_stops_while_its_clock_is_away_and_starts_cold_on_its_return(
    tmp_path: Path,
) -> None:
    """Write no triple rows while ox23 is away, then start it cold, stepped or not."""
    batch_config, clock_config = moving_deployment(tmp_path / "batch")
    run.run(batch_config, clock_config, None, ShutdownHandler())
    triple_rows = rows_of(batch_config, ("mc1", "mc1", "ox23"))
    assert epoch_indexes(triple_rows, LATE_START) == [2, 3, 4, 7, 8, 9]
    assert len(rows_of(batch_config, ("mc1", "ox23"))) == 10
    assert [(row.flags, row.segment) for row in triple_rows[1:]] == [
        ("RD", 0),
        ("ANU", 1),
        ("RD", 2),
        ("RD", 2),
        ("ANU", 3),
    ]
    stepped_config, clock_config = moving_deployment(tmp_path / "stepped")
    for _ in range(10):
        run.run(stepped_config, clock_config, 1, ShutdownHandler())
    for series_key in LOOP_SERIES:
        batch_file = registry.series_file(
            batch_config.processed.processed_path, "a", series_key
        )
        stepped_file = registry.series_file(
            stepped_config.processed.processed_path, "a", series_key
        )
        assert batch_file.read_bytes() == stepped_file.read_bytes(), series_key


# ------------------------------------------------------------ series that stop

SHORT_GAP_CLOCK_CONFIG_YAML: Final = CLOCK_CONFIG_YAML.replace(
    "gap_limit: 40", "gap_limit: 6"
)
"""The clock configuration, with every gap limit at N_break, 6 epochs."""

STOPPING_READINGS: Final = [("mc1", "mc1", 1000), ("mc1", "ox23", 50_000)]
"""mc1 measured against itself, and ox23 against mc1."""


def stopping_deployment(tmp_path: Path) -> tuple[AppConfig, ClockConfig]:
    """Give 14 epochs from E, ox23 measured at the first 3 and the last 2."""
    config, _ = make_loop_deployment(tmp_path, E)
    short_gap_file = tmp_path / "short_gap.yaml"
    short_gap_file.write_text(SHORT_GAP_CLOCK_CONFIG_YAML, encoding="utf-8")
    write_das_day(
        tmp_path,
        [STOPPING_READINGS] * 3 + [STOPPING_READINGS[:1]] * 9 + [STOPPING_READINGS] * 2,
        E,
    )
    return config, read_clock_config(short_gap_file)


def test_a_series_whose_measurements_stop_writes_no_more_rows(tmp_path: Path) -> None:
    """Predict for the gap limit, then write nothing until a measurement (13.3)."""
    config, clock_config = stopping_deployment(tmp_path)
    run.run(config, clock_config, None, ShutdownHandler())
    pair_rows = rows_of(config, ("mc1", "ox23"))
    assert epoch_indexes(pair_rows, E) == [*range(9), 12, 13]
    assert all("P" in row.flags and "D" not in row.flags for row in pair_rows[3:9]), (
        pair_rows[3:9]
    )
    assert [row.segment for row in pair_rows[8:]] == [1, 2, 2]
    assert epoch_indexes(rows_of(config, ("mc1", "mc1")), E) == list(range(14))
    assert epoch_indexes(rows_of(config, ("mc1", "mc1", "ox23")), E) == [2]
    for _, file_kind, series_key in run.data_series(config):
        assert not any(
            "D" in row.flags and "P" in row.flags for row in rows_of(config, series_key)
        ), (file_kind, series_key)


def test_a_series_that_stops_writes_the_same_files_stepped(tmp_path: Path) -> None:
    """Give the same files when a series stops, in one go or epoch by epoch (I5)."""
    batch_config, clock_config = stopping_deployment(tmp_path / "batch")
    run.run(batch_config, clock_config, None, ShutdownHandler())
    stepped_config, clock_config = stopping_deployment(tmp_path / "stepped")
    for _ in range(15):
        run.run(stepped_config, clock_config, 1, ShutdownHandler())
    batch_files = run.data_series(batch_config)
    assert len(batch_files) == 4
    assert [series_key for _, _, series_key in run.data_series(stepped_config)] == [
        series_key for _, _, series_key in batch_files
    ]
    for batch_file, _, series_key in batch_files:
        stepped_file = registry.series_file(
            stepped_config.processed.processed_path, "a", series_key
        )
        assert batch_file.read_bytes() == stepped_file.read_bytes(), series_key


def test_an_epoch_that_writes_no_row_is_not_counted_as_a_step(tmp_path: Path) -> None:
    """Pass over gap epochs no series writes, so a stepped run gets past them."""
    config, clock_config = make_loop_deployment(tmp_path)
    write_das_files(tmp_path, [LATE_START, LATE_START + T, LATE_START + 4 * T])
    run.run(config, clock_config, 2, ShutdownHandler())
    assert epoch_indexes(rows_of(config, ("mc1", "mc1")), LATE_START) == [0, 1]
    run.run(config, clock_config, 1, ShutdownHandler())
    self_rows = rows_of(config, ("mc1", "mc1"))
    assert epoch_indexes(self_rows, LATE_START) == [0, 1, 4]
    assert [(row.flags, row.segment) for row in self_rows] == [
        ("RD", 0),
        ("RD", 0),
        ("RD", 1),
    ]


def test_a_series_that_stops_is_logged_once_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a series that writes no row at INFO, and count only rows written (16.2)."""
    stopping_last_rows = {
        **REFERENCE_LAST_ROWS,
        ("mc1", "mc1"): last_row(x_fs=1_000_000, epochs_since_accept=40),
    }
    epoch = epoch_of(REFERENCE_MEASUREMENTS[1:], stopping_last_rows, tmp_path)
    log_entries = logged_events(caplog, epoch, stopping_last_rows)
    info_messages = messages_at(log_entries, "INFO")
    assert "das_a.mc1.mc1 stops: no row until it is measured again" in info_messages
    assert "das_a.mc1.mc1 dormant" not in info_messages
    assert not any(
        message.startswith("das_a.mc1.mc1:")
        for message in messages_at(log_entries, "DEBUG")
    )
    assert info_messages[-1].startswith("epoch 2025-09-23 06:00:00+00:00: 3 pairs, ")
    stopped_last_rows = {
        series_key: row
        for series_key, row in REFERENCE_LAST_ROWS.items()
        if series_key != ("mc1", "mc1")
    }
    restarted_entries = logged_events(caplog, epoch, stopped_last_rows)
    assert not any(
        message.startswith("das_a.mc1.mc1")
        for message in messages_at(restarted_entries, "INFO")
    )
