"""Tests for scripts/characterize.py.

The rules covered: a row's columns are read where the file's column table
puts them; the accepted rows of a local triple, z less step_offset, are
split at every cold start, a row flagged N after a dormant row; each value
is moved back to its epoch start by the median one-epoch rate times its
measurement time after the start; a row whose changes from its neighbours
both stray, in opposite directions, by more than five robust spreads is
dropped, and with one neighbour its one change decides; drift is taken off
as a quadratic; the Allan variance at tau = mT uses only whole sets of three
rows, m doubling up to a third of the rows' time, the stretches combined by
their terms; each reference's white phase noise is taken off and the
references combined by their terms; the noise model is fitted with every
coefficient zero or above, weighted by terms over m and the model's value
squared, until it settles; the crossover sets M, at least one; the gap limit
is the largest gap decycled with a five-sigma margin; and the script reads
a characterization run's files and prints one line per clock, references
left out.

The noise data are invented, from a seeded generator.
"""

import math
import random
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

import characterize
from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import main as das_processor_main
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.das_processor.registry import series_file

T: Final = characterize.T
"""One epoch, s."""

START: Final = datetime(2025, 9, 23, tzinfo=UTC)
"""An invented first epoch."""


def seeded(seed: int) -> random.Random:
    """Give a generator of invented noise, the same on every run."""
    return random.Random(seed)  # noqa: S311 - repeatable test data, not a secret


# ------------------------------------------------------------ steps 1 and 2


def test_cold_starts_split_the_accepted_rows_less_their_offsets() -> None:
    """Keep accepted rows, z less step_offset, and start again at N after D."""
    rows = [
        characterize.TripleRow(1, 100, 0, "AN"),
        characterize.TripleRow(2, 150, 40, "A"),
        characterize.TripleRow(3, 170, 40, "R"),
        characterize.TripleRow(4, None, 40, "P"),
        characterize.TripleRow(5, 400, 0, "AN"),
        characterize.TripleRow(6, 300, 0, "RD"),
        characterize.TripleRow(7, 900, 0, "AN"),
    ]
    assert characterize.split_at_cold_starts(rows) == [
        {1: 100.0, 2: 110.0, 5: 400.0},
        {7: 900.0},
    ]


def test_no_accepted_row_gives_no_stretch() -> None:
    """Give nothing for a series that was never accepted."""
    rows = [
        characterize.TripleRow(1, 5, 0, "RD"),
        characterize.TripleRow(2, 6, 0, "RD"),
    ]
    assert characterize.split_at_cold_starts(rows) == []


# ------------------------------------------------------------------- step 3


def test_a_value_is_moved_back_by_the_rate_times_its_delta() -> None:
    """Take off the median one-epoch change over T, times delta."""
    stretch = {epoch: 60.0 * epoch for epoch in range(5)}
    stretch[3] = 999.0
    deltas = dict.fromkeys(range(5), 100.0)
    moved = characterize.to_epoch_start(stretch, deltas)
    assert moved == {epoch: value - 0.1 * 100.0 for epoch, value in stretch.items()}


def test_values_with_no_neighbour_one_epoch_apart_stay() -> None:
    """Leave the values as they are when no rate can be worked out."""
    assert characterize.to_epoch_start({1: 5.0, 3: 7.0}, {1: 9.0, 3: 9.0}) == {
        1: 5.0,
        3: 7.0,
    }


# ------------------------------------------------------------------- step 4


def noisy_ramp(length: int, rate: float = 50.0, seed: int = 1) -> dict[int, float]:
    """Give a ramp with white noise of one ps."""
    generator = seeded(seed)
    return {epoch: rate * epoch + generator.gauss(0.0, 1.0) for epoch in range(length)}


def test_an_outlier_with_two_neighbours_is_dropped() -> None:
    """Drop a row whose two changes stray far, in opposite directions."""
    stretch = noisy_ramp(50)
    stretch[20] += 100.0
    assert sorted(set(noisy_ramp(50)) - set(characterize.drop_outliers(stretch))) == [
        20
    ]


def test_a_step_is_not_an_outlier() -> None:
    """Keep the rows on both sides of one large change in one direction."""
    stretch = noisy_ramp(50)
    for epoch in range(25, 50):
        stretch[epoch] += 100.0
    assert characterize.drop_outliers(stretch) == stretch


def test_an_outlier_beside_a_hole_is_judged_by_its_one_change() -> None:
    """Drop a row with one neighbour when that one change strays."""
    stretch = noisy_ramp(50)
    del stretch[21]
    stretch[20] += 100.0
    assert 20 not in characterize.drop_outliers(stretch)
    assert 19 in characterize.drop_outliers(stretch)


def test_two_bad_rows_together_are_not_caught() -> None:
    """Keep two neighbouring bad rows: each has one ordinary change."""
    stretch = noisy_ramp(50)
    stretch[20] += 100.0
    stretch[21] += 100.0
    kept = characterize.drop_outliers(stretch)
    assert 20 in kept
    assert 21 in kept


@pytest.mark.parametrize(
    "stretch", [{1: 1.0, 3: 2.0}, {epoch: 2.0 * epoch for epoch in range(6)}]
)
def test_no_change_or_no_stray_change_keeps_every_row(
    stretch: dict[int, float],
) -> None:
    """Keep every row when no two are one epoch apart, or every change is the same."""
    assert characterize.drop_outliers(stretch) == stretch


def test_with_no_spread_any_stray_change_counts() -> None:
    """Drop the one bad row of an otherwise perfect ramp."""
    stretch = {epoch: 2.0 * epoch for epoch in range(10)}
    stretch[5] += 1.0
    assert sorted(characterize.drop_outliers(stretch)) == [0, 1, 2, 3, 4, 6, 7, 8, 9]


# ------------------------------------------------------------------- step 5


def test_a_quadratic_is_taken_off() -> None:
    """Leave only the noise of a drifting series."""
    generator = seeded(2)
    noise = {epoch: generator.gauss(0.0, 1.0) for epoch in range(200)}
    drifting = {
        epoch: 7.0 + 3.0 * epoch + 0.01 * epoch * epoch + value
        for epoch, value in noise.items()
    }
    residual = characterize.remove_drift(drifting)
    noise_alone = characterize.remove_drift(noise)
    assert max(abs(residual[epoch] - noise_alone[epoch]) for epoch in noise) < 1e-6


@pytest.mark.parametrize("stretch", [{1: 1.0, 2: 5.0}, {1: 1.0, 2: 5.0, 3: 2.0}])
def test_too_few_rows_for_a_quadratic(stretch: dict[int, float]) -> None:
    """Fit through three rows exactly, and leave two as they are."""
    residual = characterize.remove_drift(stretch)
    if len(stretch) < 3:
        assert residual == stretch
    else:
        assert max(abs(value) for value in residual.values()) < 1e-9


# ------------------------------------------------------------------- step 6


def test_taus_double_up_to_a_third_of_the_rows_time() -> None:
    """Give 1, 2, 4, ... with 3m at most the epochs covered."""
    assert characterize.doubling_taus({0: 0.0, 48: 0.0}) == [1, 2, 4, 8, 16]
    assert characterize.doubling_taus({}) == []


def test_a_white_frequency_series_has_the_allan_variance_it_was_made_with() -> None:
    """Give sigma_y**2(tau) = h / tau for a random walk in phase."""
    generator = seeded(3)
    step_ps = 10.0
    phase, stretch = 0.0, {}
    for epoch in range(200_000):
        stretch[epoch] = phase
        phase += generator.gauss(0.0, step_ps)
    (tau_value, *_) = characterize.reference_variances([stretch])
    expected = (step_ps * 1e-12) ** 2 / T**2
    assert tau_value.m == 1
    assert tau_value.variance == pytest.approx(expected, rel=0.02)


def test_stretches_are_combined_by_their_terms() -> None:
    """Add the squares and terms of every stretch at each tau."""
    first = {0: 0.0, 1: 0.0, 2: 6.0, 3: 0.0}
    second = {10: 0.0, 11: 3.0, 12: 0.0, 13: 0.0}
    (tau_value,) = characterize.reference_variances([first, second])
    squares = 36.0 + 144.0 + 36.0 + 9.0
    assert tau_value == characterize.TauValue(
        m=1, variance=squares * 1e-24 / (2 * T**2 * 4), terms=4
    )


def test_a_stretch_too_short_for_any_tau_adds_nothing() -> None:
    """Leave out rows covering less than three epochs."""
    assert characterize.reference_variances([{10: 0.0, 11: 3.0, 12: 0.0}]) == []


def test_a_hole_leaves_out_every_set_it_falls_in() -> None:
    """Count only sets of three rows all present."""
    assert characterize.allan_sums({0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0}, 1).terms == 1


def test_white_phase_noise_is_taken_off_each_reference() -> None:
    """Take each reference's own noise off before combining, dropping none above."""
    noise = characterize.white_phase_variance(10.0, T)
    combined = characterize.clock_variances(
        [
            ([characterize.TauValue(1, noise + 2e-30, 10)], 10.0),
            ([characterize.TauValue(1, noise / 2, 30)], 10.0),
        ]
    )
    assert [(tau_value.m, tau_value.terms) for tau_value in combined] == [(1, 10)]
    assert combined[0].variance == pytest.approx(2e-30)


def model_taus(coefficients: characterize.Coefficients) -> list[characterize.TauValue]:
    """Give the model's own values at m = 1 to 2**12, with many terms each."""
    return [
        characterize.TauValue(
            m, characterize.model_variance(coefficients, m * T), 10**6
        )
        for m in (2**power for power in range(13))
    ]


@pytest.mark.parametrize(
    "coefficients", [(1e-21, 1e-29, 1e-34), (1e-21, 0.0, 1e-34), (0.0, 1e-28, 0.0)]
)
def test_the_fit_gives_back_the_model_it_was_made_with(
    coefficients: characterize.Coefficients,
) -> None:
    """Recover every coefficient, zero ones included, across many magnitudes."""
    fitted = characterize.fit_noise_model(model_taus(coefficients))
    for found, made in zip(fitted, coefficients, strict=True):
        assert found == pytest.approx(made, rel=1e-6, abs=1e-40)


def test_the_fit_never_gives_a_negative_coefficient() -> None:
    """Hold a coefficient at zero where a free fit would make it negative."""
    tau_values = [
        characterize.TauValue(m, 1e-24 / (m * T) - 1e-31 * m * T, 100)
        for m in (1, 2, 4, 8, 16)
    ]
    assert all(value >= 0 for value in characterize.fit_noise_model(tau_values))


def test_a_fit_that_does_not_settle_stops_after_its_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give the last fit when the rounds run out before it settles."""
    tau_values = model_taus((1e-21, 1e-29, 1e-34))
    monkeypatch.setattr(characterize, "FIT_ROUNDS", 1)
    assert characterize.fit_noise_model(tau_values) == characterize.nonnegative_fit(
        tau_values,
        [(tv.terms / tv.m) / tv.variance**2 for tv in tau_values],
    )


def test_no_values_fit_to_zero() -> None:
    """Give a zero model for a clock with no values."""
    assert characterize.fit_noise_model([]) == (0.0, 0.0, 0.0)


def test_a_singular_fit_is_passed_over() -> None:
    """Fit one tau with one term rather than three."""
    fitted = characterize.nonnegative_fit([characterize.TauValue(1, 2e-24, 5)], [1.0])
    assert characterize.model_variance(fitted, T) == pytest.approx(2e-24)


def test_the_weights_follow_the_model_and_independent_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weight a value by terms over m and the model's value squared."""
    calls: list[list[float]] = []
    real_fit = characterize.nonnegative_fit

    def recording_fit(
        tau_values: list[characterize.TauValue], weights: list[float]
    ) -> characterize.Coefficients:
        """Note the weights of each fit, then fit."""
        calls.append(list(weights))
        return real_fit(tau_values, weights)

    tau_values = model_taus((1e-21, 1e-29, 1e-34))
    monkeypatch.setattr(characterize, "nonnegative_fit", recording_fit)
    characterize.fit_noise_model(tau_values)
    assert calls[0] == [
        (tau_value.terms / tau_value.m) / tau_value.variance**2
        for tau_value in tau_values
    ]


# ------------------------------------------------------ settings from the fit


def test_the_crossover_is_where_the_two_noises_meet() -> None:
    """Find tau_c with sqrt(3) sigma_meas 1e-12 / tau_c = sigma_y,c(tau_c)."""
    coefficients = (1e-22, 1e-30, 1e-36)
    tau_c = characterize.crossover(coefficients, 20.0)
    assert tau_c is not None
    assert math.sqrt(3) * 20e-12 / tau_c == pytest.approx(
        math.sqrt(characterize.model_variance(coefficients, tau_c)), rel=1e-9
    )
    assert characterize.crossover((0.0, 0.0, 0.0), 20.0) is None


def test_the_gap_limit_is_the_largest_gap_decycled_with_margin() -> None:
    """Give n with 5 sigma_x,pred((n + 1) T) under P / 2 and n + 1 not."""
    coefficients = (1e-22, 1e-28, 1e-33)
    gap = characterize.gap_limit(coefficients, 10)

    def sigma(n: int) -> float:
        """Give sigma_x,pred((n + 1) T), ps."""
        tau = (n + 1) * T
        return (
            tau
            * 1e12
            * math.sqrt(
                characterize.model_variance(coefficients, tau)
                + characterize.model_variance(coefficients, 10 * T)
            )
        )

    assert 5 * sigma(gap) < 100_000 <= 5 * sigma(gap + 1)


def test_a_clock_too_noisy_for_one_epoch_has_no_gap() -> None:
    """Give -1 when even the next epoch's prediction is out of reach."""
    assert characterize.gap_limit((1e-10, 0.0, 0.0), 1) == -1


def test_a_quiet_clock_reaches_the_search_limit() -> None:
    """Stop the search at its limit for a clock that never strays."""
    assert characterize.gap_limit((0.0, 0.0, 0.0), 1) == characterize.GAP_SEARCH_LIMIT


# ------------------------------------------------- a characterization run's files

REFERENCES: Final = ("mc1", "mc2")
"""The invented deployment's references."""

CLOCKS: Final = ("hm1", "ox1")
"""Its clocks, each measured against both references."""

DAYS: Final = 2
"""How many days of data it holds."""

GAP_PAIR: Final = ("mc2", "hm1")
"""A pair the invented DAS does not measure for a few epochs."""

GAP_EPOCHS: Final = range(100, 105)
"""The epochs it is not measured in."""

CHARACTERIZATION_CONFIG: Final = (
    "rejects_before_restart: 36\n"
    "rms_limit:\n"
    "  default: 9999\n"
    "types:\n"
    "  mc:    {filter_states: 1, scale_time_constant: 50.0,"
    " initial_innovation_scale: 2000.0, gap_limit: 432}\n"
    "  clock: {filter_states: 1, scale_time_constant: 50.0,"
    " initial_innovation_scale: 2000.0, gap_limit: 432}\n"
    "clocks:\n"
    + "".join(f"  {name}: [{{type: mc, location: 1}}]\n" for name in REFERENCES)
    + "".join(f"  {name}: [{{type: clock, location: 1}}]\n" for name in CLOCKS)
)
"""Every clock with one state, as design 15.3's run has them."""


def true_phases(seed: int) -> dict[str, list[float]]:
    """Give each clock's phase at every second-of-epoch grid point, ps.

    The references stay at zero. hm1 runs 30 ps an epoch fast with a random
    walk of frequency; ox1 also drifts.
    """
    generator = seeded(seed)
    epochs = DAYS * 144
    phases: dict[str, list[float]] = {name: [0.0] * epochs for name in REFERENCES}
    for clock, drift in (("hm1", 0.0), ("ox1", 0.002)):
        phase, rate, clock_phases = 0.0, 30.0, []
        for _ in range(epochs):
            clock_phases.append(phase)
            rate += drift + generator.gauss(0.0, 0.5)
            phase += rate + generator.gauss(0.0, 5.0)
        phases[clock] = clock_phases
    return phases


def write_deployment(folder: Path, seed: int = 4) -> list[str]:
    """Write the invented DAS files and configuration; give the run's arguments."""
    for subfolder in ("das", "steering", "processed"):
        (folder / subfolder).mkdir(parents=True)
    (folder / "clock_config.yaml").write_text(CHARACTERIZATION_CONFIG)
    generator = seeded(seed + 1)
    phases = true_phases(seed)
    measured_pairs = [(r, s) for r in REFERENCES for s in REFERENCES]
    measured_pairs += [(r, c) for r in REFERENCES for c in CLOCKS]
    day_lines: dict[int, list[str]] = {}
    for epoch in range(DAYS * 144):
        epoch_start = START + epoch * timedelta(seconds=T)
        for slot, (reference, clock) in enumerate(measured_pairs):
            if (reference, clock) == GAP_PAIR and epoch in GAP_EPOCHS:
                continue
            offset = 1 + 500 * slot / len(measured_pairs) + generator.uniform(0, 60)
            rate = phases[clock][min(epoch + 1, DAYS * 144 - 1)] - phases[clock][epoch]
            true = phases[clock][epoch] + rate * offset / T - phases[reference][epoch]
            measured = round(true + generator.gauss(0.0, 3.0)) % 200_000
            measurement_mjd = round(
                datetime_to_mjd(epoch_start + timedelta(seconds=offset)), 6
            )
            line = DASMeasurement(
                measurement_mjd=measurement_mjd,
                measured_phase=measured,
                rms=3,
                switch=f"{reference[-1]}A{slot:02d}",
                clock=clock,
            )
            day_lines.setdefault(int(measurement_mjd), []).append(f"{line}\n")
    for day, lines in day_lines.items():
        (folder / "das" / f"cd5m5m_{day}.dat").write_text("".join(lines))
    return [
        "--rf", "a",
        "--cd5m5m-path", str(folder / "das"),
        "--steering-path", str(folder / "steering"),
        "--processed-path", str(folder / "processed"),
        "--clock-config-file", str(folder / "clock_config.yaml"),
        "--start-from-mjd", f"{datetime_to_mjd(START):.6f}",
        "--log-file", "None",
        "--log-level", "None",
        "--backup-count", "None",
    ]  # fmt: skip


@pytest.fixture(scope="module")
def processed_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run das_processor once over the invented deployment; give its output."""
    folder = tmp_path_factory.mktemp("characterization")
    das_arguments = write_deployment(folder)
    assert das_processor_main(das_arguments) == 0
    return folder / "processed"


def test_each_clock_has_a_local_triple_through_each_reference(
    processed_path: Path,
) -> None:
    """Find (r, r, c) for every clock and reference, references left out."""
    assert characterize.local_triples(processed_path, "a") == {
        clock: list(REFERENCES) for clock in CLOCKS
    }


def test_a_triple_s_rows_are_read_from_their_columns(processed_path: Path) -> None:
    """Read each row, its epoch numbered, its z and flags as written.

    The triple's first row is its first measurement, two epochs in, when
    its pairs cold-start; its clock pair's gap gives rows with no z.
    """
    triple_rows = characterize.read_triple_rows(
        series_file(processed_path, "a", (GAP_PAIR[0], *GAP_PAIR))
    )
    first_epoch = characterize.epoch_number(f"{datetime_to_mjd(START):.6f}")
    assert [row.epoch for row in triple_rows] == list(
        range(first_epoch + 2, first_epoch + DAYS * 144)
    )
    assert "A" in triple_rows[-1].flags
    assert triple_rows[-1].z is not None
    assert any(row.z is None for row in triple_rows)


def test_a_pair_s_unmeasured_epochs_give_no_time(processed_path: Path) -> None:
    """Leave out the rows without a measurement."""
    pair_rows = characterize.read_pair_rows(series_file(processed_path, "a", GAP_PAIR))
    assert len(pair_rows.deltas) == DAYS * 144 - len(GAP_EPOCHS)
    assert len(pair_rows.rms_values) == DAYS * 144 - len(GAP_EPOCHS)


def test_only_local_triples_of_clocks_are_found(tmp_path: Path) -> None:
    """Pass over references, remote triples, pairs, other channels and names."""
    archive = tmp_path / "ddiff"
    archive.mkdir()
    for file_name in (
        "das_a.mc1.mc1.hm1.dat",
        "das_a.mc1.mc2.hm1.dat",
        "das_a.mc1.mc1.mc2.dat",
        "das_a.mc1.hm1.dat",
        "das_b.mc2.mc2.hm1.dat",
        "notes.txt",
    ):
        (archive / file_name).touch()
    assert characterize.local_triples(tmp_path, "a") == {"hm1": ["mc1"]}


def test_a_pair_s_measurement_times_and_rms_are_read(processed_path: Path) -> None:
    """Give each measured epoch's time after its start, and every rms."""
    pair_rows = characterize.read_pair_rows(
        series_file(processed_path, "a", ("mc2", "ox1"))
    )
    assert len(pair_rows.deltas) == DAYS * 144
    assert all(0 < delta < T for delta in pair_rows.deltas.values())
    assert set(pair_rows.rms_values) == {3}


def test_each_clock_is_characterized_from_both_references(
    processed_path: Path,
) -> None:
    """Use both local triples, sigma_meas from the rms, and give M and G_max."""
    result = characterize.characterize_clock(
        processed_path, "a", "ox1", REFERENCES, drift=True
    )
    assert result.references == REFERENCES
    assert result.rows > DAYS * 144
    assert result.sigma_meas == 3.0
    assert result.variances
    assert result.M is not None and result.M >= 1
    assert result.gap is not None


def test_the_report_has_a_line_per_clock(
    processed_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the header and one line per clock, in name order."""
    assert (
        characterize.main(
            [str(processed_path), "--rf", "a", "--three-state", "ox", "--jobs", "1"]
        )
        == 0
    )
    report_lines = capsys.readouterr().out.splitlines()
    assert report_lines[0] == characterize.REPORT_HEADER
    assert [line.split()[0] for line in report_lines[1:]] == list(CLOCKS)
    assert all(line.split()[1] == "mc1,mc2" for line in report_lines[1:])


def test_clocks_are_characterized_in_parallel_alike(
    processed_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the same report from two processes as from one."""
    characterize.main([str(processed_path), "--rf", "a", "--jobs", "1"])
    one_process = capsys.readouterr().out
    characterize.main([str(processed_path), "--rf", "a", "--jobs", "2"])
    assert capsys.readouterr().out == one_process


def test_drift_is_taken_off_the_three_state_clocks_only(processed_path: Path) -> None:
    """Mark a clock for drift removal exactly when its name has a given prefix."""
    jobs = characterize.clock_jobs(processed_path, "a", ("ox",))
    assert [(job[2], job[3], job[4]) for job in jobs] == [
        ("hm1", REFERENCES, False),
        ("ox1", REFERENCES, True),
    ]
    assert not any(job[4] for job in characterize.clock_jobs(processed_path, "a", ()))


def test_one_clock_is_characterized_from_one_argument(processed_path: Path) -> None:
    """Give a pool of processes the same result as a direct call."""
    job: characterize.ClockJob = (processed_path, "a", "hm1", REFERENCES, False)
    assert characterize._characterize_one(job) == characterize.characterize_clock(*job)


def test_the_script_runs_as_a_program(
    processed_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run main from the command line and exit with its status."""
    monkeypatch.setattr(
        sys, "argv", ["characterize.py", str(processed_path), "--rf", "a"]
    )
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(characterize.__file__, run_name="__main__")
    assert program_exit.value.code == 0
    assert capsys.readouterr().out.startswith(characterize.REPORT_HEADER)


def test_a_result_with_nothing_fitted_shows_dashes() -> None:
    """Write '-' for a missing crossover, time constant and gap limit."""
    result = characterize.ClockResult(
        "hm9", ("mc1",), 0, 0.0, (0.0, 0.0, 0.0), None, None, None, ()
    )
    assert characterize.format_result(result).split()[-3:] == ["-", "-", "-"]
