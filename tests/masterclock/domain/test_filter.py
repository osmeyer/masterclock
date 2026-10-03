"""Tests for src/masterclock/domain/filter.py.

The rules covered: the fixed gains put every pole of the closed loop at
lambda = exp(-1/M), a triple pole for three states and a double pole for
two, and a 1-state series passes its measurement through; the prediction
moves the last row's state on by one epoch and adds the steering input,
with the phase exact; a dormant or missing last row gives no prediction;
the update adds the gains times the innovation, the phase exact; and a
noise-free ramp or parabola is followed to within 1 ps, whether the phase
is kept exact or each row stores it in whole femtoseconds.

The row lifecycle: a row starts from the last row moved on one epoch, or
dormant in segment 0 for a new series; an accepted row holds the update,
clears the counters and the buffer, and moves the innovation scale by the
innovation before the update, never below its floor; a held row (P, X or R)
stores the prediction and keeps the scale, the counters and the buffer, one
more epoch since an accept; a held row past the gap limit, or with no
prediction, is dormant, with no state; a cold start begins segment + 1 at
the measurement with sigma0; a warm segment start keeps the state and the
step offset and takes the new time constants, never a new model; a row of a
2- or 3-state series is unsettled while its segment is younger than five
time constants, a dormant or 1-state row never; flags are written in the
order ARXPDSNU; and a 1-state series passes its measurements through.

Steps: the gate passes an innovation of at most five innovation scales and
an rms up to the pair's limit; a counted reject enters the buffer, which
keeps three; three rejects that agree within three innovation scales are a
phase step, accepted in the same segment with the step added to the step
offset; three that lie on a line within three scales are a frequency step,
accepted in a new warm segment with the prediction moved onto the line; a
1-state series takes only phase steps.

Acquisition: a series with no valid state buffers its measurements and
cold-starts from the third of three from consecutive epochs whose second
difference is within 5 sqrt(6) sigma0; a missing epoch empties the buffer;
counted rejects reaching N_break make a series dormant; and the last
buffered measurement is what a dormant pair is decycled against.

The filter step: a configuration change starts a warm segment before the
measurement is handled, and shares the row with its outcome; then every
path of the decision flow gives its row, a component cold start makes a
triple dormant, and the result says whether the row cold-started.

The classification and acquisition limits hold exactly at their values, the
slope is fitted as the design writes it, gains exist for M = 1, the drift is
carried through holds and steps, a settings change keeps the step offset,
and each error is logged as raised.
"""

import dataclasses
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.domain import filter as estimator
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import (
    EPOCH_SECONDS,
    PHASE_MAX,
    PHASE_PERIOD,
    decycle,
    exact,
    round_even,
    to_fs,
)
from masterclock.domain.series import Row, SeriesParams, State, check_row

EPOCH_START: Final = datetime(2025, 9, 23, 5, 50, tzinfo=UTC)
"""An invented ten-minute mark."""

T: Final = EPOCH_SECONDS
"""One epoch, s."""

NO_STEERING_INPUT: Final = (Fraction(0), 0.0)
"""A steering input of nothing."""


GAIN_TABLE: Final = {
    10: (0.904837, 0.259182, 2.58751e-2, 4.30892e-4, 0.181269, 9.05592e-3),
    30: (0.967216, 9.51626e-2, 3.17150e-3, 1.76178e-5, 6.44930e-2, 1.07478e-3),
    100: (0.990050, 2.95545e-2, 2.95540e-4, 4.92562e-7, 1.98013e-2, 9.90058e-5),
    300: (0.996672, 9.95017e-3, 3.31672e-5, 1.84262e-8, 6.64449e-3, 1.10741e-5),
    1000: (0.999000, 2.99550e-3, 2.99550e-6, 4.99251e-10, 1.99800e-3, 9.99001e-7),
}
"""Design 8.4: M to lambda, 3-state g, h, k and 2-state g, h."""


def last_row(**field_changes: object) -> Row:
    """Build the worked epoch's last row, checked, with ``field_changes`` applied."""
    row_fields: dict[str, object] = {
        "interpolated_datetime": EPOCH_START,
        "innovation": 0.0,
        "x_fs": 1_234_567_000,
        "y": 0.0123,
        "d": 0.0,
        "innovation_scale": 3.0,
        "segment": 4,
        "step_offset": 0,
        "epochs_in_segment": 811,
        "epochs_since_accept": 0,
        "consecutive_rejects": 0,
        "rejects": (),
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "flags": "A",
    }
    row_fields.update(field_changes)
    row = Row(**row_fields)  # type: ignore[arg-type]
    check_row(row)
    return row


def significant_figures(number: float, digits: int = 6) -> float:
    """Round ``number`` to ``digits`` significant figures."""
    return float(f"{number:.{digits - 1}e}")


# -------------------------------------------------------------------- gains


@pytest.mark.parametrize("M", sorted(GAIN_TABLE))
def test_the_gains_reproduce_the_table(M: int) -> None:
    """Give lambda, g, h and k of design 8.4 to six significant figures (U6)."""
    lam, g3, h3, k3, g2, h2 = GAIN_TABLE[M]
    assert significant_figures(math.exp(-1 / M)) == lam
    g, h_over_t, two_k_over_t2 = estimator.gains(3, float(M))
    assert significant_figures(g) == g3
    assert significant_figures(h_over_t * T) == h3
    assert significant_figures(two_k_over_t2 * T * T / 2) == k3
    g, h_over_t, drift = estimator.gains(2, float(M))
    assert (significant_figures(g), significant_figures(h_over_t * T), drift) == (
        g2,
        h2,
        0.0,
    )


def closed_loop_matrix(filter_states: int, M: float) -> list[list[float]]:
    """Build (I - K H) Phi for the estimator's gains."""
    g, h_over_t, two_k_over_t2 = estimator.gains(filter_states, M)
    if filter_states == 2:
        gain_vector = [g, h_over_t]
        transition = [[1.0, T], [0.0, 1.0]]
    else:
        gain_vector = [g, h_over_t, two_k_over_t2]
        transition = [[1.0, T, T * T / 2], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
    state_count = len(gain_vector)
    return [
        [
            sum(
                ((1.0 if i == m else 0.0) - (gain_vector[i] if m == 0 else 0.0))
                * transition[m][j]
                for m in range(state_count)
            )
            for j in range(state_count)
        ]
        for i in range(state_count)
    ]


@pytest.mark.parametrize("M", sorted(GAIN_TABLE))
def test_every_pole_of_three_states_is_lambda(M: int) -> None:
    """Make the characteristic polynomial (z - lambda)**3 within 1e-4 (U6).

    Its coefficients are the trace, the sum of the principal 2 by 2 minors
    and the determinant, so no eigenvalue routine is needed.
    """
    a = closed_loop_matrix(3, float(M))
    lam = math.exp(-1 / M)
    trace = a[0][0] + a[1][1] + a[2][2]
    minors = (
        a[0][0] * a[1][1] - a[0][1] * a[1][0]
        + a[0][0] * a[2][2] - a[0][2] * a[2][0]
        + a[1][1] * a[2][2] - a[1][2] * a[2][1]
    )  # fmt: skip
    determinant = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    assert trace == pytest.approx(3 * lam, abs=1e-4)
    assert minors == pytest.approx(3 * lam**2, abs=1e-4)
    assert determinant == pytest.approx(lam**3, abs=1e-4)


@pytest.mark.parametrize("M", sorted(GAIN_TABLE))
def test_every_pole_of_two_states_is_lambda(M: int) -> None:
    """Make the characteristic polynomial (z - lambda)**2 within 1e-4 (U6)."""
    a = closed_loop_matrix(2, float(M))
    lam = math.exp(-1 / M)
    assert a[0][0] + a[1][1] == pytest.approx(2 * lam, abs=1e-4)
    assert a[0][0] * a[1][1] - a[0][1] * a[1][0] == pytest.approx(lam**2, abs=1e-4)


def test_one_state_passes_the_measurement_through() -> None:
    """Give a gain of one on the phase and none on rate or drift."""
    assert estimator.gains(1, None) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("filter_states", "M"),
    [(1, 10.0), (2, None), (3, None), (4, 10.0), (0, None), (2, 0.5)],
)
def test_gains_refuse_a_model_and_time_constant_that_do_not_belong(
    filter_states: int, M: float | None
) -> None:
    """Refuse a time constant on 1 state, none on 2 or 3, or another model."""
    with pytest.raises(FilterError):
        estimator.gains(filter_states, M)


# ---------------------------------------------------------------- predict


def test_the_worked_epoch_predicts_its_state() -> None:
    """Move x on by y T exactly, keep the rate, for the worked epoch."""
    prediction = estimator.predict(last_row(), NO_STEERING_INPUT)
    assert prediction == State(x=1_234_567 + exact(0.0123) * 600, y=0.0123, d=0.0)


def test_three_states_move_on_with_rate_and_drift() -> None:
    """Add y T + d T**2 / 2 to x and d T to y, and keep d."""
    previous_row = last_row(y=0.25, d=1e-6)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    assert prediction.x == 1_234_567 + exact(0.25) * T + exact(1e-6) * T * T / 2
    assert prediction.y == 0.25 + 1e-6 * T
    assert prediction.d == 1e-6


def test_two_states_move_on_with_rate_only() -> None:
    """Add y T to x and keep y; a 2-state row has no drift."""
    previous_row = last_row(filter_states=2, time_constant=30.0, y=0.25)
    assert estimator.predict(previous_row, NO_STEERING_INPUT) == State(
        x=Fraction(1_234_567 + 150), y=0.25
    )


def test_one_state_carries_the_phase() -> None:
    """Keep x, and no rate or drift, for a 1-state row."""
    previous_row = last_row(filter_states=1, time_constant=None, y=0.0)
    assert estimator.predict(previous_row, NO_STEERING_INPUT) == State(
        x=Fraction(1_234_567), y=0.0
    )


@pytest.mark.parametrize(
    ("field_changes", "expected_prediction"),
    [
        ({}, State(x=1_234_567 + exact(0.0123) * 600 + Fraction(7, 2), y=0.0123 + 0.5)),
        (
            {"filter_states": 2, "time_constant": 30.0},
            State(x=1_234_567 + exact(0.0123) * 600 + Fraction(7, 2), y=0.0123 + 0.5),
        ),
        (
            {"filter_states": 1, "time_constant": None, "y": 0.0},
            State(x=1_234_567 + Fraction(7, 2), y=0.0),
        ),
    ],
)
def test_the_steering_input_is_added(
    field_changes: dict[str, object], expected_prediction: State
) -> None:
    """Add u_x to the phase, and u_y to the rate where there is one."""
    steering_input = (Fraction(7, 2), 0.5)
    assert (
        estimator.predict(last_row(**field_changes), steering_input)
        == expected_prediction
    )


def test_a_dormant_or_missing_row_gives_no_prediction() -> None:
    """Give None without a last row, or with a dormant one."""
    dormant_last_row = last_row(
        flags="PD", innovation=None, x_fs=None, y=None, d=None, innovation_scale=None
    )
    assert estimator.predict(None, NO_STEERING_INPUT) is None
    assert estimator.predict(dormant_last_row, NO_STEERING_INPUT) is None


# ----------------------------------------------------------------- update


def test_the_worked_epoch_updates_to_its_state() -> None:
    """Reproduce Appendix A: x stored as 1 234 574.457 ps, y and d exactly."""
    prediction = estimator.predict(last_row(), NO_STEERING_INPUT)
    assert prediction is not None
    innovation = 1_234_577 - prediction.x
    assert float(innovation) == 2.62
    updated_state = estimator.update(prediction, innovation, 3, 100.0)
    assert to_fs(updated_state.x) == 1_234_574_457
    assert updated_state.y == 0.01230129052352643
    assert updated_state.d == 7.169515400974333e-12


def test_the_update_is_exact_in_phase() -> None:
    """Add g times the innovation to x as an exact Fraction."""
    prediction = State(x=Fraction(2**60) + Fraction(1, 3), y=0.0)
    g, _, _ = estimator.gains(2, 30.0)
    updated_state = estimator.update(prediction, Fraction(10), 2, 30.0)
    assert updated_state.x == Fraction(2**60) + Fraction(1, 3) + exact(g) * 10
    assert updated_state.d == 0.0


def test_one_state_takes_the_measurement_exactly() -> None:
    """Make x the measurement for a 1-state series, with no rate."""
    prediction = State(x=Fraction(2**62 + 5), y=0.0)
    updated_state = estimator.update(prediction, Fraction(-7), 1, None)
    assert updated_state == State(x=Fraction(2**62 - 2), y=0.0)


# ----------------------------------------------------- following a signal


def follow_rows(
    filter_states: int, M: float, true_phases: list[Fraction]
) -> list[Fraction]:
    """Run the estimator over noise-free measurements, as the rows would.

    The measurements are the truth rounded to whole picoseconds. The series
    cold-starts from the first, then each epoch predicts from the last
    stored row, takes the innovation, updates and stores x in whole
    femtoseconds, as a row stores it.
    """
    measurements = [round_even(true_phase) for true_phase in true_phases]
    previous_row = last_row(
        x_fs=to_fs(measurements[0]),
        y=0.0,
        d=0.0,
        filter_states=filter_states,
        time_constant=M,
        epochs_in_segment=0,
    )
    innovations: list[Fraction] = []
    for epoch_number, z in enumerate(measurements[1:], start=1):
        prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
        assert prediction is not None
        innovation = z - prediction.x
        innovations.append(innovation)
        updated_state = estimator.update(prediction, innovation, filter_states, M)
        previous_row = last_row(
            interpolated_datetime=EPOCH_START + timedelta(minutes=10 * epoch_number),
            x_fs=to_fs(updated_state.x),
            y=updated_state.y,
            d=updated_state.d,
            filter_states=filter_states,
            time_constant=M,
        )
    return innovations


def follow_exactly(M: float, true_phases: list[Fraction]) -> list[Fraction]:
    """Run the 3-state loop over exact measurements, with x never rounded.

    The transition is written out here, as Phi of design 8.2, so the loop
    can carry x as an exact Fraction from epoch to epoch.
    """
    estimator_state = State(x=true_phases[0], y=0.0, d=0.0)
    innovations: list[Fraction] = []
    for z in true_phases[1:]:
        prediction = State(
            x=estimator_state.x
            + exact(estimator_state.y) * T
            + exact(estimator_state.d) * T * T / 2,
            y=estimator_state.y + estimator_state.d * T,
            d=estimator_state.d,
        )
        innovation = z - prediction.x
        innovations.append(innovation)
        estimator_state = estimator.update(prediction, innovation, 3, M)
    return innovations


def parabola_phases(M: float) -> list[Fraction]:
    """Give a noise-free phase with rate and drift over 40 M epochs."""
    rate, drift = Fraction(1, 20), Fraction(1, 10**7)
    return [1_000 + rate * T * k + drift * (T * k) ** 2 / 2 for k in range(int(40 * M))]


@pytest.mark.parametrize("M", [10.0, 30.0])
def test_a_ramp_is_followed_within_a_picosecond(M: float) -> None:
    """Bring a 2-state innovation within 1 ps of a ramp after 20 M epochs (U7)."""
    rate = Fraction(1, 20)
    true_phases = [1_000 + rate * T * k for k in range(int(40 * M))]
    innovations = follow_rows(2, M, true_phases)
    assert max(abs(innovation) for innovation in innovations[int(20 * M) :]) <= 1


@pytest.mark.parametrize("M", [10.0, 30.0])
def test_a_parabola_is_followed_within_a_picosecond(M: float) -> None:
    """Bring a 3-state innovation within 1 ps of a parabola after 20 M epochs (U7).

    With exact values, so the filter's own dynamics are what is tested.
    """
    innovations = follow_exactly(M, parabola_phases(M))
    assert max(abs(innovation) for innovation in innovations[int(20 * M) :]) <= 1


@pytest.mark.parametrize("M", [10.0, 30.0, 100.0])
def test_a_parabola_stored_in_femtoseconds_is_followed_within_a_picosecond(
    M: float,
) -> None:
    """Bring a 3-state innovation within 1 ps of a parabola, as rows store it (U7).

    Each row stores x in whole femtoseconds. Stored in whole picoseconds,
    the rounding would come back every epoch and the 3-state loop would take
    it into its rate and drift, leaving innovations of 2 ps and more here.
    """
    innovations = follow_rows(3, M, parabola_phases(M))
    assert max(abs(innovation) for innovation in innovations[int(20 * M) :]) <= 1


# ------------------------------------------------------------ row lifecycle


def make_series_params(**field_changes: object) -> SeriesParams:
    """Build the worked epoch's settings, with ``field_changes`` applied."""
    settings_fields: dict[str, object] = {
        "filter_states": 3,
        "M": 100.0,
        "M_sigma": 50.0,
        "sigma0": 5.0,
        "gmax": 432,
        "n_break": 36,
        "rms_max": 80,
    }
    settings_fields.update(field_changes)
    return SeriesParams(**settings_fields)  # type: ignore[arg-type]


ONE_STATE_FIELDS: Final[dict[str, object]] = {
    "y": 0.0,
    "d": 0.0,
    "filter_states": 1,
    "time_constant": None,
}
"""The changes that make a last row one of a 1-state series."""

DORMANT_FIELDS: Final[dict[str, object]] = {
    "innovation": None,
    "x_fs": None,
    "y": None,
    "d": None,
    "innovation_scale": None,
}
"""The changes that make a last row a dormant one, apart from its flags."""

NEXT_EPOCH_START: Final = EPOCH_START + timedelta(minutes=10)
"""The epoch after the worked epoch's last row: the worked epoch itself."""


def moved_on(previous_row: Row, **draft_changes: object) -> estimator.RowDraft:
    """Carry ``previous_row`` on one epoch, with ``draft_changes`` made to the draft."""
    draft = estimator.carry(
        previous_row.interpolated_datetime + timedelta(minutes=10),
        previous_row,
        make_series_params(),
    )
    return dataclasses.replace(draft, **draft_changes)  # type: ignore[arg-type]


def test_a_draft_has_the_fields_of_a_row() -> None:
    """Give RowDraft exactly Row's fields, in the same order, so finish fits."""
    field_names = [
        draft_field.name for draft_field in dataclasses.fields(estimator.RowDraft)
    ]
    assert field_names == [row_field.name for row_field in dataclasses.fields(Row)]


def test_a_new_series_starts_dormant_in_segment_zero() -> None:
    """Start a series with no last row with no state and every counter at 0."""
    draft = estimator.carry(
        NEXT_EPOCH_START,
        None,
        make_series_params(filter_states=2, M=30.0, M_sigma=40.0),
    )
    assert draft == estimator.RowDraft(
        interpolated_datetime=NEXT_EPOCH_START,
        innovation=None,
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        segment=0,
        step_offset=0,
        epochs_in_segment=0,
        epochs_since_accept=0,
        consecutive_rejects=0,
        rejects=(),
        filter_states=2,
        time_constant=30.0,
        scale_time_constant=40.0,
        flags="",
    )


def test_carry_moves_the_last_row_on_one_epoch() -> None:
    """Keep the last row's fields, one more epoch in its segment, no innovation."""
    previous_row = last_row(
        innovation=-4.5,
        consecutive_rejects=2,
        rejects=((EPOCH_START - timedelta(minutes=10), 40.0), (EPOCH_START, -4.5)),
        epochs_since_accept=2,
        flags="R",
    )
    draft = estimator.carry(NEXT_EPOCH_START, previous_row, make_series_params(M=150.0))
    expected_fields = {
        **dataclasses.asdict(previous_row),
        "interpolated_datetime": NEXT_EPOCH_START,
        "innovation": None,
        "epochs_in_segment": 812,
        "flags": "",
    }
    assert dataclasses.asdict(draft) == expected_fields


def test_a_slip_correction_marks_the_row() -> None:
    """Carry S into the row when the slip check corrected its measurement."""
    draft = estimator.carry(
        NEXT_EPOCH_START, last_row(), make_series_params(), slip=True
    )
    assert draft.flags == "S"
    prediction = estimator.predict(last_row(), NO_STEERING_INPUT)
    assert prediction is not None
    row = estimator.accept(draft, prediction, 1_234_577 - prediction.x, 3)
    assert row.flags == "AS"


def test_the_worked_epoch_gives_the_rows_of_the_measurement_file() -> None:
    """Reproduce the two example rows of design 5.4: accepted, then predicted."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    accepted_row = estimator.accept(
        estimator.carry(NEXT_EPOCH_START, previous_row, make_series_params()),
        prediction,
        1_234_577 - prediction.x,
        3,
    )
    assert accepted_row == last_row(
        interpolated_datetime=NEXT_EPOCH_START,
        innovation=float(1_234_577 - prediction.x),
        x_fs=1_234_574_457,
        y=0.01230129052352643,
        d=7.169515400974333e-12,
        epochs_in_segment=812,
    )
    following_prediction = estimator.predict(accepted_row, NO_STEERING_INPUT)
    assert following_prediction is not None
    later_epoch_start = NEXT_EPOCH_START + timedelta(minutes=10)
    predicted_row = estimator.hold(
        estimator.carry(later_epoch_start, accepted_row, make_series_params()),
        following_prediction,
        "P",
        make_series_params(),
    )
    assert predicted_row == last_row(
        interpolated_datetime=later_epoch_start,
        innovation=None,
        x_fs=1_234_581_838,
        y=0.012301294825235671,
        d=7.169515400974333e-12,
        epochs_in_segment=813,
        epochs_since_accept=1,
        flags="P",
    )


def test_an_accept_clears_the_counters_and_the_buffer() -> None:
    """Set consecutive_rejects and epochs_since_accept to 0, empty the buffer."""
    previous_row = last_row(
        innovation=-4.5,
        consecutive_rejects=2,
        rejects=((EPOCH_START - timedelta(minutes=10), 40.0), (EPOCH_START, -4.5)),
        epochs_since_accept=2,
        flags="R",
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    row = estimator.accept(moved_on(previous_row), prediction, Fraction(1), 3)
    assert (row.consecutive_rejects, row.rejects, row.epochs_since_accept) == (0, (), 0)
    assert row.flags == "A"


@pytest.mark.parametrize(
    ("innovation_scale", "innovation", "scale_floor", "expected_scale"),
    [
        (4.0, 10, 3, math.sqrt(0.98 * 16 + 0.02 * 100)),
        (4.0, -10, 3, math.sqrt(0.98 * 16 + 0.02 * 100)),
        (4.0, 0, 3, math.sqrt(0.98 * 16)),
        (3.0, 1, 3, 3.0),
        (4.0, 0, 3.99, math.sqrt(3.99**2)),
    ],
)
def test_an_accept_moves_the_innovation_scale_but_not_below_its_floor(
    innovation_scale: float, innovation: int, scale_floor: float, expected_scale: float
) -> None:
    """Average the squared innovation in with w = 1/M_sigma, kept above the floor."""
    previous_row = last_row(innovation_scale=innovation_scale)
    prediction = State(x=Fraction(1_234_574), y=0.0123)
    row = estimator.accept(
        moved_on(previous_row), prediction, Fraction(innovation), scale_floor
    )
    assert row.innovation_scale == expected_scale


def test_an_accept_takes_the_innovation_before_the_update() -> None:
    """Write the innovation given, and move the scale by it, not the residual."""
    previous_row = last_row(innovation_scale=4.0)
    prediction = State(x=Fraction(1_000), y=0.0)
    row = estimator.accept(moved_on(previous_row), prediction, Fraction(21, 2), 1)
    assert row.innovation == 10.5
    assert row.innovation_scale == math.sqrt(0.98 * 16 + 0.02 * 10.5**2)


def test_an_innovation_given_as_a_float_gives_the_same_row() -> None:
    """Build the same row and state whether nu is passed in or worked out."""
    prediction = estimator.predict(last_row(), NO_STEERING_INPUT)
    assert prediction is not None
    innovation = 1_234_577 - prediction.x
    assert estimator.accept(
        moved_on(last_row()), prediction, innovation, 3, nu=float(innovation)
    ) == estimator.accept(moved_on(last_row()), prediction, innovation, 3)
    assert estimator.update(
        prediction, innovation, 3, 100.0, nu=float(innovation)
    ) == estimator.update(prediction, innovation, 3, 100.0)


def test_an_accept_needs_a_series_with_an_innovation_scale() -> None:
    """Raise FilterError for an accept on a draft that has no scale."""
    draft = moved_on(last_row(), innovation_scale=None)
    with pytest.raises(FilterError, match="no innovation scale"):
        estimator.accept(draft, State(x=Fraction(0), y=0.0), Fraction(0), 3)


@pytest.mark.parametrize("outcome", ["P", "X", "R"])
def test_a_held_row_carries_the_prediction(outcome: Literal["P", "X", "R"]) -> None:
    """Store the prediction, keep the scale and counters, one more epoch unaccepted."""
    previous_row = last_row(
        innovation=-4.5,
        consecutive_rejects=1,
        rejects=((EPOCH_START, -40.0),),
        epochs_since_accept=1,
        flags="R",
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    innovation = None if outcome == "P" else 7.25
    row = estimator.hold(
        moved_on(previous_row, innovation=innovation),
        prediction,
        outcome,
        make_series_params(),
    )
    assert row == last_row(
        interpolated_datetime=NEXT_EPOCH_START,
        innovation=innovation,
        x_fs=to_fs(prediction.x),
        y=prediction.y,
        d=prediction.d,
        epochs_in_segment=812,
        epochs_since_accept=2,
        consecutive_rejects=1,
        rejects=((EPOCH_START, -40.0),),
        flags=outcome,
    )


def test_an_excluded_measurement_within_the_gate_leaves_the_counts() -> None:
    """Give X with consecutive_rejects and the buffer unchanged (U12)."""
    reject_buffer = ((EPOCH_START - timedelta(minutes=10), 30.0), (EPOCH_START, 31.0))
    previous_row = last_row(
        innovation=31.0,
        consecutive_rejects=2,
        rejects=reject_buffer,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    row = estimator.hold(
        moved_on(previous_row, innovation=2.0), prediction, "X", make_series_params()
    )
    assert (row.flags, row.consecutive_rejects, row.rejects) == ("X", 2, reject_buffer)
    assert row.epochs_since_accept == 3
    assert row.x_fs == to_fs(prediction.x)


def run_gap(epoch_count: int, gmax: int) -> Row:
    """Hold a tracked series through ``epoch_count`` epochs with no measurement."""
    series_params = make_series_params(gmax=gmax, n_break=3)
    previous_row = last_row()
    for _ in range(epoch_count):
        prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
        draft = estimator.carry(
            previous_row.interpolated_datetime + timedelta(minutes=10),
            previous_row,
            series_params,
        )
        previous_row = estimator.hold(draft, prediction, "P", series_params)
    return previous_row


@pytest.mark.parametrize("gmax", [3, 6])
def test_a_gap_of_the_gap_limit_is_still_predicted(gmax: int) -> None:
    """Give P rows for G_max epochs, then accept the next measurement (U13)."""
    previous_row = run_gap(gmax, gmax)
    assert (previous_row.flags, previous_row.epochs_since_accept) == ("P", gmax)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    row = estimator.accept(
        moved_on(previous_row), prediction, round_even(prediction.x) - prediction.x, 3
    )
    assert (row.flags, row.epochs_since_accept, row.segment) == ("A", 0, 4)


@pytest.mark.parametrize("gmax", [3, 6])
def test_a_gap_past_the_gap_limit_goes_dormant(gmax: int) -> None:
    """Give a D row at G_max + 1 epochs, with no state and no prediction (U13)."""
    previous_row = run_gap(gmax + 1, gmax)
    assert (previous_row.flags, previous_row.epochs_since_accept) == ("PD", gmax + 1)
    assert (
        previous_row.x_fs,
        previous_row.y,
        previous_row.d,
        previous_row.innovation_scale,
    ) == (None,) * 4
    assert (previous_row.segment, previous_row.step_offset) == (4, 0)
    assert estimator.predict(previous_row, NO_STEERING_INPUT) is None
    later_row = run_gap(gmax + 3, gmax)
    assert (later_row.flags, later_row.epochs_since_accept) == ("PD", gmax + 3)


def test_a_held_row_without_a_prediction_is_dormant() -> None:
    """Make a held row dormant when the series has no prediction."""
    row = estimator.hold(moved_on(last_row()), None, "R", make_series_params())
    assert row.flags == "RD"
    assert row.x_fs is None


@pytest.mark.parametrize("outcome", ["P", "X", "R"])
def test_a_dormant_row_keeps_its_segment_and_offset(
    outcome: Literal["P", "X", "R"],
) -> None:
    """Empty the state and the buffer, keep segment and step_offset (13.4)."""
    previous_row = last_row(
        step_offset=-37, rejects=((EPOCH_START, 12.0),), consecutive_rejects=1
    )
    row = estimator.dormant(moved_on(previous_row), outcome)
    assert row.flags == f"{outcome}D"
    assert (row.x_fs, row.y, row.d, row.innovation_scale) == (None,) * 4
    assert (row.segment, row.step_offset, row.rejects) == (4, -37, ())


def test_a_dormant_row_can_keep_its_acquisition_buffer() -> None:
    """Keep the buffer when asked: it holds the measurements to acquire from."""
    buffered_entries = ((EPOCH_START, 1_234_000.0),)
    previous_row = last_row(
        flags="RD", rejects=buffered_entries, segment=0, **DORMANT_FIELDS
    )
    row = estimator.dormant(moved_on(previous_row), "R", keep_buffer=True)
    assert (row.flags, row.rejects) == ("RD", buffered_entries)


@pytest.mark.parametrize(
    ("filter_states", "M", "flags"),
    [(3, 100.0, "ANU"), (2, 30.0, "ANU"), (1, None, "AN")],
)
def test_a_cold_start_begins_a_segment_from_the_measurement(
    filter_states: Literal[1, 2, 3], M: float | None, flags: str
) -> None:
    """Start segment + 1 at [z, 0, 0] with sigma0 and step_offset 0 (8.6, 13.4)."""
    series_params = make_series_params(
        filter_states=filter_states, M=M, M_sigma=40.0, sigma0=6.5
    )
    previous_row = last_row(
        filter_states=filter_states,
        time_constant=M,
        flags="RD",
        segment=2,
        step_offset=15,
        consecutive_rejects=4,
        rejects=((EPOCH_START, 1_234_570.0),),
        epochs_since_accept=9,
        **DORMANT_FIELDS,
    )
    draft = estimator.carry(NEXT_EPOCH_START, previous_row, series_params)
    row = estimator.cold_start(draft, 1_234_577, series_params)
    assert row == Row(
        interpolated_datetime=NEXT_EPOCH_START,
        innovation=None,
        x_fs=1_234_577_000,
        y=0.0,
        d=0.0,
        innovation_scale=6.5,
        segment=3,
        step_offset=0,
        epochs_in_segment=0,
        epochs_since_accept=0,
        consecutive_rejects=0,
        rejects=(),
        filter_states=filter_states,
        time_constant=M,
        scale_time_constant=40.0,
        flags=flags,
    )


def test_a_configuration_change_starts_a_warm_segment() -> None:
    """Carry X- and step_offset into segment + 1 with the new M: N U + outcome (8.7)."""
    previous_row = last_row(step_offset=25)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    changed_params = make_series_params(M=150.0, M_sigma=60.0)
    draft = moved_on(previous_row)
    estimator.start_segment(draft, changed_params, keep_offset=True)
    row = estimator.hold(draft, prediction, "P", changed_params)
    assert (row.flags, row.segment, row.epochs_in_segment) == ("PNU", 5, 0)
    assert (row.step_offset, row.time_constant, row.scale_time_constant) == (
        25,
        150.0,
        60.0,
    )
    assert (row.x_fs, row.innovation_scale) == (to_fs(prediction.x), 3.0)


def test_a_frequency_step_starts_a_warm_segment_and_accepts() -> None:
    """Give A N U, segment + 1, step_offset carried, then the update (13.4)."""
    previous_row = last_row(step_offset=25)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(previous_row)
    estimator.start_segment(draft, make_series_params(), keep_offset=True)
    row = estimator.accept(draft, prediction, Fraction(2), 3)
    assert (row.flags, row.segment, row.step_offset) == ("ANU", 5, 25)
    assert row.x_fs == to_fs(estimator.update(prediction, Fraction(2), 3, 100.0).x)


def test_a_phase_step_keeps_the_segment() -> None:
    """Accept within the same segment, with the step offset the step gave (13.4)."""
    previous_row = last_row(step_offset=25)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    row = estimator.accept(
        moved_on(previous_row, step_offset=525), prediction, Fraction(2), 3
    )
    assert (row.flags, row.segment, row.step_offset) == ("A", 4, 525)


def test_a_segment_keeps_its_model() -> None:
    """Raise FilterError for a segment whose settings name another model."""
    with pytest.raises(FilterError, match="model"):
        estimator.start_segment(
            moved_on(last_row()),
            make_series_params(filter_states=2, M=30.0),
            keep_offset=True,
        )


@pytest.mark.parametrize(
    ("M", "epochs_in_segment", "flags"),
    [(100.0, 499, "AU"), (100.0, 500, "A"), (30.0, 149, "AU"), (30.0, 150, "A")],
)
def test_a_row_is_unsettled_until_five_time_constants(
    M: float, epochs_in_segment: int, flags: str
) -> None:
    """Carry U while epochs_in_segment < 5 M, and not from 5 M on (8.8)."""
    previous_row = last_row(time_constant=M)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(previous_row, epochs_in_segment=epochs_in_segment)
    assert estimator.accept(draft, prediction, Fraction(0), 3).flags == flags


def test_a_dormant_row_is_never_unsettled() -> None:
    """Give D without U, however young the segment."""
    row = estimator.dormant(moved_on(last_row(), epochs_in_segment=0), "P")
    assert row.flags == "PD"


def test_flags_are_written_in_their_order() -> None:
    """Order slip, new segment and outcome as ARXPDSNU, whatever was added first."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = estimator.carry(
        NEXT_EPOCH_START, previous_row, make_series_params(), slip=True
    )
    estimator.start_segment(draft, make_series_params(), keep_offset=True)
    assert draft.flags == "SN"
    assert estimator.accept(draft, prediction, Fraction(0), 3).flags == "ASNU"


def test_finish_refuses_a_row_that_breaks_a_rule() -> None:
    """Raise FilterError when the finished draft is not a valid row."""
    draft = moved_on(last_row(), x_fs=None)
    with pytest.raises(FilterError, match="invalid row"):
        estimator.finish(draft, "A")


def test_a_one_state_series_passes_its_measurements_through() -> None:
    """Make x = z exactly on every accepted row, y and d 0.0, never U (U24)."""
    series_params = make_series_params(filter_states=1, M=None)
    previous_row = last_row(epochs_in_segment=0, **ONE_STATE_FIELDS)
    measurements = [2**60 + 7, 2**60 + 7, None, 2**60 - 200_001, 5]
    for z in measurements:
        prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
        assert prediction is not None
        draft = estimator.carry(
            previous_row.interpolated_datetime + timedelta(minutes=10),
            previous_row,
            series_params,
        )
        if z is None:
            row = estimator.hold(draft, prediction, "P", series_params)
            assert row.x_fs == previous_row.x_fs
        else:
            row = estimator.accept(draft, prediction, z - prediction.x, 3)
            assert row.x_fs == z * 1000
        assert (row.y, row.d) == (0.0, 0.0)
        assert "U" not in row.flags
        previous_row = row


# ------------------------------------------------------- step classification

EPOCH_LENGTH: Final = timedelta(minutes=10)
"""One epoch."""


def reject_buffer(*innovations: float) -> tuple[tuple[datetime, float], ...]:
    """Give a reject buffer of ``innovations``, one per epoch from EPOCH_START."""
    return tuple(
        (EPOCH_START + EPOCH_LENGTH * epoch_index, innovation)
        for epoch_index, innovation in enumerate(innovations)
    )


@pytest.mark.parametrize(
    ("innovations", "sigma", "step_kind"),
    [
        ((150.0, 151.0, 149.0), 3.0, "phase"),
        ((100.0, 100.0, 112.0), 3.0, "phase"),
        ((100.0, 100.0, 121.0), 3.0, "frequency"),
        ((30.0, 60.0, 90.0), 3.0, "frequency"),
        ((0.0, 120.0, 120.0), 10.0, None),
        ((100.0, -100.0, 100.0), 3.0, None),
    ],
)
def test_classify_tells_a_phase_step_from_a_frequency_step(
    innovations: tuple[float, float, float], sigma: float, step_kind: str | None
) -> None:
    """Find a phase step within 3 sigma of the mean, else a line within 3 sigma."""
    assert estimator.classify(reject_buffer(*innovations), sigma).step_kind == step_kind


def test_classify_fits_the_line_of_a_frequency_step() -> None:
    """Give the intercept a and the slope s, per second, of the fitted line."""
    classified = estimator.classify(reject_buffer(30.0, 60.0, 90.0), 3.0)
    assert classified.a == pytest.approx(30.0)
    assert classified.s == pytest.approx(30.0 / T)


def test_classify_gives_no_line_for_a_phase_step() -> None:
    """Leave a and s empty when the innovations agree."""
    classified = estimator.classify(reject_buffer(150.0, 150.0, 150.0), 3.0)
    assert (classified.a, classified.s) == (None, None)


def test_classify_uses_the_epochs_of_the_rejects() -> None:
    """Fit against the epochs as they are, so a gap between rejects counts."""
    reject_buffer = (
        (EPOCH_START, 30.0),
        (EPOCH_START + EPOCH_LENGTH, 60.0),
        (EPOCH_START + 3 * EPOCH_LENGTH, 120.0),
    )
    classified = estimator.classify(reject_buffer, 3.0)
    assert classified.step_kind == "frequency"
    assert classified.s == pytest.approx(30.0 / T)


@pytest.mark.parametrize("reject_count", [0, 2])
def test_classify_needs_three_rejects(reject_count: int) -> None:
    """Raise FilterError for a buffer that does not hold three rejects."""
    with pytest.raises(FilterError, match="three rejects"):
        estimator.classify(reject_buffer(*[10.0] * reject_count), 3.0)


@pytest.mark.parametrize(
    ("innovation", "innovation_scale", "in_gate"),
    [
        (Fraction(15), 3.0, True),
        (Fraction(-15), 3.0, True),
        (Fraction(15) + Fraction(1, 10**9), 3.0, False),
        (Fraction(-15) - Fraction(1, 10**9), 3.0, False),
    ],
)
def test_the_gate_is_five_innovation_scales_wide(
    innovation: Fraction, innovation_scale: float, in_gate: bool
) -> None:
    """Pass an innovation of at most 5 sigma either way, compared exactly (U8)."""
    assert estimator.within_gate(innovation, innovation_scale) is in_gate


@given(
    innovation=st.fractions(max_denominator=10**12),
    innovation_scale=st.floats(min_value=0.0, max_value=1e12),
)
def test_the_gate_compares_as_exact_fractions_do(
    innovation: Fraction, innovation_scale: float
) -> None:
    """Pass exactly what a comparison of exact fractions passes (U8)."""
    assert estimator.within_gate(innovation, innovation_scale) is (
        abs(innovation) <= Fraction(estimator.K_OUT * innovation_scale)
    )


def test_the_gate_holds_exactly_at_its_edge() -> None:
    """Pass an innovation exactly five scales out, refuse one a hair past it."""
    edge = Fraction(estimator.K_OUT * 0.1)
    assert estimator.within_gate(edge, 0.1)
    assert estimator.within_gate(-edge, 0.1)
    assert not estimator.within_gate(edge + Fraction(1, 2**80), 0.1)


def test_the_gate_refuses_a_scale_that_is_not_finite() -> None:
    """Raise FilterError when five scales are not a finite number."""
    with pytest.raises(FilterError, match="not finite"):
        estimator.within_gate(Fraction(1), float("inf"))


@pytest.mark.parametrize(
    ("rms", "rms_max", "passes"), [(80, 80, True), (81, 80, False), (10**6, None, True)]
)
def test_the_gate_refuses_an_rms_over_the_limit(
    rms: int, rms_max: int | None, passes: bool
) -> None:
    """Pass an rms up to the pair's limit, and any rms without one (U8)."""
    assert estimator.rms_ok(rms, rms_max) is passes


def test_a_counted_reject_enters_the_buffer() -> None:
    """Add one to consecutive_rejects and push (E, innovation), keeping three."""
    previous_row = last_row(
        innovation=31.0,
        consecutive_rejects=3,
        rejects=reject_buffer(10.0, 20.0, 31.0),
        epochs_since_accept=3,
        flags="R",
    )
    draft = moved_on(previous_row)
    estimator.count_reject(draft, Fraction(81, 2))
    assert draft.consecutive_rejects == 4
    assert draft.rejects == (
        *reject_buffer(10.0, 20.0, 31.0)[1:],
        (EPOCH_START + EPOCH_LENGTH, 40.5),
    )


def run_epoch(
    previous_row: Row, z: int, series_params: SeriesParams, rms: int = 3
) -> Row:
    """Process one epoch's pair measurement through the filter step."""
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    filter_input = estimator.FilterInput(z=z, rms=rms)
    epoch_start = previous_row.interpolated_datetime + EPOCH_LENGTH
    return estimator.filter_step(
        epoch_start, series_params, previous_row, prediction, filter_input
    ).row


def worked_measurements(epoch_count: int, offset: int = 0) -> list[int]:
    """Give the worked epoch's noise-free measurements, ``offset`` added."""
    rate = exact(0.0123)
    return [
        round_even(1_234_567 + rate * T * k) + offset for k in range(1, epoch_count + 1)
    ]


def run_epochs(
    measurements: list[int], series_params: SeriesParams, previous_row: Row
) -> list[Row]:
    """Process a run of measurements from ``previous_row``, one epoch each."""
    step_rows = []
    for z in measurements:
        previous_row = run_epoch(previous_row, z, series_params)
        step_rows.append(previous_row)
    return step_rows


def test_an_innovation_in_the_gate_is_accepted_and_one_past_it_rejected() -> None:
    """Accept 5 sigma, reject just over it, reject an rms over the limit (U8)."""
    series_params = make_series_params()
    previous_row = last_row(y=0.0)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    gate_edge = prediction.x + 15
    assert gate_edge == 1_234_582
    assert run_epoch(previous_row, int(gate_edge), series_params).flags == "A"
    assert run_epoch(previous_row, int(gate_edge) + 1, series_params).flags == "R"
    assert run_epoch(previous_row, int(gate_edge) - 30, series_params).flags == "A"
    assert run_epoch(previous_row, int(gate_edge) - 31, series_params).flags == "R"
    assert (
        run_epoch(previous_row, int(prediction.x), series_params, rms=81).flags == "R"
    )


@pytest.mark.parametrize("filter_states", [3, 2, 1])
def test_a_phase_step_is_found_on_the_third_reject(
    filter_states: Literal[1, 2, 3],
) -> None:
    """Give R, R, then A with step_offset up by the step, same segment (U9)."""
    M = {3: 100.0, 2: 30.0, 1: None}[filter_states]
    series_params = make_series_params(filter_states=filter_states, M=M)
    if filter_states == 1:
        previous_row = last_row(**ONE_STATE_FIELDS, step_offset=12)
    else:
        previous_row = last_row(
            filter_states=filter_states, time_constant=M, step_offset=12
        )
    measurements = worked_measurements(6) if filter_states > 1 else [1_234_567] * 6
    rows_before = run_epochs(measurements[:3], series_params, previous_row)
    assert [row.flags for row in rows_before] == ["A", "A", "A"]
    step_ps = 150
    rows_after = run_epochs(
        [z + step_ps for z in measurements[3:]], series_params, rows_before[-1]
    )
    assert [row.flags for row in rows_after] == ["R", "R", "A"]
    assert abs(rows_after[-1].step_offset - 12 - step_ps) <= 1
    assert rows_after[-1].segment == rows_before[-1].segment
    assert rows_after[-1].innovation is not None
    assert abs(rows_after[-1].innovation) <= 1
    assert (rows_after[-1].consecutive_rejects, rows_after[-1].rejects) == (0, ())


def test_a_phase_step_rounds_the_mean_half_to_even() -> None:
    """Make the step the mean innovation rounded once, ties to even."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(
        previous_row, rejects=reject_buffer(100.5, 100.5, 100.5), consecutive_rejects=3
    )
    row = estimator.phase_step(draft, prediction, round_even(prediction.x) + 100, 3)
    assert row.step_offset == 100
    assert row.innovation == float(round_even(prediction.x) - prediction.x)


@pytest.mark.parametrize("filter_states", [3, 2])
def test_a_frequency_step_starts_a_new_segment(filter_states: Literal[2, 3]) -> None:
    """Give R, R, then A N U in segment + 1, the prediction on the line (U10)."""
    M = {3: 100.0, 2: 30.0}[filter_states]
    series_params = make_series_params(filter_states=filter_states, M=M)
    previous_row = last_row(
        filter_states=filter_states, time_constant=M, step_offset=12
    )
    measurements = worked_measurements(6)
    rows_before = run_epochs(measurements[:3], series_params, previous_row)
    ramp_measurements = [z + 30 * (i + 1) for i, z in enumerate(measurements[3:])]
    rows_after = run_epochs(ramp_measurements, series_params, rows_before[-1])
    assert [row.flags for row in rows_after] == ["R", "R", "ANU"]
    assert rows_after[-1].segment == rows_before[-1].segment + 1
    assert (rows_after[-1].step_offset, rows_after[-1].epochs_in_segment) == (12, 0)
    assert rows_after[-1].innovation is not None
    assert abs(rows_after[-1].innovation) <= 1
    assert rows_after[-1].y is not None
    assert rows_before[-1].y is not None
    assert rows_after[-1].y == pytest.approx(rows_before[-1].y + 30 / T, abs=1e-3)


def test_a_frequency_step_moves_the_phase_to_the_line_at_the_third_reject() -> None:
    """Add a + s t3, exactly, with t3 the time from the first reject to the third."""
    previous_row = last_row()
    prediction = State(x=Fraction(1_000), y=0.0123)
    draft = moved_on(
        previous_row, rejects=reject_buffer(30.0, 60.0, 90.0), consecutive_rejects=3
    )
    row = estimator.frequency_step(
        draft, prediction, 30.0, 0.05, 1_090, 3, make_series_params()
    )
    line_phase = Fraction(1_000) + exact(30.0) + exact(0.05) * 2 * T
    assert row.innovation == float(1_090 - line_phase)
    assert row.flags == "ANU"


def test_a_one_state_series_takes_no_frequency_step() -> None:
    """Keep rejecting a rate step on a 1-state series: it has no rate (9.4)."""
    series_params = make_series_params(filter_states=1, M=None)
    previous_row = last_row(**ONE_STATE_FIELDS)
    ramp_measurements = [1_234_567 + 30 * (i + 1) for i in range(4)]
    step_rows = run_epochs(ramp_measurements, series_params, previous_row)
    assert [row.flags for row in step_rows] == ["R", "R", "R", "R"]
    draft = moved_on(
        previous_row, rejects=reject_buffer(30.0, 60.0, 90.0), consecutive_rejects=3
    )
    prediction = State(x=Fraction(1_234_567), y=0.0)
    assert estimator.accept_step(draft, prediction, 1_234_657, 3, series_params) is None


def test_no_step_is_looked_for_before_three_rejects() -> None:
    """Give no step while fewer than three consecutive rejects are counted."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(
        previous_row, rejects=reject_buffer(150.0, 150.0), consecutive_rejects=2
    )
    assert (
        estimator.accept_step(draft, prediction, 1_234_724, 3, make_series_params())
        is None
    )


def test_scattered_rejects_show_no_step() -> None:
    """Give no step when the rejects agree on neither a phase nor a line."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(
        previous_row, rejects=reject_buffer(100.0, -100.0, 100.0), consecutive_rejects=3
    )
    assert (
        estimator.accept_step(draft, prediction, 1_234_674, 3, make_series_params())
        is None
    )


def test_a_step_needs_a_series_with_an_innovation_scale() -> None:
    """Raise FilterError for three rejects on a draft that has no scale."""
    draft = moved_on(
        last_row(),
        rejects=reject_buffer(150.0, 150.0, 150.0),
        consecutive_rejects=3,
        innovation_scale=None,
    )
    prediction = State(x=Fraction(0), y=0.0)
    with pytest.raises(FilterError, match="no innovation scale"):
        estimator.accept_step(draft, prediction, 150, 3, make_series_params())


# ---------------------------------------------------------------- acquisition


def dormant_row(*buffered_values: float, **field_changes: object) -> Row:
    """Build a dormant last row buffering ``buffered_values``, newest at EPOCH_START."""
    buffered_count = len(buffered_values)
    buffer_entries = tuple(
        (
            EPOCH_START - EPOCH_LENGTH * (buffered_count - 1 - entry_index),
            buffered_value,
        )
        for entry_index, buffered_value in enumerate(buffered_values)
    )
    row_fields: dict[str, object] = {
        "flags": "RD",
        "rejects": buffer_entries,
        "epochs_since_accept": 7,
        **DORMANT_FIELDS,
        **field_changes,
    }
    return last_row(**row_fields)


def test_a_measurement_of_a_dormant_series_is_buffered() -> None:
    """Add (E, z) to the buffer and write a dormant R row, rejects not counted."""
    previous_row = dormant_row(1_000.0, consecutive_rejects=5)
    row = estimator.acquire(moved_on(previous_row), 2_000, make_series_params())
    assert row.flags == "RD"
    assert row.rejects == ((EPOCH_START, 1_000.0), (NEXT_EPOCH_START, 2_000.0))
    assert row.consecutive_rejects == 0
    assert (row.x_fs, row.innovation_scale) == (None, None)


@pytest.mark.parametrize("filter_states", [3, 2, 1])
def test_three_consistent_measurements_cold_start_the_series(
    filter_states: Literal[1, 2, 3],
) -> None:
    """Cold-start from the third when the second difference passes (13.3)."""
    M = {3: 100.0, 2: 30.0, 1: None}[filter_states]
    series_params = make_series_params(filter_states=filter_states, M=M, sigma0=5.0)
    previous_row = dormant_row(
        1_000.0, 51_000.0, filter_states=filter_states, time_constant=M
    )
    limit_z = round_even(Fraction(101_000) + Fraction(5) * Fraction(math.sqrt(6) * 5))
    row = estimator.acquire(moved_on(previous_row), limit_z, series_params)
    assert row.flags == ("AN" if filter_states == 1 else "ANU")
    assert (row.x_fs, row.segment, row.innovation_scale) == (limit_z * 1000, 5, 5.0)
    assert (row.rejects, row.consecutive_rejects, row.epochs_since_accept) == ((), 0, 0)


def test_a_second_difference_past_the_limit_stays_dormant() -> None:
    """Keep buffering when |z3 - 2 z2 + z1| is over 5 sqrt(6) sigma0."""
    acquisition_limit = 5 * math.sqrt(6) * 5.0
    previous_row = dormant_row(1_000.0, 51_000.0)
    within_row = estimator.acquire(
        moved_on(previous_row),
        101_000 + math.floor(acquisition_limit),
        make_series_params(),
    )
    outside_row = estimator.acquire(
        moved_on(previous_row),
        101_000 + math.ceil(acquisition_limit),
        make_series_params(),
    )
    assert (within_row.flags, outside_row.flags) == ("ANU", "RD")
    assert [buffered_value for _, buffered_value in outside_row.rejects] == [
        1_000.0,
        51_000.0,
        float(101_000 + math.ceil(acquisition_limit)),
    ]


def test_three_measurements_from_epochs_apart_do_not_cold_start() -> None:
    """Need three measurements from consecutive epochs, not just three."""
    buffer_entries = ((EPOCH_START - 3 * EPOCH_LENGTH, 1_000.0), (EPOCH_START, 1_000.0))
    previous_row = last_row(flags="RD", rejects=buffer_entries, **DORMANT_FIELDS)
    row = estimator.acquire(moved_on(previous_row), 1_000, make_series_params())
    assert row.flags == "RD"
    assert len(row.rejects) == 3


def test_the_buffer_keeps_the_newest_three_measurements() -> None:
    """Drop the oldest buffered measurement when a fourth comes."""
    previous_row = dormant_row(1_000.0, -9_000.0, 40_000.0)
    row = estimator.acquire(moved_on(previous_row), 2_000, make_series_params())
    assert row.flags == "RD"
    assert [buffered_value for _, buffered_value in row.rejects] == [
        -9_000.0,
        40_000.0,
        2_000.0,
    ]


def test_a_missing_epoch_empties_the_acquisition_buffer() -> None:
    """Write D P with an empty buffer for a dormant series with no measurement."""
    row = estimator.hold(
        moved_on(dormant_row(1_000.0, 2_000.0)), None, "P", make_series_params()
    )
    assert (row.flags, row.rejects) == ("PD", ())


def test_inconsistent_outliers_make_a_series_dormant_at_n_break() -> None:
    """Go dormant when the rejects reach N_break, then need three consistent (U11)."""
    series_params = make_series_params(n_break=5)
    outliers = [
        1_234_567 + outlier_offset
        for outlier_offset in (900, -700, 1_300, -1_100, 600, -2_000, 800)
    ]
    step_rows = run_epochs(outliers, series_params, last_row())
    flags = [row.flags for row in step_rows]
    assert flags == ["R", "R", "R", "R", "RD", "RD", "RD"]
    assert [row.consecutive_rejects for row in step_rows[:4]] == [1, 2, 3, 4]
    assert step_rows[4].rejects == (
        (step_rows[4].interpolated_datetime, float(outliers[4])),
    )
    steady_measurements = [1_240_000, 1_240_010, 1_240_020]
    rows_after = run_epochs(steady_measurements, series_params, step_rows[-1])
    assert [row.flags for row in rows_after] == ["RD", "RD", "ANU"]
    assert rows_after[-1].segment == step_rows[-1].segment + 1
    assert rows_after[-1].x_fs == 1_240_020_000


def test_a_dormant_series_acquires_across_a_wrap() -> None:
    """Stay dormant through scatter, empty on a gap, cold-start on steady (U25)."""
    series_params = make_series_params(sigma0=5.0)
    previous_row = dormant_row()
    true_phases = [PHASE_MAX - 40_000 + 60_000 * k for k in range(4)]
    scattered_readings = [5_000, 150_000, 60_000, 190_000]
    step_rows: list[Row] = []

    def buffer_reading(phi: int) -> None:
        """Decycle ``phi`` against the anchor and buffer it, as a dormant pair does."""
        nonlocal previous_row
        anchor = estimator.anchor_of(previous_row)
        z = decycle(phi, Fraction(0), Fraction(0), None, anchor).z
        previous_row = estimator.acquire(moved_on(previous_row), z, series_params)
        step_rows.append(previous_row)

    for phi in scattered_readings:
        buffer_reading(phi)
    assert [row.flags for row in step_rows] == ["RD"] * 4
    previous_row = estimator.hold(moved_on(previous_row), None, "P", series_params)
    assert previous_row.rejects == ()
    step_rows.clear()
    for true_phase in true_phases[:3]:
        buffer_reading(true_phase % PHASE_PERIOD)
    assert [row.flags for row in step_rows] == ["RD", "RD", "ANU"]
    assert step_rows[-1].x_fs == true_phases[2] * 1000
    assert true_phases[1] > PHASE_MAX


def test_the_anchor_is_the_last_buffered_measurement() -> None:
    """Give the newest buffered measurement of a dormant series, as a whole ps."""
    assert estimator.anchor_of(dormant_row(1_000.0, 2_500_000_017.0)) == 2_500_000_017


@pytest.mark.parametrize(
    "previous_row",
    [
        None,
        last_row(),
        last_row(flags="PD", **DORMANT_FIELDS),
        last_row(flags="R", rejects=((EPOCH_START, 40.0),), consecutive_rejects=1),
    ],
)
def test_no_anchor_without_a_buffered_measurement(previous_row: Row | None) -> None:
    """Give None for no row, a tracked row, or a dormant row with an empty buffer."""
    assert estimator.anchor_of(previous_row) is None


# ------------------------------------------------------------- filter step


def pair_input(z: int, rms: int = 3, *, slip: bool = False) -> estimator.FilterInput:
    """Give a pair's measurement at an epoch."""
    return estimator.FilterInput(z=z, rms=rms, slip=slip)


def triple_input(
    z: int, sigma: float = 3.5, *, pair_cold_started: bool = False
) -> estimator.FilterInput:
    """Give a triple's measurement at an epoch."""
    return estimator.FilterInput(
        z=z, sigma_dd=sigma, pair_cold_started=pair_cold_started
    )


def filter_step_after(
    previous_row: Row | None,
    filter_input: estimator.FilterInput | None,
    series_params: SeriesParams | None = None,
    *,
    excluded: bool = False,
) -> estimator.StepResult:
    """Run the filter step at the epoch after ``previous_row``."""
    epoch_start = (
        NEXT_EPOCH_START
        if previous_row is None
        else previous_row.interpolated_datetime + EPOCH_LENGTH
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    return estimator.filter_step(
        epoch_start,
        series_params or make_series_params(),
        previous_row,
        prediction,
        filter_input,
        excluded=excluded,
    )


def buffer_ending_at_start(*innovations: float) -> tuple[tuple[datetime, float], ...]:
    """Give a reject buffer of ``innovations``, one per epoch, ending at EPOCH_START."""
    reject_count = len(innovations)
    return tuple(
        (EPOCH_START - EPOCH_LENGTH * (reject_count - 1 - epoch_index), innovation)
        for epoch_index, innovation in enumerate(innovations)
    )


WORKED_Z: Final = 1_234_577
"""The worked epoch's measurement, ps: z_E of Appendix A."""


def test_the_worked_epoch_is_accepted() -> None:
    """Give the accepted row of design 5.4 for the worked epoch, not cold."""
    step_result = filter_step_after(last_row(), pair_input(WORKED_Z))
    assert step_result.cold_started is False
    assert step_result.row.flags == "A"
    assert step_result.row.x_fs == 1_234_574_457


def test_no_measurement_gives_a_predicted_row() -> None:
    """Give P with the prediction when there is no measurement (9.6)."""
    step_result = filter_step_after(last_row(), None)
    assert (step_result.row.flags, step_result.cold_started) == ("P", False)
    assert step_result.row.epochs_since_accept == 1


def test_no_measurement_for_a_dormant_series_gives_a_dormant_row() -> None:
    """Give D P for a new or dormant series with no measurement (9.6)."""
    assert filter_step_after(None, None).row.flags == "PD"
    assert (
        filter_step_after(last_row(flags="PD", **DORMANT_FIELDS), None).row.flags
        == "PD"
    )


def test_a_new_series_buffers_its_first_measurement() -> None:
    """Give D R, the measurement buffered, for a series with no prediction (9.6)."""
    step_result = filter_step_after(None, pair_input(WORKED_Z))
    assert (step_result.row.flags, step_result.cold_started) == ("RD", False)
    assert step_result.row.rejects == ((NEXT_EPOCH_START, float(WORKED_Z)),)
    assert step_result.row.segment == 0


def test_a_consistent_third_measurement_cold_starts() -> None:
    """Give A N U and say the row cold-started, from the acquisition buffer (9.6)."""
    previous_row = last_row(
        flags="RD",
        rejects=(
            (EPOCH_START - EPOCH_LENGTH, float(WORKED_Z)),
            (EPOCH_START, float(WORKED_Z)),
        ),
        **DORMANT_FIELDS,
    )
    step_result = filter_step_after(previous_row, pair_input(WORKED_Z))
    assert (step_result.row.flags, step_result.cold_started) == ("ANU", True)
    assert step_result.row.segment == 5


def test_an_excluded_measurement_within_the_gate_is_held() -> None:
    """Give X, not counted, for an excluded measurement inside the gate (9.5)."""
    step_result = filter_step_after(last_row(), pair_input(WORKED_Z), excluded=True)
    assert (step_result.row.flags, step_result.row.consecutive_rejects) == ("X", 0)
    assert step_result.row.rejects == ()
    assert step_result.row.innovation == float(
        WORKED_Z - Fraction(1_234_567) - exact(0.0123) * T
    )


def test_an_excluded_measurement_outside_the_gate_is_a_counted_reject() -> None:
    """Give R, counted and buffered, for an excluded one outside the gate (9.5)."""
    step_result = filter_step_after(
        last_row(), pair_input(WORKED_Z + 100), excluded=True
    )
    assert (step_result.row.flags, step_result.row.consecutive_rejects) == ("R", 1)
    assert len(step_result.row.rejects) == 1


def test_a_measurement_outside_the_gate_is_a_counted_reject() -> None:
    """Give R and push the innovation for a measurement outside the gate (9.6)."""
    step_result = filter_step_after(last_row(), pair_input(WORKED_Z + 100))
    assert (step_result.row.flags, step_result.row.consecutive_rejects) == ("R", 1)
    assert step_result.row.innovation == step_result.row.rejects[0][1]


def test_an_rms_over_the_limit_is_a_counted_reject() -> None:
    """Give R for a pair whose rms is over its limit, inside the gate (9.1)."""
    step_result = filter_step_after(last_row(), pair_input(WORKED_Z, rms=81))
    assert (step_result.row.flags, step_result.row.consecutive_rejects) == ("R", 1)


def test_a_triple_has_no_rms_test_and_its_floor_is_sigma_dd() -> None:
    """Accept a triple with no rms limit, its scale held up by sigma_dd (12.5)."""
    step_result = filter_step_after(
        last_row(), triple_input(WORKED_Z, sigma=3.5), make_series_params(rms_max=None)
    )
    assert step_result.row.flags == "A"
    assert step_result.row.innovation_scale == 3.5


def test_a_third_agreeing_reject_is_a_phase_step() -> None:
    """Give A, step_offset up, same segment, after two rejects (9.6)."""
    previous_row = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=buffer_ending_at_start(150.0, 150.0),
        innovation=150.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    step_result = filter_step_after(
        previous_row, pair_input(round_even(prediction.x) + 150)
    )
    assert (
        step_result.row.flags,
        step_result.row.step_offset,
        step_result.row.segment,
    ) == (
        "A",
        150,
        4,
    )


def test_a_third_reject_on_a_line_is_a_frequency_step() -> None:
    """Give A N U in segment + 1 after rejects on a line (9.6)."""
    previous_row = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=buffer_ending_at_start(30.0, 60.0),
        innovation=60.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    step_result = filter_step_after(
        previous_row, pair_input(round_even(prediction.x) + 90)
    )
    assert (step_result.row.flags, step_result.row.segment) == ("ANU", 5)


def test_a_third_scattered_reject_is_held() -> None:
    """Give R when three rejects show no step and N_break is not reached (9.6)."""
    previous_row = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=buffer_ending_at_start(100.0, -100.0),
        innovation=-100.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    step_result = filter_step_after(
        previous_row, pair_input(round_even(prediction.x) + 100)
    )
    assert (step_result.row.flags, step_result.row.consecutive_rejects) == ("R", 3)


def test_rejects_reaching_n_break_make_the_series_dormant() -> None:
    """Give D R with only the current measurement buffered at N_break (9.6)."""
    previous_row = last_row(
        flags="R",
        consecutive_rejects=4,
        rejects=buffer_ending_at_start(100.0, -100.0, 300.0),
        innovation=300.0,
        epochs_since_accept=4,
    )
    step_result = filter_step_after(
        previous_row, pair_input(WORKED_Z - 500), make_series_params(n_break=5)
    )
    assert (step_result.row.flags, step_result.cold_started) == ("RD", False)
    assert step_result.row.rejects == ((NEXT_EPOCH_START, float(WORKED_Z - 500)),)
    assert step_result.row.consecutive_rejects == 0


def test_a_component_cold_start_makes_a_triple_dormant() -> None:
    """Give D R and restart acquisition when a component pair cold-started (12.6)."""
    previous_row = last_row(
        rejects=buffer_ending_at_start(90.0),
        consecutive_rejects=1,
        flags="R",
        innovation=90.0,
    )
    step_result = filter_step_after(
        previous_row,
        triple_input(WORKED_Z, pair_cold_started=True),
        make_series_params(rms_max=None),
    )
    assert (step_result.row.flags, step_result.cold_started) == ("RD", False)
    assert step_result.row.rejects == ((NEXT_EPOCH_START, float(WORKED_Z)),)


def test_a_slip_corrected_measurement_carries_s() -> None:
    """Carry S onto the row of a measurement the slip check corrected."""
    assert (
        filter_step_after(last_row(), pair_input(WORKED_Z, slip=True)).row.flags == "AS"
    )


def test_a_configuration_change_warm_starts_before_the_measurement() -> None:
    """Start segment + 1 with the new M, then accept with its gains (8.7, U14)."""
    changed_params = make_series_params(M=150.0, M_sigma=60.0)
    previous_row = last_row()
    step_result = filter_step_after(previous_row, pair_input(WORKED_Z), changed_params)
    row = step_result.row
    assert (row.flags, row.segment, row.epochs_in_segment) == ("ANU", 5, 0)
    assert (row.time_constant, row.scale_time_constant, row.step_offset) == (
        150.0,
        60.0,
        0,
    )
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    expected_state = estimator.update(prediction, WORKED_Z - prediction.x, 3, 150.0)
    assert row.x_fs == to_fs(expected_state.x)
    assert row.y == expected_state.y


def test_a_configuration_change_shares_a_row_with_no_measurement() -> None:
    """Give N U P on a configuration change at an epoch with no measurement (8.7)."""
    assert (
        filter_step_after(last_row(), None, make_series_params(M_sigma=60.0)).row.flags
        == "PNU"
    )


def test_a_dormant_series_takes_new_settings_at_its_cold_start() -> None:
    """Start no warm segment for a dormant series: its cold start takes them."""
    previous_row = last_row(flags="PD", **DORMANT_FIELDS)
    assert (
        filter_step_after(previous_row, None, make_series_params(M=150.0)).row.flags
        == "PD"
    )


def test_unchanged_settings_start_no_segment() -> None:
    """Keep the segment when M and M_sigma are as in the last row."""
    assert filter_step_after(last_row(), pair_input(WORKED_Z)).row.segment == 4


@pytest.mark.parametrize(
    "input_fields",
    [
        {"z": 1, "rms": 3, "sigma_dd": 3.0},
        {"z": 1},
        {"z": 1, "sigma_dd": 3.0, "slip": True},
        {"z": 1, "rms": 3, "pair_cold_started": True},
    ],
)
def test_a_measurement_is_a_pair_s_or_a_triple_s(
    input_fields: dict[str, object],
) -> None:
    """Refuse a measurement that is neither a pair's (rms) nor a triple's (sigma_dd)."""
    with pytest.raises(FilterError):
        estimator.FilterInput(**input_fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filter_input", "scale_floor"),
    [(pair_input(5, rms=4), 4.0), (triple_input(5, sigma=3.25), 3.25)],
)
def test_a_measurement_gives_its_floor(
    filter_input: estimator.FilterInput, scale_floor: float
) -> None:
    """Give the rms of a pair and sigma_dd of a triple as the scale's floor (9.2)."""
    assert filter_input.scale_floor == scale_floor


def test_a_prediction_for_a_series_with_no_scale_is_refused() -> None:
    """Raise FilterError when a dormant last row is given a prediction anyway."""
    previous_row = last_row(flags="PD", **DORMANT_FIELDS)
    prediction = State(x=Fraction(WORKED_Z), y=0.0)
    with pytest.raises(FilterError, match="no scale"):
        estimator.filter_step(
            NEXT_EPOCH_START,
            make_series_params(),
            previous_row,
            prediction,
            pair_input(WORKED_Z),
        )


# ------------------------------------------------- edges and what is logged


def test_classify_fits_the_slope_as_the_design_writes_it() -> None:
    """Give s as sum((t - tbar)(v - vbar)) / sum((t - tbar)^2), to the last bit."""
    innovations = (31.3, 60.7, 90.1)
    classified = estimator.classify(reject_buffer(*innovations), 3.0)
    ts = [0.0, T, 2 * T]
    vbar, tbar = sum(innovations) / 3, sum(ts) / 3
    s = sum(
        (t - tbar) * (v - vbar) for t, v in zip(ts, innovations, strict=True)
    ) / sum((t - tbar) ** 2 for t in ts)
    assert classified.step_kind == "frequency"
    assert classified.s == s
    assert classified.a == vbar - s * tbar


def test_a_phase_step_needs_the_rejects_strictly_within_three_scales() -> None:
    """Give no phase step when the furthest reject is exactly 3 scales from the mean."""
    assert (
        estimator.classify(reject_buffer(100.0, 100.0, 109.0), 2.0).step_kind
        == "frequency"
    )


def test_a_frequency_step_needs_the_rejects_strictly_within_three_scales() -> None:
    """Give no step when the furthest reject is exactly 3 scales from the line."""
    assert estimator.classify(reject_buffer(0.0, 0.0, 18.0), 2.0).step_kind is None


def test_acquisition_takes_a_second_difference_at_its_limit() -> None:
    """Cold-start when the second difference is exactly the acquisition limit."""
    sigma0 = 40 / estimator._ACQUIRE_LIMIT
    assert estimator._ACQUIRE_LIMIT * sigma0 == 40.0
    previous_row = dormant_row(1_000.0, 2_000.0)
    row = estimator.acquire(
        moved_on(previous_row), 3_040, make_series_params(sigma0=sigma0)
    )
    assert row.flags == "ANU"


@pytest.mark.parametrize("filter_states", [2, 3])
def test_a_time_constant_of_one_epoch_has_gains(filter_states: int) -> None:
    """Give the gains for M = 1, the shortest time constant."""
    lam = math.exp(-1.0)
    expected_gains = (
        (1 - lam**2, (1 - lam) ** 2 / T, 0.0)
        if filter_states == 2
        else (1 - lam**3, 1.5 * (1 - lam) ** 2 * (1 + lam) / T, (1 - lam) ** 3 / T**2)
    )
    assert estimator.gains(filter_states, 1.0) == expected_gains


CARRIED_DRIFT: Final = 2.5e-12
"""A drift, ps/s^2, for rows whose d must be carried."""


def test_a_held_row_carries_the_drift() -> None:
    """Store the predicted d in a held row."""
    previous_row = last_row(d=CARRIED_DRIFT)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    assert prediction.d != 0.0
    row = estimator.hold(
        moved_on(previous_row, innovation=None), prediction, "P", make_series_params()
    )
    assert row.d == prediction.d


def test_a_phase_step_keeps_the_drift() -> None:
    """Correct only the phase in a phase step, the drift carried into the update."""
    previous_row = last_row(d=CARRIED_DRIFT)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(
        previous_row, rejects=reject_buffer(100.0, 100.0, 100.0), consecutive_rejects=3
    )
    z = round_even(prediction.x) + 100
    row = estimator.phase_step(draft, prediction, z, 3)
    corrected_prediction = State(x=prediction.x + 100, y=prediction.y, d=prediction.d)
    expected_state = estimator.update(
        corrected_prediction, z - corrected_prediction.x, 3, 100.0
    )
    assert row.d == expected_state.d


def test_a_frequency_step_keeps_the_drift() -> None:
    """Correct phase and rate in a frequency step, the drift carried into the update."""
    previous_row = last_row(d=CARRIED_DRIFT)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    draft = moved_on(
        previous_row, rejects=reject_buffer(30.0, 60.0, 90.0), consecutive_rejects=3
    )
    z = round_even(prediction.x) + 90
    row = estimator.frequency_step(
        draft, prediction, 30.0, 0.05, z, 3, make_series_params()
    )
    corrected_prediction = State(
        x=prediction.x + 30 + exact(0.05) * 2 * T,
        y=prediction.y + 0.05,
        d=prediction.d,
    )
    expected_state = estimator.update(
        corrected_prediction, z - corrected_prediction.x, 3, 100.0
    )
    assert row.d == expected_state.d


def test_a_configuration_change_in_the_filter_step_keeps_the_step_offset() -> None:
    """Carry step_offset into the warm segment a settings change starts (8.7)."""
    previous_row = last_row(step_offset=25)
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    filter_input = estimator.FilterInput(z=round_even(prediction.x), rms=3)
    changed_params = make_series_params(M=150.0, M_sigma=60.0)
    row = estimator.filter_step(
        NEXT_EPOCH_START, changed_params, previous_row, prediction, filter_input
    ).row
    assert (row.segment, row.step_offset) == (5, 25)
    assert "N" in row.flags


def test_an_excluded_measurement_in_the_gate_is_written_x() -> None:
    """Write X alone for a measurement in the gate that screening excluded."""
    previous_row = last_row()
    prediction = estimator.predict(previous_row, NO_STEERING_INPUT)
    assert prediction is not None
    filter_input = estimator.FilterInput(z=round_even(prediction.x), rms=3)
    row = estimator.filter_step(
        NEXT_EPOCH_START,
        make_series_params(),
        previous_row,
        prediction,
        filter_input,
        excluded=True,
    ).row
    assert row.flags == "X"


def test_every_filter_error_is_logged_as_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log each FilterError at ERROR in the words it is raised with."""
    no_scale_draft = moved_on(last_row(), innovation_scale=None)
    prediction = State(x=Fraction(0), y=0.0)
    failing_calls: list[Callable[[], object]] = [
        lambda: estimator.accept(no_scale_draft, prediction, Fraction(0), 3),
        lambda: estimator.accept_step(
            dataclasses.replace(no_scale_draft, consecutive_rejects=3),
            prediction,
            0,
            3,
            make_series_params(),
        ),
        lambda: estimator.classify(reject_buffer(1.0), 3.0),
        lambda: estimator.gains(2, 0.5),
        lambda: estimator._gate(
            no_scale_draft,
            prediction,
            estimator.FilterInput(z=0, rms=3),
            make_series_params(),
            excluded=False,
        ),
        lambda: estimator.start_segment(
            moved_on(last_row()),
            make_series_params(filter_states=2, M=30.0),
            keep_offset=True,
        ),
        lambda: estimator.finish(moved_on(last_row(), x_fs=None), "A"),
        lambda: estimator.FilterInput(z=0),
    ]
    for failing_call in failing_calls:
        caplog.clear()
        with pytest.raises(FilterError) as raised_error:
            failing_call()
        assert [log_record.getMessage() for log_record in caplog.records] == [
            str(raised_error.value)
        ]
        assert str(raised_error.value)
    caplog.clear()
    with pytest.raises(FilterError) as raised_error:
        estimator.gains(2, 0.5)
    assert (
        str(raised_error.value) == "no gains for a 2-state model with time constant 0.5"
    )
