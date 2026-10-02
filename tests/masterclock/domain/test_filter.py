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
from pydantic import ValidationError

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
from masterclock.domain.series import Row, SeriesParams, State

MARK: Final = datetime(2025, 9, 23, 5, 50, tzinfo=UTC)
"""An invented ten-minute mark."""

T: Final = EPOCH_SECONDS
"""One epoch, s."""

NO_INPUT: Final = (Fraction(0), 0.0)
"""A steering input of nothing."""


TABLE: Final = {
    10: (0.904837, 0.259182, 2.58751e-2, 4.30892e-4, 0.181269, 9.05592e-3),
    30: (0.967216, 9.51626e-2, 3.17150e-3, 1.76178e-5, 6.44930e-2, 1.07478e-3),
    100: (0.990050, 2.95545e-2, 2.95540e-4, 4.92562e-7, 1.98013e-2, 9.90058e-5),
    300: (0.996672, 9.95017e-3, 3.31672e-5, 1.84262e-8, 6.64449e-3, 1.10741e-5),
    1000: (0.999000, 2.99550e-3, 2.99550e-6, 4.99251e-10, 1.99800e-3, 9.99001e-7),
}
"""Design 8.4: M to lambda, 3-state g, h, k and 2-state g, h."""


def last_row(**changes: object) -> Row:
    """Build the last row of the worked epoch, with ``changes`` applied."""
    values: dict[str, object] = {
        "interpolated_datetime": MARK,
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
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "flags": "A",
    }
    values.update(changes)
    return Row.model_validate(values)


def significant(value: float, digits: int = 6) -> float:
    """Round ``value`` to ``digits`` significant figures."""
    return float(f"{value:.{digits - 1}e}")


# -------------------------------------------------------------------- gains


@pytest.mark.parametrize("M", sorted(TABLE))
def test_the_gains_reproduce_the_table(M: int) -> None:
    """Give lambda, g, h and k of design 8.4 to six significant figures (U6)."""
    lam, g3, h3, k3, g2, h2 = TABLE[M]
    assert significant(math.exp(-1 / M)) == lam
    g, h_over_t, two_k_over_t2 = estimator.gains(3, float(M))
    assert significant(g) == g3
    assert significant(h_over_t * T) == h3
    assert significant(two_k_over_t2 * T * T / 2) == k3
    g, h_over_t, drift = estimator.gains(2, float(M))
    assert (significant(g), significant(h_over_t * T), drift) == (g2, h2, 0.0)


def closed_loop(model: int, M: float) -> list[list[float]]:
    """Build (I - K H) Phi for the estimator's gains."""
    g, h_over_t, two_k_over_t2 = estimator.gains(model, M)
    if model == 2:
        gain = [g, h_over_t]
        transition = [[1.0, T], [0.0, 1.0]]
    else:
        gain = [g, h_over_t, two_k_over_t2]
        transition = [[1.0, T, T * T / 2], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
    size = len(gain)
    return [
        [
            sum(
                ((1.0 if i == m else 0.0) - (gain[i] if m == 0 else 0.0))
                * transition[m][j]
                for m in range(size)
            )
            for j in range(size)
        ]
        for i in range(size)
    ]


@pytest.mark.parametrize("M", sorted(TABLE))
def test_every_pole_of_three_states_is_lambda(M: int) -> None:
    """Make the characteristic polynomial (z - lambda)**3 within 1e-4 (U6).

    Its coefficients are the trace, the sum of the principal 2 by 2 minors
    and the determinant, so no eigenvalue routine is needed.
    """
    a = closed_loop(3, float(M))
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


@pytest.mark.parametrize("M", sorted(TABLE))
def test_every_pole_of_two_states_is_lambda(M: int) -> None:
    """Make the characteristic polynomial (z - lambda)**2 within 1e-4 (U6)."""
    a = closed_loop(2, float(M))
    lam = math.exp(-1 / M)
    assert a[0][0] + a[1][1] == pytest.approx(2 * lam, abs=1e-4)
    assert a[0][0] * a[1][1] - a[0][1] * a[1][0] == pytest.approx(lam**2, abs=1e-4)


def test_one_state_passes_the_measurement_through() -> None:
    """Give a gain of one on the phase and none on rate or drift."""
    assert estimator.gains(1, None) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("model", "M"),
    [(1, 10.0), (2, None), (3, None), (4, 10.0), (0, None), (2, 0.5)],
)
def test_gains_refuse_a_model_and_time_constant_that_do_not_belong(
    model: int, M: float | None
) -> None:
    """Refuse a time constant on 1 state, none on 2 or 3, or another model."""
    with pytest.raises(FilterError):
        estimator.gains(model, M)


# ---------------------------------------------------------------- predict


def test_the_worked_epoch_predicts_its_state() -> None:
    """Move x on by y T exactly, keep the rate, for the worked epoch."""
    prediction = estimator.predict(last_row(), NO_INPUT)
    assert prediction == State(x=1_234_567 + exact(0.0123) * 600, y=0.0123, d=0.0)


def test_three_states_move_on_with_rate_and_drift() -> None:
    """Add y T + d T**2 / 2 to x and d T to y, and keep d."""
    last = last_row(y=0.25, d=1e-6)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    assert prediction.x == 1_234_567 + exact(0.25) * T + exact(1e-6) * T * T / 2
    assert prediction.y == 0.25 + 1e-6 * T
    assert prediction.d == 1e-6


def test_two_states_move_on_with_rate_only() -> None:
    """Add y T to x and keep y; a 2-state row has no drift."""
    last = last_row(filter_states=2, time_constant=30.0, y=0.25)
    assert estimator.predict(last, NO_INPUT) == State(
        x=Fraction(1_234_567 + 150), y=0.25
    )


def test_one_state_carries_the_phase() -> None:
    """Keep x, and no rate or drift, for a 1-state row."""
    last = last_row(filter_states=1, time_constant=None, y=0.0)
    assert estimator.predict(last, NO_INPUT) == State(x=Fraction(1_234_567), y=0.0)


@pytest.mark.parametrize(
    ("changes", "expected"),
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
    changes: dict[str, object], expected: State
) -> None:
    """Add u_x to the phase, and u_y to the rate where there is one."""
    input_ = (Fraction(7, 2), 0.5)
    assert estimator.predict(last_row(**changes), input_) == expected


def test_a_dormant_or_missing_row_gives_no_prediction() -> None:
    """Give None without a last row, or with a dormant one."""
    dormant = last_row(
        flags="PD", innovation=None, x_fs=None, y=None, d=None, innovation_scale=None
    )
    assert estimator.predict(None, NO_INPUT) is None
    assert estimator.predict(dormant, NO_INPUT) is None


# ----------------------------------------------------------------- update


def test_the_worked_epoch_updates_to_its_state() -> None:
    """Reproduce Appendix A: x stored as 1 234 574.457 ps, y and d exactly."""
    prediction = estimator.predict(last_row(), NO_INPUT)
    assert prediction is not None
    innovation = 1_234_577 - prediction.x
    assert float(innovation) == 2.62
    state = estimator.update(prediction, innovation, 3, 100.0)
    assert to_fs(state.x) == 1_234_574_457
    assert state.y == 0.01230129052352643
    assert state.d == 7.169515400974333e-12


def test_the_update_is_exact_in_phase() -> None:
    """Add g times the innovation to x as an exact Fraction."""
    prediction = State(x=Fraction(2**60) + Fraction(1, 3), y=0.0)
    g, _, _ = estimator.gains(2, 30.0)
    state = estimator.update(prediction, Fraction(10), 2, 30.0)
    assert state.x == Fraction(2**60) + Fraction(1, 3) + exact(g) * 10
    assert state.d == 0.0


def test_one_state_takes_the_measurement_exactly() -> None:
    """Make x the measurement for a 1-state series, with no rate."""
    prediction = State(x=Fraction(2**62 + 5), y=0.0)
    state = estimator.update(prediction, Fraction(-7), 1, None)
    assert state == State(x=Fraction(2**62 - 2), y=0.0)


# ----------------------------------------------------- following a signal


def follow_rows(model: int, M: float, truth: list[Fraction]) -> list[Fraction]:
    """Run the estimator over noise-free measurements, as the rows would.

    The measurements are the truth rounded to whole picoseconds. The series
    cold-starts from the first, then each epoch predicts from the last
    stored row, takes the innovation, updates and stores x in whole
    femtoseconds, as a row stores it.
    """
    measurements = [round_even(value) for value in truth]
    last = last_row(
        x_fs=to_fs(measurements[0]),
        y=0.0,
        d=0.0,
        filter_states=model,
        time_constant=M,
        epochs_in_segment=0,
    )
    innovations: list[Fraction] = []
    for epoch, z in enumerate(measurements[1:], start=1):
        prediction = estimator.predict(last, NO_INPUT)
        assert prediction is not None
        innovation = z - prediction.x
        innovations.append(innovation)
        state = estimator.update(prediction, innovation, model, M)
        last = last_row(
            interpolated_datetime=MARK + timedelta(minutes=10 * epoch),
            x_fs=to_fs(state.x),
            y=state.y,
            d=state.d,
            filter_states=model,
            time_constant=M,
        )
    return innovations


def follow_exactly(M: float, truth: list[Fraction]) -> list[Fraction]:
    """Run the 3-state loop over exact measurements, with x never rounded.

    The transition is written out here, as Phi of design 8.2, so the loop
    can carry x as an exact Fraction from epoch to epoch.
    """
    state = State(x=truth[0], y=0.0, d=0.0)
    innovations: list[Fraction] = []
    for z in truth[1:]:
        prediction = State(
            x=state.x + exact(state.y) * T + exact(state.d) * T * T / 2,
            y=state.y + state.d * T,
            d=state.d,
        )
        innovation = z - prediction.x
        innovations.append(innovation)
        state = estimator.update(prediction, innovation, 3, M)
    return innovations


def parabola(M: float) -> list[Fraction]:
    """Give a noise-free phase with rate and drift over 40 M epochs."""
    rate, drift = Fraction(1, 20), Fraction(1, 10**7)
    return [1_000 + rate * T * k + drift * (T * k) ** 2 / 2 for k in range(int(40 * M))]


@pytest.mark.parametrize("M", [10.0, 30.0])
def test_a_ramp_is_followed_within_a_picosecond(M: float) -> None:
    """Bring a 2-state innovation within 1 ps of a ramp after 20 M epochs (U7)."""
    rate = Fraction(1, 20)
    truth = [1_000 + rate * T * k for k in range(int(40 * M))]
    innovations = follow_rows(2, M, truth)
    assert max(abs(v) for v in innovations[int(20 * M) :]) <= 1


@pytest.mark.parametrize("M", [10.0, 30.0])
def test_a_parabola_is_followed_within_a_picosecond(M: float) -> None:
    """Bring a 3-state innovation within 1 ps of a parabola after 20 M epochs (U7).

    With exact values, so the filter's own dynamics are what is tested.
    """
    innovations = follow_exactly(M, parabola(M))
    assert max(abs(v) for v in innovations[int(20 * M) :]) <= 1


@pytest.mark.parametrize("M", [10.0, 30.0, 100.0])
def test_a_parabola_stored_in_femtoseconds_is_followed_within_a_picosecond(
    M: float,
) -> None:
    """Bring a 3-state innovation within 1 ps of a parabola, as rows store it (U7).

    Each row stores x in whole femtoseconds. Stored in whole picoseconds,
    the rounding would come back every epoch and the 3-state loop would take
    it into its rate and drift, leaving innovations of 2 ps and more here.
    """
    innovations = follow_rows(3, M, parabola(M))
    assert max(abs(v) for v in innovations[int(20 * M) :]) <= 1


# ------------------------------------------------------------ row lifecycle


def params(**changes: object) -> SeriesParams:
    """Build the worked epoch's settings, with ``changes`` applied."""
    values: dict[str, object] = {
        "model": 3,
        "M": 100.0,
        "M_sigma": 50.0,
        "sigma0": 5.0,
        "gmax": 432,
        "n_break": 36,
        "rms_max": 80,
    }
    values.update(changes)
    return SeriesParams.model_validate(values)


ONE_STATE: Final[dict[str, object]] = {
    "y": 0.0,
    "d": 0.0,
    "filter_states": 1,
    "time_constant": None,
}
"""The changes that make a last row one of a 1-state series."""

DORMANT: Final[dict[str, object]] = {
    "innovation": None,
    "x_fs": None,
    "y": None,
    "d": None,
    "innovation_scale": None,
}
"""The changes that make a last row a dormant one, apart from its flags."""

NEXT: Final = MARK + timedelta(minutes=10)
"""The epoch after the worked epoch's last row: the worked epoch itself."""


def moved_on(last: Row, **changes: object) -> estimator.RowDraft:
    """Carry ``last`` on to the next epoch, with ``changes`` made to the draft."""
    draft = estimator.carry(
        last.interpolated_datetime + timedelta(minutes=10), last, params()
    )
    return dataclasses.replace(draft, **changes)  # type: ignore[arg-type]


def test_a_draft_has_the_fields_of_a_row() -> None:
    """Give RowDraft exactly Row's fields, in the same order, so finish fits."""
    names = [field.name for field in dataclasses.fields(estimator.RowDraft)]
    assert names == list(Row.model_fields)


def test_a_new_series_starts_dormant_in_segment_zero() -> None:
    """Start a series with no last row with no state and every counter at 0."""
    draft = estimator.carry(NEXT, None, params(model=2, M=30.0, M_sigma=40.0))
    assert draft == estimator.RowDraft(
        interpolated_datetime=NEXT,
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
    last = last_row(
        innovation=-4.5,
        consecutive_rejects=2,
        rejects=((MARK - timedelta(minutes=10), 40.0), (MARK, -4.5)),
        epochs_since_accept=2,
        flags="R",
    )
    draft = estimator.carry(NEXT, last, params(M=150.0))
    expected = {
        **dict(last),
        "interpolated_datetime": NEXT,
        "innovation": None,
        "epochs_in_segment": 812,
        "flags": "",
    }
    assert dataclasses.asdict(draft) == expected


def test_a_slip_correction_marks_the_row() -> None:
    """Carry S into the row when the slip check corrected its measurement."""
    draft = estimator.carry(NEXT, last_row(), params(), slip=True)
    assert draft.flags == "S"
    prediction = estimator.predict(last_row(), NO_INPUT)
    assert prediction is not None
    row = estimator.accept(draft, prediction, 1_234_577 - prediction.x, 3)
    assert row.flags == "AS"


def test_the_worked_epoch_gives_the_rows_of_the_measurement_file() -> None:
    """Reproduce the two example rows of design 5.4: accepted, then predicted."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    accepted = estimator.accept(
        estimator.carry(NEXT, last, params()), prediction, 1_234_577 - prediction.x, 3
    )
    assert accepted == last_row(
        interpolated_datetime=NEXT,
        innovation=float(1_234_577 - prediction.x),
        x_fs=1_234_574_457,
        y=0.01230129052352643,
        d=7.169515400974333e-12,
        epochs_in_segment=812,
    )
    following = estimator.predict(accepted, NO_INPUT)
    assert following is not None
    later = NEXT + timedelta(minutes=10)
    predicted = estimator.hold(
        estimator.carry(later, accepted, params()), following, "P", params()
    )
    assert predicted == last_row(
        interpolated_datetime=later,
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
    last = last_row(
        innovation=-4.5,
        consecutive_rejects=2,
        rejects=((MARK - timedelta(minutes=10), 40.0), (MARK, -4.5)),
        epochs_since_accept=2,
        flags="R",
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    row = estimator.accept(moved_on(last), prediction, Fraction(1), 3)
    assert (row.consecutive_rejects, row.rejects, row.epochs_since_accept) == (0, (), 0)
    assert row.flags == "A"


@pytest.mark.parametrize(
    ("scale", "innovation", "floor", "expected"),
    [
        (4.0, 10, 3, math.sqrt(0.98 * 16 + 0.02 * 100)),
        (4.0, -10, 3, math.sqrt(0.98 * 16 + 0.02 * 100)),
        (4.0, 0, 3, math.sqrt(0.98 * 16)),
        (3.0, 1, 3, 3.0),
        (4.0, 0, 3.99, math.sqrt(3.99**2)),
    ],
)
def test_an_accept_moves_the_innovation_scale_but_not_below_its_floor(
    scale: float, innovation: int, floor: float, expected: float
) -> None:
    """Average the squared innovation in with w = 1/M_sigma, kept above the floor."""
    last = last_row(innovation_scale=scale)
    prediction = State(x=Fraction(1_234_574), y=0.0123)
    row = estimator.accept(moved_on(last), prediction, Fraction(innovation), floor)
    assert row.innovation_scale == expected


def test_an_accept_takes_the_innovation_before_the_update() -> None:
    """Write the innovation given, and move the scale by it, not the residual."""
    last = last_row(innovation_scale=4.0)
    prediction = State(x=Fraction(1_000), y=0.0)
    row = estimator.accept(moved_on(last), prediction, Fraction(21, 2), 1)
    assert row.innovation == 10.5
    assert row.innovation_scale == math.sqrt(0.98 * 16 + 0.02 * 10.5**2)


def test_an_accept_needs_a_series_with_an_innovation_scale() -> None:
    """Raise FilterError for an accept on a draft that has no scale."""
    draft = moved_on(last_row(), innovation_scale=None)
    with pytest.raises(FilterError, match="no innovation scale"):
        estimator.accept(draft, State(x=Fraction(0), y=0.0), Fraction(0), 3)


@pytest.mark.parametrize("flag", ["P", "X", "R"])
def test_a_held_row_carries_the_prediction(flag: Literal["P", "X", "R"]) -> None:
    """Store the prediction, keep the scale and counters, one more epoch unaccepted."""
    last = last_row(
        innovation=-4.5,
        consecutive_rejects=1,
        rejects=((MARK, -40.0),),
        epochs_since_accept=1,
        flags="R",
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    innovation = None if flag == "P" else 7.25
    row = estimator.hold(
        moved_on(last, innovation=innovation), prediction, flag, params()
    )
    assert row == last_row(
        interpolated_datetime=NEXT,
        innovation=innovation,
        x_fs=to_fs(prediction.x),
        y=prediction.y,
        d=prediction.d,
        epochs_in_segment=812,
        epochs_since_accept=2,
        consecutive_rejects=1,
        rejects=((MARK, -40.0),),
        flags=flag,
    )


def test_an_excluded_measurement_within_the_gate_leaves_the_counts() -> None:
    """Give X with consecutive_rejects and the buffer unchanged (U12)."""
    rejects = ((MARK - timedelta(minutes=10), 30.0), (MARK, 31.0))
    last = last_row(
        innovation=31.0, consecutive_rejects=2, rejects=rejects, epochs_since_accept=2
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    row = estimator.hold(moved_on(last, innovation=2.0), prediction, "X", params())
    assert (row.flags, row.consecutive_rejects, row.rejects) == ("X", 2, rejects)
    assert row.epochs_since_accept == 3
    assert row.x_fs == to_fs(prediction.x)


def run_gap(epochs: int, gmax: int) -> Row:
    """Hold a tracked series through ``epochs`` epochs with no measurement."""
    settings = params(gmax=gmax, n_break=3)
    last = last_row()
    for _ in range(epochs):
        prediction = estimator.predict(last, NO_INPUT)
        draft = estimator.carry(
            last.interpolated_datetime + timedelta(minutes=10), last, settings
        )
        last = estimator.hold(draft, prediction, "P", settings)
    return last


@pytest.mark.parametrize("gmax", [3, 6])
def test_a_gap_of_the_gap_limit_is_still_predicted(gmax: int) -> None:
    """Give P rows for G_max epochs, then accept the next measurement (U13)."""
    last = run_gap(gmax, gmax)
    assert (last.flags, last.epochs_since_accept) == ("P", gmax)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    row = estimator.accept(
        moved_on(last), prediction, round_even(prediction.x) - prediction.x, 3
    )
    assert (row.flags, row.epochs_since_accept, row.segment) == ("A", 0, 4)


@pytest.mark.parametrize("gmax", [3, 6])
def test_a_gap_past_the_gap_limit_goes_dormant(gmax: int) -> None:
    """Give a D row at G_max + 1 epochs, with no state and no prediction (U13)."""
    last = run_gap(gmax + 1, gmax)
    assert (last.flags, last.epochs_since_accept) == ("PD", gmax + 1)
    assert (last.x_fs, last.y, last.d, last.innovation_scale) == (None,) * 4
    assert (last.segment, last.step_offset) == (4, 0)
    assert estimator.predict(last, NO_INPUT) is None
    later = run_gap(gmax + 3, gmax)
    assert (later.flags, later.epochs_since_accept) == ("PD", gmax + 3)


def test_a_held_row_without_a_prediction_is_dormant() -> None:
    """Make a held row dormant when the series has no prediction."""
    row = estimator.hold(moved_on(last_row()), None, "R", params())
    assert row.flags == "RD"
    assert row.x_fs is None


@pytest.mark.parametrize("flag", ["P", "X", "R"])
def test_a_dormant_row_keeps_its_segment_and_offset(
    flag: Literal["P", "X", "R"],
) -> None:
    """Empty the state and the buffer, keep segment and step_offset (13.4)."""
    last = last_row(step_offset=-37, rejects=((MARK, 12.0),), consecutive_rejects=1)
    row = estimator.dormant(moved_on(last), flag)
    assert row.flags == f"{flag}D"
    assert (row.x_fs, row.y, row.d, row.innovation_scale) == (None,) * 4
    assert (row.segment, row.step_offset, row.rejects) == (4, -37, ())


def test_a_dormant_row_can_keep_its_acquisition_buffer() -> None:
    """Keep the buffer when asked: it holds the measurements to acquire from."""
    buffered = ((MARK, 1_234_000.0),)
    last = last_row(flags="RD", rejects=buffered, segment=0, **DORMANT)
    row = estimator.dormant(moved_on(last), "R", keep_buffer=True)
    assert (row.flags, row.rejects) == ("RD", buffered)


@pytest.mark.parametrize(
    ("model", "M", "flags"), [(3, 100.0, "ANU"), (2, 30.0, "ANU"), (1, None, "AN")]
)
def test_a_cold_start_begins_a_segment_from_the_measurement(
    model: Literal[1, 2, 3], M: float | None, flags: str
) -> None:
    """Start segment + 1 at [z, 0, 0] with sigma0 and step_offset 0 (8.6, 13.4)."""
    settings = params(model=model, M=M, M_sigma=40.0, sigma0=6.5)
    last = last_row(
        filter_states=model,
        time_constant=M,
        flags="RD",
        segment=2,
        step_offset=15,
        consecutive_rejects=4,
        rejects=((MARK, 1_234_570.0),),
        epochs_since_accept=9,
        **DORMANT,
    )
    draft = estimator.carry(NEXT, last, settings)
    row = estimator.cold_start(draft, 1_234_577, settings)
    assert row == Row(
        interpolated_datetime=NEXT,
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
        filter_states=model,
        time_constant=M,
        scale_time_constant=40.0,
        flags=flags,
    )


def test_a_configuration_change_starts_a_warm_segment() -> None:
    """Carry X- and step_offset into segment + 1 with the new M: N U + outcome (8.7)."""
    last = last_row(step_offset=25)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    changed = params(M=150.0, M_sigma=60.0)
    draft = estimator.start_segment(moved_on(last), changed, keep_offset=True)
    row = estimator.hold(draft, prediction, "P", changed)
    assert (row.flags, row.segment, row.epochs_in_segment) == ("PNU", 5, 0)
    assert (row.step_offset, row.time_constant, row.scale_time_constant) == (
        25,
        150.0,
        60.0,
    )
    assert (row.x_fs, row.innovation_scale) == (to_fs(prediction.x), 3.0)


def test_a_frequency_step_starts_a_warm_segment_and_accepts() -> None:
    """Give A N U, segment + 1, step_offset carried, then the update (13.4)."""
    last = last_row(step_offset=25)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = estimator.start_segment(moved_on(last), params(), keep_offset=True)
    row = estimator.accept(draft, prediction, Fraction(2), 3)
    assert (row.flags, row.segment, row.step_offset) == ("ANU", 5, 25)
    assert row.x_fs == to_fs(estimator.update(prediction, Fraction(2), 3, 100.0).x)


def test_a_phase_step_keeps_the_segment() -> None:
    """Accept within the same segment, with the step offset the step gave (13.4)."""
    last = last_row(step_offset=25)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    row = estimator.accept(moved_on(last, step_offset=525), prediction, Fraction(2), 3)
    assert (row.flags, row.segment, row.step_offset) == ("A", 4, 525)


def test_a_segment_keeps_its_model() -> None:
    """Raise FilterError for a segment whose settings name another model."""
    with pytest.raises(FilterError, match="model"):
        estimator.start_segment(
            moved_on(last_row()), params(model=2, M=30.0), keep_offset=True
        )


@pytest.mark.parametrize(
    ("M", "epochs", "flags"),
    [(100.0, 499, "AU"), (100.0, 500, "A"), (30.0, 149, "AU"), (30.0, 150, "A")],
)
def test_a_row_is_unsettled_until_five_time_constants(
    M: float, epochs: int, flags: str
) -> None:
    """Carry U while epochs_in_segment < 5 M, and not from 5 M on (8.8)."""
    last = last_row(time_constant=M)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, epochs_in_segment=epochs)
    assert estimator.accept(draft, prediction, Fraction(0), 3).flags == flags


def test_a_dormant_row_is_never_unsettled() -> None:
    """Give D without U, however young the segment."""
    row = estimator.dormant(moved_on(last_row(), epochs_in_segment=0), "P")
    assert row.flags == "PD"


def test_flags_are_written_in_their_order() -> None:
    """Order slip, new segment and outcome as ARXPDSNU, whatever was added first."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = estimator.carry(NEXT, last, params(), slip=True)
    draft = estimator.start_segment(draft, params(), keep_offset=True)
    assert draft.flags == "SN"
    assert estimator.accept(draft, prediction, Fraction(0), 3).flags == "ASNU"


def test_finish_refuses_a_row_that_breaks_a_rule() -> None:
    """Raise FilterError when the finished draft is not a valid row."""
    draft = moved_on(last_row(), x_fs=None)
    with pytest.raises(FilterError, match="invalid row"):
        estimator.finish(draft, "A")


def test_a_one_state_series_passes_its_measurements_through() -> None:
    """Make x = z exactly on every accepted row, y and d 0.0, never U (U24)."""
    settings = params(model=1, M=None)
    last = last_row(epochs_in_segment=0, **ONE_STATE)
    measurements = [2**60 + 7, 2**60 + 7, None, 2**60 - 200_001, 5]
    for z in measurements:
        prediction = estimator.predict(last, NO_INPUT)
        assert prediction is not None
        draft = estimator.carry(
            last.interpolated_datetime + timedelta(minutes=10), last, settings
        )
        if z is None:
            row = estimator.hold(draft, prediction, "P", settings)
            assert row.x_fs == last.x_fs
        else:
            row = estimator.accept(draft, prediction, z - prediction.x, 3)
            assert row.x_fs == z * 1000
        assert (row.y, row.d) == (0.0, 0.0)
        assert "U" not in row.flags
        last = row


# ------------------------------------------------------- step classification

EPOCH: Final = timedelta(minutes=10)
"""One epoch."""


def buffer(*values: float) -> tuple[tuple[datetime, float], ...]:
    """Give a reject buffer of ``values`` at consecutive epochs from MARK."""
    return tuple((MARK + EPOCH * i, value) for i, value in enumerate(values))


@pytest.mark.parametrize(
    ("values", "sigma", "kind"),
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
    values: tuple[float, float, float], sigma: float, kind: str | None
) -> None:
    """Find a phase step within 3 sigma of the mean, else a line within 3 sigma."""
    assert estimator.classify(buffer(*values), sigma).kind == kind


def test_classify_fits_the_line_of_a_frequency_step() -> None:
    """Give the intercept a and the slope s, per second, of the fitted line."""
    classified = estimator.classify(buffer(30.0, 60.0, 90.0), 3.0)
    assert classified.a == pytest.approx(30.0)
    assert classified.s == pytest.approx(30.0 / T)


def test_classify_gives_no_line_for_a_phase_step() -> None:
    """Leave a and s empty when the innovations agree."""
    classified = estimator.classify(buffer(150.0, 150.0, 150.0), 3.0)
    assert (classified.a, classified.s) == (None, None)


def test_classify_uses_the_epochs_of_the_rejects() -> None:
    """Fit against the epochs as they are, so a gap between rejects counts."""
    rejects = ((MARK, 30.0), (MARK + EPOCH, 60.0), (MARK + 3 * EPOCH, 120.0))
    classified = estimator.classify(rejects, 3.0)
    assert classified.kind == "frequency"
    assert classified.s == pytest.approx(30.0 / T)


@pytest.mark.parametrize("count", [0, 2])
def test_classify_needs_three_rejects(count: int) -> None:
    """Raise FilterError for a buffer that does not hold three rejects."""
    with pytest.raises(FilterError, match="three rejects"):
        estimator.classify(buffer(*[10.0] * count), 3.0)


@pytest.mark.parametrize(
    ("innovation", "scale", "within"),
    [
        (Fraction(15), 3.0, True),
        (Fraction(-15), 3.0, True),
        (Fraction(15) + Fraction(1, 10**9), 3.0, False),
        (Fraction(-15) - Fraction(1, 10**9), 3.0, False),
    ],
)
def test_the_gate_is_five_innovation_scales_wide(
    innovation: Fraction, scale: float, within: bool
) -> None:
    """Pass an innovation of at most 5 sigma either way, compared exactly (U8)."""
    assert estimator.within_gate(innovation, scale) is within


@pytest.mark.parametrize(
    ("rms", "rms_max", "ok"), [(80, 80, True), (81, 80, False), (10**6, None, True)]
)
def test_the_gate_refuses_an_rms_over_the_limit(
    rms: int, rms_max: int | None, ok: bool
) -> None:
    """Pass an rms up to the pair's limit, and any rms without one (U8)."""
    assert estimator.rms_ok(rms, rms_max) is ok


def test_a_counted_reject_enters_the_buffer() -> None:
    """Add one to consecutive_rejects and push (E, innovation), keeping three."""
    last = last_row(
        innovation=31.0,
        consecutive_rejects=3,
        rejects=buffer(10.0, 20.0, 31.0),
        epochs_since_accept=3,
        flags="R",
    )
    draft = estimator.count_reject(moved_on(last), Fraction(81, 2))
    assert draft.consecutive_rejects == 4
    assert draft.rejects == (*buffer(10.0, 20.0, 31.0)[1:], (MARK + EPOCH, 40.5))


def run_epoch(last: Row, z: int, settings: SeriesParams, rms: int = 3) -> Row:
    """Process one epoch's pair measurement through the filter step."""
    prediction = estimator.predict(last, NO_INPUT)
    measured = estimator.Measured(z=z, rms=rms)
    mark = last.interpolated_datetime + EPOCH
    return estimator.filter_step(mark, settings, last, prediction, measured).row


def truth(epochs: int, offset: int = 0) -> list[int]:
    """Give the worked epoch's noise-free measurements, ``offset`` added."""
    rate = exact(0.0123)
    return [round_even(1_234_567 + rate * T * k) + offset for k in range(1, epochs + 1)]


def run(measurements: list[int], settings: SeriesParams, last: Row) -> list[Row]:
    """Process a run of measurements from ``last``, one epoch each."""
    rows = []
    for z in measurements:
        last = run_epoch(last, z, settings)
        rows.append(last)
    return rows


def test_an_innovation_in_the_gate_is_accepted_and_one_past_it_rejected() -> None:
    """Accept 5 sigma, reject just over it, reject an rms over the limit (U8)."""
    settings = params()
    last = last_row(y=0.0)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    edge = prediction.x + 15
    assert edge == 1_234_582
    assert run_epoch(last, int(edge), settings).flags == "A"
    assert run_epoch(last, int(edge) + 1, settings).flags == "R"
    assert run_epoch(last, int(edge) - 30, settings).flags == "A"
    assert run_epoch(last, int(edge) - 31, settings).flags == "R"
    assert run_epoch(last, int(prediction.x), settings, rms=81).flags == "R"


@pytest.mark.parametrize("model", [3, 2, 1])
def test_a_phase_step_is_found_on_the_third_reject(model: Literal[1, 2, 3]) -> None:
    """Give R, R, then A with step_offset up by the step, same segment (U9)."""
    M = {3: 100.0, 2: 30.0, 1: None}[model]
    settings = params(model=model, M=M)
    if model == 1:
        last = last_row(**ONE_STATE, step_offset=12)
    else:
        last = last_row(filter_states=model, time_constant=M, step_offset=12)
    measurements = truth(6) if model > 1 else [1_234_567] * 6
    before = run(measurements[:3], settings, last)
    assert [row.flags for row in before] == ["A", "A", "A"]
    step = 150
    after = run([z + step for z in measurements[3:]], settings, before[-1])
    assert [row.flags for row in after] == ["R", "R", "A"]
    assert abs(after[-1].step_offset - 12 - step) <= 1
    assert after[-1].segment == before[-1].segment
    assert after[-1].innovation is not None
    assert abs(after[-1].innovation) <= 1
    assert (after[-1].consecutive_rejects, after[-1].rejects) == (0, ())


def test_a_phase_step_rounds_the_mean_half_to_even() -> None:
    """Make the step the mean innovation rounded once, ties to even."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, rejects=buffer(100.5, 100.5, 100.5), consecutive_rejects=3)
    row = estimator.phase_step(draft, prediction, round_even(prediction.x) + 100, 3)
    assert row.step_offset == 100
    assert row.innovation == float(round_even(prediction.x) - prediction.x)


@pytest.mark.parametrize("model", [3, 2])
def test_a_frequency_step_starts_a_new_segment(model: Literal[2, 3]) -> None:
    """Give R, R, then A N U in segment + 1, the prediction on the line (U10)."""
    M = {3: 100.0, 2: 30.0}[model]
    settings = params(model=model, M=M)
    last = last_row(filter_states=model, time_constant=M, step_offset=12)
    measurements = truth(6)
    before = run(measurements[:3], settings, last)
    ramp = [z + 30 * (i + 1) for i, z in enumerate(measurements[3:])]
    after = run(ramp, settings, before[-1])
    assert [row.flags for row in after] == ["R", "R", "ANU"]
    assert after[-1].segment == before[-1].segment + 1
    assert (after[-1].step_offset, after[-1].epochs_in_segment) == (12, 0)
    assert after[-1].innovation is not None
    assert abs(after[-1].innovation) <= 1
    assert after[-1].y is not None
    assert before[-1].y is not None
    assert after[-1].y == pytest.approx(before[-1].y + 30 / T, abs=1e-3)


def test_a_frequency_step_moves_the_phase_to_the_line_at_the_third_reject() -> None:
    """Add a + s t3, exactly, with t3 the time from the first reject to the third."""
    last = last_row()
    prediction = State(x=Fraction(1_000), y=0.0123)
    draft = moved_on(last, rejects=buffer(30.0, 60.0, 90.0), consecutive_rejects=3)
    row = estimator.frequency_step(draft, prediction, 30.0, 0.05, 1_090, 3, params())
    moved = Fraction(1_000) + exact(30.0) + exact(0.05) * 2 * T
    assert row.innovation == float(1_090 - moved)
    assert row.flags == "ANU"


def test_a_one_state_series_takes_no_frequency_step() -> None:
    """Keep rejecting a rate step on a 1-state series: it has no rate (9.4)."""
    settings = params(model=1, M=None)
    last = last_row(**ONE_STATE)
    ramp = [1_234_567 + 30 * (i + 1) for i in range(4)]
    rows = run(ramp, settings, last)
    assert [row.flags for row in rows] == ["R", "R", "R", "R"]
    draft = moved_on(last, rejects=buffer(30.0, 60.0, 90.0), consecutive_rejects=3)
    prediction = State(x=Fraction(1_234_567), y=0.0)
    assert estimator.accept_step(draft, prediction, 1_234_657, 3, settings) is None


def test_no_step_is_looked_for_before_three_rejects() -> None:
    """Give no step while fewer than three consecutive rejects are counted."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, rejects=buffer(150.0, 150.0), consecutive_rejects=2)
    assert estimator.accept_step(draft, prediction, 1_234_724, 3, params()) is None


def test_scattered_rejects_show_no_step() -> None:
    """Give no step when the rejects agree on neither a phase nor a line."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, rejects=buffer(100.0, -100.0, 100.0), consecutive_rejects=3)
    assert estimator.accept_step(draft, prediction, 1_234_674, 3, params()) is None


def test_a_step_needs_a_series_with_an_innovation_scale() -> None:
    """Raise FilterError for three rejects on a draft that has no scale."""
    draft = moved_on(
        last_row(),
        rejects=buffer(150.0, 150.0, 150.0),
        consecutive_rejects=3,
        innovation_scale=None,
    )
    prediction = State(x=Fraction(0), y=0.0)
    with pytest.raises(FilterError, match="no innovation scale"):
        estimator.accept_step(draft, prediction, 150, 3, params())


# ---------------------------------------------------------------- acquisition


def dormant_row(*values: float, **changes: object) -> Row:
    """Build a dormant last row whose buffer holds ``values``, the newest at MARK."""
    count = len(values)
    entries = tuple(
        (MARK - EPOCH * (count - 1 - i), value) for i, value in enumerate(values)
    )
    fields: dict[str, object] = {
        "flags": "RD",
        "rejects": entries,
        "epochs_since_accept": 7,
        **DORMANT,
        **changes,
    }
    return last_row(**fields)


def test_a_measurement_of_a_dormant_series_is_buffered() -> None:
    """Add (E, z) to the buffer and write a dormant R row, rejects not counted."""
    last = dormant_row(1_000.0, consecutive_rejects=5)
    row = estimator.acquire(moved_on(last), 2_000, params())
    assert row.flags == "RD"
    assert row.rejects == ((MARK, 1_000.0), (NEXT, 2_000.0))
    assert row.consecutive_rejects == 0
    assert (row.x_fs, row.innovation_scale) == (None, None)


@pytest.mark.parametrize("model", [3, 2, 1])
def test_three_consistent_measurements_cold_start_the_series(
    model: Literal[1, 2, 3],
) -> None:
    """Cold-start from the third when the second difference passes (13.3)."""
    M = {3: 100.0, 2: 30.0, 1: None}[model]
    settings = params(model=model, M=M, sigma0=5.0)
    last = dormant_row(1_000.0, 51_000.0, filter_states=model, time_constant=M)
    edge = round_even(Fraction(101_000) + Fraction(5) * Fraction(math.sqrt(6) * 5))
    row = estimator.acquire(moved_on(last), edge, settings)
    assert row.flags == ("AN" if model == 1 else "ANU")
    assert (row.x_fs, row.segment, row.innovation_scale) == (edge * 1000, 5, 5.0)
    assert (row.rejects, row.consecutive_rejects, row.epochs_since_accept) == ((), 0, 0)


def test_a_second_difference_past_the_limit_stays_dormant() -> None:
    """Keep buffering when |z3 - 2 z2 + z1| is over 5 sqrt(6) sigma0."""
    limit = 5 * math.sqrt(6) * 5.0
    last = dormant_row(1_000.0, 51_000.0)
    within = estimator.acquire(moved_on(last), 101_000 + math.floor(limit), params())
    outside = estimator.acquire(moved_on(last), 101_000 + math.ceil(limit), params())
    assert (within.flags, outside.flags) == ("ANU", "RD")
    assert [value for _, value in outside.rejects] == [
        1_000.0,
        51_000.0,
        float(101_000 + math.ceil(limit)),
    ]


def test_three_measurements_from_epochs_apart_do_not_cold_start() -> None:
    """Need three measurements from consecutive epochs, not just three."""
    entries = ((MARK - 3 * EPOCH, 1_000.0), (MARK, 1_000.0))
    last = last_row(flags="RD", rejects=entries, **DORMANT)
    row = estimator.acquire(moved_on(last), 1_000, params())
    assert row.flags == "RD"
    assert len(row.rejects) == 3


def test_the_buffer_keeps_the_newest_three_measurements() -> None:
    """Drop the oldest buffered measurement when a fourth comes."""
    last = dormant_row(1_000.0, -9_000.0, 40_000.0)
    row = estimator.acquire(moved_on(last), 2_000, params())
    assert row.flags == "RD"
    assert [value for _, value in row.rejects] == [-9_000.0, 40_000.0, 2_000.0]


def test_a_missing_epoch_empties_the_acquisition_buffer() -> None:
    """Write D P with an empty buffer for a dormant series with no measurement."""
    row = estimator.hold(moved_on(dormant_row(1_000.0, 2_000.0)), None, "P", params())
    assert (row.flags, row.rejects) == ("PD", ())


def test_inconsistent_outliers_make_a_series_dormant_at_n_break() -> None:
    """Go dormant when the rejects reach N_break, then need three consistent (U11)."""
    settings = params(n_break=5)
    outliers = [1_234_567 + v for v in (900, -700, 1_300, -1_100, 600, -2_000, 800)]
    rows = run(outliers, settings, last_row())
    flags = [row.flags for row in rows]
    assert flags == ["R", "R", "R", "R", "RD", "RD", "RD"]
    assert [row.consecutive_rejects for row in rows[:4]] == [1, 2, 3, 4]
    assert rows[4].rejects == ((rows[4].interpolated_datetime, float(outliers[4])),)
    steady = [1_240_000, 1_240_010, 1_240_020]
    after = run(steady, settings, rows[-1])
    assert [row.flags for row in after] == ["RD", "RD", "ANU"]
    assert after[-1].segment == rows[-1].segment + 1
    assert after[-1].x_fs == 1_240_020_000


def test_a_dormant_series_acquires_across_a_wrap() -> None:
    """Stay dormant through scatter, empty on a gap, cold-start on steady (U25)."""
    settings = params(sigma0=5.0)
    last = dormant_row()
    true = [PHASE_MAX - 40_000 + 60_000 * k for k in range(4)]
    scattered = [5_000, 150_000, 60_000, 190_000]
    rows: list[Row] = []

    def measure(phi: int) -> None:
        """Decycle ``phi`` against the anchor and buffer it, as a dormant pair does."""
        nonlocal last
        anchor = estimator.anchor_of(last)
        z = decycle(phi, Fraction(0), Fraction(0), None, anchor).z
        last = estimator.acquire(moved_on(last), z, settings)
        rows.append(last)

    for phi in scattered:
        measure(phi)
    assert [row.flags for row in rows] == ["RD"] * 4
    last = estimator.hold(moved_on(last), None, "P", settings)
    assert last.rejects == ()
    rows.clear()
    for value in true[:3]:
        measure(value % PHASE_PERIOD)
    assert [row.flags for row in rows] == ["RD", "RD", "ANU"]
    assert rows[-1].x_fs == true[2] * 1000
    assert true[1] > PHASE_MAX


def test_the_anchor_is_the_last_buffered_measurement() -> None:
    """Give the newest buffered measurement of a dormant series, as a whole ps."""
    assert estimator.anchor_of(dormant_row(1_000.0, 2_500_000_017.0)) == 2_500_000_017


@pytest.mark.parametrize(
    "last",
    [
        None,
        last_row(),
        last_row(flags="PD", **DORMANT),
        last_row(flags="R", rejects=((MARK, 40.0),), consecutive_rejects=1),
    ],
)
def test_no_anchor_without_a_buffered_measurement(last: Row | None) -> None:
    """Give None for no row, a tracked row, or a dormant row with an empty buffer."""
    assert estimator.anchor_of(last) is None


# ------------------------------------------------------------- filter step


def pair(z: int, rms: int = 3, *, slip: bool = False) -> estimator.Measured:
    """Give a pair's measurement at an epoch."""
    return estimator.Measured(z=z, rms=rms, slip=slip)


def triple(z: int, sigma: float = 3.5, *, cold: bool = False) -> estimator.Measured:
    """Give a triple's measurement at an epoch."""
    return estimator.Measured(z=z, sigma_dd=sigma, cold=cold)


def step(
    last: Row | None,
    measured: estimator.Measured | None,
    settings: SeriesParams | None = None,
    *,
    excluded: bool = False,
) -> estimator.StepResult:
    """Run the filter step at the epoch after ``last``."""
    mark = NEXT if last is None else last.interpolated_datetime + EPOCH
    prediction = estimator.predict(last, NO_INPUT)
    return estimator.filter_step(
        mark, settings or params(), last, prediction, measured, excluded=excluded
    )


def ending(*values: float) -> tuple[tuple[datetime, float], ...]:
    """Give a reject buffer of ``values`` at consecutive epochs ending at MARK."""
    count = len(values)
    return tuple((MARK - EPOCH * (count - 1 - i), v) for i, v in enumerate(values))


WORKED: Final = 1_234_577
"""The worked epoch's measurement, ps: z_E of Appendix A."""


def test_the_worked_epoch_is_accepted() -> None:
    """Give the accepted row of design 5.4 for the worked epoch, not cold."""
    result = step(last_row(), pair(WORKED))
    assert result.cold is False
    assert result.row.flags == "A"
    assert result.row.x_fs == 1_234_574_457


def test_no_measurement_gives_a_predicted_row() -> None:
    """Give P with the prediction when there is no measurement (9.6)."""
    result = step(last_row(), None)
    assert (result.row.flags, result.cold) == ("P", False)
    assert result.row.epochs_since_accept == 1


def test_no_measurement_for_a_dormant_series_gives_a_dormant_row() -> None:
    """Give D P for a new or dormant series with no measurement (9.6)."""
    assert step(None, None).row.flags == "PD"
    assert step(last_row(flags="PD", **DORMANT), None).row.flags == "PD"


def test_a_new_series_buffers_its_first_measurement() -> None:
    """Give D R, the measurement buffered, for a series with no prediction (9.6)."""
    result = step(None, pair(WORKED))
    assert (result.row.flags, result.cold) == ("RD", False)
    assert result.row.rejects == ((NEXT, float(WORKED)),)
    assert result.row.segment == 0


def test_a_consistent_third_measurement_cold_starts() -> None:
    """Give A N U and say the row cold-started, from the acquisition buffer (9.6)."""
    last = last_row(
        flags="RD",
        rejects=((MARK - EPOCH, float(WORKED)), (MARK, float(WORKED))),
        **DORMANT,
    )
    result = step(last, pair(WORKED))
    assert (result.row.flags, result.cold) == ("ANU", True)
    assert result.row.segment == 5


def test_an_excluded_measurement_within_the_gate_is_held() -> None:
    """Give X, not counted, for an excluded measurement inside the gate (9.5)."""
    result = step(last_row(), pair(WORKED), excluded=True)
    assert (result.row.flags, result.row.consecutive_rejects) == ("X", 0)
    assert result.row.rejects == ()
    assert result.row.innovation == float(
        WORKED - Fraction(1_234_567) - exact(0.0123) * T
    )


def test_an_excluded_measurement_outside_the_gate_is_a_counted_reject() -> None:
    """Give R, counted and buffered, for an excluded one outside the gate (9.5)."""
    result = step(last_row(), pair(WORKED + 100), excluded=True)
    assert (result.row.flags, result.row.consecutive_rejects) == ("R", 1)
    assert len(result.row.rejects) == 1


def test_a_measurement_outside_the_gate_is_a_counted_reject() -> None:
    """Give R and push the innovation for a measurement outside the gate (9.6)."""
    result = step(last_row(), pair(WORKED + 100))
    assert (result.row.flags, result.row.consecutive_rejects) == ("R", 1)
    assert result.row.innovation == result.row.rejects[0][1]


def test_an_rms_over_the_limit_is_a_counted_reject() -> None:
    """Give R for a pair whose rms is over its limit, inside the gate (9.1)."""
    result = step(last_row(), pair(WORKED, rms=81))
    assert (result.row.flags, result.row.consecutive_rejects) == ("R", 1)


def test_a_triple_has_no_rms_test_and_its_floor_is_sigma_dd() -> None:
    """Accept a triple with no rms limit, its scale held up by sigma_dd (12.5)."""
    result = step(last_row(), triple(WORKED, sigma=3.5), params(rms_max=None))
    assert result.row.flags == "A"
    assert result.row.innovation_scale == 3.5


def test_a_third_agreeing_reject_is_a_phase_step() -> None:
    """Give A, step_offset up, same segment, after two rejects (9.6)."""
    last = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=ending(150.0, 150.0),
        innovation=150.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    result = step(last, pair(round_even(prediction.x) + 150))
    assert (result.row.flags, result.row.step_offset, result.row.segment) == (
        "A",
        150,
        4,
    )


def test_a_third_reject_on_a_line_is_a_frequency_step() -> None:
    """Give A N U in segment + 1 after rejects on a line (9.6)."""
    last = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=ending(30.0, 60.0),
        innovation=60.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    result = step(last, pair(round_even(prediction.x) + 90))
    assert (result.row.flags, result.row.segment) == ("ANU", 5)


def test_a_third_scattered_reject_is_held() -> None:
    """Give R when three rejects show no step and N_break is not reached (9.6)."""
    last = last_row(
        flags="R",
        consecutive_rejects=2,
        rejects=ending(100.0, -100.0),
        innovation=-100.0,
        epochs_since_accept=2,
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    result = step(last, pair(round_even(prediction.x) + 100))
    assert (result.row.flags, result.row.consecutive_rejects) == ("R", 3)


def test_rejects_reaching_n_break_make_the_series_dormant() -> None:
    """Give D R with only the current measurement buffered at N_break (9.6)."""
    last = last_row(
        flags="R",
        consecutive_rejects=4,
        rejects=ending(100.0, -100.0, 300.0),
        innovation=300.0,
        epochs_since_accept=4,
    )
    result = step(last, pair(WORKED - 500), params(n_break=5))
    assert (result.row.flags, result.cold) == ("RD", False)
    assert result.row.rejects == ((NEXT, float(WORKED - 500)),)
    assert result.row.consecutive_rejects == 0


def test_a_component_cold_start_makes_a_triple_dormant() -> None:
    """Give D R and restart acquisition when a component pair cold-started (12.6)."""
    last = last_row(
        rejects=ending(90.0), consecutive_rejects=1, flags="R", innovation=90.0
    )
    result = step(last, triple(WORKED, cold=True), params(rms_max=None))
    assert (result.row.flags, result.cold) == ("RD", False)
    assert result.row.rejects == ((NEXT, float(WORKED)),)


def test_a_slip_corrected_measurement_carries_s() -> None:
    """Carry S onto the row of a measurement the slip check corrected."""
    assert step(last_row(), pair(WORKED, slip=True)).row.flags == "AS"


def test_a_configuration_change_warm_starts_before_the_measurement() -> None:
    """Start segment + 1 with the new M, then accept with its gains (8.7, U14)."""
    changed = params(M=150.0, M_sigma=60.0)
    last = last_row()
    result = step(last, pair(WORKED), changed)
    row = result.row
    assert (row.flags, row.segment, row.epochs_in_segment) == ("ANU", 5, 0)
    assert (row.time_constant, row.scale_time_constant, row.step_offset) == (
        150.0,
        60.0,
        0,
    )
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    expected = estimator.update(prediction, WORKED - prediction.x, 3, 150.0)
    assert row.x_fs == to_fs(expected.x)
    assert row.y == expected.y


def test_a_configuration_change_shares_a_row_with_no_measurement() -> None:
    """Give N U P on a configuration change at an epoch with no measurement (8.7)."""
    assert step(last_row(), None, params(M_sigma=60.0)).row.flags == "PNU"


def test_a_dormant_series_takes_new_settings_at_its_cold_start() -> None:
    """Start no warm segment for a dormant series: its cold start takes them."""
    last = last_row(flags="PD", **DORMANT)
    assert step(last, None, params(M=150.0)).row.flags == "PD"


def test_unchanged_settings_start_no_segment() -> None:
    """Keep the segment when M and M_sigma are as in the last row."""
    assert step(last_row(), pair(WORKED)).row.segment == 4


@pytest.mark.parametrize(
    "values",
    [
        {"z": 1, "rms": 3, "sigma_dd": 3.0},
        {"z": 1},
        {"z": 1, "sigma_dd": 3.0, "slip": True},
        {"z": 1, "rms": 3, "cold": True},
        {"z": 1, "rms": -1},
        {"z": 1, "sigma_dd": -1.0},
        {"z": 1, "sigma_dd": float("nan")},
        {"z": 1.5, "rms": 3},
    ],
)
def test_a_measurement_is_a_pair_s_or_a_triple_s(values: dict[str, object]) -> None:
    """Refuse a measurement that is neither a pair's (rms) nor a triple's (sigma_dd)."""
    with pytest.raises((ValidationError, FilterError)):
        estimator.Measured.model_validate(values)


@pytest.mark.parametrize(
    ("measured", "floor"), [(pair(5, rms=4), 4.0), (triple(5, sigma=3.25), 3.25)]
)
def test_a_measurement_gives_its_floor(
    measured: estimator.Measured, floor: float
) -> None:
    """Give the rms of a pair and sigma_dd of a triple as the scale's floor (9.2)."""
    assert measured.floor == floor


def test_a_prediction_for_a_series_with_no_scale_is_refused() -> None:
    """Raise FilterError when a dormant last row is given a prediction anyway."""
    last = last_row(flags="PD", **DORMANT)
    prediction = State(x=Fraction(WORKED), y=0.0)
    with pytest.raises(FilterError, match="no scale"):
        estimator.filter_step(NEXT, params(), last, prediction, pair(WORKED))


# ------------------------------------------------- edges and what is logged


def test_classify_fits_the_slope_as_the_design_writes_it() -> None:
    """Give s as sum((t - tbar)(v - vbar)) / sum((t - tbar)^2), to the last bit."""
    values = (31.3, 60.7, 90.1)
    classified = estimator.classify(buffer(*values), 3.0)
    ts = [0.0, T, 2 * T]
    vbar, tbar = sum(values) / 3, sum(ts) / 3
    s = sum((t - tbar) * (v - vbar) for t, v in zip(ts, values, strict=True)) / sum(
        (t - tbar) ** 2 for t in ts
    )
    assert classified.kind == "frequency"
    assert classified.s == s
    assert classified.a == vbar - s * tbar


def test_a_phase_step_needs_the_rejects_strictly_within_three_scales() -> None:
    """Give no phase step when the furthest reject is exactly 3 scales from the mean."""
    assert estimator.classify(buffer(100.0, 100.0, 109.0), 2.0).kind == "frequency"


def test_a_frequency_step_needs_the_rejects_strictly_within_three_scales() -> None:
    """Give no step when the furthest reject is exactly 3 scales from the line."""
    assert estimator.classify(buffer(0.0, 0.0, 18.0), 2.0).kind is None


def test_acquisition_takes_a_second_difference_at_its_limit() -> None:
    """Cold-start when the second difference is exactly the acquisition limit."""
    sigma0 = 40 / estimator._ACQUIRE_LIMIT
    assert estimator._ACQUIRE_LIMIT * sigma0 == 40.0
    last = dormant_row(1_000.0, 2_000.0)
    row = estimator.acquire(moved_on(last), 3_040, params(sigma0=sigma0))
    assert row.flags == "ANU"


@pytest.mark.parametrize("model", [2, 3])
def test_a_time_constant_of_one_epoch_has_gains(model: int) -> None:
    """Give the gains for M = 1, the shortest time constant."""
    lam = math.exp(-1.0)
    expected = (
        (1 - lam**2, (1 - lam) ** 2 / T, 0.0)
        if model == 2
        else (1 - lam**3, 1.5 * (1 - lam) ** 2 * (1 + lam) / T, (1 - lam) ** 3 / T**2)
    )
    assert estimator.gains(model, 1.0) == expected


DRIFT: Final = 2.5e-12
"""A drift, ps/s^2, for rows whose d must be carried."""


def test_a_held_row_carries_the_drift() -> None:
    """Store the predicted d in a held row."""
    last = last_row(d=DRIFT)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    assert prediction.d != 0.0
    row = estimator.hold(moved_on(last, innovation=None), prediction, "P", params())
    assert row.d == prediction.d


def test_a_phase_step_keeps_the_drift() -> None:
    """Correct only the phase in a phase step, the drift carried into the update."""
    last = last_row(d=DRIFT)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, rejects=buffer(100.0, 100.0, 100.0), consecutive_rejects=3)
    z = round_even(prediction.x) + 100
    row = estimator.phase_step(draft, prediction, z, 3)
    corrected = State(x=prediction.x + 100, y=prediction.y, d=prediction.d)
    expected = estimator.update(corrected, z - corrected.x, 3, 100.0)
    assert row.d == expected.d


def test_a_frequency_step_keeps_the_drift() -> None:
    """Correct phase and rate in a frequency step, the drift carried into the update."""
    last = last_row(d=DRIFT)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    draft = moved_on(last, rejects=buffer(30.0, 60.0, 90.0), consecutive_rejects=3)
    z = round_even(prediction.x) + 90
    row = estimator.frequency_step(draft, prediction, 30.0, 0.05, z, 3, params())
    corrected = State(
        x=prediction.x + 30 + exact(0.05) * 2 * T,
        y=prediction.y + 0.05,
        d=prediction.d,
    )
    expected = estimator.update(corrected, z - corrected.x, 3, 100.0)
    assert row.d == expected.d


def test_a_configuration_change_in_the_filter_step_keeps_the_step_offset() -> None:
    """Carry step_offset into the warm segment a settings change starts (8.7)."""
    last = last_row(step_offset=25)
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    measured = estimator.Measured(z=round_even(prediction.x), rms=3)
    changed = params(M=150.0, M_sigma=60.0)
    row = estimator.filter_step(NEXT, changed, last, prediction, measured).row
    assert (row.segment, row.step_offset) == (5, 25)
    assert "N" in row.flags


def test_an_excluded_measurement_in_the_gate_is_written_x() -> None:
    """Write X alone for a measurement in the gate that screening excluded."""
    last = last_row()
    prediction = estimator.predict(last, NO_INPUT)
    assert prediction is not None
    measured = estimator.Measured(z=round_even(prediction.x), rms=3)
    row = estimator.filter_step(
        NEXT, params(), last, prediction, measured, excluded=True
    ).row
    assert row.flags == "X"


def test_every_filter_error_is_logged_as_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log each FilterError at ERROR in the words it is raised with."""
    no_scale = moved_on(last_row(), innovation_scale=None)
    prediction = State(x=Fraction(0), y=0.0)
    calls: list[Callable[[], object]] = [
        lambda: estimator.accept(no_scale, prediction, Fraction(0), 3),
        lambda: estimator.accept_step(
            dataclasses.replace(no_scale, consecutive_rejects=3),
            prediction,
            0,
            3,
            params(),
        ),
        lambda: estimator.classify(buffer(1.0), 3.0),
        lambda: estimator.gains(2, 0.5),
        lambda: estimator._gate(
            no_scale,
            prediction,
            estimator.Measured(z=0, rms=3),
            params(),
            excluded=False,
        ),
        lambda: estimator.start_segment(
            moved_on(last_row()), params(model=2, M=30.0), keep_offset=True
        ),
    ]
    for call in calls:
        caplog.clear()
        with pytest.raises(FilterError) as raised:
            call()
        assert [r.getMessage() for r in caplog.records] == [str(raised.value)]
        assert str(raised.value)
    caplog.clear()
    with pytest.raises(FilterError) as raised:
        estimator.gains(2, 0.5)
    assert str(raised.value) == "no gains for a 2-state model with time constant 0.5"
