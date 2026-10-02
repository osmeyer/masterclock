"""Tests for src/masterclock/domain/filter.py.

The rules covered: the fixed gains put every pole of the closed loop at
lambda = exp(-1/M), a triple pole for three states and a double pole for
two, and a 1-state series passes its measurement through; the prediction
moves the last row's state on by one epoch and adds the steering input,
with the phase exact; a dormant or missing last row gives no prediction;
the update adds the gains times the innovation, the phase exact; and a
noise-free ramp or parabola is followed to within 1 ps, whether the phase
is kept exact or each row stores it in whole femtoseconds.
"""

import math
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final

import pytest

from masterclock.domain import filter as estimator
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import EPOCH_SECONDS, exact, round_even, to_fs
from masterclock.domain.series import Row, State

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
