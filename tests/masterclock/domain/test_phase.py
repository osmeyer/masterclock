"""Tests for src/masterclock/domain/phase.py.

The rules covered: a phase period is one period of a 5 MHz signal in whole
picoseconds, and the largest phase a reading gives is one short of it; a
time offset between two datetimes is an exact fraction of seconds, and
naive datetimes are refused; a float becomes an exact fraction, and a value
that is not finite is refused; rounding is to the nearest whole number,
a tie going to the even one, exactly at any size; the estimator's phase is
held in whole femtoseconds, rounded the same way; an epoch lasts 600 s; and
a measurement is decycled against the prediction at its own time, or,
without a prediction, against the last buffered measurement, or with no
cycles added, and referred back to its epoch start with one exact rounding;
a reading outside one period, or a measurement time outside its epoch, is
refused; and a decycled measurement is frozen.

Decycling includes the drift term d delta**2 / 2, and adds the steering
inside the epoch to the prediction and takes it off again; each error is
logged as raised.
"""

import dataclasses
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest
from gmpy2 import mpq
from hypothesis import given
from hypothesis import strategies as st

from masterclock.domain import phase
from masterclock.domain.exceptions import FilterError, PhaseError
from masterclock.domain.series import State

EPOCH_START: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented ten-minute mark."""


def test_the_period_is_one_cycle_of_5_mhz_in_picoseconds() -> None:
    """Make the period exactly 1 / 5 MHz, written in picoseconds."""
    assert mpq(phase.PHASE_PERIOD, 10**12) == mpq(1, 5_000_000)


def test_the_largest_phase_is_one_short_of_a_period() -> None:
    """Keep a whole period out, since it would read as zero."""
    assert phase.PHASE_MAX == phase.PHASE_PERIOD - 1


# ----------------------------------------------------------------- seconds


def test_an_offset_is_exact_to_the_microsecond() -> None:
    """Give 137.2032 s as exactly 1 372 032 / 10 000, not a float near it."""
    later_instant = EPOCH_START + timedelta(seconds=137, microseconds=203_200)
    assert phase.seconds(later_instant, EPOCH_START) == mpq(1_372_032, 10_000)


def test_an_offset_can_be_negative() -> None:
    """Give an earlier first datetime as a negative offset."""
    assert phase.seconds(EPOCH_START, EPOCH_START + timedelta(microseconds=1)) == mpq(
        -1, 10**6
    )


@pytest.mark.parametrize(
    ("later_instant", "earlier_instant"),
    [
        (EPOCH_START.replace(tzinfo=None), EPOCH_START),
        (EPOCH_START, EPOCH_START.replace(tzinfo=None)),
        (EPOCH_START.replace(tzinfo=None), EPOCH_START.replace(tzinfo=None)),
    ],
)
def test_a_naive_datetime_is_refused(
    later_instant: datetime, earlier_instant: datetime
) -> None:
    """Refuse a datetime without a timezone, which names no one instant."""
    with pytest.raises(PhaseError, match="naive"):
        phase.seconds(later_instant, earlier_instant)


@given(st.integers(min_value=-(10**15), max_value=10**15))
def test_an_offset_is_its_microseconds_over_a_million(microseconds: int) -> None:
    """Give any offset as its whole number of microseconds over 10**6."""
    later_instant = EPOCH_START + timedelta(microseconds=microseconds)
    assert phase.seconds(later_instant, EPOCH_START) == mpq(microseconds, 10**6)


# ------------------------------------------------------------------- exact


def test_a_float_becomes_the_fraction_it_holds() -> None:
    """Give 0.1 as the binary value the float holds, not as one tenth."""
    assert phase.exact(0.1) == mpq(3_602_879_701_896_397, 2**55)
    assert phase.exact(0.1) != mpq(1, 10)


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_not_finite_is_refused(bad_number: float) -> None:
    """Refuse nan and the infinities, which no phase can hold."""
    with pytest.raises(FilterError, match="not finite"):
        phase.exact(bad_number)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_every_finite_float_is_held_exactly(number: float) -> None:
    """Give a fraction equal to the float, numerator and denominator alike."""
    assert phase.exact(number) == mpq(*number.as_integer_ratio())


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_a_float_s_ratio_is_the_fraction_it_holds(number: float) -> None:
    """Give the numerator and denominator of the value exact gives, in lowest terms."""
    numerator, denominator = phase.exact_ratio(number)
    assert mpq(numerator, denominator) == phase.exact(number)
    assert (numerator, denominator) == (
        int(phase.exact(number).numerator),
        int(phase.exact(number).denominator),
    )


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), float("-inf")])
def test_a_ratio_of_a_value_that_is_not_finite_is_refused(
    bad_number: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse nan and the infinities, and log the refusal as raised."""
    with pytest.raises(FilterError, match="not finite") as raised:
        phase.exact_ratio(bad_number)
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(raised.value)
    ]


# -------------------------------------------------------------- round_even


@pytest.mark.parametrize(
    ("exact_sum", "expected_round"),
    [
        (mpq(5, 2), 2),
        (mpq(7, 2), 4),
        (mpq(-5, 2), -2),
        (mpq(-7, 2), -4),
        (mpq(12_345_773_124, 10_000), 1_234_577),
        (mpq(2**63 * 2 + 1, 2), 2**63),
        (17, 17),
    ],
)
def test_rounding_is_to_nearest_with_ties_to_even(
    exact_sum: mpq, expected_round: int
) -> None:
    """Round 2.5 to 2, 3.5 to 4 and -2.5 to -2, exactly at any size (U2)."""
    rounded_result = phase.round_even(exact_sum)
    assert rounded_result == expected_round
    assert type(rounded_result) is int


@given(st.integers(min_value=-(2**70), max_value=2**70))
def test_every_half_rounds_to_even(whole_number: int) -> None:
    """Round every n + 1/2 to whichever of n and n + 1 is even."""
    rounded_result = phase.round_even(mpq(2 * whole_number + 1, 2))
    assert rounded_result in {whole_number, whole_number + 1}
    assert rounded_result % 2 == 0


def reference_round(numerator: int, denominator: int) -> int:
    """Round numerator / denominator by integer arithmetic alone, ties to even."""
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (
        2 * remainder == denominator and quotient % 2 == 1
    ):
        return quotient + 1
    return quotient


@given(
    st.integers(min_value=2**53, max_value=2**63),
    st.floats(min_value=-1e6, max_value=1e6),
    st.sampled_from([1, -1]),
)
def test_a_large_phase_plus_a_float_rounds_as_integers_do(
    whole_phase: int, float_term: float, sign: int
) -> None:
    """Round a phase beyond 2**53 plus a float term as exact integer sums do."""
    numerator, denominator = float_term.as_integer_ratio()
    exact_numerator = sign * whole_phase * denominator + numerator
    assert phase.round_even(
        sign * whole_phase + phase.exact(float_term)
    ) == reference_round(exact_numerator, denominator)


@given(st.integers(min_value=2**53, max_value=2**63))
def test_a_large_phase_plus_a_half_is_a_tie_to_even(whole_phase: int) -> None:
    """Treat a phase beyond 2**53 plus exactly 0.5 as a tie, not a float near it."""
    assert (
        phase.round_even(whole_phase + phase.exact(0.5))
        == whole_phase + whole_phase % 2
    )


# ------------------------------------------------------------ femtoseconds


@pytest.mark.parametrize(
    ("phase_ps", "expected_fs"),
    [
        (mpq(12_345_744_574, 10_000), 1_234_574_457),
        (mpq(1, 2000), 0),
        (mpq(3, 2000), 2),
        (mpq(-1, 2000), 0),
        (mpq(-3, 2000), -2),
        (7, 7_000),
    ],
)
def test_a_phase_rounds_to_whole_femtoseconds(phase_ps: mpq, expected_fs: int) -> None:
    """Round to the nearest femtosecond, a tie to the even one."""
    assert phase.to_fs(phase_ps) == expected_fs


@given(st.integers(min_value=-(2**70), max_value=2**70))
def test_femtoseconds_come_back_exactly(phase_fs: int) -> None:
    """Give back the same whole femtoseconds after a turn through picoseconds."""
    assert phase.to_fs(phase.from_fs(phase_fs)) == phase_fs
    assert phase.from_fs(phase_fs) == mpq(phase_fs, 1000)


# -------------------------------------------------------------- decycling

P: Final = phase.PHASE_PERIOD
"""One period, ps."""


def test_an_epoch_lasts_600_seconds() -> None:
    """Make one epoch ten minutes, in whole seconds."""
    assert phase.EPOCH_SECONDS == 600


def test_the_worked_epoch_decycles_to_six_cycles() -> None:
    """Reproduce Appendix A: n = 6 and z_E = 1 234 577."""
    prediction = State(x=1_234_567 + phase.exact(0.0123) * 600, y=0.0123, d=0.0)
    delta = mpq(1_372_032, 10_000)
    decycled = phase.decycle(34_579, delta, mpq(0), prediction, None)
    assert decycled == phase.Decycled(cycle_count=6, z=1_234_577)


def predicted_motion(prediction: State, delta: mpq) -> mpq:
    """Work out y delta + d delta**2 / 2 exactly, as decycle does."""
    y = mpq(*prediction.y.as_integer_ratio())
    d = mpq(*prediction.d.as_integer_ratio())
    return y * delta + d * delta * delta / 2


@given(
    true_phase=st.integers(min_value=-(2**62), max_value=2**62),
    rate=st.floats(min_value=-150.0, max_value=150.0),
    drift=st.floats(min_value=-1e-6, max_value=1e-6),
    microseconds=st.integers(min_value=0, max_value=599_999_999),
    prediction_error=st.integers(min_value=-90_000, max_value=90_000),
    w=st.fractions(min_value=-1000, max_value=1000, max_denominator=10**6),
)
def test_a_measurement_decycles_to_the_truth(
    true_phase: int,
    rate: float,
    drift: float,
    microseconds: int,
    prediction_error: int,
    w: mpq,
) -> None:
    """Recover the whole phase with a prediction up to 0.45 P wrong (U3).

    ``true_phase`` is the whole phase at the measurement time, of any sign and
    far beyond 2**53, and the reading is it modulo one period. The rate is
    large enough that its motion over the measurement offset reaches 0.45 P
    too, so a decycling that left the motion out would miss whole periods.
    """
    delta = mpq(microseconds, 10**6)
    reading = true_phase % P
    motion_ps = predicted_motion(State(x=mpq(0), y=rate, d=drift), delta)
    prediction = State(x=true_phase - motion_ps - w + prediction_error, y=rate, d=drift)
    decycled = phase.decycle(reading, delta, w, prediction, None)
    assert decycled.cycle_count == (true_phase - reading) // P
    assert decycled.z == round(true_phase - motion_ps - w)


@pytest.mark.parametrize("rate", [0.031, -0.031])
@pytest.mark.parametrize("start_phase", [-3 * P - 5, P - 7, 2**60 + P - 3])
def test_a_ramp_decycles_across_every_wrap(rate: float, start_phase: int) -> None:
    """Follow a ramp through the wrap at 0 and at PHASE_MAX, epoch after epoch.

    Each epoch is predicted from the last epoch's result and a rate in error
    by enough to put the prediction 0.45 P off, so every wrap is crossed with
    the decycling bound nearly used up.
    """
    epoch_seconds = mpq(phase.EPOCH_SECONDS)
    wrong_rate = rate + 0.45 * P / phase.EPOCH_SECONDS
    previous_z = start_phase
    for epoch_number in range(1, 40):
        true_phase = start_phase + phase.round_even(
            mpq(*rate.as_integer_ratio()) * epoch_seconds * epoch_number
        )
        reading = true_phase % P
        prediction = State(
            x=previous_z + phase.exact(wrong_rate) * epoch_seconds, y=wrong_rate, d=0.0
        )
        decycled = phase.decycle(reading, mpq(0), mpq(0), prediction, None)
        assert decycled.z == true_phase, epoch_number
        previous_z = decycled.z


def test_a_value_on_an_exact_half_rounds_to_even() -> None:
    """Round a z_E that falls on n + 1/2 to the even neighbour, once."""
    prediction = State(x=mpq(100), y=0.5, d=0.0)
    for reading, expected_z in ((101, 100), (103, 102)):
        decycled = phase.decycle(reading, mpq(1), mpq(0), prediction, None)
        assert decycled.z == expected_z


@given(
    whole_phase=st.integers(min_value=2**53, max_value=2**63),
    rate=st.floats(min_value=-0.1, max_value=0.1),
    microseconds=st.integers(min_value=0, max_value=599_999_999),
)
def test_large_phases_decycle_exactly(
    whole_phase: int, rate: float, microseconds: int
) -> None:
    """Match an mpq reference for phases beyond 2**53 (U27, decycling)."""
    delta = mpq(microseconds, 10**6)
    motion_ps = predicted_motion(State(x=mpq(0), y=rate), delta)
    prediction = State(x=mpq(whole_phase), y=rate)
    reading = phase.round_even(whole_phase + motion_ps) % P
    decycled = phase.decycle(reading, delta, mpq(0), prediction, None)
    n = phase.round_even((whole_phase + motion_ps - reading) / P)
    assert decycled.cycle_count == n
    assert decycled.z == phase.round_even(reading + n * P - motion_ps)


@pytest.mark.parametrize(
    ("anchor", "reading", "w", "cycles"),
    [
        (None, 123, mpq(0), 0),
        (None, 123, mpq(50), 0),
        (P + 123, 123, mpq(0), 1),
        (-P + 100, 100, mpq(0), -1),
        (5 * P + 10, P - 10, mpq(0), 4),
        (10, P - 10, mpq(0), -1),
        (P + 123, 123, mpq(P), 2),
    ],
)
def test_without_a_prediction_the_anchor_decycles(
    anchor: int | None, reading: int, w: mpq, cycles: int
) -> None:
    """Decycle against the last buffered value, or with no cycles at all."""
    decycled = phase.decycle(reading, mpq(77), w, None, anchor)
    assert decycled.cycle_count == cycles
    assert decycled.z == round(reading + cycles * P - w)


@pytest.mark.parametrize("reading", [-1, P])
def test_a_reading_outside_one_period_is_refused(reading: int) -> None:
    """Refuse a reading that is not a phase within one period."""
    with pytest.raises(PhaseError, match="not within one period"):
        phase.decycle(reading, mpq(0), mpq(0), None, None)


@pytest.mark.parametrize("delta", [mpq(-1, 10**6), mpq(600)])
def test_a_measurement_outside_its_epoch_is_refused(delta: mpq) -> None:
    """Refuse a measurement time before its epoch start or a whole epoch on."""
    with pytest.raises(PhaseError, match="not within its epoch"):
        phase.decycle(0, delta, mpq(0), None, None)


def test_a_decycled_measurement_is_frozen() -> None:
    """Refuse a change to a decycled measurement."""
    decycled = phase.Decycled(cycle_count=1, z=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decycled.z = 3  # type: ignore[misc]


def test_with_a_prediction_the_anchor_is_not_used() -> None:
    """Decycle against the prediction even when an anchor is given."""
    prediction = State(x=mpq(3 * P + 10), y=0.0)
    decycled = phase.decycle(10, mpq(0), mpq(0), prediction, -7 * P)
    assert decycled == phase.Decycled(cycle_count=3, z=3 * P + 10)


def test_a_fast_rate_is_decycled_with_its_motion() -> None:
    """Count the rate's motion over the offset when choosing the periods.

    At 150 ps/s, 500 s after the mark the phase has moved 75 000 ps; with
    the prediction 40 000 ps low as well, leaving the motion out would put
    the reading a period away.
    """
    true_phase = 7 * P + 12_345
    prediction = State(x=mpq(true_phase - 75_000 - 40_000), y=150.0)
    decycled = phase.decycle(true_phase % P, mpq(500), mpq(0), prediction, None)
    assert decycled == phase.Decycled(cycle_count=7, z=true_phase - 75_000)


# --------------------------------------------- drift, steering, what is logged


def test_decycling_takes_the_drift_half_delta_squared() -> None:
    """Refer the phase back to E with d delta**2 / 2, the drift's own term."""
    prediction = State(x=mpq(500_000), y=0.0, d=100 / 360_000)
    delta = mpq(600 - 1, 1)
    decycled = phase.decycle(1_000, delta, mpq(0), prediction, None)
    expected_z = phase.round_even(
        1_000
        + decycled.cycle_count * phase.PHASE_PERIOD
        - predicted_motion(prediction, delta)
    )
    assert predicted_motion(prediction, delta) > 40
    assert decycled.z == expected_z


def test_steering_inside_the_epoch_moves_the_prediction_its_way() -> None:
    """Add w to the prediction at t, so a large w still finds the right cycle."""
    w = mpq(60_000)
    prediction = State(x=mpq(1_000_000), y=0.0, d=0.0)
    reading = int(1_000_000 + w) % phase.PHASE_PERIOD
    decycled = phase.decycle(reading, mpq(0), w, prediction, None)
    assert decycled.z == 1_000_000


def test_every_phase_error_is_logged_as_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log each PhaseError at ERROR in the words it is raised with."""
    zero_state = State(x=mpq(0), y=0.0, d=0.0)
    failing_calls: list[Callable[[], object]] = [
        lambda: phase.decycle(phase.PHASE_PERIOD, mpq(0), mpq(0), zero_state, None),
        lambda: phase.decycle(0, mpq(600), mpq(0), zero_state, None),
        lambda: phase.seconds(EPOCH_START.replace(tzinfo=None), EPOCH_START),
    ]
    for failing_call in failing_calls:
        caplog.clear()
        with pytest.raises(PhaseError) as raised_error:
            failing_call()
        assert [log_record.getMessage() for log_record in caplog.records] == [
            str(raised_error.value)
        ]
        assert str(raised_error.value)
    caplog.clear()
    with pytest.raises(FilterError) as not_finite_error:
        phase.exact(float("nan"))
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(not_finite_error.value)
    ]
