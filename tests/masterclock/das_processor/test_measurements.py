"""Tests for src/masterclock/das_processor/measurements.py.

The rules covered: a pair measurement is a DAS measurement decycled and
referred to its epoch start, against the prediction or, without one, the
anchor; its offset after the epoch start is worked out exactly from the
datetimes and cannot be passed in; a slip correction moves the cycle count
and z by whole periods and marks it; a pair or triple measurement gives
the filter step its plain values; and a triple measurement names the
components it used, as 111, 110 or 101 only.
"""

from fractions import Fraction
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.das_processor.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain.double_difference import TripleValue
from masterclock.domain.filter import Measured
from masterclock.domain.phase import PHASE_PERIOD, exact
from masterclock.domain.series import State

RAW: Final = DASMeasurement(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    switch="2B07",
    clock="nav23",
)
"""Appendix A's raw row."""

PREDICTION: Final = State(x=1_234_567 + exact(0.0123) * 600, y=0.0123)
"""Appendix A's prediction at the epoch start."""


def test_the_worked_epoch_gives_its_pair_measurement() -> None:
    """Reproduce Appendix A: delta 137.2032 s, n = 6, z_E = 1 234 577."""
    pair = measure_pair(RAW, PREDICTION, Fraction(0), None)
    assert pair.delta == Fraction(1_372_032, 10_000)
    assert (pair.cycle_count, pair.z, pair.slip) == (6, 1_234_577, False)
    assert pair.measurement == RAW


def test_a_pair_without_a_prediction_is_decycled_against_its_anchor() -> None:
    """Decycle against the last buffered measurement when there is no prediction."""
    pair = measure_pair(RAW, None, Fraction(0), 1_234_000)
    assert (pair.cycle_count, pair.z) == (6, 34_579 + 6 * PHASE_PERIOD)


def test_steering_in_the_epoch_is_taken_off() -> None:
    """Refer the measurement to E with the steering since E taken off."""
    pair = measure_pair(RAW, None, Fraction(7, 2), None)
    assert (pair.cycle_count, pair.z) == (0, 34_576)


def test_the_offset_comes_from_the_datetimes_not_the_mjd() -> None:
    """Give delta exactly, as whole microseconds, not from the float MJDs."""
    raw = DASMeasurement(
        measurement_mjd=60941.250001, measured_phase=1, rms=1, switch="2B07", clock="c1"
    )
    expected = Fraction(
        (raw.measurement_datetime - raw.interpolated_datetime).microseconds, 10**6
    )
    assert measure_pair(raw, None, Fraction(0), None).delta == expected


def test_the_offset_cannot_be_passed_in() -> None:
    """Refuse delta given to a pair measurement: it follows from the measurement."""
    with pytest.raises(ValidationError):
        PairMeasurement.model_validate(
            {"measurement": RAW, "delta": Fraction(1), "cycle_count": 6, "z": 1}
        )


def test_a_slip_correction_moves_whole_periods() -> None:
    """Add k periods to the cycle count and z, and mark the measurement."""
    pair = measure_pair(RAW, PREDICTION, Fraction(0), None)
    corrected = pair.corrected(-2)
    assert (corrected.cycle_count, corrected.z, corrected.slip) == (
        4,
        1_234_577 - 2 * PHASE_PERIOD,
        True,
    )
    assert corrected.measurement == RAW
    assert pair.slip is False


def test_a_pair_measurement_gives_the_filter_its_values() -> None:
    """Give z, the rms and the slip mark as a Measured."""
    pair = measure_pair(RAW, PREDICTION, Fraction(0), None).corrected(1)
    assert pair.measured() == Measured(z=1_234_577 + PHASE_PERIOD, rms=3, slip=True)


def test_a_triple_measurement_comes_from_its_value() -> None:
    """Build the file's triple measurement from the double difference."""
    value = TripleValue(z=6_666_667, sigma=3.3166, components_used="110", cold=True)
    triple = TripleMeasurement.from_value(value)
    assert triple == TripleMeasurement(
        z=6_666_667, double_difference_sigma=3.3166, components_used="110", cold=True
    )
    assert triple.measured() == Measured(z=6_666_667, sigma_dd=3.3166, cold=True)


@pytest.mark.parametrize("used", ["011", "100", "1", "", "111 "])
def test_a_triple_names_only_the_components_it_can_use(used: str) -> None:
    """Refuse components_used other than 111, 110 or 101."""
    with pytest.raises(ValidationError):
        TripleMeasurement.model_validate(
            {
                "z": 1,
                "double_difference_sigma": 1.0,
                "components_used": used,
                "cold": False,
            }
        )


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_a_triple_sigma_is_positive_and_finite(sigma: float) -> None:
    """Refuse a double-difference sigma that is not a positive finite number."""
    with pytest.raises(ValidationError):
        TripleMeasurement(
            z=1, double_difference_sigma=sigma, components_used="111", cold=False
        )
