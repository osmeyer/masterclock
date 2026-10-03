"""Tests for src/masterclock/domain/measurements.py.

The rules covered: a pair measurement is a reading decycled and referred
to its epoch start, against the prediction or, without one, the anchor; it
holds the plain values it was made from, and its measurement time, epoch
start and offset after the epoch start are worked out from the MJD, the
offset exactly from the datetimes, and none of them can be passed in; an
epoch starts at midnight and every epoch after; a slip correction moves the
cycle count and z by whole periods and marks it; a pair or triple
measurement gives the filter step its plain values; and a triple's sigma
may be zero.

The ranges of a measurement's values are checked where a measurement is
read back from a file, and tested there (das_processor/test_files.py).
"""

from datetime import UTC, datetime
from typing import Final

import pytest
from gmpy2 import mpq

from masterclock.app.timeutil import mjd_to_datetime
from masterclock.domain.double_difference import TripleValue
from masterclock.domain.filter import FilterInput
from masterclock.domain.measurements import (
    PairMeasurement,
    TripleMeasurement,
    epoch_start,
    measure_pair,
)
from masterclock.domain.phase import PHASE_PERIOD, exact
from masterclock.domain.series import State

APPENDIX_A_READING: Final[dict[str, float | int]] = {
    "measurement_mjd": 60941.251588,
    "measured_phase": 34579,
    "rms": 3,
}
"""Appendix A's reading: its MJD, phase and rms."""

PREDICTION: Final = State(x=1_234_567 + exact(0.0123) * 600, y=0.0123)
"""Appendix A's prediction at the epoch start."""


def measure_appendix_a_reading(
    prediction: State | None, w: mpq, anchor: int | None
) -> PairMeasurement:
    """Decycle Appendix A's reading."""
    return measure_pair(
        measurement_mjd=60941.251588,
        measured_phase=34579,
        rms=3,
        prediction=prediction,
        w=w,
        anchor=anchor,
    )


def test_the_worked_epoch_gives_its_pair_measurement() -> None:
    """Reproduce Appendix A: delta 137.2032 s, n = 6, z_E = 1 234 577."""
    pair_measurement = measure_appendix_a_reading(PREDICTION, mpq(0), None)
    assert pair_measurement.delta == mpq(1_372_032, 10_000)
    assert (
        pair_measurement.cycle_count,
        pair_measurement.z,
        pair_measurement.slip,
    ) == (6, 1_234_577, False)
    assert (
        pair_measurement.measurement_mjd,
        pair_measurement.measured_phase,
        pair_measurement.rms,
    ) == (
        60941.251588,
        34579,
        3,
    )


def test_a_pair_without_a_prediction_is_decycled_against_its_anchor() -> None:
    """Decycle against the last buffered measurement when there is no prediction."""
    pair_measurement = measure_appendix_a_reading(None, mpq(0), 1_234_000)
    assert (pair_measurement.cycle_count, pair_measurement.z) == (
        6,
        34_579 + 6 * PHASE_PERIOD,
    )


def test_steering_in_the_epoch_is_taken_off() -> None:
    """Refer the measurement to E with the steering since E taken off."""
    pair_measurement = measure_appendix_a_reading(None, mpq(7, 2), None)
    assert (pair_measurement.cycle_count, pair_measurement.z) == (0, 34_576)


def test_the_offset_comes_from_the_datetimes_not_the_mjd() -> None:
    """Give delta exactly, as whole microseconds, not from the float MJDs."""
    measured_instant = mjd_to_datetime(60941.250001)
    expected_delta = mpq(
        (measured_instant - epoch_start(measured_instant)).microseconds, 10**6
    )
    pair_measurement = measure_pair(
        measurement_mjd=60941.250001,
        measured_phase=1,
        rms=1,
        prediction=None,
        w=mpq(0),
        anchor=None,
    )
    assert pair_measurement.delta == expected_delta


@pytest.mark.parametrize(
    "derived_field", ["delta", "measurement_datetime", "interpolated_datetime"]
)
def test_what_follows_from_the_mjd_cannot_be_passed_in(derived_field: str) -> None:
    """Refuse a value given that the measurement works out from its MJD."""
    with pytest.raises(TypeError, match=derived_field):
        PairMeasurement(
            **{  # type: ignore[arg-type]
                **APPENDIX_A_READING,
                derived_field: mpq(1),
                "cycle_count": 6,
                "z": 1,
            }
        )


def test_a_measurement_knows_its_time_and_epoch() -> None:
    """Work out the measurement time and its epoch start from the MJD."""
    pair_measurement = measure_appendix_a_reading(PREDICTION, mpq(0), None)
    assert pair_measurement.measurement_datetime == mjd_to_datetime(60941.251588)
    assert pair_measurement.interpolated_datetime == mjd_to_datetime(60941.25)


@pytest.mark.parametrize(
    ("hour", "minute", "second", "start_minute"),
    [(0, 0, 0, 0), (6, 2, 17, 0), (6, 9, 59, 0), (6, 10, 0, 10), (23, 59, 59, 50)],
)
def test_an_epoch_starts_on_its_ten_minute_mark(
    hour: int, minute: int, second: int, start_minute: int
) -> None:
    """Give the epoch start: midnight and every 600 s after."""
    instant = datetime(2025, 9, 23, hour, minute, second, 1, tzinfo=UTC)
    assert epoch_start(instant) == datetime(2025, 9, 23, hour, start_minute, tzinfo=UTC)


def test_a_slip_correction_moves_whole_periods() -> None:
    """Add k periods to the cycle count and z, and mark the measurement."""
    pair_measurement = measure_appendix_a_reading(PREDICTION, mpq(0), None)
    corrected_measurement = pair_measurement.corrected(-2)
    assert (
        corrected_measurement.cycle_count,
        corrected_measurement.z,
        corrected_measurement.slip,
    ) == (
        4,
        1_234_577 - 2 * PHASE_PERIOD,
        True,
    )
    assert corrected_measurement.measurement_mjd == pair_measurement.measurement_mjd
    assert pair_measurement.slip is False


def test_a_pair_measurement_gives_the_filter_its_values() -> None:
    """Give z, the rms and the slip mark as a FilterInput."""
    pair_measurement = measure_appendix_a_reading(PREDICTION, mpq(0), None).corrected(1)
    assert pair_measurement.filter_input() == FilterInput(
        z=1_234_577 + PHASE_PERIOD, rms=3, slip=True
    )


def test_a_triple_measurement_comes_from_its_value() -> None:
    """Build the file's triple measurement from the double difference."""
    triple_value = TripleValue(
        z=6_666_667, sigma=3.3166, components_used="110", pair_cold_started=True
    )
    triple_measurement = TripleMeasurement.from_triple_value(triple_value)
    assert triple_measurement == TripleMeasurement(
        z=6_666_667,
        double_difference_sigma=3.3166,
        components_used="110",
        pair_cold_started=True,
    )
    assert triple_measurement.filter_input() == FilterInput(
        z=6_666_667, sigma_dd=3.3166, pair_cold_started=True
    )


def test_a_triple_sigma_may_be_zero() -> None:
    """Take a sigma of 0, as a pair's rms of 0 is taken, as the scale's floor."""
    triple_measurement = TripleMeasurement(
        z=1, double_difference_sigma=0.0, components_used="111", pair_cold_started=False
    )
    assert triple_measurement.filter_input().scale_floor == 0.0
