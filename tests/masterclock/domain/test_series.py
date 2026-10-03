"""Tests for src/masterclock/domain/series.py.

The rules covered: an estimator state holds its phase exactly; a series'
settings hold what they are given; a row's flags are letters of ARXPDSNU in
that order, with exactly one outcome, a dormant row never accepted and an
unsettled row never dormant or one-state; a dormant row has no state and
every other row a whole one, with no rate or drift where its model has
none; at most three rejects are held, oldest first; a time constant is held
exactly when the model has more than one state; every float is finite and
no counter below 0; a row gives its state unless it is dormant; and every
type is frozen.

A row is built unchecked, and check_row says why it refuses one, logging
nothing.
"""

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Final

import pytest
from gmpy2 import mpq

from masterclock.domain import series
from masterclock.domain.exceptions import FilterError

EPOCH_START: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented ten-minute mark."""

EPOCH_LENGTH: Final = timedelta(minutes=10)
"""One epoch."""


def make_row(**field_changes: object) -> series.Row:
    """Build an accepted 3-state row, unchecked, with ``field_changes`` applied."""
    row_fields: dict[str, object] = {
        "interpolated_datetime": EPOCH_START,
        "innovation": 2.62,
        "x_fs": 1_234_574_457,
        "y": 0.01230129052352643,
        "d": 7.169515400974333e-12,
        "innovation_scale": 3.0,
        "segment": 4,
        "step_offset": 0,
        "epochs_in_segment": 812,
        "epochs_since_accept": 0,
        "consecutive_rejects": 0,
        "rejects": (),
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "flags": "A",
    }
    row_fields.update(field_changes)
    return series.Row(**row_fields)  # type: ignore[arg-type]


def checked_row(**field_changes: object) -> series.Row:
    """Build a row as :func:`make_row` does, and check it."""
    row = make_row(**field_changes)
    series.check_row(row)
    return row


DORMANT_FIELDS: Final[dict[str, object]] = {
    "x_fs": None,
    "y": None,
    "d": None,
    "innovation_scale": None,
}
"""The changes that make a row dormant, apart from its flags."""

ONE_STATE_FIELDS: Final[dict[str, object]] = {
    "y": 0.0,
    "d": 0.0,
    "filter_states": 1,
    "time_constant": None,
}
"""The changes that make a row one of a 1-state series."""


# ------------------------------------------------------------------- State


def test_a_state_holds_its_phase_exactly() -> None:
    """Keep x as the mpq given, and d as 0.0 when it is not given."""
    built_state = series.State(x=mpq(1, 3), y=0.5)
    assert built_state.x == mpq(1, 3)
    assert built_state.d == 0.0


# ------------------------------------------------------------- SeriesParams


def make_series_params(**field_changes: object) -> series.SeriesParams:
    """Build 3-state settings, with ``field_changes`` applied."""
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
    return series.SeriesParams(**settings_fields)  # type: ignore[arg-type]


def test_settings_hold_what_they_are_given() -> None:
    """Keep every value; the clock configuration checked them when it was read."""
    assert make_series_params(filter_states=1, M=None, rms_max=None) == (
        series.SeriesParams(
            filter_states=1,
            M=None,
            M_sigma=50.0,
            sigma0=5.0,
            gmax=432,
            n_break=36,
            rms_max=None,
        )
    )


# --------------------------------------------------------------------- Row


@pytest.mark.parametrize(
    ("flags", "field_changes"),
    [
        ("A", {}),
        ("AU", {"epochs_in_segment": 0}),
        ("ANU", {"epochs_in_segment": 0}),
        ("AS", {}),
        ("ASNU", {"epochs_in_segment": 0}),
        ("R", {}),
        ("RS", {}),
        ("X", {}),
        ("XS", {}),
        ("P", {"innovation": None}),
        ("PNU", {"innovation": None}),
        ("PD", {"innovation": None, **DORMANT_FIELDS}),
        ("RD", DORMANT_FIELDS),
        ("XD", DORMANT_FIELDS),
        ("PDN", {"innovation": None, **DORMANT_FIELDS}),
        ("A", ONE_STATE_FIELDS),
        ("R", ONE_STATE_FIELDS),
        ("PD", {"innovation": None, **ONE_STATE_FIELDS, **DORMANT_FIELDS}),
        ("A", {"filter_states": 2, "d": 0.0, "time_constant": 30.0}),
    ],
)
def test_a_row_of_each_flag_pattern_builds(
    flags: str, field_changes: dict[str, object]
) -> None:
    """Pass every outcome the measurement table allows."""
    assert checked_row(flags=flags, **field_changes).flags == flags


@pytest.mark.parametrize(
    "flags",
    [
        "",
        "N",
        "AR",
        "AX",
        "RP",
        "AA",
        "UA",
        "NA",
        "AZ",
        "a",
        "AD",
    ],
)
def test_flags_need_one_outcome_in_order(flags: str) -> None:
    """Refuse flags without exactly one of A R X P, out of order, or unknown."""
    field_changes = DORMANT_FIELDS if "D" in flags else {}
    with pytest.raises(ValueError, match="flags"):
        checked_row(flags=flags, **field_changes)


def test_an_unsettled_row_is_never_dormant() -> None:
    """Refuse U on a dormant row, which has no segment to settle."""
    with pytest.raises(ValueError, match="flags"):
        checked_row(flags="PDU", innovation=None, **DORMANT_FIELDS)


def test_a_one_state_row_is_never_unsettled() -> None:
    """Refuse U on a 1-state row, which has nothing to settle."""
    with pytest.raises(ValueError, match=r"U .* 1-state"):
        checked_row(flags="AU", **ONE_STATE_FIELDS)


@pytest.mark.parametrize("state_field", ["x_fs", "y", "d", "innovation_scale"])
def test_a_dormant_row_has_no_state(state_field: str) -> None:
    """Refuse a dormant row that holds any part of a state."""
    row_fields = {**DORMANT_FIELDS, state_field: 1 if state_field == "x_fs" else 1.0}
    with pytest.raises(ValueError, match="dormant"):
        checked_row(flags="RD", **row_fields)


@pytest.mark.parametrize("state_field", ["x_fs", "y", "d", "innovation_scale"])
def test_a_row_that_is_not_dormant_has_a_whole_state(state_field: str) -> None:
    """Refuse a row without D that lacks any part of its state."""
    with pytest.raises(ValueError, match="dormant"):
        checked_row(**{state_field: None})


@pytest.mark.parametrize(
    ("field_changes", "refusal_reason"),
    [
        ({"filter_states": 2, "time_constant": 30.0}, "d .* 2-state"),
        ({**ONE_STATE_FIELDS, "y": 0.5}, "y .* 1-state"),
        ({**ONE_STATE_FIELDS, "d": 1e-12}, "d .* 1-state"),
    ],
)
def test_a_row_has_no_rate_or_drift_its_model_lacks(
    field_changes: dict[str, object], refusal_reason: str
) -> None:
    """Refuse a drift on a 2-state row, and a rate or drift on a 1-state row."""
    with pytest.raises(ValueError, match=refusal_reason):
        checked_row(**field_changes)


def test_a_row_without_a_measurement_has_no_innovation() -> None:
    """Refuse an innovation on a predicted row, which had nothing to compare."""
    with pytest.raises(ValueError, match="no innovation"):
        checked_row(flags="P")


def test_at_most_three_rejects_are_held() -> None:
    """Refuse a fourth entry in the reject buffer."""
    reject_buffer = tuple(
        (EPOCH_START - epochs_back * EPOCH_LENGTH, 10.0) for epochs_back in (3, 2, 1, 0)
    )
    assert len(checked_row(rejects=reject_buffer[1:]).rejects) == 3
    with pytest.raises(ValueError, match="three"):
        checked_row(rejects=reject_buffer)


@pytest.mark.parametrize("epochs_back_order", [(1, 2), (1, 1)])
def test_rejects_are_held_oldest_first(epochs_back_order: tuple[int, int]) -> None:
    """Refuse a buffer whose epochs do not rise from first to last."""
    reject_buffer = tuple(
        (EPOCH_START - epochs_back * EPOCH_LENGTH, 10.0)
        for epochs_back in epochs_back_order
    )
    with pytest.raises(ValueError, match="oldest first"):
        checked_row(rejects=reject_buffer)


@pytest.mark.parametrize(
    "field_changes",
    [
        {"time_constant": None},
        {**ONE_STATE_FIELDS, "time_constant": 100.0},
    ],
)
def test_a_time_constant_is_held_exactly_when_the_model_has_more_states(
    field_changes: dict[str, object],
) -> None:
    """Refuse a 3-state row without M, and a 1-state row with one."""
    with pytest.raises(ValueError, match="time constant"):
        checked_row(**field_changes)


@pytest.mark.parametrize(
    "float_field",
    [
        "innovation",
        "y",
        "d",
        "innovation_scale",
        "time_constant",
        "scale_time_constant",
    ],
)
def test_every_float_of_a_row_is_finite(float_field: str) -> None:
    """Refuse a float that no estimator value can be."""
    with pytest.raises(ValueError, match=rf"{float_field} .* not finite"):
        checked_row(**{float_field: float("nan")})


def test_every_reject_value_is_finite() -> None:
    """Refuse a buffered value that is not finite."""
    with pytest.raises(ValueError, match=r"rejects .* not finite"):
        checked_row(rejects=((EPOCH_START, float("inf")),))


@pytest.mark.parametrize(
    "counter_field",
    ["segment", "epochs_in_segment", "epochs_since_accept", "consecutive_rejects"],
)
def test_no_counter_of_a_row_is_below_zero(counter_field: str) -> None:
    """Refuse a segment number or counter below 0."""
    with pytest.raises(ValueError, match=rf"{counter_field} -1 is below 0"):
        checked_row(**{counter_field: -1})


def test_every_type_is_frozen() -> None:
    """Refuse any change to a built state, settings or row."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        series.State(x=mpq(0), y=0.0).y = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_series_params().M = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_row().x_fs = 0  # type: ignore[misc]


def test_a_row_gives_its_state() -> None:
    """Give x, y and d of a row that is not dormant."""
    assert make_row().known_state() == (
        1_234_574_457,
        0.01230129052352643,
        7.169515400974333e-12,
    )


def test_a_dormant_row_gives_no_state() -> None:
    """Raise FilterError when asked for the state of a dormant row."""
    with pytest.raises(FilterError, match="holds no state"):
        make_row(flags="RD", **DORMANT_FIELDS).known_state()


def test_a_row_refusal_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Log nothing for a refused row: the caller says what the row was for."""
    with pytest.raises(ValueError, match="flags"):
        checked_row(flags="AR")
    assert caplog.records == []


def test_a_dormant_row_asked_for_its_state_is_logged_as_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log the FilterError for a dormant row's state as it is raised."""
    with pytest.raises(FilterError) as raised_error:
        make_row(flags="RD", **DORMANT_FIELDS).known_state()
    error_messages = [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "ERROR"
    ]
    assert error_messages == [str(raised_error.value)]


def test_a_one_state_row_with_u_is_refused_in_words() -> None:
    """Say that U is never carried by a 1-state row."""
    with pytest.raises(ValueError, match=r"^U is never carried by a 1-state row$"):
        checked_row(flags="AU", **ONE_STATE_FIELDS)
