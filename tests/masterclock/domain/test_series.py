"""Tests for src/masterclock/domain/series.py.

The rules covered: an estimator state holds its phase exactly and finite
rate and drift; a series' settings hold a model of one, two or three states,
a time constant exactly when the model has more than one state, and limits
in their ranges; a row's flags are letters of ARXPDSNU in that order, with
exactly one outcome, a dormant row never accepted and an unsettled row
never dormant or one-state; a dormant row has no state and every other row
a whole one, its phase in whole femtoseconds, with no rate or drift where
its model has none; at most three rejects are held, oldest first; a time
constant is held exactly when the model has more than one state; every
float is finite; a row gives its state unless it is dormant; every model is
frozen and strict and refuses unknown fields; and build_row builds a row
from its fields, and replace a changed row, checking it again.

Each row refusal says why and is logged as raised.
"""

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.domain import series
from masterclock.domain.exceptions import FilterError

MARK: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented ten-minute mark."""

STEP: Final = timedelta(minutes=10)
"""One epoch."""


def row(**changes: object) -> series.Row:
    """Build a valid accepted 3-state row, with ``changes`` applied."""
    values: dict[str, object] = {
        "interpolated_datetime": MARK,
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
    values.update(changes)
    return series.Row.model_validate(values)


DORMANT: Final[dict[str, object]] = {
    "x_fs": None,
    "y": None,
    "d": None,
    "innovation_scale": None,
}
"""The changes that make a row dormant, apart from its flags."""

ONE_STATE: Final[dict[str, object]] = {
    "y": 0.0,
    "d": 0.0,
    "filter_states": 1,
    "time_constant": None,
}
"""The changes that make a row one of a 1-state series."""


# ------------------------------------------------------------------- State


def test_a_state_holds_its_phase_exactly() -> None:
    """Keep x as the Fraction given, and d as 0.0 when it is not given."""
    state = series.State(x=Fraction(1, 3), y=0.5)
    assert state.x == Fraction(1, 3)
    assert state.d == 0.0


@pytest.mark.parametrize("x", [0.5, 2, "1/3"])
def test_a_state_refuses_a_phase_that_is_not_a_fraction(x: object) -> None:
    """Refuse a float, an int or text for x, so no sum is made inexact."""
    with pytest.raises(ValidationError):
        series.State.model_validate({"x": x, "y": 0.0})


@pytest.mark.parametrize("field", ["y", "d"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_state_refuses_a_rate_or_drift_that_is_not_finite(
    field: str, value: float
) -> None:
    """Raise FilterError for a rate or drift that no estimator can carry."""
    values: dict[str, object] = {"x": Fraction(0), "y": 0.0, field: value}
    with pytest.raises(FilterError, match=rf"{field} .* not finite"):
        series.State.model_validate(values)


# ------------------------------------------------------------- SeriesParams


def params(**changes: object) -> series.SeriesParams:
    """Build valid 3-state settings, with ``changes`` applied."""
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
    return series.SeriesParams.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"model": 2, "M": 30.0},
        {"model": 1, "M": None},
        {"rms_max": None},
        {"n_break": 3, "gmax": 3},
        {"M": 1.0, "M_sigma": 1.0, "gmax": 3, "n_break": 3},
    ],
)
def test_valid_settings_build(changes: dict[str, object]) -> None:
    """Build settings for each model, with and without an RMS limit."""
    assert params(**changes).model == changes.get("model", 3)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"model": 4}, "model"),
        ({"model": 0}, "model"),
        ({"model": 1}, "time constant"),
        ({"M": None}, "time constant"),
        ({"M": 0.5}, "M"),
        ({"M_sigma": 0.5}, "M_sigma"),
        ({"sigma0": 0.0}, "sigma0"),
        ({"sigma0": -1.0}, "sigma0"),
        ({"gmax": 0}, "gmax"),
        ({"n_break": 2}, "n_break"),
        ({"n_break": 433}, "n_break"),
        ({"rms_max": 0}, "rms_max"),
        ({"M": float("inf")}, "M"),
        ({"model": True}, "not a bool"),
    ],
)
def test_invalid_settings_are_refused(changes: dict[str, object], reason: str) -> None:
    """Refuse each value outside its range, naming what is wrong."""
    with pytest.raises(ValidationError, match=reason):
        params(**changes)


# --------------------------------------------------------------------- Row


@pytest.mark.parametrize(
    ("flags", "changes"),
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
        ("PD", {"innovation": None, **DORMANT}),
        ("RD", DORMANT),
        ("XD", DORMANT),
        ("PDN", {"innovation": None, **DORMANT}),
        ("A", ONE_STATE),
        ("R", ONE_STATE),
        ("PD", {"innovation": None, **ONE_STATE, **DORMANT}),
        ("A", {"filter_states": 2, "d": 0.0, "time_constant": 30.0}),
    ],
)
def test_a_row_of_each_flag_pattern_builds(
    flags: str, changes: dict[str, object]
) -> None:
    """Build a row for every outcome the measurement table allows."""
    assert row(flags=flags, **changes).flags == flags


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
    changes = DORMANT if "D" in flags else {}
    with pytest.raises(ValidationError, match="flags"):
        row(flags=flags, **changes)


def test_an_unsettled_row_is_never_dormant() -> None:
    """Refuse U on a dormant row, which has no segment to settle."""
    with pytest.raises(ValidationError, match="flags"):
        row(flags="PDU", innovation=None, **DORMANT)


def test_a_one_state_row_is_never_unsettled() -> None:
    """Refuse U on a 1-state row, which has nothing to settle."""
    with pytest.raises(ValidationError, match=r"U .* 1-state"):
        row(flags="AU", **ONE_STATE)


@pytest.mark.parametrize("field", ["x_fs", "y", "d", "innovation_scale"])
def test_a_dormant_row_has_no_state(field: str) -> None:
    """Refuse a dormant row that holds any part of a state."""
    values = {**DORMANT, field: 1 if field == "x_fs" else 1.0}
    with pytest.raises(ValidationError, match="dormant"):
        row(flags="RD", **values)


@pytest.mark.parametrize("field", ["x_fs", "y", "d", "innovation_scale"])
def test_a_row_that_is_not_dormant_has_a_whole_state(field: str) -> None:
    """Refuse a row without D that lacks any part of its state."""
    with pytest.raises(ValidationError, match="dormant"):
        row(**{field: None})


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"filter_states": 2, "time_constant": 30.0}, "d .* 2-state"),
        ({**ONE_STATE, "y": 0.5}, "y .* 1-state"),
        ({**ONE_STATE, "d": 1e-12}, "d .* 1-state"),
    ],
)
def test_a_row_has_no_rate_or_drift_its_model_lacks(
    changes: dict[str, object], reason: str
) -> None:
    """Refuse a drift on a 2-state row, and a rate or drift on a 1-state row."""
    with pytest.raises(ValidationError, match=reason):
        row(**changes)


def test_a_row_without_a_measurement_has_no_innovation() -> None:
    """Refuse an innovation on a predicted row, which had nothing to compare."""
    with pytest.raises(ValidationError, match="no innovation"):
        row(flags="P")


def test_at_most_three_rejects_are_held() -> None:
    """Refuse a fourth entry in the reject buffer."""
    rejects = tuple((MARK - n * STEP, 10.0) for n in (3, 2, 1, 0))
    assert len(row(rejects=rejects[1:]).rejects) == 3
    with pytest.raises(ValidationError, match="three"):
        row(rejects=rejects)


@pytest.mark.parametrize("order", [(1, 2), (1, 1)])
def test_rejects_are_held_oldest_first(order: tuple[int, int]) -> None:
    """Refuse a buffer whose epochs do not rise from first to last."""
    rejects = tuple((MARK - n * STEP, 10.0) for n in order)
    with pytest.raises(ValidationError, match="oldest first"):
        row(rejects=rejects)


@pytest.mark.parametrize(
    "changes",
    [
        {"time_constant": None},
        {**ONE_STATE, "time_constant": 100.0},
    ],
)
def test_a_time_constant_is_held_exactly_when_the_model_has_more_states(
    changes: dict[str, object],
) -> None:
    """Refuse a 3-state row without M, and a 1-state row with one."""
    with pytest.raises(ValidationError, match="time constant"):
        row(**changes)


@pytest.mark.parametrize(
    "field",
    [
        "innovation",
        "y",
        "d",
        "innovation_scale",
        "time_constant",
        "scale_time_constant",
    ],
)
def test_every_float_of_a_row_is_finite(field: str) -> None:
    """Raise FilterError for a float that no estimator value can be."""
    with pytest.raises(FilterError, match=rf"{field} .* not finite"):
        row(**{field: float("nan")})


def test_every_reject_value_is_finite() -> None:
    """Raise FilterError for a buffered value that is not finite."""
    with pytest.raises(FilterError, match=r"rejects .* not finite"):
        row(rejects=((MARK, float("inf")),))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("segment", -1),
        ("epochs_in_segment", -1),
        ("epochs_since_accept", -1),
        ("consecutive_rejects", -1),
        ("filter_states", 4),
        ("x_fs", 1.5),
        ("segment", True),
        ("filter_states", True),
        ("interpolated_datetime", MARK.replace(tzinfo=None)),
    ],
)
def test_a_row_refuses_values_of_the_wrong_kind(field: str, value: object) -> None:
    """Refuse negative counts, unknown models, float phases and naive marks."""
    with pytest.raises(ValidationError, match=field):
        row(**{field: value})


@pytest.mark.parametrize("model", [series.State, series.SeriesParams, series.Row])
def test_every_model_refuses_unknown_fields(model: type[object]) -> None:
    """Refuse a field the model does not declare."""
    assert isinstance(model, type)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({"colour": "red"})  # type: ignore[attr-defined]


def test_every_model_is_frozen() -> None:
    """Refuse any change to a built state, settings or row."""
    with pytest.raises(ValidationError, match="frozen"):
        series.State(x=Fraction(0), y=0.0).y = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        params().M = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        row().x_fs = 0  # type: ignore[misc]


def test_a_row_gives_its_state() -> None:
    """Give x, y and d of a row that is not dormant."""
    assert row().known_state() == (
        1_234_574_457,
        0.01230129052352643,
        7.169515400974333e-12,
    )


def test_a_dormant_row_gives_no_state() -> None:
    """Raise FilterError when asked for the state of a dormant row."""
    with pytest.raises(FilterError, match="holds no state"):
        row(flags="RD", **DORMANT).known_state()


# ----------------------------------------------------------------- replace


def test_replace_builds_a_changed_row() -> None:
    """Give a new row with the changes made and every other field kept."""
    original = row()
    changed = series.replace(original, flags="R", consecutive_rejects=1)
    assert (changed.flags, changed.consecutive_rejects) == ("R", 1)
    assert changed.model_dump(exclude={"flags", "consecutive_rejects"}) == (
        original.model_dump(exclude={"flags", "consecutive_rejects"})
    )
    assert original.flags == "A"


@pytest.mark.parametrize(
    "changes",
    [
        {"flags": "AR"},
        {"x_fs": None},
        {"colour": "red"},
        {"segment": -1},
    ],
)
def test_replace_checks_the_changed_row_again(changes: dict[str, object]) -> None:
    """Raise FilterError for a change that makes the row invalid."""
    with pytest.raises(FilterError, match="invalid row"):
        series.replace(row(), **changes)


# --------------------------------------------------------------- build_row


def test_build_row_builds_a_row_from_its_fields() -> None:
    """Give the row whose fields are the values given."""
    assert series.build_row(dict(row())) == row()


def test_build_row_checks_the_row() -> None:
    """Raise FilterError for fields that make no valid row."""
    with pytest.raises(FilterError, match="invalid row"):
        series.build_row({**dict(row()), "flags": "AR"})


def test_every_row_error_says_why_and_is_logged_as_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Give each refusal's reason in words, and log the FilterError as raised."""
    with pytest.raises(FilterError) as raised:
        series.build_row({**dict(row()), "flags": "AR"})
    assert str(raised.value).startswith("invalid row: ")
    assert "flags" in str(raised.value)
    assert [r.getMessage() for r in caplog.records] == [str(raised.value)]
    for call in (
        lambda: row(flags="RD", **DORMANT).known_state(),
        lambda: row(y=float("inf")),
    ):
        caplog.clear()
        with pytest.raises(FilterError) as raised:
            call()
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert errors == [str(raised.value)]


def test_a_one_state_row_with_u_is_refused_in_words() -> None:
    """Say that U is never carried by a 1-state row."""
    with pytest.raises(ValidationError) as raised:
        row(flags="AU", **ONE_STATE)
    (error,) = raised.value.errors()
    assert error["msg"] == "Value error, U is never carried by a 1-state row"
