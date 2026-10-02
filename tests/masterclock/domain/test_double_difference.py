"""Tests for src/masterclock/domain/double_difference.py.

The rules covered: a triple (r, s, c) has a value only when its clock pair
(s, c) is accepted; with both link directions accepted it is
z(s,c) + (z(r,s) - z(s,r)) / 2, with variance sigma_sc**2 plus a quarter of
the links' variances; with one direction accepted, the other is replaced
through the links' predicted round trip, with the variance of the two
pairs used; with neither, or with the round trip needed and a prediction
missing, there is no value; the value is summed exactly and rounded once;
a local triple (r, r, c) needs only (r, c), collapses to z(r, c) exactly and
stops with PhaseError when it does not; the value is marked cold when any
component cold-started; and a constant link offset keeps the value
continuous as the components used change.

The 110 sigma squares the forward rms, and a local triple that does not
collapse says so in full.
"""

import math
from fractions import Fraction
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.domain.double_difference import (
    Component,
    TripleValue,
    double_difference,
)
from masterclock.domain.exceptions import PhaseError

REMOTE: Final = ("mc1", "mc2", "ox23")
"""Appendix A's remote triple."""

LOCAL: Final = ("mc2", "mc2", "ox23")
"""Appendix A's local triple."""

SC: Final = Component(accepted=True, z=1_234_577, rms=3)
"""Appendix A's clock pair (mc2, ox23)."""


def link(
    z: int,
    rms: int = 2,
    *,
    accepted: bool = True,
    predicted: Fraction | None = None,
    cold: bool = False,
) -> Component:
    """Give a link pair's component."""
    return Component(
        accepted=accepted,
        z=z if accepted else None,
        rms=rms if accepted else None,
        predicted_phase=predicted,
        cold_started=cold,
    )


def missing(predicted: Fraction | None = None) -> Component:
    """Give a link direction that was not accepted at the epoch."""
    return Component(accepted=False, predicted_phase=predicted)


# ----------------------------------------------------------- both directions


def test_the_worked_epoch_gives_its_double_difference() -> None:
    """Reproduce Appendix A: dd = 6 666 667 and sigma_dd = 3.31662."""
    value = double_difference(REMOTE, SC, link(5_432_100), link(-5_432_080))
    assert value == TripleValue(
        z=6_666_667,
        sigma=math.sqrt(9 + 0.25 * (4 + 4)),
        components_used="111",
        pair_cold_started=False,
    )
    assert round(value.sigma, 5) == 3.31662


def test_a_half_period_sum_rounds_to_even() -> None:
    """Round the exact sum once, a tie to the even whole number."""
    up = double_difference(REMOTE, SC, link(1), link(0))
    down = double_difference(REMOTE, SC, link(3), link(0))
    assert up is not None
    assert down is not None
    assert (up.z, down.z) == (1_234_578, 1_234_578)


# ----------------------------------------------------------- one direction

PREDICTED_RS: Final = Fraction(54_321_005, 10)
"""A link prediction x-(r,s), ps."""

PREDICTED_SR: Final = Fraction(-54_320_795, 10)
"""A link prediction x-(s,r), ps."""


def test_a_missing_back_direction_uses_the_round_trip() -> None:
    """Give z(s,c) + z(r,s) - rho/2 with sigma_sc**2 + sigma_rs**2 (110)."""
    value = double_difference(
        REMOTE, SC, link(5_432_100, predicted=PREDICTED_RS), missing(PREDICTED_SR)
    )
    rho = PREDICTED_RS + PREDICTED_SR
    assert value == TripleValue(
        z=round(1_234_577 + 5_432_100 - rho / 2),
        sigma=math.sqrt(9 + 4),
        components_used="110",
        pair_cold_started=False,
    )


def test_a_missing_forward_direction_uses_the_round_trip() -> None:
    """Give z(s,c) - z(s,r) + rho/2 with sigma_sc**2 + sigma_sr**2 (101)."""
    value = double_difference(
        REMOTE,
        SC,
        missing(PREDICTED_RS),
        link(-5_432_080, rms=5, predicted=PREDICTED_SR),
    )
    rho = PREDICTED_RS + PREDICTED_SR
    assert value == TripleValue(
        z=round(1_234_577 + 5_432_080 + rho / 2),
        sigma=math.sqrt(9 + 25),
        components_used="101",
        pair_cold_started=False,
    )


@pytest.mark.parametrize(
    ("rs", "sr"),
    [
        (link(5_432_100), missing(PREDICTED_SR)),
        (link(5_432_100, predicted=PREDICTED_RS), missing()),
        (missing(PREDICTED_RS), link(-5_432_080)),
    ],
)
def test_no_round_trip_without_both_predictions(rs: Component, sr: Component) -> None:
    """Give no value when one direction is missing and a prediction is too."""
    assert double_difference(REMOTE, SC, rs, sr) is None


def test_no_value_without_either_direction() -> None:
    """Give no value when neither link direction is accepted."""
    assert (
        double_difference(REMOTE, SC, missing(PREDICTED_RS), missing(PREDICTED_SR))
        is None
    )


def test_no_value_without_the_clock_pair() -> None:
    """Give no value when (s, c) is not accepted."""
    sc = Component(accepted=False, predicted_phase=Fraction(1_234_574))
    assert double_difference(REMOTE, sc, link(5_432_100), link(-5_432_080)) is None


def test_a_constant_link_offset_keeps_the_value_continuous() -> None:
    """Change dd by less than sigma_dd as 111 gives way to 110 and 101 (U18)."""
    x_r, x_s, x_c = 7_000_000, 1_000_000, -234_567
    delay_rs, delay_sr = 431, -117
    z_sc = x_s - x_c
    z_rs = x_r - x_s + delay_rs
    z_sr = x_s - x_r + delay_sr
    predicted_rs = Fraction(z_rs) + Fraction(3, 10)
    predicted_sr = Fraction(z_sr) - Fraction(7, 10)
    sc = Component(accepted=True, z=z_sc, rms=3)
    both = double_difference(
        REMOTE,
        sc,
        link(z_rs, predicted=predicted_rs),
        link(z_sr, predicted=predicted_sr),
    )
    forward = double_difference(
        REMOTE, sc, link(z_rs, predicted=predicted_rs), missing(predicted_sr)
    )
    back = double_difference(
        REMOTE, sc, missing(predicted_rs), link(z_sr, predicted=predicted_sr)
    )
    assert both is not None and forward is not None and back is not None
    assert [value.components_used for value in (both, forward, back)] == [
        "111",
        "110",
        "101",
    ]
    assert abs(forward.z - both.z) < both.sigma
    assert abs(back.z - both.z) < both.sigma


# ---------------------------------------------------------------- local


@pytest.mark.parametrize("z", [1_234_577, -1, 0, 2**62 + 1, -(2**61) - 3])
def test_a_local_triple_is_its_pair_exactly(z: int) -> None:
    """Give dd = z(r,c) and sigma_dd = sigma_rc for (r, r, c) (U19)."""
    sc = Component(accepted=True, z=z, rms=4)
    rr = link(5_432_101, rms=9)
    assert double_difference(LOCAL, sc, rr, rr) == TripleValue(
        z=z, sigma=4.0, components_used="111", pair_cold_started=False
    )


def test_a_local_triple_needs_only_its_pair() -> None:
    """Give the local triple its value with the self pair not accepted."""
    rr = missing()
    value = double_difference(LOCAL, SC, rr, rr)
    assert value == TripleValue(
        z=1_234_577, sigma=3.0, components_used="111", pair_cold_started=False
    )


def test_a_local_triple_that_does_not_collapse_stops_the_run() -> None:
    """Raise PhaseError when the general formula does not give z(r,c) (U19)."""
    with pytest.raises(PhaseError, match="does not collapse"):
        double_difference(LOCAL, SC, link(5_432_102), link(5_432_100))


# ------------------------------------------------------------------- cold


@pytest.mark.parametrize("which", ["sc", "rs", "sr"])
def test_a_component_cold_start_marks_the_value_cold(which: str) -> None:
    """Mark the value cold when any of its pairs cold-started (12.6)."""
    sc = Component(accepted=True, z=1_234_577, rms=3, cold_started=which == "sc")
    rs = link(5_432_100, cold=which == "rs")
    sr = link(-5_432_080, cold=which == "sr")
    value = double_difference(REMOTE, sc, rs, sr)
    assert value is not None
    assert value.pair_cold_started is True


def test_a_local_triple_is_cold_when_its_pair_is() -> None:
    """Mark a local triple cold when its pair cold-started."""
    sc = Component(accepted=True, z=1_234_577, rms=3, cold_started=True)
    value = double_difference(LOCAL, sc, missing(), missing())
    assert value is not None
    assert value.pair_cold_started is True


# --------------------------------------------------------------- the models


@pytest.mark.parametrize(
    "values",
    [
        {"accepted": True, "z": 1},
        {"accepted": True, "rms": 3},
        {"accepted": False, "z": 1, "rms": 3},
        {"accepted": True, "z": 1, "rms": -1},
        {"accepted": True, "z": 1.0, "rms": 3},
    ],
)
def test_a_component_is_accepted_with_a_value_or_not_without_one(
    values: dict[str, object],
) -> None:
    """Refuse a component whose z and rms do not match whether it was accepted."""
    with pytest.raises(ValidationError):
        Component.model_validate(values)


def test_a_value_names_the_components_it_used() -> None:
    """Refuse components_used other than 111, 110 or 101."""
    with pytest.raises(ValidationError):
        TripleValue.model_validate(
            {"z": 1, "sigma": 1.0, "components_used": "011", "pair_cold_started": False}
        )


def test_a_missing_back_direction_squares_the_forward_rms() -> None:
    """Combine the clock pair's and the forward link's rms in quadrature (110)."""
    value = double_difference(
        REMOTE,
        SC,
        link(5_432_100, rms=5, predicted=PREDICTED_RS),
        missing(PREDICTED_SR),
    )
    assert value is not None
    assert value.sigma == math.sqrt(9 + 25)


def test_a_local_triple_that_does_not_collapse_says_so_in_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Name the triple and both values, and log it at ERROR as raised (U19)."""
    with pytest.raises(PhaseError) as raised:
        double_difference(LOCAL, SC, link(5_432_106), link(5_432_100))
    assert str(raised.value) == (
        "local triple ('mc2', 'mc2', 'ox23') does not collapse to its pair:"
        " 1234580 != 1234577"
    )
    assert [r.getMessage() for r in caplog.records] == [str(raised.value)]
