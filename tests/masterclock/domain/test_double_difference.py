"""Tests for src/masterclock/domain/double_difference.py.

The rules covered: a triple (r, s, c) has a value only when its clock pair
(s, c) is accepted; with both link directions accepted it is z(s,c) +
(z(r,s) - z(s,r)) / 2, with variance sigma_sc**2 plus a quarter of the
links' variances; with one direction accepted, the other is replaced through
the links' predicted round trip, with the variance of the two pairs used;
with neither, or with the round trip needed and a prediction missing, there
is no value; the value is summed exactly and rounded once; a local triple
(r, r, c) needs only (r, c), collapses to z(r, c) exactly and stops with
PhaseError when it does not; a pair whose value the triple uses restarted
when any of its three pairs cold-started, or, for a local triple, its pair
(r, c), the self pair cancelling, whether or not the triple has a value, and
the value is then marked cold; a constant link offset keeps the value
continuous as the components used change; and a component has its z and rms
exactly when it was accepted, and is refused otherwise, logged as raised.

The 110 sigma combines the clock pair's and the forward link's rms in
quadrature, and a local triple that does not collapse says so in full.
"""

import math
from typing import Final

import pytest
from gmpy2 import mpq

from masterclock.domain.double_difference import (
    Component,
    TripleValue,
    double_difference,
    pair_restarted,
)
from masterclock.domain.exceptions import PhaseError
from masterclock.domain.phase import round_even

REMOTE_TRIPLE: Final = ("mc1", "mc2", "ox23")
"""Appendix A's remote triple."""

LOCAL_TRIPLE: Final = ("mc2", "mc2", "ox23")
"""Appendix A's local triple."""

SC: Final = Component(accepted=True, z=1_234_577, rms=3)
"""Appendix A's clock pair (mc2, ox23)."""


def link_component(
    z: int,
    rms: int = 2,
    *,
    accepted: bool = True,
    predicted_phase: mpq | None = None,
    cold_started: bool = False,
) -> Component:
    """Give a link pair's component."""
    return Component(
        accepted=accepted,
        z=z if accepted else None,
        rms=rms if accepted else None,
        predicted_phase=predicted_phase,
        cold_started=cold_started,
    )


def unaccepted_link(predicted_phase: mpq | None = None) -> Component:
    """Give a link direction that was not accepted at the epoch."""
    return Component(accepted=False, predicted_phase=predicted_phase)


# ----------------------------------------------------------- both directions


def test_the_worked_epoch_gives_its_double_difference() -> None:
    """Reproduce Appendix A: dd = 6 666 667 and sigma_dd = 3.31662."""
    triple_value = double_difference(
        REMOTE_TRIPLE, SC, link_component(5_432_100), link_component(-5_432_080)
    )
    assert triple_value == TripleValue(
        z=6_666_667,
        sigma=math.sqrt(9 + 0.25 * (4 + 4)),
        components_used="111",
        pair_cold_started=False,
    )
    assert round(triple_value.sigma, 5) == 3.31662


def test_a_half_period_sum_rounds_to_even() -> None:
    """Round the exact sum once, a tie to the even whole number."""
    rounded_up = double_difference(
        REMOTE_TRIPLE, SC, link_component(1), link_component(0)
    )
    rounded_down = double_difference(
        REMOTE_TRIPLE, SC, link_component(3), link_component(0)
    )
    assert rounded_up is not None
    assert rounded_down is not None
    assert (rounded_up.z, rounded_down.z) == (1_234_578, 1_234_578)


# ----------------------------------------------------------- one direction

PREDICTED_RS: Final = mpq(54_321_005, 10)
"""A link prediction x-(r,s), ps."""

PREDICTED_SR: Final = mpq(-54_320_795, 10)
"""A link prediction x-(s,r), ps."""


def test_a_missing_back_direction_uses_the_round_trip() -> None:
    """Give z(s,c) + z(r,s) - rho/2 with sigma_sc**2 + sigma_rs**2 (110)."""
    triple_value = double_difference(
        REMOTE_TRIPLE,
        SC,
        link_component(5_432_100, predicted_phase=PREDICTED_RS),
        unaccepted_link(PREDICTED_SR),
    )
    rho = PREDICTED_RS + PREDICTED_SR
    assert triple_value == TripleValue(
        z=round_even(1_234_577 + 5_432_100 - rho / 2),
        sigma=math.sqrt(9 + 4),
        components_used="110",
        pair_cold_started=False,
    )


def test_a_missing_forward_direction_uses_the_round_trip() -> None:
    """Give z(s,c) - z(s,r) + rho/2 with sigma_sc**2 + sigma_sr**2 (101)."""
    triple_value = double_difference(
        REMOTE_TRIPLE,
        SC,
        unaccepted_link(PREDICTED_RS),
        link_component(-5_432_080, rms=5, predicted_phase=PREDICTED_SR),
    )
    rho = PREDICTED_RS + PREDICTED_SR
    assert triple_value == TripleValue(
        z=round_even(1_234_577 + 5_432_080 + rho / 2),
        sigma=math.sqrt(9 + 25),
        components_used="101",
        pair_cold_started=False,
    )


@pytest.mark.parametrize(
    ("rs", "sr"),
    [
        (link_component(5_432_100), unaccepted_link(PREDICTED_SR)),
        (link_component(5_432_100, predicted_phase=PREDICTED_RS), unaccepted_link()),
        (unaccepted_link(PREDICTED_RS), link_component(-5_432_080)),
    ],
)
def test_no_round_trip_without_both_predictions(rs: Component, sr: Component) -> None:
    """Give no value when one direction is missing and a prediction is too."""
    assert double_difference(REMOTE_TRIPLE, SC, rs, sr) is None


def test_no_value_without_either_direction() -> None:
    """Give no value when neither link direction is accepted."""
    assert (
        double_difference(
            REMOTE_TRIPLE,
            SC,
            unaccepted_link(PREDICTED_RS),
            unaccepted_link(PREDICTED_SR),
        )
        is None
    )


def test_no_value_without_the_clock_pair() -> None:
    """Give no value when (s, c) is not accepted."""
    sc = Component(accepted=False, predicted_phase=mpq(1_234_574))
    assert (
        double_difference(
            REMOTE_TRIPLE, sc, link_component(5_432_100), link_component(-5_432_080)
        )
        is None
    )


def test_a_constant_link_offset_keeps_the_value_continuous() -> None:
    """Change dd by less than sigma_dd as 111 gives way to 110 and 101 (U18)."""
    x_r, x_s, x_c = 7_000_000, 1_000_000, -234_567
    delay_rs, delay_sr = 431, -117
    z_sc = x_s - x_c
    z_rs = x_r - x_s + delay_rs
    z_sr = x_s - x_r + delay_sr
    predicted_rs = mpq(z_rs) + mpq(3, 10)
    predicted_sr = mpq(z_sr) - mpq(7, 10)
    sc = Component(accepted=True, z=z_sc, rms=3)
    both_links = double_difference(
        REMOTE_TRIPLE,
        sc,
        link_component(z_rs, predicted_phase=predicted_rs),
        link_component(z_sr, predicted_phase=predicted_sr),
    )
    forward_only = double_difference(
        REMOTE_TRIPLE,
        sc,
        link_component(z_rs, predicted_phase=predicted_rs),
        unaccepted_link(predicted_sr),
    )
    back_only = double_difference(
        REMOTE_TRIPLE,
        sc,
        unaccepted_link(predicted_rs),
        link_component(z_sr, predicted_phase=predicted_sr),
    )
    assert both_links is not None and forward_only is not None and back_only is not None
    assert [
        triple_value.components_used
        for triple_value in (both_links, forward_only, back_only)
    ] == [
        "111",
        "110",
        "101",
    ]
    assert abs(forward_only.z - both_links.z) < both_links.sigma
    assert abs(back_only.z - both_links.z) < both_links.sigma


# ---------------------------------------------------------------- local


@pytest.mark.parametrize("z", [1_234_577, -1, 0, 2**62 + 1, -(2**61) - 3])
def test_a_local_triple_is_its_pair_exactly(z: int) -> None:
    """Give dd = z(r,c) and sigma_dd = sigma_rc for (r, r, c) (U19)."""
    sc = Component(accepted=True, z=z, rms=4)
    rr = link_component(5_432_101, rms=9)
    assert double_difference(LOCAL_TRIPLE, sc, rr, rr) == TripleValue(
        z=z, sigma=4.0, components_used="111", pair_cold_started=False
    )


def test_a_local_triple_needs_only_its_pair() -> None:
    """Give the local triple its value with the self pair not accepted."""
    rr = unaccepted_link()
    triple_value = double_difference(LOCAL_TRIPLE, SC, rr, rr)
    assert triple_value == TripleValue(
        z=1_234_577, sigma=3.0, components_used="111", pair_cold_started=False
    )


def test_a_local_triple_that_does_not_collapse_stops_the_run() -> None:
    """Raise PhaseError when the general formula does not give z(r,c) (U19)."""
    with pytest.raises(PhaseError, match="does not collapse"):
        double_difference(
            LOCAL_TRIPLE, SC, link_component(5_432_102), link_component(5_432_100)
        )


# ------------------------------------------------------------------- cold


@pytest.mark.parametrize("cold_pair", ["sc", "rs", "sr"])
def test_a_component_cold_start_marks_the_value_cold(cold_pair: str) -> None:
    """Mark the value cold when any of its pairs cold-started (12.6)."""
    sc = Component(accepted=True, z=1_234_577, rms=3, cold_started=cold_pair == "sc")
    rs = link_component(5_432_100, cold_started=cold_pair == "rs")
    sr = link_component(-5_432_080, cold_started=cold_pair == "sr")
    triple_value = double_difference(REMOTE_TRIPLE, sc, rs, sr)
    assert triple_value is not None
    assert triple_value.pair_cold_started is True


def test_a_local_triple_is_not_cold_when_its_self_pair_is() -> None:
    """Leave a local triple warm when only its self pair cold-started (12.6)."""
    rr = link_component(5_432_101, cold_started=True)
    triple_value = double_difference(LOCAL_TRIPLE, SC, rr, rr)
    assert triple_value is not None
    assert triple_value.pair_cold_started is False


@pytest.mark.parametrize(
    ("triple", "cold_pair", "restarted"),
    [
        (REMOTE_TRIPLE, "sc", True),
        (REMOTE_TRIPLE, "rs", True),
        (REMOTE_TRIPLE, "sr", True),
        (REMOTE_TRIPLE, None, False),
        (LOCAL_TRIPLE, "sc", True),
        (LOCAL_TRIPLE, "rs", False),
        (LOCAL_TRIPLE, None, False),
    ],
)
def test_a_pair_restart_is_one_of_a_pair_the_triple_uses(
    triple: tuple[str, str, str], cold_pair: str | None, restarted: bool
) -> None:
    """Count any pair of a remote triple, and only (r, c) of a local one (12.6)."""
    sc = Component(accepted=True, z=1_234_577, rms=3, cold_started=cold_pair == "sc")
    rs = link_component(5_432_100, cold_started=cold_pair == "rs")
    sr = link_component(-5_432_080, cold_started=cold_pair == "sr")
    if triple == LOCAL_TRIPLE:
        rs = sr = link_component(5_432_100, cold_started=cold_pair == "rs")
    assert pair_restarted(triple, sc, rs, sr) is restarted


def test_a_link_restart_counts_without_the_clock_pair() -> None:
    """Count a link's restart at an epoch whose (s, c) was not accepted (12.6)."""
    sc = Component(accepted=False, predicted_phase=mpq(1_234_574))
    rs = link_component(5_432_100, cold_started=True)
    assert double_difference(REMOTE_TRIPLE, sc, rs, link_component(-5_432_080)) is None
    assert pair_restarted(REMOTE_TRIPLE, sc, rs, link_component(-5_432_080))


def test_a_local_triple_is_cold_when_its_pair_is() -> None:
    """Mark a local triple cold when its pair cold-started."""
    sc = Component(accepted=True, z=1_234_577, rms=3, cold_started=True)
    triple_value = double_difference(
        LOCAL_TRIPLE, sc, unaccepted_link(), unaccepted_link()
    )
    assert triple_value is not None
    assert triple_value.pair_cold_started is True


# --------------------------------------------------------------- the models


@pytest.mark.parametrize(
    "component_fields",
    [
        {"accepted": True, "z": 1},
        {"accepted": True, "rms": 3},
        {"accepted": False, "z": 1, "rms": 3},
    ],
)
def test_a_component_is_accepted_with_a_value_or_not_without_one(
    component_fields: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse a component whose z and rms do not match whether it was accepted."""
    with pytest.raises(PhaseError, match="exactly when it was accepted") as raised:
        Component(**component_fields)  # type: ignore[arg-type]
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(raised.value)
    ]


def test_a_missing_back_direction_squares_the_forward_rms() -> None:
    """Combine the clock pair's and the forward link's rms in quadrature (110)."""
    triple_value = double_difference(
        REMOTE_TRIPLE,
        SC,
        link_component(5_432_100, rms=5, predicted_phase=PREDICTED_RS),
        unaccepted_link(PREDICTED_SR),
    )
    assert triple_value is not None
    assert triple_value.sigma == math.sqrt(9 + 25)


def test_a_local_triple_that_does_not_collapse_says_so_in_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Name the triple and both values, and log it at ERROR as raised (U19)."""
    with pytest.raises(PhaseError) as raised:
        double_difference(
            LOCAL_TRIPLE, SC, link_component(5_432_106), link_component(5_432_100)
        )
    assert str(raised.value) == (
        "local triple ('mc2', 'mc2', 'ox23') does not collapse to its pair:"
        " 1234580 != 1234577"
    )
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(raised.value)
    ]
