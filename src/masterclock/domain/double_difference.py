"""Double differences: a clock measured against a remote reference.

A triple (r, s, c) turns the measurement of clock c against its local
reference s into a measurement against reference r, by adding the two-way
comparison of r and s:

    dd = z(s, c) + (z(r, s) - z(s, r)) / 2 = x_r - x_c + const.

Only accepted measurements enter it, never the pairs' estimates. When one
link direction was not accepted, it is replaced through the round trip the
links' predictions give, rho = x-(r, s) + x-(s, r). A local triple (r, r, c)
uses the self pair (r, r) for both directions, so the link term cancels and
the value is z(r, c) exactly; that is checked every time, and a mismatch is
a fault. Every value is summed exactly and rounded once, a tie to even.
"""

import math
from fractions import Fraction
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import PhaseError
from masterclock.domain.phase import round_even
from masterclock.domain.series import TripleKey

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


class Component(BaseModel):
    """One pair's part in a triple at an epoch.

    Parameters
    ----------
    accepted : bool
        Whether the pair's measurement was accepted at the epoch (row
        flag A).
    z : int or None, optional
        The accepted measurement z_E, ps; ``None`` when not accepted.
    rms : int or None, optional
        Its rms, ps, at least 0; ``None`` when not accepted.
    predicted : Fraction or None, optional
        The pair's predicted phase x- at the epoch, ps; ``None`` when it
        has no valid prediction.
    cold : bool, optional
        Whether the pair cold-started at the epoch.

    Raises
    ------
    pydantic.ValidationError
        If ``z`` and ``rms`` are not given exactly when ``accepted``, or a
        value is of the wrong kind or out of range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    accepted: bool
    z: int | None = None
    rms: Annotated[int, Field(ge=0)] | None = None
    predicted: Fraction | None = None
    cold: bool = False

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        """Refuse a value without an acceptance, or an acceptance without one.

        Returns
        -------
        Self
            The component, unchanged.

        Raises
        ------
        ValueError
            If ``z`` or ``rms`` is given and not accepted, or missing and
            accepted.
        """
        given = (self.z is not None, self.rms is not None)
        if given != (self.accepted, self.accepted):
            message = "a component has its z and rms exactly when it was accepted"
            raise ValueError(message)
        return self


class TripleValue(BaseModel):
    """A triple's measurement at an epoch.

    Parameters
    ----------
    z : int
        The double difference dd, ps.
    sigma : float
        Its measurement sigma, sigma_dd, ps.
    components_used : {'111', '110', '101'}
        Which of (s, c), (r, s) and (s, r) gave it, in that order.
    cold : bool
        Whether one of its pairs cold-started at the epoch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    z: int
    sigma: float
    components_used: Literal["111", "110", "101"]
    cold: bool


def double_difference(
    triple: TripleKey, sc: Component, rs: Component, sr: Component
) -> TripleValue | None:
    """Give a triple's measurement from its pairs' (design 12).

    Parameters
    ----------
    triple : (str, str, str)
        The triple (r, s, c).
    sc : Component
        The clock pair (s, c).
    rs, sr : Component
        The link pairs (r, s) and (s, r); for a local triple (r, r, c),
        the self pair (r, r) for both.

    Returns
    -------
    TripleValue or None
        With both links accepted: z(s,c) + (z(r,s) - z(s,r)) / 2, with
        sigma_sc**2 + (sigma_rs**2 + sigma_sr**2) / 4 (111). With one, the
        other replaced through rho = x-(r,s) + x-(s,r):
        z(s,c) + z(r,s) - rho/2 with sigma_sc**2 + sigma_rs**2 (110), or
        z(s,c) - z(s,r) + rho/2 with sigma_sc**2 + sigma_sr**2 (101). A
        local triple needs only (r, c), and is z(r, c) with sigma_rc. Cold
        when any pair cold-started. ``None`` when (s, c) is not accepted,
        neither link is, or rho is needed and a prediction is missing.

    Raises
    ------
    PhaseError
        If a local triple does not come out as its pair exactly.

    Examples
    --------
    The worked epoch's remote triple:

    >>> sc = Component(accepted=True, z=1_234_577, rms=3)
    >>> rs = Component(accepted=True, z=5_432_100, rms=2)
    >>> sr = Component(accepted=True, z=-5_432_080, rms=2)
    >>> double_difference(("mc1", "mc2", "nav23"), sc, rs, sr).z
    6666667
    """
    r, s, _ = triple
    clock = _measured(sc)
    if clock is None:
        return None
    cold = sc.cold or rs.cold or sr.cold
    if r == s:
        return _local(triple, *clock, rs, sr, cold=cold)
    return _remote(*clock, rs, sr, cold=cold)


def _measured(component: Component) -> tuple[int, int] | None:
    """Give an accepted component's measurement and rms.

    Parameters
    ----------
    component : Component
        The pair's part in the triple.

    Returns
    -------
    tuple of (int, int) or None
        z and rms, ps; ``None`` when the pair was not accepted.
    """
    if component.z is None or component.rms is None:
        return None
    return component.z, component.rms


def _remote(
    z: int, rms: int, rs: Component, sr: Component, *, cold: bool
) -> TripleValue | None:
    """Give a remote triple's measurement (design 12.1-12.3).

    Parameters
    ----------
    z : int
        The accepted measurement of (s, c), ps.
    rms : int
        Its rms, ps.
    rs, sr : Component
        The link pairs (r, s) and (s, r).
    cold : bool
        Whether a pair cold-started.

    Returns
    -------
    TripleValue or None
        As :func:`double_difference` describes; ``None`` when neither link
        is accepted, or rho is needed and a prediction is missing.
    """
    forward, back = _measured(rs), _measured(sr)
    if forward is not None and back is not None:
        dd = z + Fraction(forward[0] - back[0], 2)
        variance = rms**2 + 0.25 * (forward[1] ** 2 + back[1] ** 2)
        return _value(dd, variance, "111", cold=cold)
    if rs.predicted is None or sr.predicted is None:
        return None
    rho = rs.predicted + sr.predicted
    if forward is not None:
        return _value(
            z + forward[0] - rho / 2, rms**2 + forward[1] ** 2, "110", cold=cold
        )
    if back is not None:
        return _value(z - back[0] + rho / 2, rms**2 + back[1] ** 2, "101", cold=cold)
    return None


def _local(
    triple: TripleKey,
    z: int,
    rms: int,
    rs: Component,
    sr: Component,
    *,
    cold: bool,
) -> TripleValue:
    """Give a local triple's measurement, checking it is its pair's (design 12.4).

    Parameters
    ----------
    triple : (str, str, str)
        The triple (r, r, c).
    z : int
        The accepted measurement of (r, c), ps.
    rms : int
        Its rms, ps.
    rs, sr : Component
        The self pair (r, r), given for both link directions; a direction
        not accepted counts as 0, which cancels too.
    cold : bool
        Whether a pair cold-started.

    Returns
    -------
    TripleValue
        z with sigma rms, components 111.

    Raises
    ------
    PhaseError
        If the general formula does not give z exactly.
    """
    forward = rs.z if rs.z is not None else 0
    back = sr.z if sr.z is not None else 0
    dd = round_even(z + Fraction(forward - back, 2))
    if dd != z:
        message = f"local triple {triple} does not collapse to its pair: {dd} != {z}"
        _log.error(message)
        raise PhaseError(message)
    return TripleValue(z=dd, sigma=float(rms), components_used="111", cold=cold)


def _value(
    dd: Fraction, variance: float, used: Literal["111", "110", "101"], *, cold: bool
) -> TripleValue:
    """Round a double difference once and give it with its sigma.

    Parameters
    ----------
    dd : Fraction
        The exact double difference, ps.
    variance : float
        Its variance, ps**2.
    used : {'111', '110', '101'}
        The components used.
    cold : bool
        Whether a pair cold-started.

    Returns
    -------
    TripleValue
        dd rounded half to even, and the square root of the variance.
    """
    return TripleValue(
        z=round_even(dd), sigma=math.sqrt(variance), components_used=used, cold=cold
    )
