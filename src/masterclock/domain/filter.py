"""The forward estimator every pair and triple runs: gains, prediction, update.

Each series runs a critically damped state-space filter with fixed gains:
three states (phase, rate, drift) with a triple pole at lambda = exp(-1/M),
two states (phase, rate) with a double pole there, or one state, which
passes an accepted measurement through unchanged. The pole sits at lambda
for every M, so the closed loop (I - K H) Phi has every eigenvalue equal to
lambda.

The phase is summed exactly, as a :class:`~fractions.Fraction`, and rounded
only when a row stores it, to whole femtoseconds (see
:data:`~masterclock.domain.phase.FS_PER_PS`); rates, drifts and gains are
floats.
"""

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Final

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import EPOCH_SECONDS, exact, from_fs
from masterclock.domain.series import State

if TYPE_CHECKING:
    from masterclock.domain.series import Row

_T: Final[int] = EPOCH_SECONDS
"""One epoch, s: whole, so a phase moved on by it stays exact."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def gains(model: int, M: float | None) -> tuple[float, float, float]:
    """Give the fixed gains of a model with time constant M.

    Parameters
    ----------
    model : int
        How many states the estimator has: 1, 2 or 3.
    M : float or None
        The time constant, epochs, at least 1; ``None`` for one state.

    Returns
    -------
    tuple of float
        The gains on phase, rate and drift: g, h/T and 2k/T**2. For three
        states, g = 1 - lambda**3, h = 1.5 (1 - lambda)**2 (1 + lambda) and
        k = (1 - lambda)**3 / 2; for two, g = 1 - lambda**2 and
        h = (1 - lambda)**2, with no drift gain; for one, (1, 0, 0).

    Raises
    ------
    FilterError
        If ``model`` is not 1, 2 or 3, or ``M`` is given for one state or is
        not at least 1 for two or three.

    Examples
    --------
    >>> gains(1, None)
    (1.0, 0.0, 0.0)
    >>> g, h_over_t, drift = gains(2, 10.0)
    >>> round(g, 6), drift
    (0.181269, 0.0)
    """
    if model == 1 and M is None:
        return (1.0, 0.0, 0.0)
    if model not in {2, 3} or M is None or not M >= 1:
        message = f"no gains for a {model}-state model with time constant {M}"
        _log.error(message)
        raise FilterError(message)
    lam = math.exp(-1.0 / M)
    if model == 3:
        return (
            1 - lam**3,
            1.5 * (1 - lam) ** 2 * (1 + lam) / _T,
            (1 - lam) ** 3 / _T**2,
        )
    return (1 - lam**2, (1 - lam) ** 2 / _T, 0.0)


def predict(last: Row | None, u: tuple[Fraction, float]) -> State | None:
    """Predict a series' state at the next epoch from its last row.

    Parameters
    ----------
    last : Row or None
        The series' last row, or ``None`` for a series with none.
    u : tuple of (Fraction, float)
        The steering input over the epoch: u_x, exact, and u_y.

    Returns
    -------
    State or None
        Phi X + u: the phase moved on by y T + d T**2 / 2 and u_x, exactly;
        the rate by d T and u_y; the drift kept. A 1-state row keeps its
        phase and has no rate; a 2-state row has no drift. ``None`` when
        there is no last row, or it is dormant.
    """
    if last is None or "D" in last.flags:
        return None
    x_fs, y, d = last.known_state()
    x = from_fs(x_fs)
    ux, uy = u
    if last.filter_states == 1:
        return State(x=x + ux, y=0.0)
    if last.filter_states == 3:
        return State(
            x=x + exact(y) * _T + exact(d) * _T * _T / 2 + ux,
            y=y + d * _T + uy,
            d=d,
        )
    return State(x=x + exact(y) * _T + ux, y=y + uy)


def update(
    prediction: State, innovation: Fraction, model: int, M: float | None
) -> State:
    """Correct a prediction by the gains times the innovation.

    Parameters
    ----------
    prediction : State
        The predicted state at the epoch.
    innovation : Fraction
        The measurement less the predicted phase, exact.
    model : int
        How many states the estimator has.
    M : float or None
        The time constant; ``None`` for one state.

    Returns
    -------
    State
        X + K nu: the phase exact, to be rounded when it is stored; the
        rate and, for three states, the drift as floats.

    Raises
    ------
    FilterError
        If ``model`` and ``M`` do not belong together (see :func:`gains`).
    """
    g, h_over_t, two_k_over_t2 = gains(model, M)
    nu = float(innovation)
    return State(
        x=prediction.x + exact(g) * innovation,
        y=prediction.y + h_over_t * nu,
        d=prediction.d + two_k_over_t2 * nu if model == 3 else 0.0,
    )
