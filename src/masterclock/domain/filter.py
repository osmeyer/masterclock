"""The forward estimator every pair and triple runs, and the rows it writes.

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

Each epoch gives every series one row. The row starts as a
:class:`RowDraft` carried on from the last row (:func:`carry`), may begin a
new segment (:func:`start_segment`), and is finished by what the epoch did:
an accepted measurement updates the state (:func:`accept`); a rejected,
excluded or missing one leaves the prediction standing (:func:`hold`); a
series with no valid state is dormant (:func:`dormant`): it buffers its
measurements (:func:`acquire`, decycled against :func:`anchor_of`) until
three agree, and starts again from the third alone (:func:`cold_start`).
Only the finished row is checked, by :func:`finish`.

A measurement is accepted when it passes the gate (:func:`within_gate`,
:func:`rms_ok`). One that fails is a counted reject (:func:`count_reject`),
and three in a row are looked at for a step (:func:`classify`): agreeing,
they are a phase step; on a line, a frequency step (:func:`accept_step`).
"""

import dataclasses
import math
from dataclasses import dataclass, fields
from datetime import datetime
from fractions import Fraction
from itertools import pairwise
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import (
    EPOCH_SECONDS,
    exact,
    from_fs,
    round_even,
    seconds,
    to_fs,
)
from masterclock.domain.series import (
    FLAG_ORDER,
    MAX_REJECTS,
    FilterStates,
    Reject,
    SeriesParams,
    State,
    build_row,
)

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


# ------------------------------------------------------------ row lifecycle

type Outcome = Literal["A", "R", "X", "P"]
"""What an epoch did to a series; every row carries exactly one."""

type Held = Literal["R", "X", "P"]
"""An outcome that leaves the state unupdated: rejected, excluded, predicted."""

SETTLE_FACTOR: Final[int] = 5
"""A segment is unsettled while it has run fewer than this many times M rows."""


@dataclass(frozen=True, slots=True)
class RowDraft:
    """A row being built for an epoch: the fields of a :class:`Row`, unchecked.

    An epoch builds its row a field or two at a time, and part way it is no
    valid row: it has no outcome flag until the last step. So the steps
    pass a draft along, each building a new one with
    :func:`dataclasses.replace`, and :func:`finish` checks the last into a
    :class:`Row`. The fields are a row's, with the same names and meanings.
    """

    interpolated_datetime: datetime
    innovation: float | None
    x_fs: int | None
    y: float | None
    d: float | None
    innovation_scale: float | None
    segment: int
    step_offset: int
    epochs_in_segment: int
    epochs_since_accept: int
    consecutive_rejects: int
    rejects: tuple[Reject, ...]
    filter_states: FilterStates
    time_constant: float | None
    scale_time_constant: float
    flags: str


def carry(
    mark: datetime, last: Row | None, params: SeriesParams, *, slip: bool = False
) -> RowDraft:
    """Start an epoch's row from the series' last row.

    Parameters
    ----------
    mark : datetime
        The epoch start E.
    last : Row or None
        The series' last row, or ``None`` for a new series.
    params : SeriesParams
        The series' settings at E; used only for a new series.
    slip : bool, optional
        Whether the slip check corrected this epoch's measurement; the row
        then carries S.

    Returns
    -------
    RowDraft
        For a new series: dormant, with no state, in segment 0, every
        counter 0, the model and time constants of ``params``. Otherwise
        the last row moved on to ``mark``, with no innovation, one more
        epoch in its segment and no flags. S when ``slip``.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> settings = SeriesParams(
    ...     model=1, M=None, M_sigma=50.0, sigma0=5.0, gmax=432, n_break=36,
    ...     rms_max=80,
    ... )
    >>> draft = carry(datetime(2025, 9, 23, 6, 0, tzinfo=UTC), None, settings)
    >>> draft.segment, draft.x_fs, draft.flags
    (0, None, '')
    """
    flags = "S" if slip else ""
    if last is None:
        return RowDraft(
            interpolated_datetime=mark,
            innovation=None,
            x_fs=None,
            y=None,
            d=None,
            innovation_scale=None,
            segment=0,
            step_offset=0,
            epochs_in_segment=0,
            epochs_since_accept=0,
            consecutive_rejects=0,
            rejects=(),
            filter_states=params.model,
            time_constant=params.M,
            scale_time_constant=params.M_sigma,
            flags=flags,
        )
    return RowDraft(
        **{
            **dict(last),
            "interpolated_datetime": mark,
            "innovation": None,
            "epochs_in_segment": last.epochs_in_segment + 1,
            "flags": flags,
        }
    )


def start_segment(
    draft: RowDraft, params: SeriesParams, *, keep_offset: bool
) -> RowDraft:
    """Begin a new segment on a draft.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    params : SeriesParams
        The settings in force at the epoch, whose M and M_sigma the segment
        takes.
    keep_offset : bool
        Whether the segment keeps the step offset: a warm start does, a
        cold start sets it to 0.

    Returns
    -------
    RowDraft
        The segment number one more, no rows in it yet, the time constants
        of ``params``, and N among the flags. The state is not touched.

    Raises
    ------
    FilterError
        If ``params`` is for another model than the series': a series keeps
        its model for the life of its file.
    """
    if params.model != draft.filter_states:
        message = (
            f"a {draft.filter_states}-state series keeps its model;"
            f" its settings name model {params.model}"
        )
        _log.error(message)
        raise FilterError(message)
    return dataclasses.replace(
        draft,
        segment=draft.segment + 1,
        epochs_in_segment=0,
        step_offset=draft.step_offset if keep_offset else 0,
        time_constant=params.M,
        scale_time_constant=params.M_sigma,
        flags=draft.flags + "N",
    )


def finish(draft: RowDraft, flag: Outcome) -> Row:
    """Add the epoch's outcome to a draft and check it into a row.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    flag : {'A', 'R', 'X', 'P'}
        The outcome.

    Returns
    -------
    Row
        The checked row, its flags in
        :data:`~masterclock.domain.series.FLAG_ORDER`, with U added to a row
        of a 2- or 3-state series that is not dormant while its segment has
        run fewer than :data:`SETTLE_FACTOR` times M rows.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    flags = draft.flags + flag
    if (
        "D" not in flags
        and draft.time_constant is not None
        and draft.epochs_in_segment < SETTLE_FACTOR * draft.time_constant
    ):
        flags += "U"
    ordered = "".join(letter for letter in FLAG_ORDER if letter in flags)
    values = {field.name: getattr(draft, field.name) for field in fields(draft)}
    return build_row({**values, "flags": ordered})


def dormant(draft: RowDraft, flag: Held, *, keep_buffer: bool = False) -> Row:
    """Finish a draft as a dormant row: one with no valid state.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    flag : {'R', 'X', 'P'}
        The outcome: a measurement buffered or rejected, or none.
    keep_buffer : bool, optional
        Whether the buffer stays: it does while a dormant series gathers
        measurements to acquire from; otherwise it is emptied.

    Returns
    -------
    Row
        The row with no phase, rate, drift or innovation scale, flag D
        beside ``flag``, and segment and step offset as they were.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    return finish(
        dataclasses.replace(
            draft,
            x_fs=None,
            y=None,
            d=None,
            innovation_scale=None,
            rejects=draft.rejects if keep_buffer else (),
            flags=draft.flags + "D",
        ),
        flag,
    )


def hold(
    draft: RowDraft, prediction: State | None, flag: Held, params: SeriesParams
) -> Row:
    """Finish a draft as a held row: no update, the prediction carried.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, with its innovation when there was a
        measurement.
    prediction : State or None
        The series' prediction at the epoch, or ``None`` when it has none.
    flag : {'R', 'X', 'P'}
        Why the state is not updated: a counted reject, an excluded
        measurement, or no measurement.
    params : SeriesParams
        The settings in force, whose ``gmax`` is the gap limit.

    Returns
    -------
    Row
        The prediction stored, x in whole femtoseconds, with the innovation
        scale, counters and buffer kept and one more epoch since an accept.
        A dormant row instead (see :func:`dormant`) when there is no
        prediction, or the series has gone more than ``gmax`` epochs without
        an accept.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    draft = dataclasses.replace(
        draft, epochs_since_accept=draft.epochs_since_accept + 1
    )
    if prediction is None or draft.epochs_since_accept > params.gmax:
        return dormant(draft, flag)
    held = dataclasses.replace(
        draft, x_fs=to_fs(prediction.x), y=prediction.y, d=prediction.d
    )
    return finish(held, flag)


def accept(
    draft: RowDraft, prediction: State, innovation: Fraction, floor: float
) -> Row:
    """Finish a draft as an accepted row: the prediction updated.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a series with an innovation scale.
    prediction : State
        The series' prediction at the epoch.
    innovation : Fraction
        The measurement less the predicted phase, exact.
    floor : float
        The lowest the innovation scale may go, ps: the measurement's own
        rms for a pair, sigma_dd for a triple.

    Returns
    -------
    Row
        The updated state, x in whole femtoseconds; the innovation; the
        innovation scale moved by the innovation, as
        sqrt(max((1 - w) scale**2 + w innovation**2, floor**2)) with
        w = 1/M_sigma; the counters and the buffer cleared.

    Raises
    ------
    FilterError
        If the draft has no innovation scale, which a series with a
        prediction always has, or the finished row breaks a rule of
        :class:`Row`.
    """
    if draft.innovation_scale is None:
        message = f"an accept at {draft.interpolated_datetime} has no innovation scale"
        _log.error(message)
        raise FilterError(message)
    state = update(prediction, innovation, draft.filter_states, draft.time_constant)
    nu = float(innovation)
    w = 1.0 / draft.scale_time_constant
    scale = math.sqrt(max((1 - w) * draft.innovation_scale**2 + w * nu**2, floor**2))
    accepted = dataclasses.replace(
        draft,
        innovation=nu,
        x_fs=to_fs(state.x),
        y=state.y,
        d=state.d,
        innovation_scale=scale,
        consecutive_rejects=0,
        rejects=(),
        epochs_since_accept=0,
    )
    return finish(accepted, "A")


def cold_start(draft: RowDraft, z: int, params: SeriesParams) -> Row:
    """Finish a draft as a cold start: a new segment from the measurement alone.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a dormant series.
    z : int
        The measurement at the epoch, ps.
    params : SeriesParams
        The settings in force, whose time constants and ``sigma0`` the new
        segment takes.

    Returns
    -------
    Row
        Segment one more, at phase ``z`` with no rate or drift, innovation
        scale ``sigma0``, step offset, counters and buffer at 0, flags A
        and N, and U for a 2- or 3-state series.

    Raises
    ------
    FilterError
        If ``params`` is for another model than the series', or the
        finished row breaks a rule of :class:`Row`.
    """
    started = start_segment(draft, params, keep_offset=False)
    return finish(
        dataclasses.replace(
            started,
            x_fs=to_fs(z),
            y=0.0,
            d=0.0,
            innovation_scale=params.sigma0,
            consecutive_rejects=0,
            rejects=(),
            epochs_since_accept=0,
        ),
        "A",
    )


# ------------------------------------------------------- step classification

K_OUT: Final[float] = 5.0
"""How many innovation scales wide the gate is, either way."""

K_STEP: Final[float] = 3.0
"""How many innovation scales three rejects may stray from a step and show it."""

_STEP_REJECTS: Final[int] = 3
"""How many consecutive counted rejects a step is looked for in."""


class Classified(BaseModel):
    """What three consecutive rejects show: a phase step, a frequency step, or neither.

    Parameters
    ----------
    kind : {'phase', 'frequency'} or None
        The kind of step; ``None`` when the rejects show neither.
    a : float or None, optional
        For a frequency step, the fitted line's value at the first reject,
        ps; ``None`` otherwise.
    s : float or None, optional
        For a frequency step, the fitted line's slope, ps/s; ``None``
        otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["phase", "frequency"] | None
    a: float | None = None
    s: float | None = None


def classify(rejects: tuple[Reject, ...], sigma: float) -> Classified:
    """Tell what three consecutive rejects show (design 9.4).

    Parameters
    ----------
    rejects : tuple of (datetime, float)
        The reject buffer: three (epoch, innovation) entries, oldest first,
        the newest the current one.
    sigma : float
        The innovation scale, ps.

    Returns
    -------
    Classified
        A phase step when every innovation lies within :data:`K_STEP`
        scales of their mean. Otherwise a frequency step, with the line
        fitted by least squares against the time since the first reject,
        when every innovation lies within :data:`K_STEP` scales of it.
        Otherwise neither.

    Raises
    ------
    FilterError
        If the buffer does not hold three rejects.

    Examples
    --------
    >>> from datetime import UTC, datetime, timedelta
    >>> start = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
    >>> epochs = [start + timedelta(minutes=10 * i) for i in range(3)]
    >>> classify(tuple(zip(epochs, (150.0, 151.0, 149.0))), 3.0).kind
    'phase'
    >>> classify(tuple(zip(epochs, (30.0, 60.0, 90.0))), 3.0).kind
    'frequency'
    """
    if len(rejects) != _STEP_REJECTS:
        message = f"a step is looked for in three rejects, not {len(rejects)}"
        _log.error(message)
        raise FilterError(message)
    first = rejects[0][0]
    ts = [float(seconds(when, first)) for when, _ in rejects]
    vs = [value for _, value in rejects]
    vbar = sum(vs) / _STEP_REJECTS
    if max(abs(v - vbar) for v in vs) < K_STEP * sigma:
        return Classified(kind="phase")
    tbar = sum(ts) / _STEP_REJECTS
    s = sum((t - tbar) * (v - vbar) for t, v in zip(ts, vs, strict=True)) / sum(
        (t - tbar) ** 2 for t in ts
    )
    a = vbar - s * tbar
    if max(abs(v - a - s * t) for t, v in zip(ts, vs, strict=True)) < K_STEP * sigma:
        return Classified(kind="frequency", a=a, s=s)
    return Classified(kind=None)


def within_gate(innovation: Fraction, scale: float) -> bool:
    """Tell whether an innovation passes the gate's width (design 9.1).

    Parameters
    ----------
    innovation : Fraction
        The measurement less the predicted phase, exact.
    scale : float
        The innovation scale, ps.

    Returns
    -------
    bool
        Whether the innovation is at most :data:`K_OUT` scales either way,
        compared exactly.

    Examples
    --------
    >>> within_gate(Fraction(15), 3.0), within_gate(Fraction(-31, 2), 3.0)
    (True, False)
    """
    return abs(innovation) <= exact(K_OUT * scale)


def rms_ok(rms: int, rms_max: int | None) -> bool:
    """Tell whether a pair measurement's rms passes the gate (design 9.1).

    Parameters
    ----------
    rms : int
        The rms the DAS gave with the measurement, ps.
    rms_max : int or None
        The pair's rms limit, ps; ``None`` for no limit.

    Returns
    -------
    bool
        Whether ``rms`` is at most ``rms_max``, or there is no limit.

    Examples
    --------
    >>> rms_ok(80, 80), rms_ok(81, 80), rms_ok(81, None)
    (True, False, True)
    """
    return rms_max is None or rms <= rms_max


def count_reject(draft: RowDraft, innovation: Fraction) -> RowDraft:
    """Count a rejected measurement and put it in the reject buffer (design 9.3).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    innovation : Fraction
        The rejected measurement less the predicted phase.

    Returns
    -------
    RowDraft
        One more consecutive reject, and (epoch, innovation) added to the
        buffer, which keeps the newest
        :data:`~masterclock.domain.series.MAX_REJECTS`.
    """
    entry = (draft.interpolated_datetime, float(innovation))
    return dataclasses.replace(
        draft,
        consecutive_rejects=draft.consecutive_rejects + 1,
        rejects=(*draft.rejects, entry)[-MAX_REJECTS:],
    )


def phase_step(draft: RowDraft, prediction: State, z: int, floor: float) -> Row:
    """Accept a measurement after a phase step the rejects agree on (design 9.4).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, its buffer holding the three rejects.
    prediction : State
        The series' prediction at the epoch.
    z : int
        The current measurement, ps.
    floor : float
        The lowest the innovation scale may go, ps (see :func:`accept`).

    Returns
    -------
    Row
        The step, the rejects' mean innovation rounded once half to even,
        added to the step offset and to the predicted phase; then the
        measurement accepted against the corrected prediction, in the same
        segment.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    total = sum((exact(value) for _, value in draft.rejects), Fraction(0))
    step = round_even(total / _STEP_REJECTS)
    corrected = State(x=prediction.x + step, y=prediction.y, d=prediction.d)
    stepped = dataclasses.replace(draft, step_offset=draft.step_offset + step)
    return accept(stepped, corrected, z - corrected.x, floor)


def frequency_step(
    draft: RowDraft,
    prediction: State,
    a: float,
    s: float,
    z: int,
    floor: float,
    params: SeriesParams,
) -> Row:
    """Accept a measurement after a frequency step the rejects lie on (design 9.4).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, its buffer holding the three rejects.
    prediction : State
        The series' prediction at the epoch.
    a : float
        The fitted line's value at the first reject, ps.
    s : float
        The fitted line's slope, ps/s.
    z : int
        The current measurement, ps.
    floor : float
        The lowest the innovation scale may go, ps (see :func:`accept`).
    params : SeriesParams
        The settings in force, whose time constants the new segment takes.

    Returns
    -------
    Row
        The prediction moved onto the line at the current epoch: a + s t3
        added to the phase, exactly, with t3 the time from the first reject
        to the current one, and s added to the rate. Then a warm segment
        start (see :func:`start_segment`), keeping the step offset, and the
        measurement accepted against the corrected prediction.

    Raises
    ------
    FilterError
        If ``params`` is for another model than the series', or the
        finished row breaks a rule of :class:`Row`.
    """
    t3 = seconds(draft.rejects[-1][0], draft.rejects[0][0])
    corrected = State(
        x=prediction.x + exact(a) + exact(s) * t3, y=prediction.y + s, d=prediction.d
    )
    started = start_segment(draft, params, keep_offset=True)
    return accept(started, corrected, z - corrected.x, floor)


def accept_step(
    draft: RowDraft, prediction: State, z: int, floor: float, params: SeriesParams
) -> Row | None:
    """Accept a measurement after a step, when the rejects show one (design 9.4).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, the current reject counted.
    prediction : State
        The series' prediction at the epoch.
    z : int
        The current measurement, ps.
    floor : float
        The lowest the innovation scale may go, ps (see :func:`accept`).
    params : SeriesParams
        The settings in force.

    Returns
    -------
    Row or None
        The accepted row after a phase step (see :func:`phase_step`) or, for
        a 2- or 3-state series, a frequency step (see
        :func:`frequency_step`). ``None`` while fewer than three
        consecutive rejects are counted, when the rejects show neither, and
        for a frequency step of a 1-state series, which has no rate.

    Raises
    ------
    FilterError
        If the draft has no innovation scale, or the finished row breaks a
        rule of :class:`Row`.
    """
    if draft.consecutive_rejects < _STEP_REJECTS:
        return None
    if draft.innovation_scale is None:
        message = f"a reject at {draft.interpolated_datetime} has no innovation scale"
        _log.error(message)
        raise FilterError(message)
    classified = classify(draft.rejects, draft.innovation_scale)
    if classified.kind == "phase":
        return phase_step(draft, prediction, z, floor)
    if (
        classified.kind == "frequency"
        and draft.filter_states > 1
        and classified.a is not None
        and classified.s is not None
    ):
        return frequency_step(
            draft, prediction, classified.a, classified.s, z, floor, params
        )
    return None


# ---------------------------------------------------------------- acquisition

_ACQUIRE_LIMIT: Final[float] = K_OUT * math.sqrt(6)
"""The second-difference limit of acquisition, in initial innovation scales.

The second difference of three measurements with independent noise sigma0
has standard deviation sqrt(6) sigma0; the limit is :data:`K_OUT` of those.
"""


def _consecutive(rejects: tuple[Reject, ...]) -> bool:
    """Tell whether buffered measurements come from consecutive epochs.

    Parameters
    ----------
    rejects : tuple of (datetime, float)
        The buffer, oldest first.

    Returns
    -------
    bool
        Whether each entry's epoch is one epoch after the one before.
    """
    return all(
        seconds(later, earlier) == _T for (earlier, _), (later, _) in pairwise(rejects)
    )


def acquire(draft: RowDraft, z: int, params: SeriesParams) -> Row:
    """Buffer a dormant series' measurement; cold-start when it is consistent.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a series with no prediction, its buffer
        holding the measurements it has gathered, as (epoch, z).
    z : int
        The measurement at the epoch, ps, decycled against the last buffered
        one (see :func:`anchor_of`).
    params : SeriesParams
        The settings in force, whose ``sigma0`` sets the test.

    Returns
    -------
    Row
        A cold start from ``z`` (see :func:`cold_start`) when the buffer,
        with ``z`` added and the newest
        :data:`~masterclock.domain.series.MAX_REJECTS` kept, holds three
        measurements from consecutive epochs whose second difference
        z3 - 2 z2 + z1 is at most 5 sqrt(6) sigma0 either way (design 13.3).
        Otherwise a dormant R row keeping that buffer, with no counted
        rejects.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    entry = (draft.interpolated_datetime, float(z))
    rejects = (*draft.rejects, entry)[-MAX_REJECTS:]
    buffered = dataclasses.replace(draft, rejects=rejects)
    if len(rejects) == MAX_REJECTS and _consecutive(rejects):
        z1, z2, z3 = (exact(value) for _, value in rejects)
        if abs(z3 - 2 * z2 + z1) <= exact(_ACQUIRE_LIMIT * params.sigma0):
            return cold_start(buffered, z, params)
    return dormant(
        dataclasses.replace(buffered, consecutive_rejects=0), "R", keep_buffer=True
    )


def anchor_of(last: Row | None) -> int | None:
    """Give what a series with no prediction is decycled against (design 7.5).

    Parameters
    ----------
    last : Row or None
        The series' last row, or ``None`` for a new series.

    Returns
    -------
    int or None
        The newest measurement in a dormant row's buffer, ps; ``None`` for
        no row, a row that is not dormant, or an empty buffer.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> from masterclock.domain.series import Row
    >>> row = Row(
    ...     interpolated_datetime=datetime(2025, 9, 23, 6, 0, tzinfo=UTC),
    ...     innovation=None, x_fs=None, y=None, d=None, innovation_scale=None,
    ...     segment=0, step_offset=0, epochs_in_segment=0,
    ...     epochs_since_accept=1, consecutive_rejects=0,
    ...     rejects=((datetime(2025, 9, 23, 6, 0, tzinfo=UTC), 1234577.0),),
    ...     filter_states=1, time_constant=None, scale_time_constant=50.0,
    ...     flags="RD",
    ... )
    >>> anchor_of(row)
    1234577
    """
    if last is None or "D" not in last.flags or not last.rejects:
        return None
    return round_even(exact(last.rejects[-1][1]))
