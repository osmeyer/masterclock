"""The forward estimator every pair and triple runs, and the rows it writes.

Each series runs a critically damped state-space filter with fixed gains:
three states (phase, rate, drift) with a triple pole at lambda = exp(-1/M),
two states (phase, rate) with a double pole there, or one state, which
passes an accepted measurement through unchanged. The pole sits at lambda
for every M, so the closed loop (I - K H) Phi has every eigenvalue equal to
lambda.

The phase is summed exactly, as a :class:`~gmpy2.mpq`, and rounded
only when a row stores it, to whole femtoseconds (see
:data:`~masterclock.domain.phase.FS_PER_PS`); rates, drifts and gains are
floats.

Each epoch gives every series one row. The row starts as a
:class:`RowDraft` carried on from the last row (:func:`carry`), which each
step changes in place: it may begin a new segment (:func:`start_segment`),
and is finished by what the epoch did: an accepted measurement updates the
state (:func:`accept`); a rejected, excluded or missing one leaves the
prediction standing (:func:`hold`); a series with no valid state is dormant
(:func:`dormant`): it buffers its measurements (:func:`acquire`, decycled
against :func:`anchor_of`) until three agree, and starts again from the
third alone (:func:`cold_start`). :func:`finish` builds the row from the
draft once, and checks it.

A measurement is accepted when it passes the gate (:func:`within_gate`,
:func:`rms_ok`). One that fails is a counted reject (:func:`count_reject`),
and three in a row are looked at for a step (:func:`classify`): agreeing,
they are a phase step; on a line, a frequency step (:func:`accept_step`).

:func:`filter_step` puts it all together for one series at one epoch: it
takes the series' measurement as plain values (:class:`FilterInput`) and
gives its row and whether it cold-started (:class:`StepResult`).
"""

import functools
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Final, Literal, cast

from gmpy2 import mpq

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError
from masterclock.domain.phase import (
    EPOCH_SECONDS,
    exact,
    exact_ratio,
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
    Row,
    SeriesParams,
    State,
    check_row,
)

_T: Final[int] = EPOCH_SECONDS
"""One epoch, s: whole, so a phase moved on by it stays exact."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def gains(filter_states: int, M: float | None) -> tuple[float, float, float]:
    """Give the fixed gains of a model with time constant M.

    Parameters
    ----------
    filter_states : int
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
        If ``filter_states`` is not 1, 2 or 3, or ``M`` is given for one state or is
        not at least 1 for two or three.

    Examples
    --------
    >>> gains(1, None)
    (1.0, 0.0, 0.0)
    >>> g, h_over_t, drift = gains(2, 10.0)
    >>> round(g, 6), drift
    (0.181269, 0.0)
    """
    if filter_states == 1 and M is None:
        return (1.0, 0.0, 0.0)
    if filter_states not in {2, 3} or M is None or not M >= 1:
        message = f"no gains for a {filter_states}-state model with time constant {M}"
        _log.error(message)
        raise FilterError(message)
    lam = math.exp(-1.0 / M)
    if filter_states == 3:
        return (
            1 - lam**3,
            1.5 * (1 - lam) ** 2 * (1 + lam) / _T,
            (1 - lam) ** 3 / _T**2,
        )
    return (1 - lam**2, (1 - lam) ** 2 / _T, 0.0)


@functools.cache
def exact_gains(filter_states: int, M: float | None) -> tuple[mpq, float, float]:
    """Give the gains of :func:`gains`, the phase gain as the fraction it holds.

    Each model's gains are worked out once in a run and the same values
    given after, since every accepted row of a series uses them.

    Parameters
    ----------
    filter_states : int
        How many states the estimator has: 1, 2 or 3.
    M : float or None
        The time constant, epochs; ``None`` for one state.

    Returns
    -------
    tuple of (mpq, float, float)
        g exactly, as :func:`~masterclock.domain.phase.exact` gives it, then
        h/T and 2k/T**2.

    Raises
    ------
    FilterError
        As :func:`gains`, each time it is asked.

    Examples
    --------
    >>> exact_gains(1, None)
    (mpq(1,1), 0.0, 0.0)
    """
    g, h_over_t, two_k_over_t2 = gains(filter_states, M)
    return exact(g), h_over_t, two_k_over_t2


def predict(last_row: Row | None, u: tuple[mpq, float]) -> State | None:
    """Predict a series' state at the next epoch from its last row.

    Parameters
    ----------
    last_row : Row or None
        The series' last row, or ``None`` for a series with none.
    u : tuple of (mpq, float)
        The steering input over the epoch: u_x, exact, and u_y.

    Returns
    -------
    State or None
        Phi X + u: the phase moved on by y T + d T**2 / 2 and u_x, exactly;
        the rate by d T and u_y; the drift kept. A 1-state row keeps its
        phase and has no rate; a 2-state row has no drift. ``None`` when
        there is no last row, or it is dormant.
    """
    if last_row is None or "D" in last_row.flags:
        return None
    x_fs, y, d = last_row.known_state()
    x = from_fs(x_fs)
    ux, uy = u
    if last_row.filter_states == 1:
        return State(x=x + ux, y=0.0)
    if last_row.filter_states == 3:
        return State(
            x=x + exact(y) * _T + exact(d) * _T * _T / 2 + ux,
            y=y + d * _T + uy,
            d=d,
        )
    return State(x=x + exact(y) * _T + ux, y=y + uy)


def update(
    prediction: State,
    innovation: mpq,
    filter_states: int,
    M: float | None,
    *,
    nu: float | None = None,
) -> State:
    """Correct a prediction by the gains times the innovation.

    Parameters
    ----------
    prediction : State
        The predicted state at the epoch.
    innovation : mpq
        The measurement less the predicted phase, exact.
    filter_states : int
        How many states the estimator has.
    M : float or None
        The time constant; ``None`` for one state.
    nu : float or None, optional
        ``float(innovation)``, when the caller has it already; worked out
        here when ``None``.

    Returns
    -------
    State
        X + K nu: the phase exact, to be rounded when it is stored; the
        rate and, for three states, the drift as floats.

    Raises
    ------
    FilterError
        If ``filter_states`` and ``M`` do not belong together (see :func:`gains`).
    """
    g, h_over_t, two_k_over_t2 = exact_gains(filter_states, M)
    if nu is None:
        nu = float(innovation)
    return State(
        x=prediction.x + g * innovation,
        y=prediction.y + h_over_t * nu,
        d=prediction.d + two_k_over_t2 * nu if filter_states == 3 else 0.0,
    )


# ------------------------------------------------------------ row lifecycle

type Outcome = Literal["A", "R", "X", "P"]
"""What an epoch did to a series; every row carries exactly one."""

type Held = Literal["R", "X", "P"]
"""An outcome that leaves the state unupdated: rejected, excluded, predicted."""

SETTLE_FACTOR: Final[int] = 5
"""A segment is unsettled while it has run fewer than this many times M rows."""


@dataclass(slots=True)
class RowDraft:
    """A row being built for an epoch: the fields of a :class:`Row`, unchecked.

    An epoch builds its row a field or two at a time, and part way it is no
    valid row: it has no outcome flag until the last step. So the steps
    change one draft in place, and :func:`finish` builds the :class:`Row`
    from it once, and checks it. The fields are a row's, with the same names
    and meanings. A draft belongs to the one epoch of one series that made
    it, and is not used once it is finished.
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
    epoch_start: datetime,
    last_row: Row | None,
    series_params: SeriesParams,
    *,
    slip: bool = False,
) -> RowDraft:
    """Start an epoch's row from the series' last row.

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    last_row : Row or None
        The series' last row, or ``None`` for a new series.
    series_params : SeriesParams
        The series' settings at E; used only for a new series.
    slip : bool, optional
        Whether the slip check corrected this epoch's measurement; the row
        then carries S.

    Returns
    -------
    RowDraft
        For a new series: dormant, with no state, in segment 0, every
        counter 0, the model and time constants of ``series_params``. Otherwise
        the last row moved on to ``epoch_start``, with no innovation, one more
        epoch in its segment and no flags. S when ``slip``.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> settings = SeriesParams(
    ...     filter_states=1, M=None, M_sigma=50.0, sigma0=5.0, gmax=432, n_break=36,
    ...     rms_max=80,
    ... )
    >>> draft = carry(datetime(2025, 9, 23, 6, 0, tzinfo=UTC), None, settings)
    >>> draft.segment, draft.x_fs, draft.flags
    (0, None, '')
    """
    flags = "S" if slip else ""
    if last_row is None:
        return RowDraft(
            interpolated_datetime=epoch_start,
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
            filter_states=series_params.filter_states,
            time_constant=series_params.M,
            scale_time_constant=series_params.M_sigma,
            flags=flags,
        )
    return RowDraft(
        interpolated_datetime=epoch_start,
        innovation=None,
        x_fs=last_row.x_fs,
        y=last_row.y,
        d=last_row.d,
        innovation_scale=last_row.innovation_scale,
        segment=last_row.segment,
        step_offset=last_row.step_offset,
        epochs_in_segment=last_row.epochs_in_segment + 1,
        epochs_since_accept=last_row.epochs_since_accept,
        consecutive_rejects=last_row.consecutive_rejects,
        rejects=last_row.rejects,
        filter_states=last_row.filter_states,
        time_constant=last_row.time_constant,
        scale_time_constant=last_row.scale_time_constant,
        flags=flags,
    )


def start_segment(
    draft: RowDraft, series_params: SeriesParams, *, keep_offset: bool
) -> None:
    """Begin a new segment on a draft, in place.

    The draft's segment number becomes one more, with no rows in it yet,
    the time constants of ``series_params`` and N among the flags. The
    state is not touched.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    series_params : SeriesParams
        The settings in force at the epoch, whose M and M_sigma the segment
        takes.
    keep_offset : bool
        Whether the segment keeps the step offset: a warm start does, a
        cold start sets it to 0.

    Raises
    ------
    FilterError
        If ``series_params`` is for another model than the series': a series keeps
        its model for the life of its file.
    """
    if series_params.filter_states != draft.filter_states:
        message = (
            f"a {draft.filter_states}-state series keeps its model;"
            f" its settings name model {series_params.filter_states}"
        )
        _log.error(message)
        raise FilterError(message)
    draft.segment += 1
    draft.epochs_in_segment = 0
    if not keep_offset:
        draft.step_offset = 0
    draft.time_constant = series_params.M
    draft.scale_time_constant = series_params.M_sigma
    draft.flags += "N"


def finish(draft: RowDraft, outcome: Outcome) -> Row:
    """Add the epoch's outcome to a draft, and build and check its row.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    outcome : {'A', 'R', 'X', 'P'}
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
        If the finished row breaks a rule of :class:`Row` (see
        :func:`~masterclock.domain.series.check_row`).
    """
    flags = draft.flags + outcome
    if (
        "D" not in flags
        and draft.time_constant is not None
        and draft.epochs_in_segment < SETTLE_FACTOR * draft.time_constant
    ):
        flags += "U"
    row = Row(
        interpolated_datetime=draft.interpolated_datetime,
        innovation=draft.innovation,
        x_fs=draft.x_fs,
        y=draft.y,
        d=draft.d,
        innovation_scale=draft.innovation_scale,
        segment=draft.segment,
        step_offset=draft.step_offset,
        epochs_in_segment=draft.epochs_in_segment,
        epochs_since_accept=draft.epochs_since_accept,
        consecutive_rejects=draft.consecutive_rejects,
        rejects=draft.rejects,
        filter_states=draft.filter_states,
        time_constant=draft.time_constant,
        scale_time_constant=draft.scale_time_constant,
        flags="".join(letter for letter in FLAG_ORDER if letter in flags),
    )
    try:
        check_row(row)
    except ValueError as exc:
        message = f"invalid row of {row.interpolated_datetime}: {exc}"
        _log.error(message)
        raise FilterError(message) from exc
    return row


def dormant(draft: RowDraft, outcome: Held, *, keep_buffer: bool = False) -> Row:
    """Finish a draft as a dormant row: one with no valid state.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    outcome : {'R', 'X', 'P'}
        The outcome: a measurement buffered or rejected, or none.
    keep_buffer : bool, optional
        Whether the buffer stays: it does while a dormant series gathers
        measurements to acquire from; otherwise it is emptied.

    Returns
    -------
    Row
        The row with no phase, rate, drift or innovation scale, flag D
        beside ``outcome``, and segment and step offset as they were.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    draft.x_fs = draft.y = draft.d = draft.innovation_scale = None
    if not keep_buffer:
        draft.rejects = ()
    draft.flags += "D"
    return finish(draft, outcome)


def hold(
    draft: RowDraft,
    prediction: State | None,
    outcome: Held,
    series_params: SeriesParams,
) -> Row:
    """Finish a draft as a held row: no update, the prediction carried.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, with its innovation when there was a
        measurement.
    prediction : State or None
        The series' prediction at the epoch, or ``None`` when it has none.
    outcome : {'R', 'X', 'P'}
        Why the state is not updated: a counted reject, an excluded
        measurement, or no measurement.
    series_params : SeriesParams
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
    draft.epochs_since_accept += 1
    if prediction is None or draft.epochs_since_accept > series_params.gmax:
        return dormant(draft, outcome)
    draft.x_fs, draft.y, draft.d = to_fs(prediction.x), prediction.y, prediction.d
    return finish(draft, outcome)


def accept(
    draft: RowDraft,
    prediction: State,
    innovation: mpq,
    scale_floor: float,
    *,
    nu: float | None = None,
) -> Row:
    """Finish a draft as an accepted row: the prediction updated.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a series with an innovation scale.
    prediction : State
        The series' prediction at the epoch.
    innovation : mpq
        The measurement less the predicted phase, exact.
    scale_floor : float
        The lowest the innovation scale may go, ps: the measurement's own
        rms for a pair, sigma_dd for a triple.
    nu : float or None, optional
        ``float(innovation)``, when the caller has it already; worked out
        here when ``None``, once for the row.

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
    if nu is None:
        nu = float(innovation)
    updated_state = update(
        prediction, innovation, draft.filter_states, draft.time_constant, nu=nu
    )
    w = 1.0 / draft.scale_time_constant
    new_scale = math.sqrt(
        max((1 - w) * draft.innovation_scale**2 + w * nu**2, scale_floor**2)
    )
    draft.innovation = nu
    draft.x_fs, draft.y, draft.d = (
        to_fs(updated_state.x),
        updated_state.y,
        updated_state.d,
    )
    draft.innovation_scale = new_scale
    draft.consecutive_rejects = draft.epochs_since_accept = 0
    draft.rejects = ()
    return finish(draft, "A")


def cold_start(draft: RowDraft, z: int, series_params: SeriesParams) -> Row:
    """Finish a draft as a cold start: a new segment from the measurement alone.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a dormant series.
    z : int
        The measurement at the epoch, ps.
    series_params : SeriesParams
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
        If ``series_params`` is for another model than the series', or the
        finished row breaks a rule of :class:`Row`.
    """
    start_segment(draft, series_params, keep_offset=False)
    draft.x_fs, draft.y, draft.d = to_fs(z), 0.0, 0.0
    draft.innovation_scale = series_params.sigma0
    draft.consecutive_rejects = draft.epochs_since_accept = 0
    draft.rejects = ()
    return finish(draft, "A")


# ------------------------------------------------------- step classification

K_OUT: Final[float] = 5.0
"""How many innovation scales wide the gate is, either way."""

K_STEP: Final[float] = 3.0
"""How many innovation scales three rejects may stray from a step and show it."""

_STEP_REJECTS: Final[int] = 3
"""How many consecutive counted rejects a step is looked for in."""


@dataclass(frozen=True, slots=True)
class Classified:
    """What three consecutive rejects show: a phase step, a frequency step, or neither.

    Parameters
    ----------
    step_kind : {'phase', 'frequency'} or None
        The kind of step; ``None`` when the rejects show neither.
    a : float or None, optional
        For a frequency step, the fitted line's value at the first reject,
        ps; ``None`` otherwise.
    s : float or None, optional
        For a frequency step, the fitted line's slope, ps/s; ``None``
        otherwise.
    """

    step_kind: Literal["phase", "frequency"] | None
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
    >>> classify(tuple(zip(epochs, (150.0, 151.0, 149.0))), 3.0).step_kind
    'phase'
    >>> classify(tuple(zip(epochs, (30.0, 60.0, 90.0))), 3.0).step_kind
    'frequency'
    """
    if len(rejects) != _STEP_REJECTS:
        message = f"a step is looked for in three rejects, not {len(rejects)}"
        _log.error(message)
        raise FilterError(message)
    first_epoch = rejects[0][0]
    ts = [float(seconds(reject_epoch, first_epoch)) for reject_epoch, _ in rejects]
    vs = [reject_innovation for _, reject_innovation in rejects]
    vbar = sum(vs) / _STEP_REJECTS
    if max(abs(v - vbar) for v in vs) < K_STEP * sigma:
        return Classified(step_kind="phase")
    tbar = sum(ts) / _STEP_REJECTS
    s = sum((t - tbar) * (v - vbar) for t, v in zip(ts, vs, strict=True)) / sum(
        (t - tbar) ** 2 for t in ts
    )
    a = vbar - s * tbar
    if max(abs(v - a - s * t) for t, v in zip(ts, vs, strict=True)) < K_STEP * sigma:
        return Classified(step_kind="frequency", a=a, s=s)
    return Classified(step_kind=None)


def within_gate(innovation: mpq, innovation_scale: float) -> bool:
    """Tell whether an innovation passes the gate's width (design 9.1).

    Parameters
    ----------
    innovation : mpq
        The measurement less the predicted phase, exact.
    innovation_scale : float
        The innovation scale, ps.

    Returns
    -------
    bool
        Whether the innovation is at most :data:`K_OUT` scales either way,
        compared exactly: the float K_OUT times the scale is taken at the
        value it holds, and both sides are compared as whole numbers.

    Raises
    ------
    FilterError
        If K_OUT times the scale is not finite.

    Examples
    --------
    >>> within_gate(mpq(15), 3.0), within_gate(mpq(-31, 2), 3.0)
    (True, False)
    """
    gate_numerator, gate_denominator = exact_ratio(K_OUT * innovation_scale)
    return (
        abs(innovation.numerator) * gate_denominator
        <= gate_numerator * innovation.denominator
    )


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


def count_reject(draft: RowDraft, innovation: mpq) -> None:
    """Count a rejected measurement and put it in the reject buffer (design 9.3).

    The draft gets one more consecutive reject, and (epoch, innovation)
    added to its buffer, which keeps the newest
    :data:`~masterclock.domain.series.MAX_REJECTS`.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far; changed in place.
    innovation : mpq
        The rejected measurement less the predicted phase.
    """
    reject_entry = (draft.interpolated_datetime, float(innovation))
    draft.consecutive_rejects += 1
    draft.rejects = (*draft.rejects, reject_entry)[-MAX_REJECTS:]


def phase_step(draft: RowDraft, prediction: State, z: int, scale_floor: float) -> Row:
    """Accept a measurement after a phase step the rejects agree on (design 9.4).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, its buffer holding the three rejects.
    prediction : State
        The series' prediction at the epoch.
    z : int
        The current measurement, ps.
    scale_floor : float
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
    innovation_sum = sum(
        (exact(reject_innovation) for _, reject_innovation in draft.rejects),
        mpq(0),
    )
    step_ps = round_even(innovation_sum / _STEP_REJECTS)
    corrected_prediction = State(
        x=prediction.x + step_ps, y=prediction.y, d=prediction.d
    )
    draft.step_offset += step_ps
    return accept(draft, corrected_prediction, z - corrected_prediction.x, scale_floor)


def frequency_step(
    draft: RowDraft,
    prediction: State,
    a: float,
    s: float,
    z: int,
    scale_floor: float,
    series_params: SeriesParams,
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
    scale_floor : float
        The lowest the innovation scale may go, ps (see :func:`accept`).
    series_params : SeriesParams
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
        If ``series_params`` is for another model than the series', or the
        finished row breaks a rule of :class:`Row`.
    """
    t3 = seconds(draft.rejects[-1][0], draft.rejects[0][0])
    corrected_prediction = State(
        x=prediction.x + exact(a) + exact(s) * t3, y=prediction.y + s, d=prediction.d
    )
    start_segment(draft, series_params, keep_offset=True)
    return accept(draft, corrected_prediction, z - corrected_prediction.x, scale_floor)


def accept_step(
    draft: RowDraft,
    prediction: State,
    z: int,
    scale_floor: float,
    series_params: SeriesParams,
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
    scale_floor : float
        The lowest the innovation scale may go, ps (see :func:`accept`).
    series_params : SeriesParams
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
    if classified.step_kind == "phase":
        return phase_step(draft, prediction, z, scale_floor)
    if (
        classified.step_kind == "frequency"
        and draft.filter_states > 1
        and classified.a is not None
        and classified.s is not None
    ):
        return frequency_step(
            draft, prediction, classified.a, classified.s, z, scale_floor, series_params
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
        seconds(later_epoch, earlier_epoch) == _T
        for (earlier_epoch, _), (later_epoch, _) in pairwise(rejects)
    )


def acquire(draft: RowDraft, z: int, series_params: SeriesParams) -> Row:
    """Buffer a dormant series' measurement; cold-start when it is consistent.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a series with no prediction, its buffer
        holding the measurements it has gathered, as (epoch, z).
    z : int
        The measurement at the epoch, ps, decycled against the last buffered
        one (see :func:`anchor_of`).
    series_params : SeriesParams
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
    buffer_entry = (draft.interpolated_datetime, float(z))
    draft.rejects = (*draft.rejects, buffer_entry)[-MAX_REJECTS:]
    if len(draft.rejects) == MAX_REJECTS and _consecutive(draft.rejects):
        z1, z2, z3 = (exact(buffered_z) for _, buffered_z in draft.rejects)
        if abs(z3 - 2 * z2 + z1) <= exact(_ACQUIRE_LIMIT * series_params.sigma0):
            return cold_start(draft, z, series_params)
    draft.consecutive_rejects = 0
    return dormant(draft, "R", keep_buffer=True)


def anchor_of(last_row: Row | None) -> int | None:
    """Give what a series with no prediction is decycled against (design 7.5).

    Parameters
    ----------
    last_row : Row or None
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
    if last_row is None or "D" not in last_row.flags or not last_row.rejects:
        return None
    return round_even(exact(last_row.rejects[-1][1]))


# ------------------------------------------------------------- filter step


@dataclass(frozen=True, slots=True)
class FilterInput:
    """What one series measured at an epoch: a pair's or a triple's value.

    Parameters
    ----------
    z : int
        The measurement at the epoch start, ps: a pair's decycled phase or
        a triple's double difference.
    rms : int or None, optional
        For a pair, the rms the DAS gave, ps, at least 0; ``None`` for a
        triple.
    sigma_dd : float or None, optional
        For a triple, the double difference's sigma, ps, zero or more, as
        a pair's rms may be; ``None`` for a pair.
    slip : bool, optional
        Whether the slip check corrected a pair's cycle count. Never set
        for a triple.
    pair_cold_started : bool, optional
        Whether a pair that gave a triple's value cold-started at the
        epoch. Never set for a pair.

    Raises
    ------
    FilterError
        If both or neither of ``rms`` and ``sigma_dd`` are given, a triple is
        marked as slip corrected, or a pair as following a cold start.

    Examples
    --------
    >>> FilterInput(z=1_234_577, rms=3).scale_floor
    3.0
    >>> FilterInput(z=6_666_667, sigma_dd=3.3166).scale_floor
    3.3166
    """

    z: int
    rms: int | None = None
    sigma_dd: float | None = None
    slip: bool = False
    pair_cold_started: bool = False

    def __post_init__(self) -> None:
        """Refuse a measurement that is not wholly a pair's or a triple's.

        Raises
        ------
        FilterError
            If both or neither of ``rms`` and ``sigma_dd`` are given, a
            triple is slip corrected or a pair follows a cold start.
        """
        message = ""
        if (self.rms is None) == (self.sigma_dd is None):
            message = "a measurement has an rms (a pair) or a sigma_dd (a triple)"
        elif self.sigma_dd is not None and self.slip:
            message = "the slip check corrects pairs only"
        elif self.rms is not None and self.pair_cold_started:
            message = "only a triple follows its component pairs' cold starts"
        if message:
            _log.error(message)
            raise FilterError(message)

    @property
    def scale_floor(self) -> float:
        """The lowest the innovation scale may go: a pair's rms, a triple's sigma_dd."""
        if self.rms is not None:
            return float(self.rms)
        return cast("float", self.sigma_dd)  # a triple's, given when rms is not


@dataclass(frozen=True, slots=True)
class StepResult:
    """The row a series writes at an epoch, and whether it cold-started there.

    Parameters
    ----------
    row : Row
        The series' row at the epoch.
    cold_started : bool
        Whether the row is a cold start: a triple built from this pair's
        value then goes dormant (design 12.6).
    """

    row: Row
    cold_started: bool


def params_changed(series_params: SeriesParams, last_row: Row) -> bool:
    """Tell whether a series' settings changed its time constants (design 8.7).

    Parameters
    ----------
    series_params : SeriesParams
        The settings in force at the epoch.
    last_row : Row
        The series' last row, which carries its segment's time constants.

    Returns
    -------
    bool
        Whether M or M_sigma differ from the last row's.
    """
    return (series_params.M, series_params.M_sigma) != (
        last_row.time_constant,
        last_row.scale_time_constant,
    )


def filter_step(
    epoch_start: datetime,
    series_params: SeriesParams,
    last_row: Row | None,
    prediction: State | None,
    measurement: FilterInput | None,
    *,
    excluded: bool = False,
) -> StepResult:
    """Give a series' row at an epoch: the whole of the decision flow (design 9.6).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    series_params : SeriesParams
        The settings in force at E.
    last_row : Row or None
        The series' last row, or ``None`` for a new series.
    prediction : State or None
        The prediction at E from ``last_row`` (see :func:`predict`), or
        ``None`` when the series has none.
    measurement : FilterInput or None
        The series' measurement at E, or ``None`` when there is none.
    excluded : bool, optional
        Whether screening or the slip check excluded the measurement.

    Returns
    -------
    StepResult
        The row, and whether it cold-started. First, a tracked series whose
        time constants changed starts a warm segment (see
        :func:`start_segment`); the row then also carries the outcome.
        With no measurement, the row holds the prediction (see :func:`hold`).
        A series with no prediction, or a triple one of whose pairs
        cold-started, acquires (see :func:`acquire`). Otherwise the
        measurement goes through the gate.

    Raises
    ------
    FilterError
        If ``series_params`` is for another model than the series', or a row
        breaks a rule of :class:`Row`.
    """
    draft = carry(
        epoch_start,
        last_row,
        series_params,
        slip=measurement is not None and measurement.slip,
    )
    if (
        last_row is not None
        and "D" not in last_row.flags
        and params_changed(series_params, last_row)
    ):
        start_segment(draft, series_params, keep_offset=True)
    if measurement is None:
        return StepResult(
            row=hold(draft, prediction, "P", series_params), cold_started=False
        )
    if measurement.pair_cold_started:
        draft.rejects = ()
        prediction = None
    if prediction is None:
        row = acquire(draft, measurement.z, series_params)
        return StepResult(row=row, cold_started="D" not in row.flags)
    row = _gate(draft, prediction, measurement, series_params, excluded=excluded)
    return StepResult(row=row, cold_started=False)


def _gate(
    draft: RowDraft,
    prediction: State,
    measurement: FilterInput,
    series_params: SeriesParams,
    *,
    excluded: bool,
) -> Row:
    """Accept, hold or reject a measurement against its prediction (design 9.6).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    prediction : State
        The series' prediction at the epoch.
    measurement : FilterInput
        The measurement.
    series_params : SeriesParams
        The settings in force.
    excluded : bool
        Whether screening or the slip check excluded the measurement.

    Returns
    -------
    Row
        Accepted when the measurement passes the gate and is not excluded.
        Held as X when it is excluded inside the gate, not counted.
        Otherwise a counted reject, which may show a step (see
        :func:`accept_step`), make the series dormant when the rejects
        reach ``n_break`` (the measurement then starts its acquisition
        buffer), or is held as R.

    Raises
    ------
    FilterError
        If a row breaks a rule of :class:`Row`.
    """
    innovation = measurement.z - prediction.x
    nu = float(innovation)
    draft.innovation = nu
    if draft.innovation_scale is None:
        message = f"a measurement at {draft.interpolated_datetime} has no scale"
        _log.error(message)
        raise FilterError(message)
    in_gate = within_gate(innovation, draft.innovation_scale)
    rms_passes = measurement.rms is None or rms_ok(
        measurement.rms, series_params.rms_max
    )
    if in_gate and rms_passes and not excluded:
        return accept(draft, prediction, innovation, measurement.scale_floor, nu=nu)
    if in_gate and excluded:
        return hold(draft, prediction, "X", series_params)
    count_reject(draft, innovation)
    step_row = accept_step(
        draft, prediction, measurement.z, measurement.scale_floor, series_params
    )
    if step_row is not None:
        return step_row
    if draft.consecutive_rejects >= series_params.n_break:
        draft.rejects = ()
        return acquire(draft, measurement.z, series_params)
    return hold(draft, prediction, "R", series_params)
