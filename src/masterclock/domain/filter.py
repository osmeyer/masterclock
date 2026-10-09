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
third, at the rate the three show (:func:`cold_start`). :func:`finish`
builds the row from the draft once, and checks it. Every row goes into the
series' file but a dormant one with no measurement (:func:`writes_row`). A
disabled pair is not tracked at all: :func:`disabled_step` gives its row, of
flag O alone.

A measurement is accepted when it passes the gate (:func:`within_gate`,
:func:`rms_ok`). One that fails is a counted reject (:func:`count_reject`),
and three in a row are looked at for a step (:func:`classify`): agreeing,
they are a phase step; on a line, a frequency step (:func:`accept_step`). A
pair's reading over its rms limit is never accepted, not by a step and not
by acquisition: it is a counted reject that empties the buffer, so it is
never one of the three a step or a cold start is found in. Each counted
reject raises the series' reject fraction and each accepted reading lowers
it; a series goes dormant when its rejects in a row reach N_break or its
reject fraction passes its limit, so rejects spaced by single accepts send
it back to acquisition as a run of rejects does.

:func:`filter_step` puts it all together for one series at one epoch: it
takes the series' measurement as plain values (:class:`FilterInput`) and
gives its row, whether it cold-started, which kind of step it accepted and
why it went dormant, when it did (:class:`StepResult`).
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

    Each model's gains are worked out once in each process and the same
    values given after, since every accepted row of a series uses them.

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
        the rate by d T and u_y; the drift kept. A 1-state row's phase
        moves by u_x alone and its rate is 0; a 2-state row has no drift.
        ``None`` when there is no last row, or it is dormant.
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

type Outcome = Literal["A", "R", "X", "P", "O"]
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
    step_offset: int
    epochs_in_segment: int
    epochs_since_accept: int
    consecutive_rejects: int
    reject_fraction: float
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
        The series' last row, or ``None`` for a new series, or one that
        starts again after epochs it had no row for.
    series_params : SeriesParams
        The series' settings at E; used only for a new series.
    slip : bool, optional
        Whether the slip check corrected this epoch's measurement; the row
        then carries S.

    Returns
    -------
    RowDraft
        For a new series: dormant, with no state, every counter 0, the model
        and time constants of ``series_params``. Otherwise the last row's
        fields, its state unchanged and its datetime set to ``epoch_start``,
        with no innovation, one more epoch in its segment and no flags. S
        when ``slip``.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> settings = SeriesParams(
    ...     filter_states=1, M=None, M_sigma=50.0, sigma0=5.0, gmax=432, n_break=36,
    ...     reject_fraction_weight=0.04, reject_fraction_limit=0.5, rms_max=80,
    ... )
    >>> draft = carry(datetime(2025, 9, 23, 6, 0, tzinfo=UTC), None, settings)
    >>> draft.x_fs, draft.flags
    (None, '')
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
            step_offset=0,
            epochs_in_segment=0,
            epochs_since_accept=0,
            consecutive_rejects=0,
            reject_fraction=0.0,
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
        step_offset=last_row.step_offset,
        epochs_in_segment=last_row.epochs_in_segment + 1,
        epochs_since_accept=last_row.epochs_since_accept,
        consecutive_rejects=last_row.consecutive_rejects,
        reject_fraction=last_row.reject_fraction,
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

    The draft has no rows in its segment yet, the time constants of
    ``series_params`` and N among the flags. The state is not touched.

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
    outcome : {'A', 'R', 'X', 'P', 'O'}
        The outcome.

    Returns
    -------
    Row
        The checked row, its flags in
        :data:`~masterclock.domain.series.FLAG_ORDER`, with U added to a row
        of a 2- or 3-state series that is neither dormant nor disabled while
        its segment has run fewer than :data:`SETTLE_FACTOR` times M rows.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row` (see
        :func:`~masterclock.domain.series.check_row`).
    """
    flags = draft.flags + outcome
    if (
        "D" not in flags
        and outcome != "O"
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
        step_offset=draft.step_offset,
        epochs_in_segment=draft.epochs_in_segment,
        epochs_since_accept=draft.epochs_since_accept,
        consecutive_rejects=draft.consecutive_rejects,
        reject_fraction=draft.reject_fraction,
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


def dormant(
    draft: RowDraft,
    outcome: Held,
    *,
    keep_buffer: bool = False,
    keep_scale: bool = False,
) -> Row:
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
    keep_scale : bool, optional
        Whether the innovation scale stays, as the draft has it: it does
        while a dormant series acquires, so a scale its reject fraction
        left it is kept for its cold start; otherwise it is emptied.

    Returns
    -------
    Row
        The row with no phase, rate or drift, no innovation scale unless it
        is kept, a reject fraction of 0, flag D beside ``outcome``, and the
        step offset as it was.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    draft.x_fs = draft.y = draft.d = None
    if not keep_scale:
        draft.innovation_scale = None
    draft.reject_fraction = 0.0
    if not keep_buffer:
        draft.rejects = ()
    draft.flags += "D"
    return finish(draft, outcome)


def writes_row(row: Row) -> bool:
    """Tell whether a series' row at an epoch goes into its file (design 13.3).

    Parameters
    ----------
    row : Row
        The series' row at the epoch.

    Returns
    -------
    bool
        Whether the row is written: every row is, except a dormant one with
        no measurement (flags D and P). So a series whose measurements stop
        writes its predicted rows up to its gap limit, and then nothing until
        it is measured again.
    """
    return not ("D" in row.flags and "P" in row.flags)


def disabled_step(
    epoch_start: datetime, series_params: SeriesParams, *, measured: bool
) -> StepResult:
    """Give the row of a disabled series at an epoch: no tracking (design 13.6).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    series_params : SeriesParams
        The settings in force at E, whose model and time constants the row
        takes.
    measured : bool
        Whether the series has a reading at E.

    Returns
    -------
    StepResult
        With a reading, a row of flag O alone: no state, innovation,
        counters or buffer. With none, a dormant row with no measurement,
        which is not written (see :func:`writes_row`). Never a cold start,
        and never a step.

    Raises
    ------
    FilterError
        If the row breaks a rule of :class:`Row`.
    """
    draft = RowDraft(
        interpolated_datetime=epoch_start,
        innovation=None,
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        step_offset=0,
        epochs_in_segment=0,
        epochs_since_accept=0,
        consecutive_rejects=0,
        reject_fraction=0.0,
        rejects=(),
        filter_states=series_params.filter_states,
        time_constant=series_params.M,
        scale_time_constant=series_params.M_sigma,
        flags="",
    )
    row = finish(draft, "O") if measured else dormant(draft, "P")
    return StepResult(row=row, cold_started=False, step=None, dormant_reason=None)


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
    reject_weight: float,
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
    reject_weight : float
        The weight of the newest reading in the reject fraction.
    nu : float or None, optional
        ``float(innovation)``, when the caller has it already; worked out
        here when ``None``, once for the row.

    Returns
    -------
    Row
        The updated state, x in whole femtoseconds; the innovation; the
        innovation scale moved by the innovation, as
        sqrt(max((1 - w) scale**2 + w innovation**2, floor**2)) with
        w = 1/M_sigma; the reject fraction lowered to (1 - reject_weight)
        times itself, and to 0 once below :data:`REJECT_FRACTION_FLOOR`;
        the consecutive rejects and the epochs since an accept set to 0,
        and the buffer emptied.

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
    draft.reject_fraction *= 1.0 - reject_weight
    if draft.reject_fraction < REJECT_FRACTION_FLOOR:
        draft.reject_fraction = 0.0
    draft.consecutive_rejects = draft.epochs_since_accept = 0
    draft.rejects = ()
    return finish(draft, "A")


def cold_start(
    draft: RowDraft, z: int, series_params: SeriesParams, *, rate: float = 0.0
) -> Row:
    """Finish a draft as a cold start: a new segment from the measurement.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a dormant series; its innovation scale
        is the one its reject fraction left it, or ``None``.
    z : int
        The measurement at the epoch, ps.
    series_params : SeriesParams
        The settings in force, whose time constants and ``sigma0`` the new
        segment takes.
    rate : float, optional
        The rate the measurements acquired from show, ps/s; a 1-state
        series has none and starts at 0 whatever is given.

    Returns
    -------
    Row
        A new segment at phase ``z`` with rate ``rate`` (0 for one state)
        and no drift, the innovation scale the draft kept, else ``sigma0``
        (design 8.6), step offset, counters, reject fraction and buffer at
        0, flags A and N, and U for a 2- or 3-state series.

    Raises
    ------
    FilterError
        If ``series_params`` is for another model than the series', or the
        finished row breaks a rule of :class:`Row`.
    """
    start_segment(draft, series_params, keep_offset=False)
    draft.x_fs, draft.d = to_fs(z), 0.0
    draft.y = 0.0 if draft.filter_states == 1 else rate
    if draft.innovation_scale is None:
        draft.innovation_scale = series_params.sigma0
    draft.reject_fraction = 0.0
    draft.consecutive_rejects = draft.epochs_since_accept = 0
    draft.rejects = ()
    return finish(draft, "A")


# ------------------------------------------------------- step classification

K_OUT: Final[float] = 5.0
"""How many innovation scales wide the gate is, either way."""

K_STEP: Final[float] = 3.0
"""How many innovation scales each of three rejects may lie from a step's fit.

The fit is the rejects' mean for a phase step, their fitted line for a
frequency step. The rejects show the step only when every one of them lies
less than this many scales from the fit.
"""

_STEP_REJECTS: Final[int] = 3
"""How many consecutive counted rejects a step is looked for in."""

REJECT_FRACTION_FLOOR: Final[float] = 1e-9
"""A reject fraction below this is 0.

The fraction only ever decays toward 0 between rejects, and far below any
limit it says nothing; set to 0 it also never shrinks past what its file
column can hold.
"""

type StepKind = Literal["phase", "frequency"]
"""The kinds of step three rejects can show."""


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

    step_kind: StepKind | None
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


def count_reject(draft: RowDraft, innovation: mpq, reject_weight: float) -> None:
    """Count a rejected measurement and put it in the reject buffer (design 9.3).

    The draft gets one more consecutive reject, its reject fraction raised
    to (1 - reject_weight) times itself plus reject_weight, and (epoch,
    innovation) added to its buffer, which keeps the newest
    :data:`~masterclock.domain.series.MAX_REJECTS`.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far; changed in place.
    innovation : mpq
        The rejected measurement less the predicted phase.
    reject_weight : float
        The weight of the newest reading in the reject fraction.
    """
    reject_entry = (draft.interpolated_datetime, float(innovation))
    draft.consecutive_rejects += 1
    draft.reject_fraction = (
        1.0 - reject_weight
    ) * draft.reject_fraction + reject_weight
    draft.rejects = (*draft.rejects, reject_entry)[-MAX_REJECTS:]


def phase_step(
    draft: RowDraft, prediction: State, z: int, scale_floor: float, reject_weight: float
) -> Row:
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
    reject_weight : float
        The weight of the newest reading in the reject fraction.

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
    return accept(
        draft,
        corrected_prediction,
        z - corrected_prediction.x,
        scale_floor,
        reject_weight,
    )


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
    return accept(
        draft,
        corrected_prediction,
        z - corrected_prediction.x,
        scale_floor,
        series_params.reject_fraction_weight,
    )


def accept_step(
    draft: RowDraft,
    prediction: State,
    z: int,
    scale_floor: float,
    series_params: SeriesParams,
) -> tuple[Row, StepKind] | None:
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
    tuple of (Row, {'phase', 'frequency'}) or None
        The accepted row and the kind of step: a phase step (see
        :func:`phase_step`) or, for a 2- or 3-state series, a frequency
        step (see :func:`frequency_step`). ``None`` while the buffer holds fewer than
        three rejects, which a reject over the rms limit empties, when the
        rejects show neither, and for a frequency step of a 1-state series,
        which has no rate.

    Raises
    ------
    FilterError
        If the draft has no innovation scale, ``series_params`` is for
        another model than the series' (on a frequency step), or the
        finished row breaks a rule of :class:`Row`.
    """
    if len(draft.rejects) < _STEP_REJECTS:
        return None
    if draft.innovation_scale is None:
        message = f"a reject at {draft.interpolated_datetime} has no innovation scale"
        _log.error(message)
        raise FilterError(message)
    classified = classify(draft.rejects, draft.innovation_scale)
    if classified.step_kind == "phase":
        row = phase_step(
            draft, prediction, z, scale_floor, series_params.reject_fraction_weight
        )
        return row, "phase"
    if (
        classified.step_kind == "frequency"
        and draft.filter_states > 1
        and classified.a is not None
        and classified.s is not None
    ):
        row = frequency_step(
            draft, prediction, classified.a, classified.s, z, scale_floor, series_params
        )
        return row, "frequency"
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


def acquire(
    draft: RowDraft, z: int, series_params: SeriesParams, *, in_limit: bool = True
) -> Row:
    """Buffer a dormant series' measurement; cold-start when it is consistent.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, of a series with no prediction or one that
        has just gone dormant on its rejects, its buffer holding the
        measurements it has gathered, as (epoch, z). Its innovation scale,
        when it has one, is kept to the cold start: the caller leaves one
        only for a series its reject fraction made dormant.
    z : int
        The measurement at the epoch, ps. A pair's is decycled against its
        prediction, or, with none, against the last buffered measurement
        (see :func:`anchor_of`).
    series_params : SeriesParams
        The settings in force, whose ``sigma0`` sets the test.
    in_limit : bool, optional
        Whether a pair's reading is within its rms limit; a triple's always
        is.

    Returns
    -------
    Row
        A dormant R row with an empty buffer when the reading is not within
        its rms limit: it is never buffered, so it cannot start the series.
        Otherwise a cold start from ``z`` (see :func:`cold_start`) when the buffer,
        with ``z`` added and the newest
        :data:`~masterclock.domain.series.MAX_REJECTS` kept, holds three
        measurements from consecutive epochs whose second difference
        z3 - 2 z2 + z1 is at most :data:`_ACQUIRE_LIMIT` sigma0 either way
        (design 13.3), at their mean rate (z3 - z1) / 2T (design 8.6).
        Otherwise a dormant R row keeping that buffer, with no counted
        rejects and epochs since an accept as they were.

    Raises
    ------
    FilterError
        If the finished row breaks a rule of :class:`Row`.
    """
    draft.consecutive_rejects = 0
    draft.reject_fraction = 0.0
    if not in_limit:
        return dormant(draft, "R", keep_scale=True)
    buffer_entry = (draft.interpolated_datetime, float(z))
    draft.rejects = (*draft.rejects, buffer_entry)[-MAX_REJECTS:]
    if len(draft.rejects) == MAX_REJECTS and _consecutive(draft.rejects):
        z1, z2, z3 = (exact(buffered_z) for _, buffered_z in draft.rejects)
        if abs(z3 - 2 * z2 + z1) <= exact(_ACQUIRE_LIMIT * series_params.sigma0):
            rate = float((z3 - z1) / (2 * _T))
            return cold_start(draft, z, series_params, rate=rate)
    return dormant(draft, "R", keep_buffer=True, keep_scale=True)


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
    ...     step_offset=0, epochs_in_segment=0, epochs_since_accept=1,
    ...     consecutive_rejects=0, reject_fraction=0.0,
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
    """The row a series writes at an epoch, and what the epoch did to the series.

    Parameters
    ----------
    row : Row
        The series' row at the epoch.
    cold_started : bool
        Whether the row is a cold start: a triple built from this pair's
        value then goes dormant (design 12.6).
    step : {'phase', 'frequency'} or None
        The kind of step the row accepted (design 9.4); ``None`` for every
        other row.
    dormant_reason : str or None
        Why a series that was tracked went dormant at this row, in words
        for the log (design 13.3): its rejects in a row, its reject
        fraction over the limit, the gap limit passed, or a pair it uses
        started again; ``None`` for a row that is not that.
    """

    row: Row
    cold_started: bool
    step: StepKind | None
    dormant_reason: str | None


_PAIR_RESTARTED: Final[str] = "a pair it uses started again"
"""Why a triple went dormant when a pair whose value it uses cold-started."""

_GAP_LIMIT_PASSED: Final[str] = "gap limit passed"
"""Why a series went dormant when its held rows outran its gap limit."""


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
    held: bool = False,
    pair_cold_started: bool = False,
) -> StepResult:
    """Give a series' row at an epoch: the whole of the decision flow (design 9.6).

    Parameters
    ----------
    epoch_start : datetime
        The epoch start E.
    series_params : SeriesParams
        The settings in force at E.
    last_row : Row or None
        The series' last row, or ``None`` for a new series, or one that
        starts again after epochs it had no row for.
    prediction : State or None
        The prediction at E from ``last_row`` (see :func:`predict`), or
        ``None`` when the series has none.
    measurement : FilterInput or None
        The series' measurement at E, or ``None`` when there is none.
    excluded : bool, optional
        Whether screening or the slip check excluded the measurement.
    held : bool, optional
        Whether screening held the measurement, for a fault of the reference
        that made it: held as X whatever its innovation (design 9.5).
    pair_cold_started : bool, optional
        For a triple, whether a pair it uses cold-started at E, whether or
        not the triple has a measurement there (design 12.6).

    Returns
    -------
    StepResult
        The row, whether it cold-started, the kind of step it accepted, if
        any, and why it went dormant, when it was tracked and did. First, a
        tracked series whose
        time constants changed starts a warm segment (see
        :func:`start_segment`); the row then also carries the outcome.
        With no measurement, the row holds the prediction (see :func:`hold`),
        or, for a triple one of whose pairs cold-started, is dormant with an
        empty buffer, and not written. A series with no prediction, or a
        triple one of whose pairs cold-started, acquires (see
        :func:`acquire`), a pair's reading over its rms limit never
        buffered. Otherwise the measurement goes through the gate.

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
        return _unmeasured(draft, prediction, series_params, pair_cold_started)
    tracked = prediction is not None
    if measurement.pair_cold_started:
        draft.rejects = ()
        draft.innovation_scale = None
        prediction = None
    if prediction is None:
        row = acquire(
            draft,
            measurement.z,
            series_params,
            in_limit=_within_rms_limit(measurement, series_params),
        )
        return StepResult(
            row=row,
            cold_started="D" not in row.flags,
            step=None,
            dormant_reason=_PAIR_RESTARTED if tracked else None,
        )
    row, step_kind, dormant_reason = (
        _held_reading(draft, prediction, measurement, series_params)
        if held
        else _gate(draft, prediction, measurement, series_params, excluded=excluded)
    )
    return StepResult(
        row=row, cold_started=False, step=step_kind, dormant_reason=dormant_reason
    )


def _unmeasured(
    draft: RowDraft,
    prediction: State | None,
    series_params: SeriesParams,
    pair_cold_started: bool,
) -> StepResult:
    """Give a series' row at an epoch with no measurement (design 9.6).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    prediction : State or None
        The series' prediction at the epoch, or ``None`` when it has none.
    series_params : SeriesParams
        The settings in force.
    pair_cold_started : bool
        For a triple, whether a pair it uses cold-started at the epoch.

    Returns
    -------
    StepResult
        For a triple one of whose pairs cold-started, a dormant row with an
        empty buffer, which is not written; otherwise the prediction held
        (see :func:`hold`). Never a cold start or a step. A series that had
        a prediction and went dormant says why: a pair it uses started
        again, or its gap limit passed.
    """
    tracked = prediction is not None
    if pair_cold_started:
        return StepResult(
            row=dormant(draft, "P"),
            cold_started=False,
            step=None,
            dormant_reason=_PAIR_RESTARTED if tracked else None,
        )
    row = hold(draft, prediction, "P", series_params)
    return StepResult(
        row=row,
        cold_started=False,
        step=None,
        dormant_reason=_GAP_LIMIT_PASSED if tracked and "D" in row.flags else None,
    )


def _gate(
    draft: RowDraft,
    prediction: State,
    measurement: FilterInput,
    series_params: SeriesParams,
    *,
    excluded: bool,
) -> tuple[Row, StepKind | None, str | None]:
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
    tuple of (Row, {'phase', 'frequency'} or None, str or None)
        The row, the kind of step it accepted, if any, and why the series
        went dormant, if it did. Accepted when the measurement passes the
        gate and is not excluded. Held as X when it is excluded inside the
        gate, not counted. Otherwise a counted reject, which empties the
        buffer when a pair's reading is over its rms limit: accepted after
        a step when the rejects show one (see :func:`accept_step`); else,
        once the rejects in a row reach ``n_break`` or the reject fraction
        passes its limit, a dormant row whose measurement starts the
        acquisition buffer, unless it is over the rms limit; else held as
        R. A held row past the gap limit is dormant too.

    Raises
    ------
    FilterError
        If the draft has no innovation scale, ``series_params`` is for
        another model than the series', or a row breaks a rule of
        :class:`Row`.
    """
    innovation = measurement.z - prediction.x
    nu = float(innovation)
    draft.innovation = nu
    if draft.innovation_scale is None:
        message = f"a measurement at {draft.interpolated_datetime} has no scale"
        _log.error(message)
        raise FilterError(message)
    in_gate = within_gate(innovation, draft.innovation_scale)
    in_limit = _within_rms_limit(measurement, series_params)
    if in_gate and in_limit and not excluded:
        row = accept(
            draft,
            prediction,
            innovation,
            measurement.scale_floor,
            series_params.reject_fraction_weight,
            nu=nu,
        )
        return row, None, None
    if in_gate and excluded:
        return _held(draft, prediction, "X", series_params)
    count_reject(draft, innovation, series_params.reject_fraction_weight)
    if not in_limit:
        draft.rejects = ()
    accepted_step = accept_step(
        draft, prediction, measurement.z, measurement.scale_floor, series_params
    )
    if accepted_step is not None:
        return *accepted_step, None
    dormant_reason = _dormant_reason(draft, series_params)
    if dormant_reason is not None:
        draft.rejects = ()
        row = acquire(draft, measurement.z, series_params, in_limit=in_limit)
        return row, None, dormant_reason
    return _held(draft, prediction, "R", series_params)


def _held_reading(
    draft: RowDraft,
    prediction: State,
    measurement: FilterInput,
    series_params: SeriesParams,
) -> tuple[Row, None, str | None]:
    """Hold a measurement screening held for a fault of its reference (design 9.5).

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

    Returns
    -------
    tuple of (Row, None, str or None)
        The row held as X with the measurement's innovation, whatever its
        size, not counted; dormant past the gap limit, and then why.
    """
    draft.innovation = float(measurement.z - prediction.x)
    return _held(draft, prediction, "X", series_params)


def _held(
    draft: RowDraft, prediction: State, outcome: Held, series_params: SeriesParams
) -> tuple[Row, None, str | None]:
    """Hold a draft, saying so when the gap limit made the row dormant.

    Parameters
    ----------
    draft : RowDraft
        The row as built so far.
    prediction : State
        The series' prediction at the epoch.
    outcome : {'R', 'X', 'P'}
        Why the state is not updated.
    series_params : SeriesParams
        The settings in force.

    Returns
    -------
    tuple of (Row, None, str or None)
        The held row (see :func:`hold`), no step, and the gap limit as the
        reason when the row came back dormant.
    """
    row = hold(draft, prediction, outcome, series_params)
    return row, None, _GAP_LIMIT_PASSED if "D" in row.flags else None


def _dormant_reason(draft: RowDraft, series_params: SeriesParams) -> str | None:
    """Say why a counted reject makes the series dormant, or that it does not.

    A series made dormant by its rejects in a row loses its innovation
    scale here, so its cold start takes ``sigma0``; one made dormant by its
    reject fraction keeps it (design 8.6).

    Parameters
    ----------
    draft : RowDraft
        The row as built so far, the current reject counted; changed in
        place.
    series_params : SeriesParams
        The settings in force.

    Returns
    -------
    str or None
        The rejects in a row when they reach ``n_break``; else the reject
        fraction and its limit, with decimals enough to tell them apart (see
        :func:`_written_apart`), when the fraction is above the limit; else
        ``None``.
    """
    if draft.consecutive_rejects >= series_params.n_break:
        draft.innovation_scale = None
        return f"{draft.consecutive_rejects} rejects in a row"
    if draft.reject_fraction > series_params.reject_fraction_limit:
        fraction_text, limit_text = _written_apart(
            draft.reject_fraction, series_params.reject_fraction_limit
        )
        return f"reject fraction {fraction_text} over the limit {limit_text}"
    return None


def _written_apart(above: float, below: float) -> tuple[str, str]:
    """Write two numbers with as many decimals as it takes to tell them apart.

    Parameters
    ----------
    above : float
        The larger number.
    below : float
        The smaller number.

    Returns
    -------
    tuple of (str, str)
        Both, with the same number of decimals: three, or more when three
        write them alike. Two different floats always come apart, since
        enough decimals write a float exactly.

    Examples
    --------
    >>> _written_apart(0.52, 0.5)
    ('0.520', '0.500')
    >>> _written_apart(0.50004, 0.5)
    ('0.50004', '0.50000')
    """
    decimals = 3
    while f"{above:.{decimals}f}" == f"{below:.{decimals}f}":
        decimals += 1
    return f"{above:.{decimals}f}", f"{below:.{decimals}f}"


def _within_rms_limit(measurement: FilterInput, series_params: SeriesParams) -> bool:
    """Tell whether a measurement is within its rms limit; a triple's always is.

    Parameters
    ----------
    measurement : FilterInput
        The measurement.
    series_params : SeriesParams
        The settings in force, whose ``rms_max`` is a pair's limit.

    Returns
    -------
    bool
        Whether the measurement has no rms, being a triple's, or its rms is
        within the limit (see :func:`rms_ok`).
    """
    return measurement.rms is None or rms_ok(measurement.rms, series_params.rms_max)
