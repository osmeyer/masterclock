"""Steering of the reference clocks: a known input to every series they are in.

A reference is steered by changing its phase and rate. That change is known,
so it enters the estimator's prediction and never its innovation. A series
is moved by every reference in its effective difference, with the sign that
reference has there (:func:`signs`): a pair (a, b) is x_a - x_b, a triple
(r, s, c) is x_r - x_c, and a self pair (r, r) is moved by nothing.

Two sums are worked out for a series at an epoch start E:

* :func:`steer_u`, the input over the previous epoch: every event in
  (E - T, E], moved on to E. It enters the prediction from E - T to E.
* :func:`steer_w`, the steering inside the epoch: every event in (E, t],
  moved on to the measurement time t. It is added to the prediction at t
  and taken off again when the measurement is referred back to E; the same
  events enter :func:`steer_u` in full at the next epoch, so nothing is
  counted twice.

Every phase term is exact (see :mod:`masterclock.domain.phase`); the rate
term is a float.
"""

from datetime import timedelta
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from masterclock.domain.phase import EPOCH_SECONDS, exact, seconds
from masterclock.domain.references import is_reference

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from datetime import datetime

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch, T."""


class SteerEvent(BaseModel):
    """One change applied to a reference clock's phase and rate.

    Parameters
    ----------
    applied_datetime : AwareDatetime
        When the change was applied.
    dx : float
        The change to the reference's phase, ps, in the sign convention of
        the measurements: a pair (a, b) is x_a - x_b.
    dy : float
        The change to the reference's rate, ps/s.

    Raises
    ------
    pydantic.ValidationError
        If a change is not a finite number, the instant has no timezone, or
        a field is unknown.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> at = datetime(2025, 9, 23, tzinfo=UTC)
    >>> SteerEvent(applied_datetime=at, dx=0.0, dy=-0.00012).dy
    -0.00012
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    applied_datetime: AwareDatetime
    dx: Annotated[float, Field(allow_inf_nan=False)]
    dy: Annotated[float, Field(allow_inf_nan=False)]


def signs(series_key: Sequence[str]) -> dict[str, int]:
    """Name the references that steer a series, each with its sign.

    Parameters
    ----------
    series_key : sequence of str
        A pair (a, b) or a triple (r, s, c).

    Returns
    -------
    dict of str to int
        For a triple, r with +1. For a self pair (r, r), nothing: its two
        terms cancel. For any other pair, a with +1 and b with -1, each only
        if it is a reference.

    Examples
    --------
    >>> signs(("mc1", "mc2")), signs(("mc2", "nav23")), signs(("mc1", "mc2", "nav23"))
    ({'mc1': 1, 'mc2': -1}, {'mc2': 1}, {'mc1': 1})
    """
    if len(series_key) == 3:
        return {series_key[0]: 1}
    first_clock, second_clock = series_key
    if first_clock == second_clock:
        return {}
    return {
        clock_name: sign
        for clock_name, sign in ((first_clock, 1), (second_clock, -1))
        if is_reference(clock_name)
    }


def _signed_events(
    steering: Mapping[str, Sequence[SteerEvent]],
    series_key: Sequence[str],
    window_start: datetime,
    window_end: datetime,
) -> Iterator[tuple[int, SteerEvent]]:
    """Yield the events that move a series in (after, through], with their signs.

    Parameters
    ----------
    steering : mapping of str to sequence of SteerEvent
        Each reference's events, in time order.
    series_key : sequence of str
        The series.
    window_start : datetime
        The start of the interval, left out.
    window_end : datetime
        The end of the interval, counted.

    Yields
    ------
    tuple of (int, SteerEvent)
        The sign of each event's reference in the series, and the event, by
        reference in the order :func:`signs` gives and then in time order.
    """
    for reference, sign in signs(series_key).items():
        for steer_event in steering.get(reference, ()):
            if window_start < steer_event.applied_datetime <= window_end:
                yield sign, steer_event


def steer_u(
    series_key: Sequence[str],
    epoch_start: datetime,
    steering: Mapping[str, Sequence[SteerEvent]],
) -> tuple[Fraction, float]:
    """Give the steering input over the epoch before a mark.

    Parameters
    ----------
    series_key : sequence of str
        The series.
    epoch_start : datetime
        The epoch start E.
    steering : mapping of str to sequence of SteerEvent
        Each reference's events, in time order. A reference that is not in
        it was never steered.

    Returns
    -------
    tuple of (Fraction, float)
        u_x, the sum over the events in (E - T, E] of the sign times
        dx + dy (E - t_m), exact; and u_y, the sum of the sign times dy.

    Raises
    ------
    PhaseError
        If ``epoch_start`` or an event's instant has no timezone.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> mark = datetime(2025, 9, 23, 6, tzinfo=UTC)
    >>> event = SteerEvent(applied_datetime=mark, dx=2.0, dy=0.5)
    >>> steer_u(("mc1", "mc2"), mark, {"mc2": (event,)})
    (Fraction(-2, 1), -0.5)
    """
    ux = Fraction(0)
    uy = 0.0
    for sign, steer_event in _signed_events(
        steering, series_key, epoch_start - _EPOCH, epoch_start
    ):
        ux += sign * (
            exact(steer_event.dx)
            + exact(steer_event.dy) * seconds(epoch_start, steer_event.applied_datetime)
        )
        uy += sign * steer_event.dy
    return ux, uy


def steer_w(
    series_key: Sequence[str],
    epoch_start: datetime,
    steering: Mapping[str, Sequence[SteerEvent]],
    measured_at: datetime,
) -> Fraction:
    """Give the steering applied between a mark and a measurement.

    Parameters
    ----------
    series_key : sequence of str
        The series.
    epoch_start : datetime
        The epoch start E.
    steering : mapping of str to sequence of SteerEvent
        Each reference's events, in time order.
    measured_at : datetime
        The measurement time t.

    Returns
    -------
    Fraction
        w(t), the sum over the events in (E, t] of the sign times
        dx + dy (t - t_m), exact.

    Raises
    ------
    PhaseError
        If ``measured_at`` or an event's instant has no timezone.
    """
    w = Fraction(0)
    for sign, steer_event in _signed_events(
        steering, series_key, epoch_start, measured_at
    ):
        w += sign * (
            exact(steer_event.dx)
            + exact(steer_event.dy) * seconds(measured_at, steer_event.applied_datetime)
        )
    return w
