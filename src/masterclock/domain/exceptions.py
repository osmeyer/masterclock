"""The errors of the subject matter: the measurements and the clocks.

Every one descends from :class:`~masterclock.app.exceptions.MasterClockError`,
so the one ``except`` clause that catches every error the project raises
catches these too.
"""

from masterclock.app.exceptions import MasterClockError


class PhaseError(MasterClockError):
    """Raised when the phase mathematics is given values it cannot use.

    Raised wherever a value would otherwise produce a plausible number from
    the wrong input, instead of computing that number: an input outside the
    range the calculation is defined for (a reading outside one period of the
    5 MHz signal, say), or inputs that do not belong together (a decycled
    phase from a different pair, say).

    Examples
    --------
    >>> try:
    ...     raise PhaseError("measured phase 200000 is not within one period")
    ... except MasterClockError as exc:
    ...     str(exc)
    'measured phase 200000 is not within one period'
    """


class FilterError(MasterClockError):
    """Raised when the forward estimator is given values it cannot use.

    Raised for a setting outside its range, such as a model that is not one,
    two or three states, and for any two things that were built for
    different models but are used together. The second matters most: a
    three-state gain applied to a two-state filter would give a number of
    exactly the right shape, computed from a pole that filter never had.

    Examples
    --------
    >>> try:
    ...     raise FilterError("4 is not a one, two or three state model")
    ... except MasterClockError as exc:
    ...     str(exc)
    '4 is not a one, two or three state model'
    """
