"""What one series of the forward estimator holds: its state, settings and rows.

A series is the run of rows of one pair or one triple, one row per epoch.
Each row is the whole state of the series at its epoch: the estimate, the
counters and the reject buffer the next epoch starts from, so nothing else
carries between epochs.

Every model here is frozen, strict and refuses unknown fields. Strict means
a value of the wrong kind is refused rather than converted: a phase that is
not a :class:`~fractions.Fraction` or an ``int``, a ``bool`` for a number, a
list for a tuple. A float that is not finite raises
:class:`~masterclock.domain.exceptions.FilterError`, since no estimator value
can be one. :func:`build_row` builds a row from the values of its fields,
checked. A row is never changed: :func:`replace` builds a new one and checks
it again.
"""

import math
from collections.abc import Mapping
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, ClassVar, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from masterclock.app.exceptions import describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.domain.exceptions import FilterError


def _refuse_bool(given_states: object) -> object:
    """Refuse a bool where a number of states is meant.

    Parameters
    ----------
    given_states : object
        The value given for the number of states.

    Returns
    -------
    object
        ``given_states``, unchanged.

    Raises
    ------
    ValueError
        If ``given_states`` is a bool, which pydantic would otherwise take as the
        literal it equals: ``True`` as 1.
    """
    if isinstance(given_states, bool):
        message = f"the number of states is 1, 2 or 3, not a bool: {given_states}"
        raise ValueError(message)
    return given_states


type FilterStates = Annotated[Literal[1, 2, 3], BeforeValidator(_refuse_bool)]
"""How many states a series' estimator has: phase; rate; drift."""

type Reject = tuple[AwareDatetime, float]
"""One entry of a row's reject buffer: an epoch and a value.

While a series is tracked, the value is a rejected innovation. While it is
dormant, it is a measurement kept to acquire from.
"""

type PairKey = tuple[str, str]
"""A pair (a, b): reference a measured against clock or reference b."""

type TripleKey = tuple[str, str, str]
"""A triple (r, s, c): clock c against remote reference r, through s."""

type SeriesKey = PairKey | TripleKey
"""Either kind of series."""

FLAG_ORDER: Final[str] = "ARXPDSNU"
"""Every flag a row can carry, in the order a row writes them."""

OUTCOMES: Final[frozenset[str]] = frozenset("ARXP")
"""The flags of which every row carries exactly one: what the epoch did."""

MAX_REJECTS: Final[int] = 3
"""How many entries a row's reject buffer holds at most."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


def _finite(field_name: str, number: float | None) -> None:
    """Refuse a float that is not finite.

    Parameters
    ----------
    field_name : str
        The field the value is for, named in the error.
    number : float or None
        The value; ``None`` passes.

    Raises
    ------
    FilterError
        If ``number`` is nan or infinite.
    """
    if number is not None and not math.isfinite(number):
        message = f"{field_name} {number} is not finite"
        _log.error(message)
        raise FilterError(message)


class State(BaseModel):
    """The estimator's state at one epoch: phase, rate and drift.

    Parameters
    ----------
    x : Fraction
        Phase, ps. Exact within the epoch; rounded once when it is stored.
    y : float
        Rate, ps/s; 0.0 for a 1-state series.
    d : float, optional
        Drift, ps/s²; 0.0, the default, for a 1- or 2-state series.

    Raises
    ------
    FilterError
        If ``y`` or ``d`` is not finite.
    pydantic.ValidationError
        If ``x`` is not a Fraction, or a field is of the wrong kind or
        unknown.

    Examples
    --------
    >>> State(x=Fraction(123457438, 100), y=0.0123)
    State(x=Fraction(61728719, 50), y=0.0123, d=0.0)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    x: Fraction
    y: float
    d: float = 0.0

    @field_validator("y", "d")
    @classmethod
    def _check_finite(cls, rate_or_drift: float, info: ValidationInfo) -> float:
        """Refuse a rate or drift that is not finite.

        Parameters
        ----------
        rate_or_drift : float
            The rate or drift.
        info : ValidationInfo
            Pydantic's validation information, naming the field.

        Returns
        -------
        float
            ``rate_or_drift``, unchanged.
        """
        _finite(str(info.field_name), rate_or_drift)
        return rate_or_drift


class SeriesParams(BaseModel):
    """Everything one series needs from the configuration at one epoch.

    Parameters
    ----------
    filter_states : {1, 2, 3}
        How many states the estimator has, fixed for the life of the series.
    M : float or None
        Estimator time constant, epochs, at least 1; ``None`` exactly for a
        1-state series, which passes its measurements through.
    M_sigma : float
        Averaging constant of the innovation scale, epochs, at least 1.
    sigma0 : float
        Innovation scale a cold start begins with, ps; above zero.
    gmax : int
        Gap limit: how many held rows a prediction may run, at least 1.
    n_break : int
        Counted rejects that make the series dormant, from 3 to ``gmax``.
    rms_max : int or None
        RMS limit of the gate, ps, above zero, for a pair; ``None`` for a
        triple, whose gate has no RMS test.

    Raises
    ------
    pydantic.ValidationError
        If a value is outside its range, ``M`` is given for a 1-state series
        or missing for another, or a field is of the wrong kind or unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filter_states: FilterStates
    M: Annotated[float, Field(ge=1, allow_inf_nan=False)] | None
    M_sigma: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    sigma0: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    gmax: PositiveInt
    n_break: Annotated[int, Field(ge=3)]
    rms_max: PositiveInt | None

    @model_validator(mode="after")
    def _check_together(self) -> Self:
        """Refuse settings whose values do not fit each other.

        Returns
        -------
        Self
            The settings, unchanged.

        Raises
        ------
        ValueError
            If ``M`` is given for a 1-state series or missing for another,
            or ``n_break`` is above ``gmax``.
        """
        if (self.M is None) != (self.filter_states == 1):
            message = "a time constant M is given exactly when filter_states is 2 or 3"
            raise ValueError(message)
        if self.n_break > self.gmax:
            message = f"n_break {self.n_break} is above gmax {self.gmax}"
            raise ValueError(message)
        return self


class Row(BaseModel):
    """The estimator's columns of one row: a series at one epoch.

    Parameters
    ----------
    interpolated_datetime : AwareDatetime
        The epoch start E.
    innovation : float or None
        The measurement less the prediction, ps; ``None`` without a
        measurement or without a prediction.
    x_fs : int or None
        Estimated phase at E, in whole femtoseconds (see
        :data:`~masterclock.domain.phase.FS_PER_PS`); ``None`` when dormant.
    y : float or None
        Estimated rate, ps/s; ``None`` when dormant, 0.0 for a 1-state
        series.
    d : float or None
        Estimated drift, ps/s²; ``None`` when dormant, 0.0 for a 1- or
        2-state series.
    innovation_scale : float or None
        The innovation scale, ps; ``None`` when dormant.
    segment : int
        Segment number; 0 until the first cold start.
    step_offset : int
        Sum of the phase steps accepted in this segment, ps.
    epochs_in_segment : int
        Rows since the segment started; 0 on its first row.
    epochs_since_accept : int
        Rows since the last accepted measurement; 0 on an accepted row.
    consecutive_rejects : int
        Consecutive counted rejects.
    rejects : tuple of (AwareDatetime, float), optional
        The reject buffer, oldest first, at most :data:`MAX_REJECTS`
        entries; empty by default.
    filter_states : {1, 2, 3}
        How many states the estimator has.
    time_constant : float or None
        Estimator time constant M of this segment; ``None`` exactly for a
        1-state series.
    scale_time_constant : float
        Averaging constant M_sigma of this segment.
    flags : str
        Letters of :data:`FLAG_ORDER`, in that order: exactly one of A, R, X
        and P; D never with A, U never with D or on a 1-state series.

    Raises
    ------
    FilterError
        If a float is not finite.
    pydantic.ValidationError
        If the flags, the state, the reject buffer or the time constant
        break a rule above, or a field is of the wrong kind or unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    interpolated_datetime: AwareDatetime
    innovation: float | None
    x_fs: int | None
    y: float | None
    d: float | None
    innovation_scale: float | None
    segment: NonNegativeInt
    step_offset: int
    epochs_in_segment: NonNegativeInt
    epochs_since_accept: NonNegativeInt
    consecutive_rejects: NonNegativeInt
    rejects: tuple[Reject, ...] = ()
    filter_states: FilterStates
    time_constant: float | None
    scale_time_constant: float
    flags: str

    _STATE: ClassVar[tuple[str, ...]] = ("x_fs", "y", "d", "innovation_scale")
    """The fields a dormant row leaves empty and every other row fills."""

    @field_validator(
        "innovation",
        "y",
        "d",
        "innovation_scale",
        "time_constant",
        "scale_time_constant",
    )
    @classmethod
    def _check_finite(
        cls, field_value: float | None, info: ValidationInfo
    ) -> float | None:
        """Refuse a float that is not finite.

        Parameters
        ----------
        field_value : float or None
            The value.
        info : ValidationInfo
            Pydantic's validation information, naming the field.

        Returns
        -------
        float or None
            ``field_value``, unchanged.
        """
        _finite(str(info.field_name), field_value)
        return field_value

    @field_validator("rejects")
    @classmethod
    def _check_rejects(cls, reject_buffer: tuple[Reject, ...]) -> tuple[Reject, ...]:
        """Refuse a reject buffer too long, out of order or not finite.

        Parameters
        ----------
        reject_buffer : tuple of (AwareDatetime, float)
            The buffer.

        Returns
        -------
        tuple of (AwareDatetime, float)
            ``reject_buffer``, unchanged.

        Raises
        ------
        ValueError
            If it holds more than three entries, or its epochs do not rise
            from first to last.
        """
        for _, reject_value in reject_buffer:
            _finite("rejects", reject_value)
        if len(reject_buffer) > MAX_REJECTS:
            message = (
                f"the reject buffer holds at most three entries: {len(reject_buffer)}"
            )
            raise ValueError(message)
        if any(
            later_epoch <= earlier_epoch
            for (earlier_epoch, _), (later_epoch, _) in pairwise(reject_buffer)
        ):
            message = "the reject buffer is held oldest first"
            raise ValueError(message)
        return reject_buffer

    @field_validator("flags")
    @classmethod
    def _check_flags(cls, row_flags: str) -> str:
        """Refuse flags that are unknown, repeated, out of order or no outcome.

        Parameters
        ----------
        row_flags : str
            The flags.

        Returns
        -------
        str
            ``row_flags``, unchanged.

        Raises
        ------
        ValueError
            If a letter is not one of :data:`FLAG_ORDER`, appears twice or
            out of order, the flags hold other than exactly one of A, R, X
            and P, or D stands with A or U.
        """
        ordered_flags = "".join(letter for letter in FLAG_ORDER if letter in row_flags)
        if row_flags != ordered_flags:
            message = (
                f"flags {row_flags!r} are not distinct letters of {FLAG_ORDER} in order"
            )
            raise ValueError(message)
        if len(OUTCOMES & set(row_flags)) != 1:
            message = (
                f"flags {row_flags!r} hold other than exactly one of A, R, X and P"
            )
            raise ValueError(message)
        if "D" in row_flags and ("A" in row_flags or "U" in row_flags):
            message = (
                f"flags {row_flags!r}: a dormant row is never accepted or unsettled"
            )
            raise ValueError(message)
        return row_flags

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        """Refuse a state that does not fit the row's flags or model.

        Returns
        -------
        Self
            The row, unchanged.

        Raises
        ------
        ValueError
            If a dormant row holds any part of a state, another row lacks
            one, a row without a measurement holds an innovation, a row has
            a rate or drift its model lacks, the time constant is given for
            a 1-state series or missing for another, or a 1-state row is
            unsettled.
        """
        is_dormant = "D" in self.flags
        empty_fields = [
            field_name
            for field_name in self._STATE
            if getattr(self, field_name) is None
        ]
        if empty_fields != (list(self._STATE) if is_dormant else []):
            message = (
                "a dormant row has no x_fs, y, d or innovation_scale and every other"
                f" row has all four; flags {self.flags!r}, empty {empty_fields}"
            )
            raise ValueError(message)
        if "P" in self.flags and self.innovation is not None:
            message = "a row with no measurement (P) has no innovation"
            raise ValueError(message)
        if (self.time_constant is None) != (self.filter_states == 1):
            message = "a time constant is given exactly when filter_states is 2 or 3"
            raise ValueError(message)
        self._check_model()
        return self

    def known_state(self) -> tuple[int, float, float]:
        """Give the row's phase, rate and drift.

        Returns
        -------
        tuple of (int, float, float)
            The phase in whole femtoseconds, the rate and the drift.

        Raises
        ------
        FilterError
            If the row is dormant, and so holds no state.
        """
        if self.x_fs is None or self.y is None or self.d is None:
            message = f"a dormant row of {self.interpolated_datetime} holds no state"
            _log.error(message)
            raise FilterError(message)
        return self.x_fs, self.y, self.d

    def _check_model(self) -> None:
        """Refuse a rate, drift or settling the row's model does not have.

        Raises
        ------
        ValueError
            If a 2-state row has a drift other than 0.0, a 1-state row a
            rate or drift other than 0.0, or a 1-state row carries U.
        """
        if self.filter_states < 3 and self.d not in {None, 0.0}:
            message = f"d is 0.0 in a {self.filter_states}-state row: {self.d}"
            raise ValueError(message)
        if self.filter_states == 1 and self.y not in {None, 0.0}:
            message = f"y is 0.0 in a 1-state row: {self.y}"
            raise ValueError(message)
        if self.filter_states == 1 and "U" in self.flags:
            message = "U is never carried by a 1-state row"
            raise ValueError(message)


def build_row(field_values: Mapping[str, object]) -> Row:
    """Build a row from the values of its fields, checked.

    Parameters
    ----------
    field_values : Mapping of str to object
        Every field of the row, by name, with its value.

    Returns
    -------
    Row
        The row the values make.

    Raises
    ------
    FilterError
        If the values break any rule of :class:`Row`, miss a field, or name
        a field a row does not have.

    Examples
    --------
    >>> from datetime import UTC, datetime
    >>> build_row({
    ...     "interpolated_datetime": datetime(2025, 9, 23, 6, 0, tzinfo=UTC),
    ...     "innovation": None, "x_fs": None, "y": None, "d": None,
    ...     "innovation_scale": None, "segment": 0, "step_offset": 0,
    ...     "epochs_in_segment": 0, "epochs_since_accept": 0,
    ...     "consecutive_rejects": 0, "rejects": (), "filter_states": 1,
    ...     "time_constant": None, "scale_time_constant": 50.0, "flags": "PD",
    ... }).flags
    'PD'
    """
    try:
        return Row.model_validate(dict(field_values))
    except ValidationError as exc:
        message = f"invalid row: {describe_error(exc)}"
        _log.error(message)
        raise FilterError(message) from exc


def replace(row: Row, **changed_fields: object) -> Row:
    """Build a row from another with some fields changed, checked again.

    Parameters
    ----------
    row : Row
        The row to start from; it is not changed.
    **changed_fields : object
        The fields to change, by name, with their new values.

    Returns
    -------
    Row
        A new row with ``changed_fields`` made and every other field as in ``row``.

    Raises
    ------
    FilterError
        If the changed row breaks any rule of :class:`Row`, or ``changed_fields``
        names a field a row does not have.

    Notes
    -----
    Pydantic's ``model_copy(update=...)`` would build the new row without
    checking it, so a change that broke a rule would go unnoticed.
    """
    return build_row({**dict(row), **changed_fields})
