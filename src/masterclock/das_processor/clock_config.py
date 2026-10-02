"""The clock configuration: each clock's estimator settings and each pair's RMS limit.

One YAML file per deployment says what is known about its clocks. A clock
has a type, whose default entry gives its estimator: how many states, its
time constants, its initial innovation scale and its gap limit. A clock can
then override any of those except its number of states, from the start or
from a given MJD. The file also gives how many counted rejects make a series
dormant, and the RMS limit of the pairs' gate.

The file is read once, at the start of a run, with a safe YAML loader that
refuses a key repeated in any mapping, and validated into frozen models
that refuse unknown keys. Everything the design lists as wrong with a file
is refused then, with a :class:`~masterclock.app.exceptions.ConfigError`,
so a run never starts on settings it could not use.

A series takes the settings of its clock side: a pair (a, b) those of b, a
triple (r, s, c) those of c (:meth:`ClockConfig.params_for`).
"""

import re
from collections.abc import Hashable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Final, NoReturn, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from masterclock.app.exceptions import ConfigError, describe_error
from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor.cli import DataMjd
from masterclock.domain.references import REFERENCE_PATTERN, REFERENCE_PREFIX
from masterclock.domain.series import FilterStates, PairKey, SeriesKey, SeriesParams

REFERENCE_TYPE: Final[str] = "mc"
"""The type every reference clock has."""

_PAIR: Final[int] = 2
"""How many clocks a pair key names."""

_MERGE_TAG: Final[str] = "tag:yaml.org,2002:merge"
"""The tag of a YAML merge key."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

type _TimeConstant = Annotated[float, Field(ge=1, allow_inf_nan=False)]
"""A time constant, epochs: at least one."""

type _Scale = Annotated[float, Field(gt=0, allow_inf_nan=False)]
"""An innovation scale, ps: above zero."""


class _Loader(yaml.SafeLoader):
    """A safe YAML loader that refuses repeated keys and merge keys."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:  # Any: PyYAML's own type for a mapping's values
        """Build a mapping, refusing a key it holds twice.

        Parameters
        ----------
        node : MappingNode
            The mapping's node.
        deep : bool, optional
            Whether to build the values at once, as PyYAML's own loader.

        Returns
        -------
        dict
            The mapping.

        Raises
        ------
        ConstructorError
            If a key is repeated, or is a merge key, whose values would
            come from elsewhere in the file.
        """
        seen: list[object] = []
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                message = "merge keys are not read"
                raise ConstructorError(None, None, message, key_node.start_mark)
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                message = f"repeated key {key!r}"
                raise ConstructorError(None, None, message, key_node.start_mark)
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


def _tuples(value: object) -> object:
    """Turn every list in loaded YAML into a tuple, so frozen models hold it.

    Parameters
    ----------
    value : object
        A value PyYAML loaded.

    Returns
    -------
    object
        The same value with lists, at any depth, as tuples.
    """
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


class TypeDefault(BaseModel):
    """The default entry of a clock type.

    Parameters
    ----------
    filter_states : {1, 2, 3}
        How many states the estimator has.
    time_constant : float or None, optional
        The estimator time constant M, epochs, at least 1; given exactly
        for 2 or 3 states.
    scale_time_constant : float
        The innovation scale's averaging constant M_sigma, epochs, at
        least 1.
    initial_innovation_scale : float
        The innovation scale a cold start begins with, sigma0, ps.
    gap_limit : int
        G_max: how many held rows a prediction may run, at least 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filter_states: FilterStates
    time_constant: _TimeConstant | None = None
    scale_time_constant: _TimeConstant
    initial_innovation_scale: _Scale
    gap_limit: PositiveInt


class Entry(BaseModel):
    """One entry of a clock: the type, or the values it overrides from a date.

    Parameters
    ----------
    type : str or None, optional
        The clock's type; given by its first entry and by no other.
    effective_mjd : float or None, optional
        The MJD the entry applies from; ``None`` for from the start.
    filter_states, time_constant, scale_time_constant : optional
        As in :class:`TypeDefault`; ``None`` to keep the value.
    initial_innovation_scale, gap_limit : optional
        As in :class:`TypeDefault`; ``None`` to keep the value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: str | None = None
    effective_mjd: DataMjd | None = None
    filter_states: FilterStates | None = None
    time_constant: _TimeConstant | None = None
    scale_time_constant: _TimeConstant | None = None
    initial_innovation_scale: _Scale | None = None
    gap_limit: PositiveInt | None = None

    def overrides(self) -> dict[str, object]:
        """Give the estimator values the entry sets.

        Returns
        -------
        dict of str to object
            Every value the entry gives other than its type and date.
        """
        return self.model_dump(exclude_none=True, exclude={"type", "effective_mjd"})


class ClockEntry(BaseModel):
    """A clock's estimator settings at one mark, every value settled.

    Parameters
    ----------
    filter_states : {1, 2, 3}
        How many states the estimator has.
    time_constant : float or None
        M, epochs; ``None`` exactly for one state.
    scale_time_constant : float
        M_sigma, epochs.
    initial_innovation_scale : float
        sigma0, ps.
    gap_limit : int
        G_max.

    Raises
    ------
    pydantic.ValidationError
        If the time constant is given for one state or missing for two or
        three.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filter_states: FilterStates
    time_constant: _TimeConstant | None
    scale_time_constant: _TimeConstant
    initial_innovation_scale: _Scale
    gap_limit: PositiveInt

    @model_validator(mode="after")
    def _check_time_constant(self) -> Self:
        """Refuse a time constant that does not fit the number of states.

        Returns
        -------
        Self
            The entry, unchanged.

        Raises
        ------
        ValueError
            If ``time_constant`` is given for one state or missing for more.
        """
        if (self.time_constant is None) != (self.filter_states == 1):
            message = "a time constant is given exactly when filter_states is 2 or 3"
            raise ValueError(message)
        return self


class RmsLimits(BaseModel):
    """The RMS limits of the pairs' gate, ps.

    Parameters
    ----------
    default : int
        The limit of every pair not named otherwise.
    references : dict of str to int, optional
        The limit of every pair measured against a reference.
    pairs : dict of str to int, optional
        The limit of one pair, written ``reference.clock``.

    Raises
    ------
    pydantic.ValidationError
        If a limit is not a positive whole number, a reference is not named
        as one, or a pair is not written as a reference and a clock.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    default: PositiveInt
    references: dict[str, PositiveInt] = Field(default_factory=dict)
    pairs: dict[str, PositiveInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        """Refuse references and pairs that are not named as such.

        Returns
        -------
        Self
            The limits, unchanged.

        Raises
        ------
        ValueError
            If a reference does not match the reference name pattern, or a
            pair is not a reference, a dot and a clock.
        """
        for name in self.references:
            if not _is_reference(name):
                message = f"rms_limit reference {name} is not a reference name"
                raise ValueError(message)
        for name in self.pairs:
            reference, dot, clock = name.partition(".")
            if not (dot and clock and _is_reference(reference)):
                message = f"rms_limit pair {name} is not written reference.clock"
                raise ValueError(message)
        return self


def _is_reference(name: str) -> bool:
    """Tell whether a name is a reference clock's.

    Parameters
    ----------
    name : str
        A clock name.

    Returns
    -------
    bool
        Whether the whole name matches the reference name pattern.
    """
    return re.fullmatch(REFERENCE_PATTERN, name) is not None


def _in_order(entries: Iterable[Entry]) -> list[Entry]:
    """Put a clock's entries in the order they apply.

    Parameters
    ----------
    entries : iterable of Entry
        The entries as the file lists them.

    Returns
    -------
    list of Entry
        Undated entries first, in file order, then dated ones by date.
    """
    return sorted(
        entries,
        key=lambda entry: (entry.effective_mjd is not None, entry.effective_mjd or 0.0),
    )


def _settled(default: TypeDefault, entries: Iterable[Entry], where: str) -> ClockEntry:
    """Apply entries to a type default and check the result.

    Parameters
    ----------
    default : TypeDefault
        The clock's type default.
    entries : iterable of Entry
        The entries to apply, in order.
    where : str
        What the settings are of, for the message.

    Returns
    -------
    ClockEntry
        The settled values.

    Raises
    ------
    ValueError
        If the values do not make a valid entry.
    """
    values: dict[str, object] = default.model_dump()
    for entry in entries:
        values |= entry.overrides()
    try:
        return ClockEntry.model_validate(values)
    except ValidationError as exc:
        message = f"{where}: {describe_error(exc)}"
        raise ValueError(message) from exc


class ClockConfig(BaseModel):
    """A deployment's clock configuration, checked.

    Parameters
    ----------
    rejects_before_restart : int
        N_break: the counted rejects that make a series dormant, at least 3
        and at most every clock's gap limit.
    rms_limit : RmsLimits
        The RMS limits of the pairs' gate; held as ``limits``, so the name
        is free for :meth:`rms_limit`.
    types : dict of str to TypeDefault
        The default entry of each clock type.
    clocks : dict of str to tuple of Entry
        Each clock's entries, the first giving its type.

    Raises
    ------
    pydantic.ValidationError
        If any check of the clock configuration fails: a clock with no
        entry, or whose first entry gives no type, or a type the file does
        not give; a later entry giving a type; a reference not of type mc;
        an entry changing the number of states; a time constant missing or
        given where the number of states says otherwise; a value out of its
        range; or N_break above a gap limit at any date.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rejects_before_restart: Annotated[int, Field(ge=3)]
    limits: RmsLimits = Field(alias="rms_limit")
    types: dict[str, TypeDefault]
    clocks: dict[str, tuple[Entry, ...]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a configuration that breaks any rule of design 15.2.

        Returns
        -------
        Self
            The configuration, unchanged.

        Raises
        ------
        ValueError
            Naming the first rule broken.
        """
        for name, default in self.types.items():
            self._check_gap(_settled(default, (), f"type {name}"), f"type {name}")
        for name, entries in self.clocks.items():
            default = self._type_default(name, entries)
            ordered = _in_order(entries)
            for count in range(len(ordered) + 1):
                where = f"clock {name}"
                self._check_gap(_settled(default, ordered[:count], where), name)
        return self

    def _type_default(self, name: str, entries: Sequence[Entry]) -> TypeDefault:
        """Give a clock's type default, checking its entries name it rightly.

        Parameters
        ----------
        name : str
            The clock.
        entries : sequence of Entry
            Its entries as the file lists them.

        Returns
        -------
        TypeDefault
            The default of the type its first entry gives.

        Raises
        ------
        ValueError
            If the clock has no entry, its first entry no type or a type
            the file does not give, a later entry gives a type, a reference
            is not of type mc, or an entry changes the number of states.
        """
        if not entries:
            message = f"clock {name} has no entry"
            raise ValueError(message)
        kind = entries[0].type
        if kind is None:
            message = f"the first entry of {name} gives no type"
            raise ValueError(message)
        if any(entry.type is not None for entry in entries[1:]):
            message = f"the type of {name} is given by its first entry only"
            raise ValueError(message)
        if kind not in self.types:
            message = f"clock {name} has type {kind}, which types does not give"
            raise ValueError(message)
        if name.startswith(REFERENCE_PREFIX) and kind != REFERENCE_TYPE:
            message = f"clock {name} is a reference, so of type {REFERENCE_TYPE}"
            raise ValueError(f"{message}, not {kind}")
        default = self.types[kind]
        for entry in entries:
            if entry.filter_states not in {None, default.filter_states}:
                message = (
                    f"an entry changes the filter_states of {name}"
                    f" from {default.filter_states} to {entry.filter_states}"
                )
                raise ValueError(message)
        return default

    def _check_gap(self, entry: ClockEntry, name: str) -> None:
        """Refuse a gap limit below N_break.

        Parameters
        ----------
        entry : ClockEntry
            Settled values of a clock or type.
        name : str
            What they are of, for the message.

        Raises
        ------
        ValueError
            If ``rejects_before_restart`` is above the gap limit.
        """
        if self.rejects_before_restart > entry.gap_limit:
            message = (
                f"rejects_before_restart {self.rejects_before_restart} is above"
                f" the gap_limit {entry.gap_limit} of {name}"
            )
            raise ValueError(message)

    def entry_for(self, clock: str, mark: datetime) -> ClockEntry:
        """Give a clock's settings at a mark (design 15.2).

        Parameters
        ----------
        clock : str
            The clock.
        mark : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        ClockEntry
            The type default of the clock's first entry, with every entry in
            force at ``mark`` applied in order of ``effective_mjd``: an entry
            without one from the start, one with one from the first mark at
            or after it.

        Raises
        ------
        ConfigError
            If the configuration has no entry for ``clock``.
        """
        entries = self.clocks.get(clock)
        if entries is None:
            message = f"the clock configuration has no entry for clock {clock}"
            _log.error(message)
            raise ConfigError(message)
        in_force = [
            entry
            for entry in _in_order(entries)
            if entry.effective_mjd is None
            or mjd_to_datetime(entry.effective_mjd) <= _aware(mark)
        ]
        default = self._type_default(clock, entries)
        return _settled(default, in_force, f"clock {clock}")

    def rms_limit(self, pair: PairKey) -> int:
        """Give a pair's RMS limit (design 9.1).

        Parameters
        ----------
        pair : (str, str)
            The pair (a, b).

        Returns
        -------
        int
            The pair's own limit, else its reference a's, else the default.
        """
        reference, clock = pair
        own = self.limits.pairs.get(f"{reference}.{clock}")
        if own is not None:
            return own
        return self.limits.references.get(reference, self.limits.default)

    def params_for(self, key: SeriesKey, mark: datetime) -> SeriesParams:
        """Give a series' settings at a mark (design 8.1).

        Parameters
        ----------
        key : (str, str) or (str, str, str)
            A pair (a, b), which takes the entry of b and its RMS limit, or
            a triple (r, s, c), which takes the entry of c and has none.
        mark : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        SeriesParams
            The series' settings.

        Raises
        ------
        ConfigError
            If the configuration has no entry for the series' clock.
        """
        entry = self.entry_for(key[-1], mark)
        rms_max = self.rms_limit((key[0], key[1])) if len(key) == _PAIR else None
        return SeriesParams(
            model=entry.filter_states,
            M=entry.time_constant,
            M_sigma=entry.scale_time_constant,
            sigma0=entry.initial_innovation_scale,
            gmax=entry.gap_limit,
            n_break=self.rejects_before_restart,
            rms_max=rms_max,
        )


def _aware(mark: datetime) -> datetime:
    """Refuse a mark without a timezone.

    Parameters
    ----------
    mark : datetime
        The mark.

    Returns
    -------
    datetime
        ``mark``, unchanged.

    Raises
    ------
    ConfigError
        If ``mark`` has no timezone, and so names no one instant.
    """
    if mark.tzinfo is None:
        message = f"the mark {mark} has no timezone"
        _log.error(message)
        raise ConfigError(message)
    return mark


def read_clock_config(path: Path) -> ClockConfig:
    """Read and check a clock configuration file (design 15.2).

    Parameters
    ----------
    path : Path
        The YAML file.

    Returns
    -------
    ClockConfig
        The checked configuration.

    Raises
    ------
    ConfigError
        If the file cannot be read as UTF-8, does not parse as YAML, holds
        a repeated or merge key or a tag the safe loader does not build, is
        not a mapping, or breaks any rule of :class:`ClockConfig`.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        _fail(path, f"cannot read: {exc}", exc)
    loader = _Loader(text)
    try:
        data = loader.get_single_data()
    except yaml.YAMLError as exc:
        _fail(path, describe_error(exc), exc)
    finally:
        loader.dispose()
    if not isinstance(data, dict):
        _fail(path, "is not a mapping of the sections", None)
    try:
        return ClockConfig.model_validate(_tuples(data))
    except ValidationError as exc:
        _fail(path, describe_error(exc), exc)


def _fail(path: Path, problem: str, cause: Exception | None) -> NoReturn:
    """Log and raise a clock configuration error.

    Parameters
    ----------
    path : Path
        The file.
    problem : str
        What is wrong with it.
    cause : Exception or None
        The error it came from, if any.

    Raises
    ------
    ConfigError
        Always, naming the file and the problem.
    """
    message = f"clock configuration {path}: {problem}"
    _log.error(message)
    raise ConfigError(message) from cause
