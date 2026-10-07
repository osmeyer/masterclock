"""The clock configuration: each clock's estimator settings and each pair's RMS limit.

One YAML file per deployment says what is known about its clocks. A clock
has a type, whose default entry gives its estimator: how many states, its
time constants, its initial innovation scale and its gap limit. A clock can
then override any of those except its number of states, from the start or
from a given MJD. A clock's entries may also give its location, the number
of the building it is in, which a type never gives: a clock that moves gets
an entry with the new building from the MJD of the move. An entry may also
disable a clock from its date, or enable it again: while a clock is
disabled, no series tracks it. The file also gives how many counted rejects
in a row make a series dormant, the averaging length and the limit of the
reject fraction that makes one dormant too, the RMS limit of the pairs'
gate, and the clocks to ignore: measured, but of no use.

The file is read once, at the start of a run, with a safe YAML loader that
refuses a key repeated in any mapping, and validated into frozen models
that refuse unknown keys. Everything the design lists as wrong with a file
is refused then, with a :class:`~masterclock.app.exceptions.ConfigError`,
so a run never starts on settings it could not use.

A series takes the settings of its clock side: a pair (a, b) those of b, a
triple (r, s, c) those of c (:meth:`ClockConfig.params_for`).
"""

from collections.abc import Hashable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Final, NoReturn, Self

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
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
from masterclock.domain.references import is_reference
from masterclock.domain.series import FilterStates, PairKey, SeriesKey, SeriesParams

REFERENCE_TYPE: Final[str] = "mc"
"""The type every reference clock has."""

_PAIR: Final[int] = 2
"""How many clocks a pair key names."""

_MERGE_TAG: Final[str] = "tag:yaml.org,2002:merge"
"""The tag of a YAML merge key."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


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


type _FilterStates = Annotated[FilterStates, BeforeValidator(_refuse_bool)]
"""How many states an estimator has, as the file gives it: 1, 2 or 3, never a bool."""

type _TimeConstant = Annotated[float, Field(ge=1, allow_inf_nan=False)]
"""A time constant, epochs: at least one."""

type _Scale = Annotated[float, Field(gt=0, allow_inf_nan=False)]
"""An innovation scale, ps: above zero."""

type _Location = Annotated[int, Field(gt=0)]
"""A clock's location: the number of the building it is in, above zero."""


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
            Whether to build the values at once, as in PyYAML's own loader.

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
        seen_keys: list[object] = []
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                message = "merge keys are not read"
                raise ConstructorError(None, None, message, key_node.start_mark)
            mapping_key = self.construct_object(key_node, deep=deep)
            if mapping_key in seen_keys:
                message = f"repeated key {mapping_key!r}"
                raise ConstructorError(None, None, message, key_node.start_mark)
            seen_keys.append(mapping_key)
        return super().construct_mapping(node, deep=deep)


def _tuples(yaml_value: object) -> object:
    """Turn every list in loaded YAML into a tuple, so frozen models hold it.

    Parameters
    ----------
    yaml_value : object
        A value PyYAML loaded.

    Returns
    -------
    object
        The same value with lists, at any depth, as tuples.
    """
    if isinstance(yaml_value, list):
        return tuple(_tuples(yaml_item) for yaml_item in yaml_value)
    if isinstance(yaml_value, dict):
        return {
            mapping_key: _tuples(yaml_item)
            for mapping_key, yaml_item in yaml_value.items()
        }
    return yaml_value


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
        G_max: how many held rows a prediction may run; no lower than the
        configuration's ``rejects_before_restart``, the only bound on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filter_states: _FilterStates
    time_constant: _TimeConstant | None = None
    scale_time_constant: _TimeConstant
    initial_innovation_scale: _Scale
    gap_limit: int


_OWN_SETTINGS: Final[tuple[str, ...]] = (
    "filter_states",
    "scale_time_constant",
    "initial_innovation_scale",
    "gap_limit",
)
"""The settings a first entry without a type must give; the time constant
too, when the number of states asks for one, as for any entry."""


class Entry(BaseModel):
    """One entry of a clock: the type, or the values it overrides from a date.

    Parameters
    ----------
    type : str or None, optional
        The clock's type; given by its first entry and by no other. A first
        entry may give every setting itself instead.
    effective_mjd : float or None, optional
        The MJD the entry applies from; ``None`` for from the start.
    filter_states, time_constant, scale_time_constant : optional
        As in :class:`TypeDefault`; ``None`` to keep the value.
    initial_innovation_scale, gap_limit : optional
        As in :class:`TypeDefault`; ``None`` to keep the value.
    location : int or None, optional
        The building the clock is in from the entry's date, a positive
        whole number; ``None`` to keep the location.
    disabled : bool or None, optional
        Whether the clock is disabled from the entry's date; ``None`` to
        keep it as it was.
    enabled : bool or None, optional
        The opposite of ``disabled``, for an entry that reads better so;
        ``None`` to keep it as it was.

    Raises
    ------
    pydantic.ValidationError
        If the entry gives both ``disabled`` and ``enabled``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: str | None = None
    effective_mjd: DataMjd | None = None
    filter_states: _FilterStates | None = None
    time_constant: _TimeConstant | None = None
    scale_time_constant: _TimeConstant | None = None
    initial_innovation_scale: _Scale | None = None
    gap_limit: int | None = None
    location: _Location | None = None
    disabled: bool | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def _check_disabled(self) -> Self:
        """Refuse an entry that says both whether the clock is disabled and enabled.

        Returns
        -------
        Self
            The entry, unchanged.

        Raises
        ------
        ValueError
            If both ``disabled`` and ``enabled`` are given.
        """
        if self.disabled is not None and self.enabled is not None:
            message = "an entry gives disabled or enabled, not both"
            raise ValueError(message)
        return self

    def overrides(self) -> dict[str, object]:
        """Give the values the entry sets.

        Returns
        -------
        dict of str to object
            Every value the entry gives other than its type and date, with
            ``enabled`` given as the ``disabled`` it means.
        """
        entry_values = self.model_dump(
            exclude_none=True, exclude={"type", "effective_mjd", "enabled"}
        )
        if self.enabled is not None:
            entry_values["disabled"] = not self.enabled
        return entry_values


class ClockEntry(BaseModel):
    """A clock's estimator settings and location at one mark, every value settled.

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
    location : int or None, optional
        The building the clock is in; ``None`` when no entry in force gives
        one.
    disabled : bool, optional
        Whether the clock is disabled; false when no entry in force says.

    Raises
    ------
    pydantic.ValidationError
        If the time constant is given for one state or missing for two or
        three.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    filter_states: _FilterStates
    time_constant: _TimeConstant | None
    scale_time_constant: _TimeConstant
    initial_innovation_scale: _Scale
    gap_limit: int
    location: _Location | None = None
    disabled: bool = False

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
            pair is not a reference, a dot and a clock, whose name holds no
            dot or slash.
        """
        for reference_name in self.references:
            if not is_reference(reference_name):
                message = (
                    f"rms_limit reference {reference_name} is not a reference name"
                )
                raise ValueError(message)
        for pair_name in self.pairs:
            reference, dot, clock = pair_name.partition(".")
            # A clock name holds no dot or slash: its series' files are named
            # with dots between the names (see registry.series_file).
            clock_named = clock and not {".", "/"} & set(clock)
            if not (dot and clock_named and is_reference(reference)):
                message = f"rms_limit pair {pair_name} is not written reference.clock"
                raise ValueError(message)
        return self


def _in_order(clock_entries: Iterable[Entry]) -> list[Entry]:
    """Put a clock's entries in the order they apply.

    Parameters
    ----------
    clock_entries : iterable of Entry
        The entries as the file lists them.

    Returns
    -------
    list of Entry
        Undated entries first, in file order, then dated ones by date.
    """
    return sorted(
        clock_entries,
        key=lambda clock_entry: (
            clock_entry.effective_mjd is not None,
            clock_entry.effective_mjd or 0.0,
        ),
    )


def _settled(
    type_default: TypeDefault, clock_entries: Iterable[Entry], settings_owner: str
) -> ClockEntry:
    """Apply entries to a type default and check the result.

    Parameters
    ----------
    type_default : TypeDefault
        The clock's type default.
    clock_entries : iterable of Entry
        The entries to apply, in order.
    settings_owner : str
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
    entry_values: dict[str, object] = type_default.model_dump()
    for clock_entry in clock_entries:
        entry_values |= clock_entry.overrides()
    try:
        return ClockEntry.model_validate(entry_values)
    except ValidationError as exc:
        message = f"{settings_owner}: {describe_error(exc)}"
        raise ValueError(message) from exc


class ClockConfig(BaseModel):
    """A deployment's clock configuration, checked.

    Parameters
    ----------
    rejects_before_restart : int
        N_break: the counted rejects in a row that make a series dormant,
        at least 3 and at most every clock's gap limit.
    reject_fraction_epochs : float
        The averaging length of the reject fraction, readings, at least 1:
        the newest reading weighs 1 over this (design 9.4).
    reject_fraction_limit : float
        The reject fraction above which a series goes dormant, between 0
        and 1, neither included.
    rms_limit : RmsLimits
        The RMS limits of the pairs' gate; held as ``rms_limits``, so the name
        is free for :meth:`rms_limit`.
    types : dict of str to TypeDefault
        The default entry of each clock type.
    clocks : dict of str to tuple of Entry
        Each clock's entries, the first giving its type or every setting.
    ignore : tuple of str, optional
        The clocks whose measurements and series are left out with no
        warning logged; none when the file names none.

    Raises
    ------
    pydantic.ValidationError
        If any check of the clock configuration fails: a clock ignored twice,
        or ignored and given entries; a clock with no
        entry, or whose first entry gives neither a type the file gives nor,
        undated, every setting itself; a later entry giving a type; a
        reference not of type mc; an entry changing the number of states;
        a time constant missing or
        given where the number of states says otherwise; a value out of its
        range, the reject fraction's length below 1 or its limit not between
        0 and 1 among them; or N_break above a gap limit at any date.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rejects_before_restart: Annotated[int, Field(ge=3)]
    reject_fraction_epochs: _TimeConstant
    reject_fraction_limit: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    rms_limits: RmsLimits = Field(alias="rms_limit")
    types: dict[str, TypeDefault]
    clocks: dict[str, tuple[Entry, ...]]
    ignore: tuple[str, ...] = ()

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
        self._check_ignored()
        for type_name, type_default in self.types.items():
            self._check_gap(
                _settled(type_default, (), f"type {type_name}"), f"type {type_name}"
            )
        for clock_name, clock_entries in self.clocks.items():
            type_default = self._type_default(clock_name, clock_entries)
            ordered_entries = _in_order(clock_entries)
            for entries_applied in range(len(ordered_entries) + 1):
                settings_owner = f"clock {clock_name}"
                self._check_gap(
                    _settled(
                        type_default, ordered_entries[:entries_applied], settings_owner
                    ),
                    clock_name,
                )
        return self

    def _check_ignored(self) -> None:
        """Refuse a clock ignored twice, or ignored and given entries.

        Raises
        ------
        ValueError
            Naming the first such clock.
        """
        seen: set[str] = set()
        for clock_name in self.ignore:
            if clock_name in seen:
                message = f"clock {clock_name} is ignored twice"
                raise ValueError(message)
            if clock_name in self.clocks:
                message = f"clock {clock_name} is ignored but has entries"
                raise ValueError(message)
            seen.add(clock_name)

    def _type_default(
        self, clock_name: str, clock_entries: Sequence[Entry]
    ) -> TypeDefault:
        """Give a clock's type default, checking its entries name it rightly.

        Parameters
        ----------
        clock_name : str
            The clock.
        clock_entries : sequence of Entry
            Its entries as the file lists them.

        Returns
        -------
        TypeDefault
            The default of the type its first entry gives, or the settings
            it gives itself when it gives no type.

        Raises
        ------
        ValueError
            If the clock has no entry, its first entry neither a type the
            file gives nor every setting, a later entry gives a type, a
            reference is not of type mc, or an entry changes the number of
            states.
        """
        if not clock_entries:
            message = f"clock {clock_name} has no entry"
            raise ValueError(message)
        clock_type = clock_entries[0].type
        if any(clock_entry.type is not None for clock_entry in clock_entries[1:]):
            message = f"the type of {clock_name} is given by its first entry only"
            raise ValueError(message)
        if is_reference(clock_name) and clock_type != REFERENCE_TYPE:
            message = f"clock {clock_name} is a reference, so of type {REFERENCE_TYPE}"
            raise ValueError(f"{message}, not {clock_type}")
        type_default = self._default_of(clock_name, clock_entries[0])
        for clock_entry in clock_entries:
            if clock_entry.filter_states not in {None, type_default.filter_states}:
                message = (
                    f"an entry changes the filter_states of {clock_name}"
                    f" from {type_default.filter_states} to {clock_entry.filter_states}"
                )
                raise ValueError(message)
        return type_default

    def _default_of(self, clock_name: str, first_entry: Entry) -> TypeDefault:
        """Give a clock's default: its type's, or its first entry's own settings.

        Parameters
        ----------
        clock_name : str
            The clock.
        first_entry : Entry
            Its first entry.

        Returns
        -------
        TypeDefault
            The default of the type the first entry gives; with no type,
            the settings the first entry gives itself.

        Raises
        ------
        ValueError
            If the type is one the file does not give, or a first entry
            with no type has a date or leaves out a setting.
        """
        if first_entry.type is not None:
            if first_entry.type not in self.types:
                message = (
                    f"clock {clock_name} has type {first_entry.type},"
                    " which types does not give"
                )
                raise ValueError(message)
            return self.types[first_entry.type]
        if first_entry.effective_mjd is not None:
            message = (
                f"the first entry of {clock_name} gives no type, so it holds from the"
                " start and has no effective_mjd"
            )
            raise ValueError(message)
        missing_settings = [
            setting_name
            for setting_name in _OWN_SETTINGS
            if getattr(first_entry, setting_name) is None
        ]
        if missing_settings:
            message = (
                f"the first entry of {clock_name} gives no type, so it gives every"
                f" setting; it leaves out {', '.join(missing_settings)}"
            )
            raise ValueError(message)
        own_settings = first_entry.overrides()
        own_settings.pop("location", None)
        own_settings.pop("disabled", None)
        return TypeDefault.model_validate(own_settings)

    def _check_gap(self, settled_entry: ClockEntry, settings_owner: str) -> None:
        """Refuse a gap limit below N_break.

        Parameters
        ----------
        settled_entry : ClockEntry
            Settled values of a clock or type.
        settings_owner : str
            What they are of, for the message.

        Raises
        ------
        ValueError
            If ``rejects_before_restart`` is above the gap limit.
        """
        if self.rejects_before_restart > settled_entry.gap_limit:
            message = (
                f"rejects_before_restart {self.rejects_before_restart} is above"
                f" the gap_limit {settled_entry.gap_limit} of {settings_owner}"
            )
            raise ValueError(message)

    def entry_for(self, clock: str, epoch_start: datetime) -> ClockEntry:
        """Give a clock's settings at a mark (design 15.2).

        Parameters
        ----------
        clock : str
            The clock.
        epoch_start : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        ClockEntry
            The clock's default (its type's, or its first entry's own
            settings when that entry gives no type), with every entry in
            force at ``epoch_start`` applied in order of ``effective_mjd``: an entry
            without one from the start, one with one from the first mark at
            or after it.

        Raises
        ------
        ConfigError
            If the configuration has no entry for ``clock``, or
            ``epoch_start`` has no timezone; the error is logged first.
        """
        clock_entries = self.clocks.get(clock)
        if clock_entries is None:
            message = f"the clock configuration has no entry for clock {clock}"
            _log.error(message)
            raise ConfigError(message)
        in_force = [
            clock_entry
            for clock_entry in _in_order(clock_entries)
            if clock_entry.effective_mjd is None
            or mjd_to_datetime(clock_entry.effective_mjd) <= _aware(epoch_start)
        ]
        type_default = self._type_default(clock, clock_entries)
        return _settled(type_default, in_force, f"clock {clock}")

    def changes_between(self, earlier: datetime, later: datetime) -> bool:
        """Tell whether any clock's settings change after one mark, up to another.

        Parameters
        ----------
        earlier : datetime
            The earlier mark; must carry a timezone.
        later : datetime
            The later mark; must carry a timezone.

        Returns
        -------
        bool
            Whether an entry of any clock takes effect in (``earlier``,
            ``later``]: the first mark at or after its ``effective_mjd``
            lies there. Without one, every clock's settings at ``later``
            are those at ``earlier`` (see :meth:`entry_for`).

        Raises
        ------
        ConfigError
            If either mark has no timezone.
        """
        earlier, later = _aware(earlier), _aware(later)
        return any(
            earlier < mjd_to_datetime(clock_entry.effective_mjd) <= later
            for clock_entries in self.clocks.values()
            for clock_entry in clock_entries
            if clock_entry.effective_mjd is not None
        )

    def locations_at(self, epoch_start: datetime) -> dict[str, int | None]:
        """Give every clock's location at a mark.

        Parameters
        ----------
        epoch_start : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        dict of str to int or None
            For each clock the file names, the location of the last of its
            entries in force at ``epoch_start`` to give one, in the order
            :meth:`entry_for` applies them; ``None`` when none does.

        Raises
        ------
        ConfigError
            If ``epoch_start`` has no timezone.
        """
        epoch_start = _aware(epoch_start)
        locations: dict[str, int | None] = {}
        for clock, clock_entries in self.clocks.items():
            location = None
            for clock_entry in _in_order(clock_entries):
                if clock_entry.effective_mjd is not None and (
                    mjd_to_datetime(clock_entry.effective_mjd) > epoch_start
                ):
                    break
                if clock_entry.location is not None:
                    location = clock_entry.location
            locations[clock] = location
        return locations

    def disabled_at(self, epoch_start: datetime) -> frozenset[str]:
        """Give the clocks disabled at a mark.

        Parameters
        ----------
        epoch_start : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        frozenset of str
            Every clock the file names whose entries in force at
            ``epoch_start``, applied in the order :meth:`entry_for` applies
            them, leave it disabled.

        Raises
        ------
        ConfigError
            If ``epoch_start`` has no timezone.
        """
        epoch_start = _aware(epoch_start)
        disabled_clocks: set[str] = set()
        for clock, clock_entries in self.clocks.items():
            disabled = False
            for clock_entry in _in_order(clock_entries):
                if clock_entry.effective_mjd is not None and (
                    mjd_to_datetime(clock_entry.effective_mjd) > epoch_start
                ):
                    break
                disabled = bool(clock_entry.overrides().get("disabled", disabled))
            if disabled:
                disabled_clocks.add(clock)
        return frozenset(disabled_clocks)

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
        pair_limit = self.rms_limits.pairs.get(f"{reference}.{clock}")
        if pair_limit is not None:
            return pair_limit
        return self.rms_limits.references.get(reference, self.rms_limits.default)

    def params_for(self, series_key: SeriesKey, epoch_start: datetime) -> SeriesParams:
        """Give a series' settings at a mark (design 8.1).

        Parameters
        ----------
        series_key : (str, str) or (str, str, str)
            A pair (a, b), which takes the entry of b and its RMS limit, or
            a triple (r, s, c), which takes the entry of c and has none.
        epoch_start : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        SeriesParams
            The series' settings; a pair is disabled when either of its
            clocks is, a triple never: to a triple a disabled clock is
            missing.

        Raises
        ------
        ConfigError
            If the configuration has no entry for the series' clock, or
            ``epoch_start`` has no timezone; the error is logged first.
        """
        return self._series_params(
            series_key,
            self.entry_for(series_key[-1], epoch_start),
            self.disabled_at(epoch_start),
        )

    def params_for_series(
        self, series_keys: Iterable[SeriesKey], epoch_start: datetime
    ) -> dict[SeriesKey, SeriesParams]:
        """Give every series' settings at a mark, each clock's entry settled once.

        Parameters
        ----------
        series_keys : iterable of series key
            The series.
        epoch_start : datetime
            The epoch start; must carry a timezone.

        Returns
        -------
        dict of series key to SeriesParams
            What :meth:`params_for` gives each series.

        Raises
        ------
        ConfigError
            If the configuration has no entry for a series' clock, or
            ``epoch_start`` has no timezone; the error is logged first.
        """
        clock_entries: dict[str, ClockEntry] = {}
        disabled_clocks = self.disabled_at(epoch_start)
        series_params = {}
        for series_key in series_keys:
            clock = series_key[-1]
            if clock not in clock_entries:
                clock_entries[clock] = self.entry_for(clock, epoch_start)
            series_params[series_key] = self._series_params(
                series_key, clock_entries[clock], disabled_clocks
            )
        return series_params

    def _series_params(
        self,
        series_key: SeriesKey,
        clock_entry: ClockEntry,
        disabled_clocks: frozenset[str],
    ) -> SeriesParams:
        """Give a series its settings from its clock's entry.

        Parameters
        ----------
        series_key : (str, str) or (str, str, str)
            The series.
        clock_entry : ClockEntry
            The entry of its clock side at the mark.
        disabled_clocks : frozenset of str
            The clocks disabled at the mark.

        Returns
        -------
        SeriesParams
            The entry's values, N_break, the reject fraction's weight, 1
            over its averaging length, and limit, and, for a pair, its RMS
            limit and whether either of its clocks is disabled.
        """
        is_pair = len(series_key) == _PAIR
        rms_max = self.rms_limit((series_key[0], series_key[1])) if is_pair else None
        return SeriesParams(
            filter_states=clock_entry.filter_states,
            M=clock_entry.time_constant,
            M_sigma=clock_entry.scale_time_constant,
            sigma0=clock_entry.initial_innovation_scale,
            gmax=clock_entry.gap_limit,
            n_break=self.rejects_before_restart,
            reject_fraction_weight=1.0 / self.reject_fraction_epochs,
            reject_fraction_limit=self.reject_fraction_limit,
            rms_max=rms_max,
            disabled=is_pair and not disabled_clocks.isdisjoint(series_key),
        )


def _aware(epoch_start: datetime) -> datetime:
    """Refuse a mark without a timezone.

    Parameters
    ----------
    epoch_start : datetime
        The mark.

    Returns
    -------
    datetime
        ``epoch_start``, unchanged.

    Raises
    ------
    ConfigError
        If ``epoch_start`` has no timezone, and so names no one instant.
    """
    if epoch_start.tzinfo is None:
        message = f"the mark {epoch_start} has no timezone"
        _log.error(message)
        raise ConfigError(message)
    return epoch_start


def read_clock_config(config_file: Path) -> ClockConfig:
    """Read and check a clock configuration file (design 15.2).

    Parameters
    ----------
    config_file : Path
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
        yaml_text = config_file.read_text(encoding="utf-8")
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        _fail(config_file, f"cannot read: {exc}", exc)
    yaml_loader = _Loader(yaml_text)
    try:
        loaded_yaml = yaml_loader.get_single_data()
    except yaml.YAMLError as exc:
        _fail(config_file, describe_error(exc), exc)
    finally:
        yaml_loader.dispose()
    if not isinstance(loaded_yaml, dict):
        _fail(config_file, "is not a mapping of the sections", None)
    try:
        return ClockConfig.model_validate(_tuples(loaded_yaml))
    except ValidationError as exc:
        _fail(config_file, describe_error(exc), exc)


def _fail(config_file: Path, problem: str, cause: Exception | None) -> NoReturn:
    """Log and raise a clock configuration error.

    Parameters
    ----------
    config_file : Path
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
    message = f"clock configuration {config_file}: {problem}"
    _log.error(message)
    raise ConfigError(message) from cause
