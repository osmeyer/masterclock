"""Merging a program's settings from its file and its command line.

Every program in the project reads an INI file of program settings and takes
the same settings on its command line, where the command line wins. What
differs is which settings there are; how they are named, merged, refused and
grouped does not.

A setting is named in four places: the command-line field it arrives on, the
INI section and entry it may also arrive in, the flag a message names it by,
and the group of the validated configuration it lands in. Those never vary
independently. :class:`Setting` holds them together, so a program describes
each of its settings once and everything else is derived from that
description.

A file naming anything the program does not read is refused rather than
ignored, the ``DEFAULT`` section included, and so is a value that runs over
more than one line. Otherwise a misspelled entry, or one an indented line
was joined onto, would be passed over in silence and the setting it was
meant to give would go unset, which is the hardest kind of configuration
mistake to find.

A value read from the file is text, and the model it is validated by reads
it as the command line reads the same setting, so a setting accepts the same
text from either source.
"""

import argparse
import configparser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PositiveInt,
)

from masterclock.app.cli import (
    NONE_LITERAL,
    UNSET,
    LogLevelName,
    Unset,
    positive_int,
)
from masterclock.app.exceptions import ConfigError, MissingSettingsError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

_NO_DEFAULT_SECTION: Final[str] = "\n"
"""The parser's name for its default section, one no section header can give.

A header is read from one line, so no file can name a section with a line
break in it. With this name, a ``[DEFAULT]`` header is an ordinary section,
and is refused as unknown even when it holds no entries.
"""


@dataclass(frozen=True, slots=True)
class Setting:
    """Everything that distinguishes one configuration setting from another.

    A setting is named in four places: the command-line options field it
    arrives on, the INI section and entry it may also arrive in, the
    command-line flag a message names it by, and the sub-model of the
    program's configuration it lands in. Those never vary independently.
    Holding them together makes a program's table of these the one place a
    setting is described, so adding one is a single row rather than an edit
    in four places that must agree.

    Parameters
    ----------
    options_field : str
        Name of the field on the program's command-line options, which is
        also the field name in the configuration sub-model.
    config_group : str
        The field of the program's configuration model the setting lands in,
        which is also the name of the section's sub-model.
    ini_section : str
        The INI section the setting may be read from.
    ini_entry : str
        The INI entry name within that section, in lowercase, since
        :mod:`configparser` lowercases the entry names it reads.
    cli_flag : str
        The command-line flag, as a missing-settings message spells it.
    allow_none : bool
        Whether the literal token ``"None"`` is meaningful for this setting
        and converts to ``None``.
    required : bool, optional
        Whether some source must provide it. A required setting absent from
        both raises
        :class:`~masterclock.app.exceptions.MissingSettingsError`;
        an optional one becomes ``None``.
    """

    options_field: str
    config_group: str
    ini_section: str
    ini_entry: str
    cli_flag: str
    allow_none: bool
    required: bool = False


def as_on_command_line[ValueT](
    cli_conversion: Callable[[str], ValueT],
) -> BeforeValidator:
    """Make a model field read text as a command-line conversion reads it.

    Text is given to ``cli_conversion``, and its refusal becomes the field's
    validation error with the same reason. Anything that is not text, such
    as a value the command line has already converted, is passed on to the
    field's own validation unchanged.

    Parameters
    ----------
    cli_conversion : callable
        One of the command-line conversions in
        :mod:`masterclock.app.cli`, taking the text and raising
        :class:`argparse.ArgumentTypeError` for text it refuses.

    Returns
    -------
    pydantic.BeforeValidator
        The validator, to go in the field's ``Annotated`` type.
    """

    def read(field_input: object) -> object:
        """Convert text with ``cli_conversion``; pass anything else on unchanged."""
        if not isinstance(field_input, str):
            return field_input
        try:
            return cli_conversion(field_input)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc

    return BeforeValidator(read)


def _absolute(given_path: Path) -> Path:
    """Refuse a path that is not absolute, as the command line does.

    Parameters
    ----------
    given_path : Path
        The path a field was given.

    Returns
    -------
    Path
        ``given_path`` unchanged.

    Raises
    ------
    ValueError
        If ``given_path`` is relative, the empty path included.
    """
    if not given_path.is_absolute():
        raise ValueError(f"path must be absolute: {str(given_path)!r}")
    return given_path


type AbsolutePath = Annotated[Path, AfterValidator(_absolute)]
"""A path a model refuses unless it is absolute, as the command line does."""


class LoggingConfig(BaseModel):
    """Validated logging settings.

    Parameters
    ----------
    log_file : Path or None
        Absolute path of the log file, or ``None`` when file logging is
        disabled.
    log_level : LogLevelName or None
        Name of the logging level, or ``None`` when logging is disabled
        entirely.
    backup_count : PositiveInt or None
        Number of rotated daily log files to keep, or ``None`` to keep every
        rotated file. Text is read as
        :func:`~masterclock.app.cli.positive_int` reads it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    log_file: AbsolutePath | None
    log_level: LogLevelName | None
    backup_count: Annotated[PositiveInt | None, as_on_command_line(positive_int)]


def _checked(settings: Sequence[Setting]) -> Sequence[Setting]:
    """Refuse a table of settings that names a setting badly.

    Parameters
    ----------
    settings : Sequence of Setting
        A program's settings.

    Returns
    -------
    Sequence of Setting
        ``settings`` unchanged.

    Raises
    ------
    ValueError
        If two settings share an ``options_field``, or a section and entry, since
        one would quietly take the other's value; or if a setting's entry
        is not in lowercase, or its section is ``DEFAULT``, since no file
        could then give it.
    """
    attribute_counts = Counter(setting.options_field for setting in settings)
    place_counts = Counter(
        (setting.ini_section, setting.ini_entry) for setting in settings
    )
    for setting in settings:
        if attribute_counts[setting.options_field] > 1:
            naming_problem = (
                f"setting {setting.options_field!r} is described more than once"
            )
        elif place_counts[setting.ini_section, setting.ini_entry] > 1:
            naming_problem = (
                f"entry [{setting.ini_section}] {setting.ini_entry} is read by"
                " more than one setting"
            )
        elif setting.ini_entry != setting.ini_entry.lower():
            naming_problem = (
                f"entry [{setting.ini_section}] {setting.ini_entry} is not in"
                " lowercase, so no file can give it"
            )
        elif setting.ini_section == configparser.DEFAULTSECT:
            naming_problem = (
                f"section [{setting.ini_section}] is refused in every file, so no"
                " file can give it"
            )
        else:
            continue
        raise ValueError(naming_problem)
    return settings


def _unknown_names(
    parser: configparser.ConfigParser, settings: Sequence[Setting]
) -> list[str]:
    """Name every section and entry the program does not read.

    Parameters
    ----------
    parser : configparser.ConfigParser
        A parser that has already read the configuration file.
    settings : Sequence of Setting
        The program's settings, which say what is known.

    Returns
    -------
    list[str]
        One ``[SECTION]`` per unknown section and then one
        ``[SECTION] entry`` per unknown entry of a known section, each in
        the order of the file. Empty when the file names nothing the program
        does not read.

    Notes
    -----
    Case is treated as :mod:`configparser` treats it: section names as
    written, entry names lowercased. A ``DEFAULT`` section is an unknown
    section like any other (see :data:`_NO_DEFAULT_SECTION`), since its
    entries would otherwise be merged into every section.
    """
    known_section_names = known_sections(settings)
    known_entry_names = known_entries(settings)
    unknown_names = [
        f"[{ini_section}]"
        for ini_section in parser.sections()
        if ini_section not in known_section_names
    ]
    unknown_names.extend(
        f"[{ini_section}] {ini_entry}"
        for ini_section in parser.sections()
        if ini_section in known_section_names
        for ini_entry in parser[ini_section]
        if ini_entry not in known_entry_names[ini_section]
    )
    return unknown_names


def _reject_unknown(
    config_file: Path, parser: configparser.ConfigParser, settings: Sequence[Setting]
) -> None:
    """Refuse a configuration file naming anything the program does not read.

    Parameters
    ----------
    config_file : Path
        The file being read, named in the error.
    parser : configparser.ConfigParser
        A parser that has already read it.
    settings : Sequence of Setting
        The program's settings, which say what is known.

    Raises
    ------
    ConfigError
        If the file contains any section or entry the program does not read.
    """
    unknown_names = _unknown_names(parser, settings)
    if unknown_names:
        raise ConfigError(
            f"config file {config_file} names what the program does not read: "
            f"{', '.join(unknown_names)}"
        )


def _reject_several_lines(config_file: Path, parser: configparser.ConfigParser) -> None:
    """Refuse a configuration file with a value that runs over several lines.

    :mod:`configparser` joins an indented line onto the value above it, so
    an entry indented by mistake would become part of another's value.

    Parameters
    ----------
    config_file : Path
        The file being read, named in the error.
    parser : configparser.ConfigParser
        A parser that has already read it.

    Raises
    ------
    ConfigError
        If any value, as written, holds a line break.
    """
    for ini_section in parser.sections():
        for ini_entry, entry_text in parser.items(ini_section, raw=True):
            if "\n" in entry_text:
                raise ConfigError(
                    f"config file {config_file} gives [{ini_section}] {ini_entry}"
                    " on more than one line"
                )


def read_entries(
    config_file: Path, settings: Sequence[Setting]
) -> dict[str, dict[str, str]]:
    """Read an INI file into a plain section/entry mapping.

    Missing sections and entries simply do not appear in the result; the
    file is not required to be complete. It is required to name nothing the
    program does not read, and to give each value on one line.

    Values go through :mod:`configparser`'s basic interpolation:
    ``%(entry)s`` is replaced by the value of another entry of the same
    section, ``%%`` stands for one ``%``, and any other ``%`` makes the file
    invalid.

    Parameters
    ----------
    config_file : Path
        Path to the INI configuration file.
    settings : Sequence of Setting
        The program's settings, which say what the file may name.

    Returns
    -------
    dict[str, dict[str, str]]
        The entries, keyed by section and then entry name.

    Raises
    ------
    ConfigError
        If the file cannot be read or is not UTF-8, is not valid INI syntax,
        names a section or entry the program does not read, or gives a value
        on more than one line.
    ValueError
        If ``settings`` names a setting badly (see :func:`known_sections`).
    """
    _checked(settings)
    parser = configparser.ConfigParser(default_section=_NO_DEFAULT_SECTION)
    try:
        with config_file.open(encoding="utf-8") as open_file:
            parser.read_file(open_file)
        _reject_unknown(config_file, parser, settings)
        _reject_several_lines(config_file, parser)
        return {
            ini_section: dict(parser.items(ini_section))
            for ini_section in parser.sections()
        }
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        raise ConfigError(f"cannot read config file {config_file}: {exc}") from exc
    except configparser.Error as exc:
        raise ConfigError(f"config file {config_file} is not valid INI: {exc}") from exc


def _file_value(
    file_entries: dict[str, dict[str, str]],
    ini_section: str,
    ini_entry: str,
    *,
    allow_none: bool,
) -> str | Unset | None:
    """Look up an entry in the config-file mapping.

    Parameters
    ----------
    file_entries : dict[str, dict[str, str]]
        The entries read by :func:`read_entries`.
    ini_section : str
        The section name.
    ini_entry : str
        The entry name within the section.
    allow_none : bool
        Whether the literal token ``"None"`` is meaningful for this entry
        and converts to ``None``.

    Returns
    -------
    str, None, or Unset
        The entry's value; ``None`` if it is the token ``"None"`` and
        ``allow_none`` is set; :data:`~masterclock.app.cli.UNSET`
        if the entry is absent.
    """
    if ini_entry not in file_entries.get(ini_section, {}):
        return UNSET
    entry_text = file_entries[ini_section][ini_entry]
    return None if allow_none and entry_text == NONE_LITERAL else entry_text


def _pick[ValueT](
    cli_value: ValueT | Unset, file_value: str | Unset | None
) -> ValueT | str | Unset | None:
    """Choose the command-line value when provided, otherwise the file value.

    Parameters
    ----------
    cli_value : ValueT or Unset
        The value from the command line,
        :data:`~masterclock.app.cli.UNSET` when the option was
        omitted.
    file_value : str, None, or Unset
        The value from the config file,
        :data:`~masterclock.app.cli.UNSET` when the entry is
        absent.

    Returns
    -------
    ValueT, str, None, or Unset
        The effective value; :data:`~masterclock.app.cli.UNSET`
        only when neither source provided one.
    """
    return file_value if isinstance(cli_value, Unset) else cli_value


def _missing_settings(
    effective_values: Mapping[str, object], settings: Sequence[Setting]
) -> list[str]:
    """Name every required setting that neither source provided.

    Parameters
    ----------
    effective_values : Mapping[str, object]
        The effective value of every setting, keyed by ``options_field``;
        :data:`~masterclock.app.cli.UNSET` where neither source
        provided one.
    settings : Sequence of Setting
        The program's settings, in the order a message lists them.

    Returns
    -------
    list[str]
        One ``[SECTION] entry (--flag)`` per missing required setting, in
        the order of ``settings``. Empty when every required setting was
        provided.
    """
    return [
        f"[{setting.ini_section}] {setting.ini_entry} ({setting.cli_flag})"
        for setting in settings
        if setting.required
        and isinstance(effective_values[setting.options_field], Unset)
    ]


def _grouped_values(
    effective_values: Mapping[str, object], settings: Sequence[Setting]
) -> dict[str, dict[str, object]]:
    """Sort the effective values into the sub-model each one belongs to.

    A setting neither source provided becomes ``None``, which means no
    value; what that means is up to the setting. Required settings are
    always provided by the time this runs (see :func:`_missing_settings`).

    Parameters
    ----------
    effective_values : Mapping[str, object]
        The effective value of every setting, keyed by ``options_field``.
    settings : Sequence of Setting
        The program's settings, which say which group each lands in.

    Returns
    -------
    dict[str, dict[str, object]]
        The values keyed by the configuration model's field and then by
        setting, ready for :meth:`~pydantic.BaseModel.model_validate`.
    """
    grouped: dict[str, dict[str, object]] = {
        setting.config_group: {} for setting in settings
    }
    for setting in settings:
        effective_value = effective_values[setting.options_field]
        grouped[setting.config_group][setting.options_field] = (
            None if isinstance(effective_value, Unset) else effective_value
        )
    return grouped


def known_sections(settings: Sequence[Setting]) -> frozenset[str]:
    """Name every INI section a program reads.

    Parameters
    ----------
    settings : Sequence of Setting
        The program's settings.

    Returns
    -------
    frozenset of str
        The sections, derived from the settings themselves so that adding one
        cannot leave this behind.

    Raises
    ------
    ValueError
        If two settings share an ``options_field``, or a section and entry, or a
        setting's entry is not in lowercase or its section is ``DEFAULT``.
    """
    return frozenset(setting.ini_section for setting in _checked(settings))


def known_entries(settings: Sequence[Setting]) -> dict[str, frozenset[str]]:
    """Name every INI entry a program reads, by section.

    Parameters
    ----------
    settings : Sequence of Setting
        The program's settings.

    Returns
    -------
    dict of str to frozenset of str
        The entries of each section, derived from the settings themselves.

    Raises
    ------
    ValueError
        If ``settings`` names a setting badly (see :func:`known_sections`).
    """
    return {
        ini_section: frozenset(
            setting.ini_entry
            for setting in settings
            if setting.ini_section == ini_section
        )
        for ini_section in known_sections(settings)
    }


def required_settings(settings: Sequence[Setting]) -> tuple[tuple[str, str, str], ...]:
    """Name every setting some source must provide.

    Parameters
    ----------
    settings : Sequence of Setting
        The program's settings.

    Returns
    -------
    tuple of tuple
        Each required setting as ``(section, entry, flag)``, in order.

    Raises
    ------
    ValueError
        If ``settings`` names a setting badly (see :func:`known_sections`).
    """
    return tuple(
        (setting.ini_section, setting.ini_entry, setting.cli_flag)
        for setting in _checked(settings)
        if setting.required
    )


def merge(
    settings: Sequence[Setting],
    cli_options: object,
    config_file: Path | None,
    *,
    config_groups: Collection[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Merge a program's settings from its file and its command line.

    Parameters
    ----------
    settings : Sequence of Setting
        The program's settings.
    cli_options : object
        Its parsed command-line options, carrying a field per setting.
    config_file : Path or None
        Its configuration file, or None where the command line named none and
        every required setting must come from the command line.
    config_groups : Collection of str or None, optional
        The parts of the configuration to merge, such as only the logging
        settings so logging can start before the rest are checked; ``None``,
        the default, for every part. The file is still checked against
        every setting.

    Returns
    -------
    dict of str to dict
        The effective values, grouped by the part of the configuration each
        belongs to, ready to be validated; only those of ``config_groups`` when
        given.

    Raises
    ------
    ConfigError
        If the file cannot be read, is not valid INI, names anything the
        program does not read, or gives a value on more than one line.
    MissingSettingsError
        If a required setting of ``config_groups`` was provided by neither source.
        A ConfigError itself, so one clause still catches both.
    ValueError
        If ``settings`` names a setting badly (see :func:`known_sections`).

    Notes
    -----
    The command line wins where both sources give a setting, and a setting
    neither gives is ``None``.
    """
    _checked(settings)
    file_entries = (
        read_entries(config_file, settings) if config_file is not None else {}
    )
    chosen_settings = [
        setting
        for setting in settings
        if config_groups is None or setting.config_group in config_groups
    ]
    effective_values = {
        setting.options_field: _pick(
            getattr(cli_options, setting.options_field),
            _file_value(
                file_entries,
                setting.ini_section,
                setting.ini_entry,
                allow_none=setting.allow_none,
            ),
        )
        for setting in chosen_settings
    }
    missing_names = _missing_settings(effective_values, chosen_settings)
    if missing_names:
        raise MissingSettingsError(
            "these settings must be provided by the config file or the "
            f"command line: {', '.join(missing_names)}"
        )
    return _grouped_values(effective_values, chosen_settings)
