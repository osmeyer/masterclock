"""das_processor's settings, merged from its INI file and its command line.

Each setting is described once, in :data:`SETTINGS`, and merged by the rules
every program shares (see :mod:`masterclock.app.config`): the command line
wins where both sources give a setting, a required setting given by neither
is a usage error, and an optional one given by neither is ``None``. The
merged values are validated into the frozen :class:`AppConfig`, reading text
from the file as the command line reads the same setting, so a setting
accepts the same text from either source.

The one setting with a value of its own is ``start_from_mjd``: given by
neither source, it is :data:`~masterclock.das_processor.cli.START_FROM_MJD`.

:func:`check_paths` is separate on purpose. Building the configuration says
whether the settings are well formed and nothing about the machine they are
read on; checking the paths asks whether the deployment they describe is one
a run could use. It is called before a run starts, so a wrong path is
refused the way every other wrong setting is.

Every problem is a :class:`~masterclock.app.exceptions.ConfigError`. A
required setting given by neither source is its
:class:`~masterclock.app.exceptions.MissingSettingsError` subclass, so the
entry point can report it as a command-line usage error (full help, exit
status 2) rather than a configuration failure.
"""

import os
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from masterclock.app.config import (
    AbsolutePath,
    LoggingConfig,
    Setting,
    as_on_command_line,
    known_entries,
    known_sections,
    merge,
    required_settings,
)
from masterclock.app.exceptions import ConfigError, describe_error
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.cli import START_FROM_MJD, DataMjd, data_mjd

if TYPE_CHECKING:
    from pathlib import Path

    from masterclock.das_processor.cli import CliOptions

SETTINGS: Final[tuple[Setting, ...]] = (
    Setting("rf", "das", "DAS", "rf", "--rf", allow_none=False, required=True),
    Setting(
        "cd5m5m_path",
        "das",
        "DAS",
        "cd5m5m_path",
        "--cd5m5m-path",
        allow_none=False,
        required=True,
    ),
    Setting(
        "steering_path",
        "das",
        "DAS",
        "steering_path",
        "--steering-path",
        allow_none=False,
        required=True,
    ),
    Setting(
        "processed_path",
        "processed",
        "PROCESSED",
        "processed_path",
        "--processed-path",
        allow_none=False,
        required=True,
    ),
    Setting(
        "clock_config_file",
        "processed",
        "PROCESSED",
        "clock_config_file",
        "--clock-config-file",
        allow_none=False,
        required=True,
    ),
    Setting(
        "log_file",
        "logging",
        "LOGGING",
        "log_file",
        "--log-file",
        allow_none=True,
        required=True,
    ),
    Setting(
        "log_level",
        "logging",
        "LOGGING",
        "log_level",
        "--log-level",
        allow_none=True,
        required=True,
    ),
    Setting(
        "start_from_mjd",
        "processed",
        "PROCESSED",
        "start_from_mjd",
        "--start-from-mjd",
        allow_none=False,
    ),
    Setting(
        "backup_count",
        "logging",
        "LOGGING",
        "backup_count",
        "--backup-count",
        allow_none=True,
    ),
)
"""Every setting the program takes, required ones first.

The required ones lead so that a missing-settings message lists them in the
order the command-line help does.
"""

REQUIRED_SETTINGS: Final[tuple[tuple[str, str, str], ...]] = required_settings(SETTINGS)
"""Every required setting as ``(section, entry, flag)``, from :data:`SETTINGS`."""

KNOWN_SECTIONS: Final[frozenset[str]] = known_sections(SETTINGS)
"""Every INI section the program reads, from :data:`SETTINGS`."""

KNOWN_ENTRIES: Final[dict[str, frozenset[str]]] = known_entries(SETTINGS)
"""Every INI entry the program reads, by section, from :data:`SETTINGS`."""

MEAS_SUBDIRECTORY: Final[str] = "meas"
"""The ``processed_path`` subdirectory holding the measurement files."""

DDIFF_SUBDIRECTORY: Final[str] = "ddiff"
"""The ``processed_path`` subdirectory holding the double-difference files."""

LOCK_FILE_TEMPLATE: Final[str] = "das_processor_{rf}.lock"
"""Name of the run lock file, directly in ``processed_path``.

Beside the subdirectories rather than inside one, so the scans that read
their files never meet it. One per RF channel, because the channels write
separate files and their runs may overlap.
"""

JOURNAL_FILE_TEMPLATE: Final[str] = "das_processor_{rf}.writing"
"""The name of an RF channel's write journal in ``processed_path``.

It is there only while the day's rows are being written, and holds the
first epoch being written, so a run that finds it knows a write stopped
part way and rolls every file back to before that epoch.
"""


def _start_when_not_given(value: object) -> object:
    """Give :data:`~masterclock.das_processor.cli.START_FROM_MJD` for no value.

    Parameters
    ----------
    value : object
        The merged ``start_from_mjd``: ``None`` when neither source gave it,
        since the setting does not accept the literal ``None``.

    Returns
    -------
    object
        :data:`~masterclock.das_processor.cli.START_FROM_MJD` for ``None``,
        otherwise ``value`` unchanged.
    """
    return START_FROM_MJD if value is None else value


type DayMjd = Annotated[DataMjd, as_on_command_line(data_mjd)]
"""An MJD on a day a data file covers, from text read as the command line reads it."""


class DasConfig(BaseModel):
    """Validated DAS settings.

    Parameters
    ----------
    rf : RfChannel
        Which RF channel to process.
    cd5m5m_path : Path
        Absolute path of the directory holding the DAS 5 MHz phase data.
    steering_path : Path
        Absolute path of the directory holding the steering file of each
        reference clock.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rf: RfChannel
    cd5m5m_path: AbsolutePath
    steering_path: AbsolutePath


class ProcessedConfig(BaseModel):
    """Validated processed-output settings.

    Parameters
    ----------
    processed_path : Path
        Absolute path under which processed results are written: each kind
        of file in a subdirectory of its own (:attr:`meas_path`,
        :attr:`ddiff_path`), with the run lock directly in it beside them.
    start_from_mjd : DataMjd
        MJD to start processing from when there are no processed files to
        read a previous measurement from; once there are, it has no effect.
        ``None`` becomes :data:`~masterclock.das_processor.cli.START_FROM_MJD`.
    clock_config_file : Path
        Absolute path of the YAML file giving each clock's estimator
        parameters and each pair's RMS limit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    processed_path: AbsolutePath
    start_from_mjd: Annotated[DayMjd, BeforeValidator(_start_when_not_given)]
    clock_config_file: AbsolutePath

    @property
    def meas_path(self) -> Path:
        """Path: The directory holding the measurement files."""
        return self.processed_path / MEAS_SUBDIRECTORY

    @property
    def ddiff_path(self) -> Path:
        """Path: The directory holding the double-difference files."""
        return self.processed_path / DDIFF_SUBDIRECTORY


class AppConfig(BaseModel):
    """The complete validated configuration of das_processor.

    Parameters
    ----------
    das : DasConfig
        The DAS settings.
    processed : ProcessedConfig
        The processed-output settings.
    logging : LoggingConfig
        The logging settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    das: DasConfig
    processed: ProcessedConfig
    logging: LoggingConfig


def _check_input_directory(path: Path, setting: str) -> None:
    """Refuse an input directory a run could not read.

    Parameters
    ----------
    path : Path
        The configured directory.
    setting : str
        The setting that names it, as ``[SECTION] entry``, for the message.

    Raises
    ------
    ConfigError
        If ``path`` is not a directory, or cannot be listed.

    Notes
    -----
    Listed rather than judged by its permission bits, so whatever would
    stop the run from listing it, stops it here.
    """
    if not path.is_dir():
        raise ConfigError(f"{setting}: {path} is not a directory to read from")
    try:
        next(path.iterdir(), None)
    except OSError as exc:
        raise ConfigError(f"{setting}: {path} cannot be listed: {exc}") from exc


def _check_input_file(path: Path, setting: str) -> None:
    """Refuse an input file a run could not read.

    Parameters
    ----------
    path : Path
        The configured file.
    setting : str
        The setting that names it, as ``[SECTION] entry``, for the message.

    Raises
    ------
    ConfigError
        If ``path`` is not a regular file, or a link to one, or cannot be
        opened for reading.

    Notes
    -----
    Opened rather than judged by its permission bits, so whatever would
    stop the run from reading it, stops it here.
    """
    if not path.is_file():
        raise ConfigError(f"{setting}: {path} is not a file to read from")
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        raise ConfigError(f"{setting}: {path} cannot be read: {exc}") from exc


def _check_processed_directory(path: Path) -> None:
    """Refuse a processed directory a run could not write into.

    Parameters
    ----------
    path : Path
        The configured ``processed_path``.

    Raises
    ------
    ConfigError
        If something other than a directory is at ``path``, or the directory
        there does not let this process make and open files in it.

    Notes
    -----
    A ``processed_path`` that is not there is not refused: a first run makes
    its own directory.
    """
    if not path.exists():
        return
    if not path.is_dir():
        raise ConfigError(
            f"[PROCESSED] processed_path: {path} is not a directory to write "
            "processed files into"
        )
    if not os.access(path, os.W_OK | os.X_OK):
        raise ConfigError(
            f"[PROCESSED] processed_path: {path} is a directory this process "
            "cannot write into"
        )


def check_paths(config: AppConfig) -> None:
    """Refuse a configuration whose paths describe a deployment that cannot run.

    Parameters
    ----------
    config : AppConfig
        The configuration to check.

    Raises
    ------
    ConfigError
        Naming the setting whose path cannot be used: a data or steering
        directory that is not one or cannot be listed, a clock configuration
        file that is not a regular file or cannot be read, or a processed
        directory that is something else or cannot be written into.

    Notes
    -----
    Separate from building the configuration, which validates the settings
    themselves and says nothing about the machine they are read on. This is
    the other half: whether the deployment those settings describe is one a
    run could use, so a run is refused before it starts rather than failing
    part way through.
    """
    _check_input_directory(config.das.cd5m5m_path, "[DAS] cd5m5m_path")
    _check_input_directory(config.das.steering_path, "[DAS] steering_path")
    _check_input_file(
        config.processed.clock_config_file, "[PROCESSED] clock_config_file"
    )
    _check_processed_directory(config.processed.processed_path)


def build_logging_config(options: CliOptions) -> LoggingConfig:
    """Merge and validate the logging settings alone, so logging can start first.

    Parameters
    ----------
    options : CliOptions
        The parsed command-line options.

    Returns
    -------
    LoggingConfig
        The effective, validated logging settings.

    Raises
    ------
    ConfigError
        If the INI file cannot be read or parsed, names anything the
        program does not read, or a logging value is invalid.
    MissingSettingsError
        If a required logging setting is given by neither source.
    """
    values = merge(SETTINGS, options, options.config_file, groups={"logging"})
    try:
        return LoggingConfig.model_validate(values["logging"])
    except ValidationError as exc:
        raise ConfigError(
            f"invalid configuration values: logging: {describe_error(exc)}"
        ) from exc


def build_config(options: CliOptions) -> AppConfig:
    """Merge and validate the effective configuration.

    Reads the INI file named by ``options.config_file``, if any, overlays the
    command-line options (the command line wins), checks that every required
    setting was given by at least one source, and validates the result.

    Parameters
    ----------
    options : CliOptions
        The parsed command-line options.

    Returns
    -------
    AppConfig
        The effective, validated configuration.

    Raises
    ------
    ConfigError
        If the INI file cannot be read or parsed, or a merged value is
        invalid.
    MissingSettingsError
        If a required setting is given by neither source. A
        :class:`~masterclock.app.exceptions.ConfigError` itself, so an
        ``except ConfigError`` still catches it.
    """
    values = merge(SETTINGS, options, options.config_file)
    try:
        return AppConfig.model_validate(values)
    except ValidationError as exc:
        raise ConfigError(
            f"invalid configuration values: {describe_error(exc)}"
        ) from exc
