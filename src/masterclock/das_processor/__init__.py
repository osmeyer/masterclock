"""das_processor: the masterclock program that conditions the DAS phase data.

A scheduler starts it for each RF channel every ten minutes. Each run
processes its channel's epochs, in order, up to the end of the DAS data, or
``--steps`` of them, and exits: it reads the raw ten-minute comparisons,
decycles and screens them, runs every pair's and triple's estimator, and
appends one row per epoch to each series' file.

:func:`main` is the entry point. Its exit status is 0 for a run that
finished, 2 for a usage error such as a required setting given by neither
the INI file nor the command line, and 1 for any other failure. Logging
starts from the logging settings before anything else is checked, so a
failure is logged; only one that keeps logging from starting, or any when
the log level is None, is printed on standard error instead.
"""

import logging
import sys
from collections.abc import Sequence
from typing import Final

from masterclock.app.exceptions import MasterClockError, MissingSettingsError
from masterclock.app.lock import RunLock
from masterclock.app.log import MasterClockLogger, configure_logging, get_logger
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor.cli import CliOptions, parse_args, usage_error
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.config import (
    LOCK_FILE_TEMPLATE,
    AppConfig,
    build_config,
    build_logging_config,
    check_paths,
)
from masterclock.das_processor.epochs import floor_to_ten_minutes
from masterclock.das_processor.files import redo_from
from masterclock.das_processor.run import data_series
from masterclock.das_processor.run import run as run_channel

SUCCESS: Final[int] = 0
"""The exit status of a run that finished."""

FAILURE: Final[int] = 1
"""The exit status of a run that failed; the failure is already reported."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""

_SILENT: Final[int] = logging.CRITICAL + 1
"""A logging level above every record's, for a run with logging set to None."""


def _read_settings(argv: Sequence[str] | None) -> tuple[AppConfig, CliOptions] | None:
    """Read the run's settings: the logging ones first, then the rest.

    Parameters
    ----------
    argv : Sequence of str or None
        The arguments, without the program name; ``None`` for
        :data:`sys.argv`.

    Returns
    -------
    tuple of (AppConfig, CliOptions) or None
        The settings, and the command line for what only it gives
        (``--steps`` and ``--redo-from-mjd``); ``None`` when a setting
        cannot be used, after saying why.

    Raises
    ------
    SystemExit
        With status 2 and the full help for a usage error, above all a
        required setting given by neither source.

    Notes
    -----
    Logging starts from the logging settings alone, so an error in any
    other setting or path is logged at ERROR before the run stops. Only an
    error that keeps logging from starting is printed on standard error
    alone, and so is any error when the log level is None, since nothing is
    then logged.
    """
    cli_options = parse_args(argv)
    logging_silenced = _start_logging(cli_options)
    if logging_silenced is None:
        return None
    try:
        config = build_config(cli_options)
        check_paths(config)
    except MissingSettingsError as exc:
        _log.error("%s", exc)
        usage_error(str(exc))
    except (
        MasterClockError,
        OSError,
    ) as exc:
        _log.error("%s", exc)
        if logging_silenced:
            print(f"das_processor: error: {exc}", file=sys.stderr)
        return None
    return config, cli_options


def _start_logging(cli_options: CliOptions) -> bool | None:
    """Start logging from the logging settings alone.

    Parameters
    ----------
    cli_options : CliOptions
        The command line.

    Returns
    -------
    bool or None
        Whether logging is silenced (log level None); ``None`` when it
        cannot start, after saying why on standard error.

    Raises
    ------
    SystemExit
        With status 2 and the full help when a required logging setting is
        given by neither source.
    """
    try:
        logging_config = build_logging_config(cli_options)
        log_level_name = logging_config.log_level
        configure_logging(
            root_level=_SILENT
            if log_level_name is None
            else logging.getLevelNamesMapping()[log_level_name],
            log_file=logging_config.log_file,
            backup_count=logging_config.backup_count,
        )
    except MissingSettingsError as exc:
        usage_error(str(exc))
    except (
        MasterClockError,
        OSError,
    ) as exc:
        print(f"das_processor: error: {exc}", file=sys.stderr)
        return None
    return log_level_name is None


def main(argv: Sequence[str] | None = None) -> int:
    """Run das_processor and give its exit status (design 6.1).

    Parameters
    ----------
    argv : Sequence of str or None, optional
        The arguments, without the program name; ``None``, the default, for
        :data:`sys.argv`.

    Returns
    -------
    int
        :data:`SUCCESS` when the run finished; :data:`FAILURE` when it
        stopped on a MasterClockError, already logged where it was raised,
        or a setting that cannot be used, printed on standard error.

    Raises
    ------
    SystemExit
        With status 2 for a usage error, after the full help, and with
        status 0 after ``--help``.

    Notes
    -----
    The run holds its channel's lock throughout, so a second run of the
    same channel is refused, and turns a termination signal into a request
    to stop between epochs. A redo, when one is asked for, deletes its rows
    before the run starts, and the run then computes them again.
    """
    run_settings = _read_settings(argv)
    if run_settings is None:
        return FAILURE
    config, cli_options = run_settings
    try:
        clock_config = read_clock_config(config.processed.clock_config_file)
        lock_name = LOCK_FILE_TEMPLATE.format(rf=config.das.rf)
        with (
            RunLock(config.processed.processed_path, lock_name),
            ShutdownHandler() as shutdown,
        ):
            redo_mjd = cli_options.redo_from_mjd
            if redo_mjd is not None:
                redo_from(
                    [
                        (data_file, file_kind)
                        for data_file, file_kind, _ in data_series(config)
                    ],
                    floor_to_ten_minutes(mjd_to_datetime(redo_mjd)),
                    config.das.rf,
                )
            run_channel(config, clock_config, cli_options.steps, shutdown)
    except MasterClockError:
        return FAILURE
    return SUCCESS
