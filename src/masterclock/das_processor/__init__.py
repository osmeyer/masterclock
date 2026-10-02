"""das_processor: the masterclock program that conditions the DAS phase data.

A scheduler starts it for each RF channel every ten minutes. Each run
processes its channel's epochs, in order, up to the end of the DAS data, or
``--steps`` of them, and exits: it reads the raw ten-minute comparisons,
decycles and screens them, runs every pair's and triple's estimator, and
appends one row per epoch to each series' file.

:func:`main` is the entry point. Its exit status is 0 for a run that
finished, 2 for a usage error such as a required setting given by neither
the INI file nor the command line, and 1 for any other failure, which is
logged where it happened (or, before logging starts, printed on standard
error).
"""

import logging
import sys
from collections.abc import Sequence
from typing import Final

from masterclock.app.exceptions import MasterClockError, MissingSettingsError
from masterclock.app.lock import RunLock
from masterclock.app.log import configure_logging
from masterclock.app.shutdown import ShutdownHandler
from masterclock.app.timeutil import mjd_to_datetime
from masterclock.das_processor.cli import parse_args, usage_error
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.config import (
    LOCK_FILE_TEMPLATE,
    AppConfig,
    build_config,
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

_SILENT: Final[int] = logging.CRITICAL + 1
"""A logging level above every record's, for a run with logging set to None."""


def _settings(argv: Sequence[str] | None) -> tuple[AppConfig, int | None] | None:
    """Read and check the run's settings, before logging starts.

    Parameters
    ----------
    argv : Sequence of str or None
        The arguments, without the program name; ``None`` for
        :data:`sys.argv`.

    Returns
    -------
    tuple of (AppConfig, int or None) or None
        The settings and ``--steps``; ``None`` when a setting cannot be
        used, after saying why on standard error.

    Raises
    ------
    SystemExit
        With status 2 and the full help for a usage error, above all a
        required setting given by neither source.
    """
    options = parse_args(argv)
    try:
        config = build_config(options)
        check_paths(config)
        level = config.logging.log_level
        configure_logging(
            level=_SILENT if level is None else logging.getLevelNamesMapping()[level],
            log_file=config.logging.log_file,
            backup_count=config.logging.backup_count,
        )
    except MissingSettingsError as exc:
        usage_error(str(exc))
    except (
        MasterClockError,
        OSError,
    ) as exc:
        print(f"das_processor: error: {exc}", file=sys.stderr)
        return None
    return config, options.steps


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
    settings = _settings(argv)
    if settings is None:
        return FAILURE
    config, steps = settings
    try:
        clock_config = read_clock_config(config.processed.clock_config_file)
        lock_name = LOCK_FILE_TEMPLATE.format(rf=config.das.rf)
        with (
            RunLock(config.processed.processed_path, lock_name),
            ShutdownHandler() as shutdown,
        ):
            redo_mjd = config.processed.redo_from_mjd
            if redo_mjd is not None:
                redo_from(
                    [(path, kind) for path, kind, _ in data_series(config)],
                    floor_to_ten_minutes(mjd_to_datetime(redo_mjd)),
                )
            run_channel(config, clock_config, steps, shutdown)
    except MasterClockError:
        return FAILURE
    return SUCCESS
