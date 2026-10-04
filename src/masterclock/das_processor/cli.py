"""The command line of das_processor.

It is read with :mod:`argparse` by the rules every program shares (see
:mod:`masterclock.app.cli`) and validated into the frozen
:class:`CliOptions`, since the command line is an external boundary.

Each option that has a configuration-file entry of the same name overrides
it, and one left out is :data:`~masterclock.app.cli.UNSET`, deferring to the
file. ``--config-file`` names that file rather than
overriding an entry in it. ``--steps`` and ``--redo-from-mjd`` back no
entry: how many epochs to process and whether to reprocess are ways of
running the program once, not properties of the deployment, so they have
nothing to defer to and the literal ``None`` means nothing to them. A redo
left in a configuration file would reprocess at every scheduled run. Without
``--config-file``, every required setting must be given on the command line.

An MJD option must fall on a day a data file can cover, from
:data:`~masterclock.das_processor.read_cd5m5m.FIRST_DAY` to
:data:`~masterclock.das_processor.read_cd5m5m.LAST_DAY`. The MJD to start
from is needed only when there are no processed files to read a previous
measurement from; if it is needed and neither the command line nor the
configuration file gives it, the run starts from :data:`START_FROM_MJD`.

A command line with no arguments at all, an argument error, and a usage
error found only after parsing (see :func:`usage_error`) all print the full
help and exit with status 2.
"""

import argparse
import sys
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING, Annotated, Final, NoReturn

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from masterclock.app.cli import (
    LOG_LEVEL_NAMES,
    NONE_LITERAL,
    UNSET,
    HelpfulArgumentParser,
    LogLevelName,
    Unset,
    absolute_path,
    optional_path,
    positive_int,
    positive_int_or_none,
    positive_mjd,
)
from masterclock.app.config import AbsolutePath
from masterclock.das_processor.channels import RF_CHOICES, RfChannel
from masterclock.das_processor.read_cd5m5m import FIRST_DAY, LAST_DAY

if TYPE_CHECKING:
    from collections.abc import Sequence

START_FROM_MJD: Final[float] = 59_500.0
"""The MJD a run starts from when it must start somewhere and is not told where.

Used only when there are no processed files to read a previous measurement
from and neither the command line nor the configuration file gives
``start_from_mjd``.
"""

type DataMjd = Annotated[float, Field(ge=FIRST_DAY, lt=LAST_DAY + 1)]
"""An MJD on a day a data file can cover."""


def data_mjd(cli_token: str) -> float:
    """Convert a command-line token to an MJD on a day a data file can cover.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    float
        The token as :func:`~masterclock.app.cli.positive_mjd` reads it.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is not a finite number above zero, or is before day
        :data:`~masterclock.das_processor.read_cd5m5m.FIRST_DAY` or after
        day :data:`~masterclock.das_processor.read_cd5m5m.LAST_DAY`.
    """
    parsed_mjd = positive_mjd(cli_token)
    if not FIRST_DAY <= parsed_mjd < LAST_DAY + 1:
        message = f"MJD must be on a day from {FIRST_DAY} to {LAST_DAY}: {cli_token!r}"
        raise argparse.ArgumentTypeError(message)
    return parsed_mjd


class CliOptions(BaseModel):
    """Validated command-line options.

    Immutable snapshot of everything the user provided on the command line,
    validated by Pydantic.

    Parameters
    ----------
    config_file : Path or None
        Absolute path to the INI configuration file (``--config-file``), or
        ``None`` when it was omitted (in which case every required setting
        must be given on the command line).
    rf : RfChannel or Unset
        The RF channel to process, or :data:`UNSET` when the option was
        omitted (defer to the config file).
    cd5m5m_path : Path or Unset
        Absolute path to the DAS 5 MHz phase measurement data, or :data:`UNSET` when
        the option was omitted (defer to the config file).
    steering_path : Path or Unset
        Absolute path of the directory holding the steering file of each
        reference clock, or :data:`UNSET` when the option was omitted (defer
        to the config file).
    processed_path : Path or Unset
        Absolute path under which processed results are written, each kind of file in
        a subdirectory of its own, or :data:`UNSET` when the option was
        omitted (defer to the config file).
    redo_from_mjd : DataMjd or None, optional
        Reprocess data starting from this MJD before the run
        (``--redo-from-mjd``); ``None`` (the default) reprocesses nothing.
        Command line only, like ``steps``.
    start_from_mjd : DataMjd or Unset
        MJD to start processing from when there are no processed files to
        read a previous measurement from, or :data:`UNSET` when the option
        was omitted (defer to the config file, and failing that
        :data:`START_FROM_MJD`).
    clock_config_file : Path or Unset
        Absolute path to the YAML clock configuration (each clock's
        estimator parameters and location, the pairs' RMS limits, and the
        clocks to ignore), or :data:`UNSET` when the option was omitted
        (defer to the config file).
    log_file : Path, None, or Unset
        Absolute path to the log file, ``None`` when file logging is explicitly
        disabled, or :data:`UNSET` when the option was omitted (defer to the
        config file).
    log_level : LogLevelName, None, or Unset
        Name of the logging level, ``None`` when logging is explicitly
        disabled, or :data:`UNSET` when the option was omitted (defer to the
        config file).
    num_workers : PositiveInt, None, or Unset
        Number of worker processes to work each epoch's series, ``None``
        when the main process works them alone, or :data:`UNSET` when the
        option was omitted (defer to the config file).
    backup_count : PositiveInt, None, or Unset
        Number of rotated daily log files to keep, ``None`` when every rotated
        file is kept, or :data:`UNSET` when the option was omitted (defer to
        the config file).
    steps : PositiveInt or None, optional
        How many ten-minute epochs that write rows the run processes
        before it shuts down (``--steps``); an epoch of a data gap that
        writes no row is not counted. ``None`` (the default) processes every
        new epoch.
        No config entry backs it, so it has no :data:`UNSET` state: there is
        nothing to defer to, and no ``"None"`` token to accept.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_file: AbsolutePath | None = None
    rf: RfChannel | Unset = UNSET
    cd5m5m_path: AbsolutePath | Unset = UNSET
    steering_path: AbsolutePath | Unset = UNSET
    processed_path: AbsolutePath | Unset = UNSET
    redo_from_mjd: DataMjd | None = None
    start_from_mjd: DataMjd | Unset = UNSET
    clock_config_file: AbsolutePath | Unset = UNSET
    log_file: AbsolutePath | Unset | None = UNSET
    log_level: LogLevelName | Unset | None = UNSET
    num_workers: PositiveInt | Unset | None = UNSET
    backup_count: PositiveInt | Unset | None = UNSET
    steps: PositiveInt | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the das_processor command line.

    Returns
    -------
    argparse.ArgumentParser
        A :class:`HelpfulArgumentParser`, which prints the full help rather
        than a usage line on an argument error, carrying every option this
        module describes along with ``--help`` and ``--version``.
    """
    parser = HelpfulArgumentParser(
        prog="das_processor",
        description=(
            "Process 5 MHz phase measurements from the Data Acquisition System (DAS)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version('masterclock')}",
    )
    parser.add_argument(
        "--config-file",
        type=absolute_path,
        default=None,
        metavar="PATH",
        help=(
            "path to the INI configuration file (optional when every required "
            "setting is given on the command line)"
        ),
    )
    parser.add_argument(
        "--rf",
        choices=RF_CHOICES,
        default=UNSET,
        help=(
            "RF channel to process, overriding the config file's [DAS] rf "
            "(default: use the config file)"
        ),
    )
    parser.add_argument(
        "--cd5m5m-path",
        type=absolute_path,
        default=UNSET,
        metavar="PATH",
        help=(
            "path to the DAS 5 MHz phase data, overriding the config file's "
            "[DAS] cd5m5m_path (default: use the config file)"
        ),
    )
    parser.add_argument(
        "--steering-path",
        type=absolute_path,
        default=UNSET,
        metavar="PATH",
        help=(
            "path to the directory of steering files, one per reference clock, "
            "overriding the config file's [DAS] steering_path (default: use "
            "the config file)"
        ),
    )
    parser.add_argument(
        "--processed-path",
        type=absolute_path,
        default=UNSET,
        metavar="PATH",
        help=(
            "path for processed results, each kind of file in a subdirectory "
            "of its own, overriding the config file's [PROCESSED] "
            "processed_path (default: use the config file)"
        ),
    )
    parser.add_argument(
        "--redo-from-mjd",
        type=data_mjd,
        default=None,
        metavar="MJD",
        help=(
            "reprocess data starting from this MJD, before the run; command "
            "line only, with no config-file entry (default: no reprocessing)"
        ),
    )
    parser.add_argument(
        "--start-from-mjd",
        type=data_mjd,
        default=UNSET,
        metavar="MJD",
        help=(
            "MJD to start processing from when there are no processed files "
            "to read a previous measurement from (floored to its ten-minute "
            "mark), overriding the config file's [PROCESSED] start_from_mjd "
            f"(default: use the config file, else MJD {START_FROM_MJD:g})"
        ),
    )
    parser.add_argument(
        "--clock-config-file",
        type=absolute_path,
        default=UNSET,
        metavar="PATH",
        help=(
            "path to the YAML clock configuration (each clock's estimator "
            "parameters and location, the pairs' RMS limits, and the clocks to "
            "ignore), overriding the config file's [PROCESSED] "
            "clock_config_file (default: use the config file)"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=positive_int_or_none,
        default=UNSET,
        metavar="N",
        help=(
            "number of worker processes to work each epoch's series, "
            "overriding the config file's [PROCESSED] num_workers; pass "
            f"{NONE_LITERAL} to work them in the main process alone "
            "(default: use the config file)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        metavar="N",
        help=(
            "process exactly this many ten-minute epochs that write rows and "
            "shut down, instead of every epoch not yet processed; an epoch of a "
            "data gap that writes no row is not counted; command line only, "
            "with no config-file entry (default: process every new epoch)"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=optional_path,
        default=UNSET,
        metavar="PATH",
        help=(
            "path to the log file, overriding the config file's [LOGGING] "
            f"log_file; pass {NONE_LITERAL} to disable file logging "
            "(default: use the config file)"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=(*LOG_LEVEL_NAMES, NONE_LITERAL),
        default=UNSET,
        help=(
            "logging level, overriding the config file's [LOGGING] log_level; "
            f"pass {NONE_LITERAL} to disable logging entirely "
            "(default: use the config file)"
        ),
    )
    parser.add_argument(
        "--backup-count",
        type=positive_int_or_none,
        default=UNSET,
        metavar="N",
        help=(
            "number of rotated daily log files to keep, "
            "overriding the config file's [LOGGING] backup_count; pass "
            f"{NONE_LITERAL} to keep every rotated file (default: use the "
            "config file)"
        ),
    )
    return parser


def usage_error(message: str) -> NoReturn:
    """Report a command-line usage error the argparse way: full help, then exit 2.

    Prints the complete help - the argparse-style list of every option (see
    :class:`HelpfulArgumentParser`) - followed by ``message`` to standard
    error, then exits with status 2, exactly as a malformed argument does.
    Used for usage errors detected only after parsing, above all a required
    setting that neither the config file nor the command line supplied.

    Parameters
    ----------
    message : str
        The error message, rendered after the help as
        ``das_processor: error: <message>``.

    Returns
    -------
    NoReturn
        This function never returns; it always raises :class:`SystemExit`.

    Raises
    ------
    SystemExit
        Always, with status 2.
    """
    build_parser().error(message)


def parse_args(argv: Sequence[str] | None = None) -> CliOptions:
    """Parse and validate the command line.

    A bare invocation with no arguments at all cannot supply any setting, so
    it is treated as a request for help: the full help is printed (like an
    argument error, see :class:`HelpfulArgumentParser`) and the program
    exits with status 2, rather than parsing an empty command line and
    failing later on the missing settings.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        The arguments to parse, without the program name. ``None`` (the
        default) parses :data:`sys.argv`.

    Returns
    -------
    CliOptions
        The validated options.

    Raises
    ------
    SystemExit
        With status 2 if the arguments are invalid (raised by argparse) or no
        arguments are given at all (after printing the full help), or with
        status 0 after ``--help`` or ``--version`` output.
    """
    parser = build_parser()
    if not (sys.argv[1:] if argv is None else argv):
        parser.print_help(sys.stderr)
        parser.exit(2)
    namespace = parser.parse_args(argv)
    return CliOptions(
        config_file=namespace.config_file,
        rf=namespace.rf,
        cd5m5m_path=namespace.cd5m5m_path,
        steering_path=namespace.steering_path,
        processed_path=namespace.processed_path,
        redo_from_mjd=namespace.redo_from_mjd,
        start_from_mjd=namespace.start_from_mjd,
        clock_config_file=namespace.clock_config_file,
        log_file=namespace.log_file,
        log_level=None if namespace.log_level == NONE_LITERAL else namespace.log_level,
        num_workers=namespace.num_workers,
        backup_count=namespace.backup_count,
        steps=namespace.steps,
    )
