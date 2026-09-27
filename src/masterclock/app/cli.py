"""The command line every program in the project shares.

A program states its own options, but what an option may be is the same for
all of them: a setting left out defers to the configuration file, one given
the literal ``None`` has no value, on the options that accept it, and a
value that is not what it should be stops the program with the full help
rather than a usage line. A path must be absolute, since no program depends
on the working directory.

Nothing here knows which program is running or what its options are.
"""

import argparse
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Final, Literal, NoReturn

type LogLevelName = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Name of a log level the log understands, including its own TRACE."""

LOG_LEVEL_NAMES: Final[tuple[LogLevelName, ...]] = (
    "TRACE",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)
"""All accepted ``--log-level`` names, most to least verbose."""

NONE_LITERAL: Final[str] = "None"
"""Command-line token that stands for "no value" on options that accept it."""


class Unset(Enum):
    """Marks an option left out of the command line.

    It is not ``None``: ``None`` is what the literal token ``"None"`` gives,
    which sets no value. An option left out, marked :data:`UNSET`, takes
    what the configuration file says.
    """

    UNSET = "unset"


UNSET: Final[Unset] = Unset.UNSET
"""The one value of :class:`Unset`."""


def optional_path(text: str) -> Path | None:
    """Convert a command-line token to an absolute path, treating ``"None"`` as no path.

    A relative path, the empty token included, is refused, since it would
    name a different file from each working directory.

    Parameters
    ----------
    text : str
        The raw command-line token.

    Returns
    -------
    Path or None
        ``None`` if ``text`` is the literal token ``"None"``, otherwise the
        token as a :class:`~pathlib.Path`.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is neither the token ``"None"`` nor an absolute path.
    """
    if text == NONE_LITERAL:
        return None
    path = Path(text)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"path must be absolute: {text!r}")
    return path


def positive_mjd_or_none(text: str) -> float | None:
    """Convert a command-line token to a positive MJD, treating ``"None"`` as no value.

    Parameters
    ----------
    text : str
        The raw command-line token.

    Returns
    -------
    float or None
        ``None`` if ``text`` is the literal token ``"None"``, otherwise the
        token read as :func:`positive_mjd` reads it.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is neither the token ``"None"`` nor a finite number
        above zero.
    """
    return None if text == NONE_LITERAL else positive_mjd(text)


def positive_mjd(text: str) -> float:
    """Convert a command-line token to a positive MJD.

    The token is read as :class:`float` reads it, so surrounding white
    space, a plus sign, underscores between digits and the digits of other
    scripts are accepted. NaN and the infinities are refused, and so is a
    number too large for a float, which reads as infinity. Unlike
    :func:`positive_mjd_or_none`, the token ``"None"`` has no special
    meaning and is refused like any other text that is not a number.

    Parameters
    ----------
    text : str
        The raw command-line token.

    Returns
    -------
    float
        The token as a finite float above zero.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not a finite number above zero.
    """
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid positive float value: {text!r}"
        ) from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"MJD must be finite: {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"MJD must be positive: {text!r}")
    return value


def positive_int_or_none(text: str) -> int | None:
    """Convert a command-line token to a positive int, treating ``"None"`` as no value.

    What no value means is up to the option; for a limit it means there is
    none.

    Parameters
    ----------
    text : str
        The raw command-line token.

    Returns
    -------
    int or None
        ``None`` if ``text`` is the literal token ``"None"``, otherwise the
        token read as :func:`positive_int` reads it.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is neither the token ``"None"`` nor a positive integer.
    """
    return None if text == NONE_LITERAL else positive_int(text)


def positive_int(text: str) -> int:
    """Convert a command-line token to a positive int.

    The token is read as :class:`int` reads it, so surrounding white space,
    a plus sign, underscores between digits and the digits of other scripts
    are accepted. Unlike :func:`positive_int_or_none`, the token ``"None"``
    has no special meaning and is refused like any other text that is not a
    whole number.

    Parameters
    ----------
    text : str
        The raw command-line token.

    Returns
    -------
    int
        The token as a positive int.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not a positive integer.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid positive int value: {text!r}"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"value must be a positive integer: {text!r}")
    return value


class HelpfulArgumentParser(argparse.ArgumentParser):
    """Argument parser that prints the full help, not just a usage line, on error.

    :class:`argparse.ArgumentParser` reports an argument error with a bare
    one-line usage summary. This subclass prints the complete help instead,
    every option with its description, so a mistyped command shows what the
    program accepts. Subcommand parsers made from it with
    :meth:`~argparse.ArgumentParser.add_subparsers` are of this class too,
    and print the help of the subcommand.
    """

    def error(self, message: str) -> NoReturn:
        """Print the full help and the error message, then exit with status 2.

        Overrides :meth:`argparse.ArgumentParser.error`, which prints only the
        usage line. The help and the error both go to standard error, and the
        exit status stays argparse's conventional 2 for a command-line error.

        Parameters
        ----------
        message : str
            The error message argparse composed for the offending argument.

        Returns
        -------
        NoReturn
            This method never returns; it always raises
            :class:`SystemExit`.

        Raises
        ------
        SystemExit
            Always, with status 2.
        """
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")
