"""The command line every program in the project shares.

A program states its own options, but what an option may be is the same for
all of them: a setting left out defers to the configuration file, one given
the literal ``None`` has no value, on the options that accept it, and a
value that is not what it should be stops the program with the full help
rather than a usage line. A path must be absolute, since no program depends
on the working directory. An option is written out in full and given at most
once, so what a command says is never decided by guessing which option was
meant or which of two values wins.

Nothing here depends on which program is running or what its options are.
"""

import argparse
import math
import sys
import weakref
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence

type LogLevelName = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Name of a log level the log accepts, including the project's own TRACE."""

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


def absolute_path(cli_token: str) -> Path:
    """Convert a command-line token to an absolute path.

    A relative path, the empty token included, is refused, since it would
    name a different file from each working directory. The token ``"None"``
    has no special meaning: it is a relative path and is refused.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    Path
        The token as a :class:`~pathlib.Path`.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is not an absolute path.
    """
    token_path = Path(cli_token)
    if not token_path.is_absolute():
        raise argparse.ArgumentTypeError(f"path must be absolute: {cli_token!r}")
    return token_path


def optional_path(cli_token: str) -> Path | None:
    """Convert a command-line token to an absolute path, treating ``"None"`` as no path.

    A relative path, the empty token included, is refused, since it would
    name a different file from each working directory.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    Path or None
        ``None`` if ``cli_token`` is the literal token ``"None"``, otherwise the
        token as a :class:`~pathlib.Path`.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is neither the token ``"None"`` nor an absolute path.
    """
    return None if cli_token == NONE_LITERAL else absolute_path(cli_token)


def positive_mjd_or_none(cli_token: str) -> float | None:
    """Convert a command-line token to a positive MJD, treating ``"None"`` as no value.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    float or None
        ``None`` if ``cli_token`` is the literal token ``"None"``, otherwise the
        token read as :func:`positive_mjd` reads it.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is neither the token ``"None"`` nor a finite number
        above zero.
    """
    return None if cli_token == NONE_LITERAL else positive_mjd(cli_token)


def positive_mjd(cli_token: str) -> float:
    """Convert a command-line token to a positive MJD.

    The token is read as :class:`float` reads it, so surrounding white
    space, a plus sign, underscores between digits and the digits of other
    scripts are accepted. NaN and the infinities are refused, and so is a
    number too large for a float, which reads as infinity. Unlike
    :func:`positive_mjd_or_none`, the token ``"None"`` has no special
    meaning and is refused like any other text that is not a number.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    float
        The token as a finite float above zero.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is not a finite number above zero.
    """
    try:
        parsed_mjd = float(cli_token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid positive float value: {cli_token!r}"
        ) from exc
    if not math.isfinite(parsed_mjd):
        raise argparse.ArgumentTypeError(f"MJD must be finite: {cli_token!r}")
    if parsed_mjd <= 0:
        raise argparse.ArgumentTypeError(f"MJD must be positive: {cli_token!r}")
    return parsed_mjd


def positive_int_or_none(cli_token: str) -> int | None:
    """Convert a command-line token to a positive int, treating ``"None"`` as no value.

    What no value means is up to the option; for a limit it means there is
    none.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    int or None
        ``None`` if ``cli_token`` is the literal token ``"None"``, otherwise the
        token read as :func:`positive_int` reads it.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is neither the token ``"None"`` nor a positive integer.
    """
    return None if cli_token == NONE_LITERAL else positive_int(cli_token)


def positive_int(cli_token: str) -> int:
    """Convert a command-line token to a positive int.

    The token is read as :class:`int` reads it, so surrounding white space,
    a plus sign, underscores between digits and the digits of other scripts
    are accepted. Unlike :func:`positive_int_or_none`, the token ``"None"``
    has no special meaning and is refused like any other text that is not a
    whole number.

    Parameters
    ----------
    cli_token : str
        The raw command-line token.

    Returns
    -------
    int
        The token as a positive int.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``cli_token`` is not a positive integer.
    """
    try:
        parsed_int = int(cli_token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid positive int value: {cli_token!r}"
        ) from exc
    if parsed_int <= 0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive integer: {cli_token!r}"
        )
    return parsed_int


class StoreOnce(argparse.Action):
    """Store an option's value, refusing the option if it was already given.

    The action every option of a :class:`HelpfulArgumentParser` takes unless
    it names another.
    """

    # Any: argparse.Action's own parameters, passed on unchanged.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Make the action, remembering no parse yet.

        Parameters
        ----------
        *args : Any
            Passed on to :class:`argparse.Action`.
        **kwargs : Any
            Passed on to :class:`argparse.Action`.
        """
        super().__init__(*args, **kwargs)
        self._given: weakref.WeakValueDictionary[int, argparse.Namespace] = (
            weakref.WeakValueDictionary()
        )

    def __call__(
        self,
        _parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        _option_string: str | None = None,
    ) -> None:
        """Store ``values``, unless this option was already given in this parse.

        Parameters
        ----------
        _parser : argparse.ArgumentParser
            The parser doing the parsing, unused.
        namespace : argparse.Namespace
            Where the parse puts what it reads. Each parse has its own, so it
            is what tells one parse from the next.
        values : str or Sequence[Any] or None
            The option's value, already converted. Any: argparse types it so,
            since a conversion may return anything.
        _option_string : str or None, optional
            The option as written, unused.

        Raises
        ------
        argparse.ArgumentError
            If the option was already given in this parse; the parser
            reports it as it reports any other argument error.
        """
        if self._given.get(id(namespace)) is namespace:
            raise argparse.ArgumentError(self, "given more than once")
        self._given[id(namespace)] = namespace
        setattr(namespace, self.dest, values)


class HelpfulArgumentParser(argparse.ArgumentParser):
    """Argument parser that prints the full help, not just a usage line, on error.

    :class:`argparse.ArgumentParser` reports an argument error with a bare
    one-line usage summary. This subclass prints the complete help instead,
    every option with its description, so a mistyped command shows what the
    program accepts. Subcommand parsers made from it with
    :meth:`~argparse.ArgumentParser.add_subparsers` are of this class too,
    and print the help of the subcommand.

    It also accepts an option only under its full name, and only once: an
    option that stores a value takes :class:`StoreOnce` unless it names
    another action.
    """

    # Any: argparse.ArgumentParser's own parameters, passed on unchanged.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Make the parser, refusing abbreviations and repeated options.

        Parameters
        ----------
        *args : Any
            Passed on to :class:`argparse.ArgumentParser`.
        **kwargs : Any
            Passed on to :class:`argparse.ArgumentParser`. Asking for
            abbreviations to be allowed has no effect.
        """
        super().__init__(*args, **kwargs)
        self.allow_abbrev = False
        self.register("action", None, StoreOnce)
        self.register("action", "store", StoreOnce)

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
