"""Tests for src/masterclock/app/cli.py.

The rules covered: the literal None, spelt exactly so, stands for no value on
the options that accept it and is refused like any other text on those that
do not; an MJD is a finite number above zero; a count is a whole number
above zero; a path is absolute; the level names match the levels the log
knows, most to least verbose; an option left out is told apart from None;
an option is accepted only under its full name and only once in a command;
and a command-line error prints the full help and the error on standard
error and exits with status 2.

An option that names the store action is given once too, and the parser
takes positional arguments as argparse does.
"""

import argparse
import logging
from pathlib import Path
from typing import get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import cli, log


def refusal(convert: object, text: str) -> str:
    """Return the message ``convert`` refuses ``text`` with."""
    assert callable(convert)
    with pytest.raises(argparse.ArgumentTypeError) as raised:
        convert(text)
    return str(raised.value)


def test_the_literal_is_none_spelt_exactly_so() -> None:
    """Spell the no-value token None, with that capitalisation."""
    assert cli.NONE_LITERAL == "None"


@given(st.floats(min_value=0, exclude_min=True, allow_infinity=False))
def test_a_positive_mjd_is_read_as_its_float(value: float) -> None:
    """Read any finite number above zero as that number."""
    assert cli.positive_mjd(repr(value)) == value
    assert cli.positive_mjd_or_none(repr(value)) == value


@pytest.mark.parametrize("text", ["0", "-0", "0.0", "-1", "-60000.5", "1e-400"])
def test_an_mjd_of_zero_or_below_is_refused(text: str) -> None:
    """Refuse zero, a negative MJD, and a number too small to be told from zero."""
    assert refusal(cli.positive_mjd, text) == f"MJD must be positive: {text!r}"
    assert refusal(cli.positive_mjd_or_none, text) == f"MJD must be positive: {text!r}"


@pytest.mark.parametrize(
    "text", ["nan", "NaN", "-nan", "inf", "Infinity", "-inf", "1e400", "-1e400"]
)
def test_an_mjd_that_is_not_finite_is_refused(text: str) -> None:
    """Refuse NaN and the infinities, including a number too large for a float."""
    assert refusal(cli.positive_mjd, text) == f"MJD must be finite: {text!r}"
    assert refusal(cli.positive_mjd_or_none, text) == f"MJD must be finite: {text!r}"


@pytest.mark.parametrize("text", ["", "x", "0x10", "1,5", "none", "NONE", " None"])
def test_an_mjd_that_is_not_a_number_is_refused(text: str) -> None:
    """Refuse text that is not a number, including None spelt any other way."""
    message = f"invalid positive float value: {text!r}"
    assert refusal(cli.positive_mjd, text) == message
    assert refusal(cli.positive_mjd_or_none, text) == message


def test_none_is_no_mjd_only_where_it_is_accepted() -> None:
    """Read None as no value in positive_mjd_or_none, and refuse it in positive_mjd."""
    assert cli.positive_mjd_or_none("None") is None
    assert refusal(cli.positive_mjd, "None") == "invalid positive float value: 'None'"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (" 5 ", 5.0),
        ("5\n", 5.0),
        ("+5", 5.0),
        ("1_000", 1000.0),
        ("\N{FULLWIDTH DIGIT FIVE}", 5.0),
    ],
)
def test_an_mjd_is_read_in_every_form_float_reads(text: str, value: float) -> None:
    """Read spaces, a plus sign, underscores and other digit sets as float does."""
    assert cli.positive_mjd(text) == value
    assert cli.positive_mjd_or_none(text) == value


@given(st.integers(min_value=1))
def test_a_positive_count_is_read_as_its_int(value: int) -> None:
    """Read any whole number above zero as that number."""
    assert cli.positive_int(str(value)) == value
    assert cli.positive_int_or_none(str(value)) == value


@pytest.mark.parametrize("text", ["0", "-0", "-1", "-99999999999999999999"])
def test_a_count_of_zero_or_below_is_refused(text: str) -> None:
    """Refuse zero and every negative count."""
    message = f"value must be a positive integer: {text!r}"
    assert refusal(cli.positive_int, text) == message
    assert refusal(cli.positive_int_or_none, text) == message


@pytest.mark.parametrize(
    "text", ["", "x", "1.0", "1e3", "0x10", "inf", "none", "NONE", " None"]
)
def test_a_count_that_is_not_a_whole_number_is_refused(text: str) -> None:
    """Refuse text that is not a whole number, including None spelt otherwise."""
    message = f"invalid positive int value: {text!r}"
    assert refusal(cli.positive_int, text) == message
    assert refusal(cli.positive_int_or_none, text) == message


def test_none_is_no_count_only_where_it_is_accepted() -> None:
    """Read None as no value in positive_int_or_none, and refuse it in positive_int."""
    assert cli.positive_int_or_none("None") is None
    assert refusal(cli.positive_int, "None") == "invalid positive int value: 'None'"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (" 5 ", 5),
        ("5\n", 5),
        ("+5", 5),
        ("1_000", 1000),
        ("\N{FULLWIDTH DIGIT FIVE}", 5),
    ],
)
def test_a_count_is_read_in_every_form_int_reads(text: str, value: int) -> None:
    """Read spaces, a plus sign, underscores and other digit sets as int does."""
    assert cli.positive_int(text) == value
    assert cli.positive_int_or_none(text) == value


@pytest.mark.parametrize("text", ["/", "/data/run", "/data//run/", "/a/../b", "//x"])
def test_an_absolute_path_is_read_as_its_path(text: str) -> None:
    """Read an absolute path as that path."""
    assert cli.optional_path(text) == Path(text)


@pytest.mark.parametrize(
    "text", ["", ".", "..", "run", "data/run", "./run", "~/run", "none", " None"]
)
def test_a_path_that_is_not_absolute_is_refused(text: str) -> None:
    """Refuse a path that would depend on the working directory, empty included."""
    assert refusal(cli.optional_path, text) == f"path must be absolute: {text!r}"


@pytest.mark.parametrize("text", ["/", "/data/run.log", "/None"])
def test_an_absolute_path_is_read_as_its_path_where_none_is_not_accepted(
    text: str,
) -> None:
    """Read an absolute path as itself."""
    assert cli.absolute_path(text) == Path(text)


@pytest.mark.parametrize("text", ["", ".", "None", "run.log", "../x", "~/x"])
def test_a_relative_path_or_none_is_refused_where_none_is_not_accepted(
    text: str,
) -> None:
    """Refuse a relative path, the literal None included."""
    assert refusal(cli.absolute_path, text) == f"path must be absolute: {text!r}"


def test_none_is_no_path() -> None:
    """Read None as no path."""
    assert cli.optional_path("None") is None


def test_the_level_names_are_the_names_the_type_allows() -> None:
    """List the same names, in the same order, as the LogLevelName type."""
    assert get_args(cli.LogLevelName.__value__) == cli.LOG_LEVEL_NAMES


def test_the_level_names_are_the_log_levels_most_verbose_first() -> None:
    """Name only levels the log knows, TRACE included, most verbose first."""
    levels = [logging.getLevelName(name) for name in cli.LOG_LEVEL_NAMES]
    assert levels[0] == log.TRACE
    assert all(isinstance(level, int) for level in levels)
    assert levels == sorted(set(levels))
    assert levels[-1] == logging.CRITICAL


def test_left_out_is_told_apart_from_none() -> None:
    """Keep one marker for a left-out option, distinct from None and truthy."""
    assert list(cli.Unset) == [cli.UNSET]
    assert cli.UNSET is not None
    assert cli.UNSET


@pytest.fixture
def uncoloured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn off colour in argparse's output, whatever the environment asks for."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


def parser() -> cli.HelpfulArgumentParser:
    """Return a parser with one option of each kind, as a program would state them."""
    made = cli.HelpfulArgumentParser(prog="prog", description="An invented program.")
    made.add_argument("--start", type=cli.positive_mjd_or_none, default=cli.UNSET)
    made.add_argument("--count", type=cli.positive_int, default=cli.UNSET)
    made.add_argument("--log-file", type=cli.optional_path, default=cli.UNSET)
    return made


def test_options_are_read_as_given_none_or_left_out() -> None:
    """Tell a value, None and a left-out option apart after parsing."""
    given_ = parser().parse_args(["--start", "60000.5", "--log-file", "None"])
    assert given_.start == 60000.5
    assert given_.log_file is None
    assert given_.count is cli.UNSET


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["--start", "nan"], "argument --start: MJD must be finite: 'nan'"),
        (["--count", "None"], "argument --count: invalid positive int value: 'None'"),
        (
            ["--log-file", "run.log"],
            "argument --log-file: path must be absolute: 'run.log'",
        ),
        (["--colour"], "unrecognized arguments: --colour"),
        (["--sta", "60000.5"], "unrecognized arguments: --sta 60000.5"),
        (
            ["--count", "2", "--count", "2"],
            "argument --count: given more than once",
        ),
        (
            ["--log-file", "None", "--log-file", "None"],
            "argument --log-file: given more than once",
        ),
    ],
)
@pytest.mark.usefixtures("uncoloured")
def test_an_error_prints_the_full_help_and_exits_with_status_2(
    arguments: list[str],
    error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the full help, then the error, on standard error, and exit with 2."""
    made = parser()
    with pytest.raises(SystemExit) as raised:
        made.parse_args(arguments)
    assert raised.value.code == 2
    written = capsys.readouterr()
    assert written.out == ""
    assert "An invented program." in written.err
    assert written.err == f"{made.format_help()}\nprog: error: {error}\n"


@pytest.mark.usefixtures("uncoloured")
def test_a_subcommand_parser_prints_its_own_full_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Give a subcommand the same behaviour, with the help of the subcommand."""
    made = cli.HelpfulArgumentParser(prog="prog")
    commands = made.add_subparsers(dest="command")
    run = commands.add_parser("run", description="Run the invented program.")
    run.add_argument("--count", type=cli.positive_int)
    assert isinstance(run, cli.HelpfulArgumentParser)
    with pytest.raises(SystemExit) as raised:
        made.parse_args(["run", "--count", "0"])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert err == (
        f"{run.format_help()}\nprog run: error: argument --count:"
        " value must be a positive integer: '0'\n"
    )


def test_abbreviations_stay_refused_when_asked_for() -> None:
    """Refuse an abbreviation even when the parser is made to allow them."""
    made = cli.HelpfulArgumentParser(prog="prog", allow_abbrev=True)
    made.add_argument("--count", type=cli.positive_int)
    with pytest.raises(SystemExit):
        made.parse_args(["--cou", "2"])


def test_each_parse_may_give_an_option_once() -> None:
    """Let a second parse with the same parser give the option again."""
    made = parser()
    # The first result is kept, so a parser that remembered the option for
    # as long as that result lives would refuse the second parse.
    first = made.parse_args(["--count", "2"])
    second = made.parse_args(["--count", "3"])
    assert (first.count, second.count) == (2, 3)


def test_an_option_naming_its_own_action_may_be_repeated() -> None:
    """Leave an option that names another action, such as append, to it."""
    made = cli.HelpfulArgumentParser(prog="prog")
    made.add_argument("--tag", action="append")
    made.add_argument("--quiet", action="store_true")
    parsed = made.parse_args(["--tag", "a", "--tag", "b", "--quiet"])
    assert parsed.tag == ["a", "b"]
    assert parsed.quiet is True


def test_a_subcommand_option_is_also_given_once() -> None:
    """Refuse a repeated option in a subcommand too."""
    made = cli.HelpfulArgumentParser(prog="prog")
    run = made.add_subparsers(dest="command").add_parser("run")
    run.add_argument("--count", type=cli.positive_int)
    with pytest.raises(SystemExit):
        made.parse_args(["run", "--count", "2", "--count", "2"])


def test_an_option_that_names_store_is_given_once_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse an option given twice when it names the store action itself."""
    made = cli.HelpfulArgumentParser(prog="prog")
    made.add_argument("--label", action="store")
    with pytest.raises(SystemExit):
        made.parse_args(["--label", "a", "--label", "b"])
    assert "argument --label: given more than once" in capsys.readouterr().err


def test_the_parser_takes_its_arguments_as_argparse_does() -> None:
    """Pass positional arguments on, so the program's name can come first."""
    assert cli.HelpfulArgumentParser("prog").prog == "prog"
