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


def refusal_message(cli_conversion: object, cli_token: str) -> str:
    """Return the message ``cli_conversion`` refuses ``cli_token`` with."""
    assert callable(cli_conversion)
    with pytest.raises(argparse.ArgumentTypeError) as raised:
        cli_conversion(cli_token)
    return str(raised.value)


def test_the_literal_is_none_spelt_exactly_so() -> None:
    """Spell the no-value token None, with that capitalisation."""
    assert cli.NONE_LITERAL == "None"


@given(st.floats(min_value=0, exclude_min=True, allow_infinity=False))
def test_a_positive_mjd_is_read_as_its_float(mjd_number: float) -> None:
    """Read any finite number above zero as that number."""
    assert cli.positive_mjd(repr(mjd_number)) == mjd_number
    assert cli.positive_mjd_or_none(repr(mjd_number)) == mjd_number


@pytest.mark.parametrize("cli_token", ["0", "-0", "0.0", "-1", "-60000.5", "1e-400"])
def test_an_mjd_of_zero_or_below_is_refused(cli_token: str) -> None:
    """Refuse zero, a negative MJD, and a number too small to be told from zero."""
    assert (
        refusal_message(cli.positive_mjd, cli_token)
        == f"MJD must be positive: {cli_token!r}"
    )
    assert (
        refusal_message(cli.positive_mjd_or_none, cli_token)
        == f"MJD must be positive: {cli_token!r}"
    )


@pytest.mark.parametrize(
    "cli_token", ["nan", "NaN", "-nan", "inf", "Infinity", "-inf", "1e400", "-1e400"]
)
def test_an_mjd_that_is_not_finite_is_refused(cli_token: str) -> None:
    """Refuse NaN and the infinities, including a number too large for a float."""
    assert (
        refusal_message(cli.positive_mjd, cli_token)
        == f"MJD must be finite: {cli_token!r}"
    )
    assert (
        refusal_message(cli.positive_mjd_or_none, cli_token)
        == f"MJD must be finite: {cli_token!r}"
    )


@pytest.mark.parametrize("cli_token", ["", "x", "0x10", "1,5", "none", "NONE", " None"])
def test_an_mjd_that_is_not_a_number_is_refused(cli_token: str) -> None:
    """Refuse text that is not a number, including None spelt any other way."""
    expected_refusal = f"invalid positive float value: {cli_token!r}"
    assert refusal_message(cli.positive_mjd, cli_token) == expected_refusal
    assert refusal_message(cli.positive_mjd_or_none, cli_token) == expected_refusal


def test_none_is_no_mjd_only_where_it_is_accepted() -> None:
    """Read None as no value in positive_mjd_or_none, and refuse it in positive_mjd."""
    assert cli.positive_mjd_or_none("None") is None
    assert (
        refusal_message(cli.positive_mjd, "None")
        == "invalid positive float value: 'None'"
    )


@pytest.mark.parametrize(
    ("cli_token", "parsed_mjd"),
    [
        (" 5 ", 5.0),
        ("5\n", 5.0),
        ("+5", 5.0),
        ("1_000", 1000.0),
        ("\N{FULLWIDTH DIGIT FIVE}", 5.0),
    ],
)
def test_an_mjd_is_read_in_every_form_float_reads(
    cli_token: str, parsed_mjd: float
) -> None:
    """Read spaces, a plus sign, underscores and other digit sets as float does."""
    assert cli.positive_mjd(cli_token) == parsed_mjd
    assert cli.positive_mjd_or_none(cli_token) == parsed_mjd


@given(st.integers(min_value=1))
def test_a_positive_count_is_read_as_its_int(count_number: int) -> None:
    """Read any whole number above zero as that number."""
    assert cli.positive_int(str(count_number)) == count_number
    assert cli.positive_int_or_none(str(count_number)) == count_number


@pytest.mark.parametrize("cli_token", ["0", "-0", "-1", "-99999999999999999999"])
def test_a_count_of_zero_or_below_is_refused(cli_token: str) -> None:
    """Refuse zero and every negative count."""
    expected_refusal = f"value must be a positive integer: {cli_token!r}"
    assert refusal_message(cli.positive_int, cli_token) == expected_refusal
    assert refusal_message(cli.positive_int_or_none, cli_token) == expected_refusal


@pytest.mark.parametrize(
    "cli_token", ["", "x", "1.0", "1e3", "0x10", "inf", "none", "NONE", " None"]
)
def test_a_count_that_is_not_a_whole_number_is_refused(cli_token: str) -> None:
    """Refuse text that is not a whole number, including None spelt otherwise."""
    expected_refusal = f"invalid positive int value: {cli_token!r}"
    assert refusal_message(cli.positive_int, cli_token) == expected_refusal
    assert refusal_message(cli.positive_int_or_none, cli_token) == expected_refusal


def test_none_is_no_count_only_where_it_is_accepted() -> None:
    """Read None as no value in positive_int_or_none, and refuse it in positive_int."""
    assert cli.positive_int_or_none("None") is None
    assert (
        refusal_message(cli.positive_int, "None")
        == "invalid positive int value: 'None'"
    )


@pytest.mark.parametrize(
    ("cli_token", "parsed_count"),
    [
        (" 5 ", 5),
        ("5\n", 5),
        ("+5", 5),
        ("1_000", 1000),
        ("\N{FULLWIDTH DIGIT FIVE}", 5),
    ],
)
def test_a_count_is_read_in_every_form_int_reads(
    cli_token: str, parsed_count: int
) -> None:
    """Read spaces, a plus sign, underscores and other digit sets as int does."""
    assert cli.positive_int(cli_token) == parsed_count
    assert cli.positive_int_or_none(cli_token) == parsed_count


@pytest.mark.parametrize(
    "cli_token", ["/", "/data/run", "/data//run/", "/a/../b", "//x"]
)
def test_an_absolute_path_is_read_as_its_path(cli_token: str) -> None:
    """Read an absolute path as that path."""
    assert cli.optional_path(cli_token) == Path(cli_token)


@pytest.mark.parametrize(
    "cli_token", ["", ".", "..", "run", "data/run", "./run", "~/run", "none", " None"]
)
def test_a_path_that_is_not_absolute_is_refused(cli_token: str) -> None:
    """Refuse a path that would depend on the working directory, empty included."""
    assert (
        refusal_message(cli.optional_path, cli_token)
        == f"path must be absolute: {cli_token!r}"
    )


@pytest.mark.parametrize("cli_token", ["/", "/data/run.log", "/None"])
def test_an_absolute_path_is_read_as_its_path_where_none_is_not_accepted(
    cli_token: str,
) -> None:
    """Read an absolute path as itself."""
    assert cli.absolute_path(cli_token) == Path(cli_token)


@pytest.mark.parametrize("cli_token", ["", ".", "None", "run.log", "../x", "~/x"])
def test_a_relative_path_or_none_is_refused_where_none_is_not_accepted(
    cli_token: str,
) -> None:
    """Refuse a relative path, the literal None included."""
    assert (
        refusal_message(cli.absolute_path, cli_token)
        == f"path must be absolute: {cli_token!r}"
    )


def test_none_is_no_path() -> None:
    """Read None as no path."""
    assert cli.optional_path("None") is None


def test_the_level_names_are_the_names_the_type_allows() -> None:
    """List the same names, in the same order, as the LogLevelName type."""
    assert get_args(cli.LogLevelName.__value__) == cli.LOG_LEVEL_NAMES


def test_the_level_names_are_the_log_levels_most_verbose_first() -> None:
    """Name only levels the log knows, TRACE included, most verbose first."""
    level_numbers = [
        logging.getLevelName(level_name) for level_name in cli.LOG_LEVEL_NAMES
    ]
    assert level_numbers[0] == log.TRACE
    assert all(isinstance(level_number, int) for level_number in level_numbers)
    assert level_numbers == sorted(set(level_numbers))
    assert level_numbers[-1] == logging.CRITICAL


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


def invented_parser() -> cli.HelpfulArgumentParser:
    """Return a parser with one option of each kind, as a program would state them."""
    program_parser = cli.HelpfulArgumentParser(
        prog="prog", description="An invented program."
    )
    program_parser.add_argument(
        "--start", type=cli.positive_mjd_or_none, default=cli.UNSET
    )
    program_parser.add_argument("--count", type=cli.positive_int, default=cli.UNSET)
    program_parser.add_argument("--log-file", type=cli.optional_path, default=cli.UNSET)
    return program_parser


def test_options_are_read_as_given_none_or_left_out() -> None:
    """Tell a value, None and a left-out option apart after parsing."""
    parsed_options = invented_parser().parse_args(
        ["--start", "60000.5", "--log-file", "None"]
    )
    assert parsed_options.start == 60000.5
    assert parsed_options.log_file is None
    assert parsed_options.count is cli.UNSET


@pytest.mark.parametrize(
    ("command_arguments", "expected_error"),
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
    command_arguments: list[str],
    expected_error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the full help, then the error, on standard error, and exit with 2."""
    program_parser = invented_parser()
    with pytest.raises(SystemExit) as raised:
        program_parser.parse_args(command_arguments)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "An invented program." in captured.err
    assert (
        captured.err
        == f"{program_parser.format_help()}\nprog: error: {expected_error}\n"
    )


@pytest.mark.usefixtures("uncoloured")
def test_a_subcommand_parser_prints_its_own_full_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Give a subcommand the same behaviour, with the help of the subcommand."""
    program_parser = cli.HelpfulArgumentParser(prog="prog")
    subcommands = program_parser.add_subparsers(dest="command")
    run_parser = subcommands.add_parser("run", description="Run the invented program.")
    run_parser.add_argument("--count", type=cli.positive_int)
    assert isinstance(run_parser, cli.HelpfulArgumentParser)
    with pytest.raises(SystemExit) as raised:
        program_parser.parse_args(["run", "--count", "0"])
    assert raised.value.code == 2
    captured_error = capsys.readouterr().err
    assert captured_error == (
        f"{run_parser.format_help()}\nprog run: error: argument --count:"
        " value must be a positive integer: '0'\n"
    )


def test_abbreviations_stay_refused_when_asked_for() -> None:
    """Refuse an abbreviation even when the parser is made to allow them."""
    program_parser = cli.HelpfulArgumentParser(prog="prog", allow_abbrev=True)
    program_parser.add_argument("--count", type=cli.positive_int)
    with pytest.raises(SystemExit):
        program_parser.parse_args(["--cou", "2"])


def test_each_parse_may_give_an_option_once() -> None:
    """Let a second parse with the same parser give the option again."""
    program_parser = invented_parser()
    # The first result is kept, so a parser that remembered the option for
    # as long as that result lives would refuse the second parse.
    first_parse = program_parser.parse_args(["--count", "2"])
    second_parse = program_parser.parse_args(["--count", "3"])
    assert (first_parse.count, second_parse.count) == (2, 3)


def test_an_option_naming_its_own_action_may_be_repeated() -> None:
    """Leave an option that names another action, such as append, to it."""
    program_parser = cli.HelpfulArgumentParser(prog="prog")
    program_parser.add_argument("--tag", action="append")
    program_parser.add_argument("--quiet", action="store_true")
    parsed_options = program_parser.parse_args(["--tag", "a", "--tag", "b", "--quiet"])
    assert parsed_options.tag == ["a", "b"]
    assert parsed_options.quiet is True


def test_a_subcommand_option_is_also_given_once() -> None:
    """Refuse a repeated option in a subcommand too."""
    program_parser = cli.HelpfulArgumentParser(prog="prog")
    run_parser = program_parser.add_subparsers(dest="command").add_parser("run")
    run_parser.add_argument("--count", type=cli.positive_int)
    with pytest.raises(SystemExit):
        program_parser.parse_args(["run", "--count", "2", "--count", "2"])


def test_an_option_that_names_store_is_given_once_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse an option given twice when it names the store action itself."""
    program_parser = cli.HelpfulArgumentParser(prog="prog")
    program_parser.add_argument("--label", action="store")
    with pytest.raises(SystemExit):
        program_parser.parse_args(["--label", "a", "--label", "b"])
    assert "argument --label: given more than once" in capsys.readouterr().err


def test_the_parser_takes_its_arguments_as_argparse_does() -> None:
    """Pass positional arguments on, so the program's name can come first."""
    assert cli.HelpfulArgumentParser("prog").prog == "prog"
