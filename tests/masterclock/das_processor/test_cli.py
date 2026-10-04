"""Tests for src/masterclock/das_processor/cli.py.

The rules covered: every setting option is read as given, None or left out,
and validated into CliOptions; paths are absolute, and refuse None except
the log file's; the steering directory and the clock configuration file are
options and take no None; there is no option for a time-constants file; an MJD
falls on a day a data file can cover; the count of epochs and the MJD to
redo from are command line only and take no None; the RF channel is one of
the channels; the MJD to start from when none is given is 59500 and the
help says so; no arguments,
a bad argument and a usage error found after parsing all print the full help
and exit with status 2; --help and --version exit with status 0, a single
argument read from sys.argv as given; and the help reads word for word as
written.
"""

import sys
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from masterclock.app.cli import UNSET
from masterclock.das_processor import cli
from masterclock.das_processor.read_cd5m5m import FIRST_DAY, LAST_DAY

EVERY_OPTION: list[str] = [
    "--config-file", "/etc/das.ini",
    "--rf", "b",
    "--cd5m5m-path", "/data/cd5m5m",
    "--steering-path", "/data/steering",
    "--processed-path", "/data/processed",
    "--redo-from-mjd", "60010.5",
    "--start-from-mjd", "59600",
    "--clock-config-file", "/etc/clocks.yaml",
    "--num-workers", "3",
    "--steps", "6",
    "--log-file", "/logs/das.log",
    "--log-level", "DEBUG",
    "--backup-count", "None",
]  # fmt: skip
"""One command line giving every option, one of them the literal None."""


HELP: Final = """\
usage: das_processor [-h] [--version] [--config-file PATH] [--rf {a,b}]
                     [--cd5m5m-path PATH] [--steering-path PATH]
                     [--processed-path PATH] [--redo-from-mjd MJD]
                     [--start-from-mjd MJD] [--clock-config-file PATH]
                     [--num-workers N] [--steps N] [--log-file PATH]
                     [--log-level {TRACE,DEBUG,INFO,WARNING,ERROR,CRITICAL,None}]
                     [--backup-count N]

Process 5 MHz phase measurements from the Data Acquisition System (DAS).

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --config-file PATH    path to the INI configuration file (optional when
                        every required setting is given on the command line)
  --rf {a,b}            RF channel to process, overriding the config file's
                        [DAS] rf (default: use the config file)
  --cd5m5m-path PATH    path to the DAS 5 MHz phase data, overriding the
                        config file's [DAS] cd5m5m_path (default: use the
                        config file)
  --steering-path PATH  path to the directory of steering files, one per
                        reference clock, overriding the config file's [DAS]
                        steering_path (default: use the config file)
  --processed-path PATH
                        path for processed results, each kind of file in a
                        subdirectory of its own, overriding the config file's
                        [PROCESSED] processed_path (default: use the config
                        file)
  --redo-from-mjd MJD   reprocess data starting from this MJD, before the run;
                        command line only, with no config-file entry (default:
                        no reprocessing)
  --start-from-mjd MJD  MJD to start processing from when there are no
                        processed files to read a previous measurement from
                        (floored to its ten-minute mark), overriding the
                        config file's [PROCESSED] start_from_mjd (default: use
                        the config file, else MJD 59500)
  --clock-config-file PATH
                        path to the YAML file giving each clock's estimator
                        parameters and each pair's RMS limit, overriding the
                        config file's [PROCESSED] clock_config_file (default:
                        use the config file)
  --num-workers N       number of worker processes to work each epoch's
                        series, overriding the config file's [PROCESSED]
                        num_workers; pass None to work them in the main
                        process alone (default: use the config file)
  --steps N             process exactly this many ten-minute epochs that write
                        rows and shut down, instead of every epoch not yet
                        processed; an epoch of a data gap that writes no row
                        is not counted; command line only, with no config-file
                        entry (default: process every new epoch)
  --log-file PATH       path to the log file, overriding the config file's
                        [LOGGING] log_file; pass None to disable file logging
                        (default: use the config file)
  --log-level {TRACE,DEBUG,INFO,WARNING,ERROR,CRITICAL,None}
                        logging level, overriding the config file's [LOGGING]
                        log_level; pass None to disable logging entirely
                        (default: use the config file)
  --backup-count N      number of rotated daily log files to keep, overriding
                        the config file's [LOGGING] backup_count; pass None to
                        keep every rotated file (default: use the config file)
"""
"""The full help, as a person reads it at 80 columns."""


@pytest.fixture
def uncoloured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn off argparse's colour, so the help compares as plain text."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


def test_every_option_is_read_as_given() -> None:
    """Read each option into its field, the literal None as None."""
    cli_options = cli.parse_args(EVERY_OPTION)
    assert cli_options == cli.CliOptions(
        config_file=Path("/etc/das.ini"),
        rf="b",
        cd5m5m_path=Path("/data/cd5m5m"),
        steering_path=Path("/data/steering"),
        processed_path=Path("/data/processed"),
        redo_from_mjd=60010.5,
        start_from_mjd=59600.0,
        clock_config_file=Path("/etc/clocks.yaml"),
        num_workers=3,
        steps=6,
        log_file=Path("/logs/das.log"),
        log_level="DEBUG",
        backup_count=None,
    )


def test_an_option_left_out_defers_to_the_config_file() -> None:
    """Leave every setting left out UNSET, and the three others None."""
    cli_options = cli.parse_args(["--rf", "a"])
    assert cli_options.model_dump() == {
        "config_file": None,
        "rf": "a",
        "cd5m5m_path": UNSET,
        "steering_path": UNSET,
        "processed_path": UNSET,
        "redo_from_mjd": None,
        "start_from_mjd": UNSET,
        "clock_config_file": UNSET,
        "num_workers": UNSET,
        "log_file": UNSET,
        "log_level": UNSET,
        "backup_count": UNSET,
        "steps": None,
    }


@pytest.mark.parametrize(
    ("cli_option", "option_value", "options_field"),
    [
        ("--log-file", "None", "log_file"),
        ("--log-level", "None", "log_level"),
        ("--num-workers", "None", "num_workers"),
    ],
)
def test_none_sets_no_value_where_accepted(
    cli_option: str, option_value: str, options_field: str
) -> None:
    """Read the literal None as no value on the options that accept it."""
    assert getattr(cli.parse_args([cli_option, option_value]), options_field) is None


@pytest.mark.parametrize(
    ("argv", "expected_error"),
    [
        (["--config-file", "None"], "argument --config-file: path must be absolute"),
        (["--config-file", "das.ini"], "argument --config-file: path must be absolute"),
        (["--cd5m5m-path", ""], "argument --cd5m5m-path: path must be absolute"),
        (["--cd5m5m-path", "None"], "argument --cd5m5m-path: path must be absolute"),
        (
            ["--steering-path", "steer"],
            "argument --steering-path: path must be absolute",
        ),
        (
            ["--steering-path", "None"],
            "argument --steering-path: path must be absolute",
        ),
        (
            ["--clock-config-file", "clocks.yaml"],
            "argument --clock-config-file: path must be absolute",
        ),
        (
            ["--clock-config-file", "None"],
            "argument --clock-config-file: path must be absolute",
        ),
        (
            ["--time-constants-file", "/etc/tc.yaml"],
            "unrecognized arguments: --time-constants-file /etc/tc.yaml",
        ),
        (
            ["--processed-path", "out"],
            "argument --processed-path: path must be absolute",
        ),
        (
            ["--processed-path", "None"],
            "argument --processed-path: path must be absolute",
        ),
        (["--log-file", "das.log"], "argument --log-file: path must be absolute"),
        (["--rf", "None"], "argument --rf: invalid choice: 'None'"),
        (["--rf", "A"], "argument --rf: invalid choice: 'A'"),
        (["--start-from-mjd", "None"], "argument --start-from-mjd: invalid positive"),
        (["--steps", "None"], "argument --steps: invalid positive int value"),
        (["--steps", "0"], "argument --steps: value must be a positive integer"),
        (["--backup-count", "0"], "argument --backup-count: value must be a positive"),
        (["--log-level", "none"], "argument --log-level: invalid choice: 'none'"),
        (["--rf", "a", "--rf", "b"], "argument --rf: given more than once"),
        (["--cd5m", "/data"], "unrecognized arguments: --cd5m /data"),
    ],
)
@pytest.mark.usefixtures("uncoloured")
def test_a_bad_argument_prints_the_full_help_and_exits_2(
    argv: list[str], expected_error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the full help, then the error, on standard error; exit with 2."""
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args(argv)
    assert system_exit.value.code == 2
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err.startswith(cli.build_parser().format_help())
    assert f"\ndas_processor: error: {expected_error}" in captured_output.err


@pytest.mark.parametrize("mjd_option", ["--start-from-mjd", "--redo-from-mjd"])
@pytest.mark.parametrize(
    ("mjd_text", "accepted"),
    [
        (str(FIRST_DAY), True),
        (f"{LAST_DAY}.999999", True),
        (f"{FIRST_DAY - 1}.999999", False),
        (str(LAST_DAY + 1), False),
        ("1", False),
        ("1e9", False),
    ],
)
def test_an_mjd_must_fall_on_a_day_a_data_file_covers(
    mjd_option: str, mjd_text: str, accepted: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept an MJD on days 50000 to 99999 only, for both MJD options."""
    options_field = mjd_option.removeprefix("--").replace("-", "_")
    if accepted:
        assert getattr(cli.parse_args([mjd_option, mjd_text]), options_field) == float(
            mjd_text
        )
    else:
        with pytest.raises(SystemExit):
            cli.parse_args([mjd_option, mjd_text])
        assert (
            f"MJD must be on a day from {FIRST_DAY} to {LAST_DAY}: {mjd_text!r}"
            in capsys.readouterr().err
        )


@pytest.mark.parametrize("mjd_field", ["redo_from_mjd", "start_from_mjd"])
def test_the_model_also_holds_an_mjd_to_the_days(mjd_field: str) -> None:
    """Refuse an MJD outside the days when the options are built directly."""
    with pytest.raises(ValidationError, match="greater than or equal to 50000"):
        cli.CliOptions.model_validate({mjd_field: 1.0})


PATH_FIELDS: list[str] = [
    "config_file",
    "cd5m5m_path",
    "steering_path",
    "processed_path",
    "clock_config_file",
    "log_file",
]
"""Every field of CliOptions that holds a path."""


def test_the_path_fields_are_every_path_field() -> None:
    """List every field whose type admits a path, so none goes unchecked."""
    model_fields = cli.CliOptions.model_fields.items()
    assert sorted(PATH_FIELDS) == sorted(
        field_name
        for field_name, field_info in model_fields
        if "Path" in str(field_info.annotation)
    )


@pytest.mark.parametrize("path_field", PATH_FIELDS)
@pytest.mark.parametrize("path_text", ["", "das.ini", "../data", "None"])
def test_the_model_refuses_a_relative_path(path_field: str, path_text: str) -> None:
    """Refuse a relative path in any path field when the options are built directly."""
    with pytest.raises(ValidationError, match="path must be absolute"):
        cli.CliOptions.model_validate({path_field: Path(path_text)})


@pytest.mark.parametrize("path_field", PATH_FIELDS)
def test_the_model_accepts_an_absolute_path(path_field: str) -> None:
    """Accept an absolute path in any path field."""
    cli_options = cli.CliOptions.model_validate({path_field: Path("/data/x")})
    assert getattr(cli_options, path_field) == Path("/data/x")


def test_the_model_refuses_unknown_fields_and_is_frozen() -> None:
    """Refuse a field the model does not declare, and any change after."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        cli.CliOptions.model_validate({"colour": "red"})
    cli_options = cli.CliOptions()
    with pytest.raises(ValidationError, match="frozen"):
        cli_options.steps = 1  # type: ignore[misc]


def test_the_start_used_when_none_is_given_is_59500() -> None:
    """Start from MJD 59500, a day a data file can cover, and say so in the help."""
    assert cli.START_FROM_MJD == 59_500
    assert cli.data_mjd(str(cli.START_FROM_MJD)) == cli.START_FROM_MJD
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "else MJD 59500)" in help_text


def test_a_redo_is_command_line_only() -> None:
    """Give no redo when the option is left out, and take no None for one."""
    assert cli.parse_args(["--rf", "a"]).redo_from_mjd is None
    assert cli.parse_args(["--redo-from-mjd", "60010"]).redo_from_mjd == 60010.0
    with pytest.raises(SystemExit):
        cli.parse_args(["--redo-from-mjd", "None"])


@pytest.mark.parametrize("argv", [[], None])
@pytest.mark.usefixtures("uncoloured")
def test_no_arguments_print_the_full_help_and_exit_2(
    argv: list[str] | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the full help and exit with 2, given no list or an empty one."""
    monkeypatch.setattr(sys, "argv", ["das_processor"])
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args(argv)
    assert system_exit.value.code == 2
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err == cli.build_parser().format_help()


def test_the_command_line_is_read_from_sys_argv_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read sys.argv, without the program name, when given no list."""
    monkeypatch.setattr(sys, "argv", ["das_processor", "--rf", "b"])
    assert cli.parse_args().rf == "b"


@pytest.mark.usefixtures("uncoloured")
def test_a_usage_error_prints_the_full_help_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report an error found after parsing as an argument error is reported."""
    with pytest.raises(SystemExit) as system_exit:
        cli.usage_error("no rf given")
    assert system_exit.value.code == 2
    assert capsys.readouterr().err == (
        f"{cli.build_parser().format_help()}\ndas_processor: error: no rf given\n"
    )


@pytest.mark.usefixtures("uncoloured")
def test_help_and_version_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the help, and the program's name and version, then exit with 0."""
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args(["--help"])
    assert system_exit.value.code == 0
    assert capsys.readouterr().out == cli.build_parser().format_help()
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args(["--version"])
    assert system_exit.value.code == 0
    assert capsys.readouterr().out == f"das_processor {version('masterclock')}\n"


def test_a_list_given_is_judged_alone_not_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Judge a given list as empty or not by itself, whatever sys.argv holds."""
    monkeypatch.setattr(sys, "argv", ["das_processor"])
    assert cli.parse_args(["--rf", "a"]).rf == "a"
    monkeypatch.setattr(sys, "argv", ["das_processor", "--rf", "a"])
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args([])
    assert system_exit.value.code == 2
    assert capsys.readouterr().err.endswith(cli.build_parser().format_help())


@pytest.mark.usefixtures("uncoloured")
def test_the_help_reads_as_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the help word for word: every option, its value's name and its text."""
    monkeypatch.setenv("COLUMNS", "80")
    assert cli.build_parser().format_help() == HELP


def test_one_argument_alone_is_read_from_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Take a single argument on the command line, such as --version, as given."""
    monkeypatch.setattr(sys, "argv", ["das_processor", "--version"])
    with pytest.raises(SystemExit) as system_exit:
        cli.parse_args(None)
    assert system_exit.value.code == 0
    assert capsys.readouterr().out == f"das_processor {version('masterclock')}\n"
