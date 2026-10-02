"""Tests for src/masterclock/das_processor/cli.py.

The rules covered: every setting option is read as given, None or left out,
and validated into CliOptions; paths are absolute, and refuse None except
the log file's; the steering directory and the clock configuration file are
options and take no None; there is no option for a time-constants file; an MJD
falls on a day a data file can cover; the count of epochs is command line
only and takes no None; the RF channel is one of the channels; the MJD to
start from when none is given is 59500 and the help says so; no arguments,
a bad argument and a usage error found after parsing all print the full help
and exit with status 2; and --help and --version exit with status 0.
"""

import sys
from importlib.metadata import version
from pathlib import Path

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
    "--steps", "6",
    "--log-file", "/logs/das.log",
    "--log-level", "DEBUG",
    "--backup-count", "None",
]  # fmt: skip
"""One command line giving every option, one of them the literal None."""


@pytest.fixture
def uncoloured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn off argparse's colour, so the help compares as plain text."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


def test_every_option_is_read_as_given() -> None:
    """Read each option into its field, the literal None as None."""
    options = cli.parse_args(EVERY_OPTION)
    assert options == cli.CliOptions(
        config_file=Path("/etc/das.ini"),
        rf="b",
        cd5m5m_path=Path("/data/cd5m5m"),
        steering_path=Path("/data/steering"),
        processed_path=Path("/data/processed"),
        redo_from_mjd=60010.5,
        start_from_mjd=59600.0,
        clock_config_file=Path("/etc/clocks.yaml"),
        steps=6,
        log_file=Path("/logs/das.log"),
        log_level="DEBUG",
        backup_count=None,
    )


def test_an_option_left_out_defers_to_the_config_file() -> None:
    """Leave every setting left out UNSET, and the two others None."""
    options = cli.parse_args(["--rf", "a"])
    assert options.model_dump() == {
        "config_file": None,
        "rf": "a",
        "cd5m5m_path": UNSET,
        "steering_path": UNSET,
        "processed_path": UNSET,
        "redo_from_mjd": UNSET,
        "start_from_mjd": UNSET,
        "clock_config_file": UNSET,
        "log_file": UNSET,
        "log_level": UNSET,
        "backup_count": UNSET,
        "steps": None,
    }


@pytest.mark.parametrize(
    ("option", "value", "field"),
    [
        ("--redo-from-mjd", "None", "redo_from_mjd"),
        ("--log-file", "None", "log_file"),
        ("--log-level", "None", "log_level"),
    ],
)
def test_none_sets_no_value_where_accepted(option: str, value: str, field: str) -> None:
    """Read the literal None as no value on the options that accept it."""
    assert getattr(cli.parse_args([option, value]), field) is None


@pytest.mark.parametrize(
    ("argv", "error"),
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
    argv: list[str], error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the full help, then the error, on standard error; exit with 2."""
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(argv)
    assert raised.value.code == 2
    written = capsys.readouterr()
    assert written.out == ""
    assert written.err.startswith(cli.build_parser().format_help())
    assert f"\ndas_processor: error: {error}" in written.err


@pytest.mark.parametrize("option", ["--start-from-mjd", "--redo-from-mjd"])
@pytest.mark.parametrize(
    ("text", "accepted"),
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
    option: str, text: str, accepted: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept an MJD on days 50000 to 99999 only, for both MJD options."""
    field = option.removeprefix("--").replace("-", "_")
    if accepted:
        assert getattr(cli.parse_args([option, text]), field) == float(text)
    else:
        with pytest.raises(SystemExit):
            cli.parse_args([option, text])
        assert (
            f"MJD must be on a day from {FIRST_DAY} to {LAST_DAY}: {text!r}"
            in capsys.readouterr().err
        )


@pytest.mark.parametrize("field", ["redo_from_mjd", "start_from_mjd"])
def test_the_model_also_holds_an_mjd_to_the_days(field: str) -> None:
    """Refuse an MJD outside the days when the options are built directly."""
    with pytest.raises(ValidationError, match="greater than or equal to 50000"):
        cli.CliOptions.model_validate({field: 1.0})


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
    fields = cli.CliOptions.model_fields.items()
    assert sorted(PATH_FIELDS) == sorted(
        name for name, field in fields if "Path" in str(field.annotation)
    )


@pytest.mark.parametrize("field", PATH_FIELDS)
@pytest.mark.parametrize("text", ["", "das.ini", "../data", "None"])
def test_the_model_refuses_a_relative_path(field: str, text: str) -> None:
    """Refuse a relative path in any path field when the options are built directly."""
    with pytest.raises(ValidationError, match="path must be absolute"):
        cli.CliOptions.model_validate({field: Path(text)})


@pytest.mark.parametrize("field", PATH_FIELDS)
def test_the_model_accepts_an_absolute_path(field: str) -> None:
    """Accept an absolute path in any path field."""
    options = cli.CliOptions.model_validate({field: Path("/data/x")})
    assert getattr(options, field) == Path("/data/x")


def test_the_model_refuses_unknown_fields_and_is_frozen() -> None:
    """Refuse a field the model does not declare, and any change after."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        cli.CliOptions.model_validate({"colour": "red"})
    options = cli.CliOptions()
    with pytest.raises(ValidationError, match="frozen"):
        options.steps = 1  # type: ignore[misc]


def test_the_start_used_when_none_is_given_is_59500() -> None:
    """Start from MJD 59500, a day a data file can cover, and say so in the help."""
    assert cli.START_FROM_MJD == 59_500
    assert cli.data_mjd(str(cli.START_FROM_MJD)) == cli.START_FROM_MJD
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "else MJD 59500)" in help_text


def test_none_means_no_mjd_only_where_accepted() -> None:
    """Read the literal None as no MJD only through the converter that accepts it."""
    assert cli.data_mjd_or_none("None") is None
    assert cli.data_mjd_or_none("60010") == 60010.0


@pytest.mark.parametrize("argv", [[], None])
@pytest.mark.usefixtures("uncoloured")
def test_no_arguments_print_the_full_help_and_exit_2(
    argv: list[str] | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the full help and exit with 2, given no list or an empty one."""
    monkeypatch.setattr(sys, "argv", ["das_processor"])
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(argv)
    assert raised.value.code == 2
    written = capsys.readouterr()
    assert written.out == ""
    assert written.err == cli.build_parser().format_help()


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
    with pytest.raises(SystemExit) as raised:
        cli.usage_error("no rf given")
    assert raised.value.code == 2
    assert capsys.readouterr().err == (
        f"{cli.build_parser().format_help()}\ndas_processor: error: no rf given\n"
    )


@pytest.mark.usefixtures("uncoloured")
def test_help_and_version_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the help, and the program's name and version, then exit with 0."""
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(["--help"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == cli.build_parser().format_help()
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == f"das_processor {version('masterclock')}\n"


def test_a_list_given_is_judged_alone_not_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Judge a given list as empty or not by itself, whatever sys.argv holds."""
    monkeypatch.setattr(sys, "argv", ["das_processor"])
    assert cli.parse_args(["--rf", "a"]).rf == "a"
    monkeypatch.setattr(sys, "argv", ["das_processor", "--rf", "a"])
    with pytest.raises(SystemExit) as raised:
        cli.parse_args([])
    assert raised.value.code == 2
    assert capsys.readouterr().err.endswith(cli.build_parser().format_help())
