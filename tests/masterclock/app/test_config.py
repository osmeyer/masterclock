"""Tests for src/masterclock/app/config.py.

The rules covered: the command line wins where both sources give a setting,
the literal None sets no value where a setting accepts it and is text where
it does not, and a setting neither source gives is None, or a usage error
when it is required; a file naming anything the program does not read, a
DEFAULT section included, is refused, and so is a value over more than one
line; a value may name another entry of its section; a file that cannot be
read or parsed is a ConfigError; a program's table of settings names each
setting once; and the logging settings are frozen, need every field, refuse
unknown fields, relative paths, and number text the command line refuses,
and take only the level names the command line takes.

A refused relative path is named, and a value over several lines is judged
as written, so the entry holding the lines is named.

Some groups of settings can be merged alone, the whole file still checked.
"""

import argparse
import itertools
import re
import types
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from masterclock.app import cli, config
from masterclock.app.exceptions import ConfigError, MissingSettingsError

PROGRAM_SETTINGS = (
    config.Setting("log_file", "logging", "logging", "log_file", "--log-file", True),
    config.Setting(
        "log_level", "logging", "logging", "log_level", "--log-level", True, True
    ),
    config.Setting(
        "backup_count", "logging", "logging", "backup_count", "--backup-count", True
    ),
    config.Setting("channel", "run", "input", "channel", "--channel", False, True),
    config.Setting("label", "run", "input", "label", "--label", False),
)
"""An invented program's settings, over more than one group and section.

Some are required and some not, and some accept None and some do not.
"""


def cli_options(**given_options: object) -> types.SimpleNamespace:
    """Return parsed options with every setting left out except those given."""
    option_fields: dict[str, object] = {
        setting.options_field: cli.UNSET for setting in PROGRAM_SETTINGS
    }
    option_fields.update(given_options)
    return types.SimpleNamespace(**option_fields)


def write_ini(tmp_path: Path, ini_text: str) -> Path:
    """Write ``ini_text`` to an invented INI file and return its path."""
    ini_file = tmp_path / "program.ini"
    ini_file.write_text(ini_text, encoding="utf-8")
    return ini_file


REQUIRED_SETTINGS_INI = "[logging]\nlog_level = INFO\n[input]\nchannel = 3\n"
"""A file giving the two required settings and nothing else."""


def test_the_tables_are_derived_from_the_settings() -> None:
    """Name the sections, entries and required settings the table describes."""
    assert config.known_sections(PROGRAM_SETTINGS) == frozenset({"logging", "input"})
    assert config.known_entries(PROGRAM_SETTINGS) == {
        "logging": frozenset({"log_file", "log_level", "backup_count"}),
        "input": frozenset({"channel", "label"}),
    }
    assert config.required_settings(PROGRAM_SETTINGS) == (
        ("logging", "log_level", "--log-level"),
        ("input", "channel", "--channel"),
    )


@pytest.mark.parametrize(
    ("extra_setting", "naming_problem"),
    [
        (
            config.Setting("channel", "other", "other", "chan", "--chan", False),
            "setting 'channel' is described more than once",
        ),
        (
            config.Setting("chan", "run", "input", "channel", "--chan", False),
            "entry [input] channel is read by more than one setting",
        ),
        (
            config.Setting("rate", "run", "input", "Rate", "--rate", False),
            "entry [input] Rate is not in lowercase, so no file can give it",
        ),
        (
            config.Setting("rate", "run", "DEFAULT", "rate", "--rate", False),
            "section [DEFAULT] is refused in every file, so no file can give it",
        ),
    ],
)
def test_a_table_that_names_a_setting_badly_is_refused(
    extra_setting: config.Setting, naming_problem: str, tmp_path: Path
) -> None:
    """Refuse a table with a repeated setting or one no file could give."""
    setting_table = (*PROGRAM_SETTINGS, extra_setting)
    for call_with_table in (
        lambda: config.known_sections(setting_table),
        lambda: config.known_entries(setting_table),
        lambda: config.required_settings(setting_table),
        lambda: config.read_entries(
            write_ini(tmp_path, REQUIRED_SETTINGS_INI), setting_table
        ),
        lambda: config.merge(setting_table, cli_options(), None),
    ):
        with pytest.raises(ValueError, match=f"^{re.escape(naming_problem)}$"):
            call_with_table()


def test_a_file_is_read_as_its_sections_and_entries(tmp_path: Path) -> None:
    """Return the entries as text, by section, entry names in lowercase."""
    ini_file = write_ini(
        tmp_path,
        "# a comment\n[logging]\nLog_Level = INFO\n; another\n"
        "log_file: /var/log/x.log\n[input]\nchannel=3\n",
    )
    assert config.read_entries(ini_file, PROGRAM_SETTINGS) == {
        "logging": {"log_level": "INFO", "log_file": "/var/log/x.log"},
        "input": {"channel": "3"},
    }


def test_an_empty_file_gives_nothing(tmp_path: Path) -> None:
    """Read a file with no sections as giving no setting."""
    assert config.read_entries(write_ini(tmp_path, ""), PROGRAM_SETTINGS) == {}


def test_a_value_may_name_another_entry_of_its_section(tmp_path: Path) -> None:
    """Replace %(entry)s by that entry's value, and %% by one percent sign."""
    ini_file = write_ini(
        tmp_path,
        "[input]\nchannel = 3\nlabel = ch%(channel)s at 100%%\n",
    )
    assert (
        config.read_entries(ini_file, PROGRAM_SETTINGS)["input"]["label"]
        == "ch3 at 100%"
    )


@pytest.mark.parametrize(
    ("ini_text", "unknown_names"),
    [
        ("[logging]\n[extra]\n[input]\n", "[extra]"),
        ("[Logging]\nlog_level = INFO\n", "[Logging]"),
        ("[input]\nchanel = 3\n", "[input] chanel"),
        ("[DEFAULT]\nchannel = 3\n", "[DEFAULT]"),
        ("[DEFAULT]\n[input]\nchannel = 3\n", "[DEFAULT]"),
        (
            "[input]\nrate = 1\n[extra]\n[logging]\nlevel = 2\n[DEFAULT]\n",
            "[extra], [DEFAULT], [input] rate, [logging] level",
        ),
    ],
)
def test_a_file_naming_what_the_program_does_not_read_is_refused(
    ini_text: str, unknown_names: str, tmp_path: Path
) -> None:
    """Refuse unknown sections, then unknown entries, DEFAULT even when empty."""
    ini_file = write_ini(tmp_path, ini_text)
    with pytest.raises(ConfigError) as raised:
        config.read_entries(ini_file, PROGRAM_SETTINGS)
    assert str(raised.value) == (
        f"config file {ini_file} names what the program does not read: {unknown_names}"
    )


@pytest.mark.parametrize(
    ("ini_text", "named_entry"),
    [
        ("[logging]\nlog_file = /a\n  backup_count = 5\n", "[logging] log_file"),
        ("[input]\nlabel =\n  first\n", "[input] label"),
    ],
)
def test_a_value_over_several_lines_is_refused(
    ini_text: str, named_entry: str, tmp_path: Path
) -> None:
    """Refuse a value an indented line was joined onto, naming the entry."""
    ini_file = write_ini(tmp_path, ini_text)
    with pytest.raises(ConfigError) as raised:
        config.read_entries(ini_file, PROGRAM_SETTINGS)
    assert str(raised.value) == (
        f"config file {ini_file} gives {named_entry} on more than one line"
    )


@pytest.mark.parametrize(
    "ini_text",
    [
        "channel = 3\n",
        "[input]\nchannel = 3\nchannel = 4\n",
        "[input]\n[input]\n",
        "[input]\nchannel\n",
        "[input]\nlabel = 50%\n",
        "[input]\nlabel = %(nothing)s\n",
    ],
)
def test_a_file_that_is_not_valid_ini_is_refused(ini_text: str, tmp_path: Path) -> None:
    """Refuse text configparser cannot read, a lone percent sign included."""
    ini_file = write_ini(tmp_path, ini_text)
    with pytest.raises(ConfigError, match=r"^config file .* is not valid INI: "):
        config.read_entries(ini_file, PROGRAM_SETTINGS)


def test_a_file_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """Refuse a missing file, a directory, and bytes that are not UTF-8."""
    missing_file = tmp_path / "missing.ini"
    garbled_file = tmp_path / "garbled.ini"
    garbled_file.write_bytes(b"[input]\nlabel = \xff\n")
    for unreadable_path in (missing_file, tmp_path, garbled_file):
        with pytest.raises(
            ConfigError,
            match=f"^cannot read config file {re.escape(str(unreadable_path))}: ",
        ):
            config.read_entries(unreadable_path, PROGRAM_SETTINGS)


CLI_STATES = ("left out", "None", "value")
FILE_STATES = ("absent", "None", "value")


@pytest.mark.parametrize(
    ("on_command_line", "in_file"), list(itertools.product(CLI_STATES, FILE_STATES))
)
def test_the_command_line_wins_and_none_sets_no_value(
    on_command_line: str, in_file: str, tmp_path: Path
) -> None:
    """Take the command line's value or None over the file's, else the file's."""
    file_entry_by_state = {
        "absent": "",
        "None": "log_file = None\n",
        "value": "log_file = /f\n",
    }
    ini_file = write_ini(
        tmp_path, f"[logging]\nlog_level = INFO\n{file_entry_by_state[in_file]}"
    )
    cli_value_by_state = {"left out": cli.UNSET, "None": None, "value": Path("/c")}
    merged_settings = config.merge(
        PROGRAM_SETTINGS,
        cli_options(log_file=cli_value_by_state[on_command_line], channel=3),
        ini_file,
    )
    expected_log_file: object = {"None": None, "value": Path("/c")}.get(on_command_line)
    if on_command_line == "left out":
        expected_log_file = {"absent": None, "None": None, "value": "/f"}[in_file]
    assert merged_settings["logging"]["log_file"] == expected_log_file


def test_values_are_grouped_as_the_table_says(tmp_path: Path) -> None:
    """Group every setting, with None for the ones neither source gives."""
    merged_settings = config.merge(
        PROGRAM_SETTINGS,
        cli_options(label="first"),
        write_ini(tmp_path, REQUIRED_SETTINGS_INI),
    )
    assert merged_settings == {
        "logging": {"log_file": None, "log_level": "INFO", "backup_count": None},
        "run": {"channel": "3", "label": "first"},
    }


def test_none_is_text_where_a_setting_does_not_accept_it(tmp_path: Path) -> None:
    """Pass the file's None through as text for a setting that refuses it."""
    ini_file = write_ini(tmp_path, f"{REQUIRED_SETTINGS_INI}label = None\n")
    assert (
        config.merge(PROGRAM_SETTINGS, cli_options(), ini_file)["run"]["label"]
        == "None"
    )


def test_a_required_setting_given_as_none_is_given(tmp_path: Path) -> None:
    """Count None from either source as giving a required setting."""
    ini_file = write_ini(tmp_path, "[logging]\nlog_level = None\n")
    assert (
        config.merge(PROGRAM_SETTINGS, cli_options(channel=3), ini_file)["logging"][
            "log_level"
        ]
        is None
    )
    from_command_line = config.merge(
        PROGRAM_SETTINGS, cli_options(log_level=None, channel=3), None
    )
    assert from_command_line["logging"]["log_level"] is None


def test_every_missing_required_setting_is_named_in_order() -> None:
    """Name every required setting neither source gives, in the table's order."""
    with pytest.raises(MissingSettingsError) as raised:
        config.merge(PROGRAM_SETTINGS, cli_options(), None)
    assert str(raised.value) == (
        "these settings must be provided by the config file or the command"
        " line: [logging] log_level (--log-level), [input] channel (--channel)"
    )


def test_with_no_file_every_setting_comes_from_the_command_line() -> None:
    """Merge from the command line alone when no file is named."""
    merged_settings = config.merge(
        PROGRAM_SETTINGS,
        cli_options(log_level="DEBUG", channel=7, backup_count=2),
        None,
    )
    assert merged_settings == {
        "logging": {"log_file": None, "log_level": "DEBUG", "backup_count": 2},
        "run": {"channel": 7, "label": None},
    }


def test_merged_logging_settings_validate(tmp_path: Path) -> None:
    """Validate what merge gives for the logging group, from text and values."""
    ini_file = write_ini(
        tmp_path,
        "[logging]\nlog_level = WARNING\nlog_file = /var/log/p.log\n"
        "backup_count = 7\n[input]\nchannel = 1\n",
    )
    logging_config = config.LoggingConfig.model_validate(
        config.merge(PROGRAM_SETTINGS, cli_options(), ini_file)["logging"]
    )
    assert logging_config == config.LoggingConfig(
        log_file=Path("/var/log/p.log"), log_level="WARNING", backup_count=7
    )


def test_the_logging_settings_are_frozen_and_complete() -> None:
    """Refuse changes, unknown fields, and a field left out."""
    logging_config = config.LoggingConfig(
        log_file=None, log_level=None, backup_count=None
    )
    with pytest.raises(ValidationError, match="frozen"):
        logging_config.log_level = "INFO"  # type: ignore[misc]  # the refusal is what is checked
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config.LoggingConfig.model_validate(
            {"log_file": None, "log_level": None, "backup_count": None, "colour": 1}
        )
    with pytest.raises(ValidationError, match="backup_count"):
        config.LoggingConfig.model_validate({"log_file": None, "log_level": None})


@pytest.mark.parametrize("log_file_text", ["", ".", "run.log", "logs/run.log", "None"])
def test_a_log_file_that_is_not_absolute_is_refused(log_file_text: str) -> None:
    """Refuse a relative log file, the empty text and None-as-text included."""
    with pytest.raises(ValidationError, match="path must be absolute"):
        config.LoggingConfig.model_validate(
            {"log_file": log_file_text, "log_level": "INFO", "backup_count": None}
        )


@pytest.mark.parametrize("log_level_text", ["INFO", "info", "Trace", "5", ""])
def test_a_log_level_is_one_of_the_names(log_level_text: str) -> None:
    """Accept exactly the names the command line accepts."""
    logging_values = {
        "log_file": None,
        "log_level": log_level_text,
        "backup_count": None,
    }
    if log_level_text in cli.LOG_LEVEL_NAMES:
        assert (
            config.LoggingConfig.model_validate(logging_values).log_level
            == log_level_text
        )
    else:
        with pytest.raises(ValidationError):
            config.LoggingConfig.model_validate(logging_values)


def count_reading(count_conversion: object, count_text: str) -> object:
    """Return what ``count_conversion`` reads ``count_text`` as, or None if refused."""
    assert callable(count_conversion)
    try:
        return count_conversion(count_text)
    except (
        ValidationError,
        argparse.ArgumentTypeError,
    ):
        return None


def count_from_file(count_text: str) -> int | None:
    """Return the backup count the model reads from the file's ``count_text``."""
    return config.LoggingConfig.model_validate(
        {"log_file": None, "log_level": None, "backup_count": count_text}
    ).backup_count


@given(
    st.one_of(
        st.text(),
        st.integers().map(str),
        st.sampled_from(
            ["5.0", "1e3", "\N{FULLWIDTH DIGIT FIVE}", "+5", " 5 ", "1_000", "true"]
        ),
    )
)
def test_a_count_reads_the_same_from_the_file_and_command_line(count_text: str) -> None:
    """Accept the same count text as the command line does, as the same number."""
    assert count_reading(count_from_file, count_text) == count_reading(
        cli.positive_int, count_text
    )


def test_a_count_refused_names_the_command_line_reason() -> None:
    """Give the command line's reason when the file's count is refused."""
    with pytest.raises(
        ValidationError, match=re.escape("value must be a positive integer: '0'")
    ):
        count_from_file("0")
    with pytest.raises(
        ValidationError, match=re.escape("invalid positive int value: '5.0'")
    ):
        count_from_file("5.0")


def test_a_relative_log_file_is_named_in_the_refusal() -> None:
    """Say which path is not absolute."""
    with pytest.raises(
        ValidationError, match=re.escape("path must be absolute: 'run.log'")
    ):
        config.LoggingConfig.model_validate(
            {"log_file": "run.log", "log_level": "INFO", "backup_count": None}
        )


def test_the_entry_holding_the_lines_is_named_not_one_that_names_it(
    tmp_path: Path,
) -> None:
    """Judge each value as written, before another entry's value is put in it."""
    ini_file = write_ini(
        tmp_path, "[input]\nchannel = %(label)s\nlabel = two\n  lines\n"
    )
    with pytest.raises(ConfigError) as raised:
        config.read_entries(ini_file, PROGRAM_SETTINGS)
    assert str(raised.value) == (
        f"config file {ini_file} gives [input] label on more than one line"
    )


def test_some_groups_can_be_merged_alone() -> None:
    """Merge only the groups asked for, a required setting of another not missed."""
    merged_settings = config.merge(
        PROGRAM_SETTINGS, cli_options(log_level="INFO"), None, config_groups={"logging"}
    )
    assert merged_settings == {
        "logging": {"log_file": None, "log_level": "INFO", "backup_count": None}
    }
    with pytest.raises(MissingSettingsError, match="channel"):
        config.merge(PROGRAM_SETTINGS, cli_options(log_level="INFO"), None)


def test_merging_some_groups_still_checks_the_whole_file(tmp_path: Path) -> None:
    """Refuse a file naming what the program does not read, whatever is merged."""
    ini_file = write_ini(tmp_path, "[input]\ncolour = red\n")
    with pytest.raises(ConfigError, match="colour"):
        config.merge(
            PROGRAM_SETTINGS,
            cli_options(log_level="INFO"),
            ini_file,
            config_groups={"logging"},
        )
