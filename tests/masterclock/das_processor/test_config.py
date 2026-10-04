"""Tests for src/masterclock/das_processor/config.py.

The rules covered: every setting of the command line is described once,
required ones first, under the flag the parser knows; the merged settings
are validated with the command line winning, text from the file read as the
command line reads it, paths absolute, MJDs on a day a data file covers,
the literal None only where a setting accepts it, and unknown fields
refused, a time-constants file among them; a start MJD given by neither
source is 59500; a required setting given by neither source is a
MissingSettingsError; the processed subdirectories and the lock file are
named from processed_path and the channel; and check_paths refuses a data
or steering directory that is not one or cannot be listed, a clock
configuration file that is not a regular file it can read, and a processed
directory that is something else or cannot be written into.

The processed-directory refusals are word for word, and a file naming a
redo or a count of epochs, which only the command line gives, is refused.
The logging settings are built alone, before the rest.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from masterclock.app.exceptions import ConfigError, MissingSettingsError
from masterclock.app.lock import RunLock
from masterclock.das_processor import cli, config

REQUIRED_LINES: dict[str, str] = {
    "rf": "[DAS]\nrf = a\n",
    "cd5m5m_path": "[DAS]\ncd5m5m_path = /data/cd5m5m\n",
    "steering_path": "[DAS]\nsteering_path = /data/steering\n",
    "processed_path": "[PROCESSED]\nprocessed_path = /data/processed\n",
    "clock_config_file": "[PROCESSED]\nclock_config_file = /etc/clocks.yaml\n",
    "log_file": "[LOGGING]\nlog_file = /logs/das.log\n",
    "log_level": "[LOGGING]\nlog_level = INFO\n",
}
"""One invented line of each required setting, with its section header."""


def write_ini_file(tmp_path: Path, extra_entries: dict[str, str] | None = None) -> Path:
    """Write an INI file of every required setting plus ``extra_entries``.

    ``extra_entries`` maps ``"SECTION entry"`` to the value written for it.
    """
    section_lines: dict[str, list[str]] = {}
    for required_text in REQUIRED_LINES.values():
        section_header, entry_line = required_text.splitlines()
        section_lines.setdefault(section_header, []).append(entry_line)
    for section_entry, entry_value in (extra_entries or {}).items():
        ini_section, ini_entry = section_entry.split()
        section_lines.setdefault(f"[{ini_section}]", []).append(
            f"{ini_entry} = {entry_value}"
        )
    ini_file = tmp_path / "das.ini"
    ini_file.write_text(
        "".join(
            f"{section_header}\n"
            + "".join(f"{entry_line}\n" for entry_line in entry_lines)
            for section_header, entry_lines in section_lines.items()
        ),
        encoding="utf-8",
    )
    return ini_file


def build_from_ini(
    tmp_path: Path, *argv: str, **extra_entries: str
) -> config.AppConfig:
    """Build the configuration from an INI file and ``argv``.

    Keyword names are ``SECTION__entry``, written into the file.
    """
    ini_entries = {
        keyword.replace("__", " "): entry_value
        for keyword, entry_value in extra_entries.items()
    }
    ini_file = write_ini_file(tmp_path, ini_entries)
    return config.build_config(cli.parse_args(["--config-file", str(ini_file), *argv]))


# ------------------------------------------------------------- the table


def test_every_command_line_setting_is_described_once() -> None:
    """Describe every CliOptions field but the three that back no entry."""
    setting_fields = set(cli.CliOptions.model_fields) - {
        "config_file",
        "steps",
        "redo_from_mjd",
    }
    options_fields = [setting.options_field for setting in config.SETTINGS]
    assert sorted(options_fields) == sorted(setting_fields)


def test_each_setting_is_named_by_a_flag_the_parser_knows() -> None:
    """Name each setting by its flag, its entry and its command-line field alike."""
    parser_actions = cli.build_parser()._option_string_actions
    for setting in config.SETTINGS:
        assert parser_actions[setting.cli_flag].dest == setting.options_field
        assert setting.ini_entry == setting.options_field


def test_the_required_settings_come_first() -> None:
    """List the required settings first, in the order of the help."""
    required_marks = [setting.required for setting in config.SETTINGS]
    assert required_marks == sorted(required_marks, reverse=True)
    assert [ini_entry for _, ini_entry, _ in config.REQUIRED_SETTINGS] == list(
        REQUIRED_LINES
    )


def test_none_is_accepted_by_the_settings_whose_options_accept_it() -> None:
    """Accept the literal None in the file exactly where the command line does."""
    none_accepting_fields = {
        setting.options_field for setting in config.SETTINGS if setting.allow_none
    }
    assert none_accepting_fields == {
        "log_file",
        "log_level",
        "num_workers",
        "backup_count",
    }
    for setting in config.SETTINGS:
        if setting.allow_none:
            assert cli.parse_args([setting.cli_flag, "None"]) is not None
        else:
            with pytest.raises(SystemExit):
                cli.parse_args([setting.cli_flag, "None"])


def test_the_sections_and_entries_come_from_the_table() -> None:
    """Read exactly the sections and entries the table names."""
    assert {"DAS", "PROCESSED", "LOGGING"} == config.KNOWN_SECTIONS
    assert config.KNOWN_ENTRIES["DAS"] == {"rf", "cd5m5m_path", "steering_path"}


# ------------------------------------------------------------- building


def test_a_file_of_the_required_settings_builds(tmp_path: Path) -> None:
    """Build from the file alone, the optional settings None and the start 59500."""
    app_config = build_from_ini(tmp_path)
    assert app_config.das.model_dump() == {
        "rf": "a",
        "cd5m5m_path": Path("/data/cd5m5m"),
        "steering_path": Path("/data/steering"),
    }
    assert app_config.processed.model_dump() == {
        "processed_path": Path("/data/processed"),
        "start_from_mjd": cli.START_FROM_MJD,
        "clock_config_file": Path("/etc/clocks.yaml"),
        "num_workers": None,
    }
    assert app_config.logging.model_dump() == {
        "log_file": Path("/logs/das.log"),
        "log_level": "INFO",
        "backup_count": None,
    }


def test_the_command_line_alone_is_enough() -> None:
    """Build with no file when the command line gives every required setting."""
    app_config = config.build_config(
        cli.parse_args(
            [
                "--rf", "b",
                "--cd5m5m-path", "/d",
                "--steering-path", "/s",
                "--processed-path", "/p",
                "--clock-config-file", "/c.yaml",
                "--log-file", "None",
                "--log-level", "None",
            ]
        )
    )  # fmt: skip
    assert app_config.das.rf == "b"
    assert app_config.das.steering_path == Path("/s")
    assert app_config.processed.clock_config_file == Path("/c.yaml")
    assert app_config.logging.log_file is None
    assert app_config.logging.log_level is None


def test_the_command_line_wins(tmp_path: Path) -> None:
    """Take the command line's value where both sources give one."""
    app_config = build_from_ini(
        tmp_path,
        "--rf",
        "b",
        "--start-from-mjd",
        "60000",
        "--steering-path",
        "/other/steering",
        "--clock-config-file",
        "/other/clocks.yaml",
        PROCESSED__start_from_mjd="61000",
    )
    assert app_config.das.rf == "b"
    assert app_config.processed.start_from_mjd == 60000.0
    assert app_config.das.steering_path == Path("/other/steering")
    assert app_config.processed.clock_config_file == Path("/other/clocks.yaml")


def test_a_start_given_in_the_file_is_kept(tmp_path: Path) -> None:
    """Use the file's start MJD, read as the command line reads it."""
    app_config = build_from_ini(tmp_path, PROCESSED__start_from_mjd="6_0010.5")
    assert app_config.processed.start_from_mjd == cli.data_mjd("6_0010.5")


@pytest.mark.parametrize(
    ("entry_keyword", "entry_value", "expected_reason"),
    [
        ("PROCESSED__start_from_mjd", "None", "invalid positive float value"),
        ("PROCESSED__start_from_mjd", "1", "MJD must be on a day from 50000"),
        ("LOGGING__backup_count", "0", "value must be a positive integer"),
        ("PROCESSED__num_workers", "0", "value must be a positive integer"),
        ("PROCESSED__num_workers", "2.5", "invalid positive int value"),
    ],
)
def test_a_bad_value_in_the_file_is_refused(
    tmp_path: Path, entry_keyword: str, entry_value: str, expected_reason: str
) -> None:
    """Refuse a value the command line would refuse, naming the setting."""
    with pytest.raises(ConfigError, match=expected_reason):
        build_from_ini(tmp_path, **{entry_keyword: entry_value})


@pytest.mark.parametrize(
    ("required_line", "replacement_line", "expected_reason"),
    [
        ("rf = a", "rf = A", "das.rf"),
        ("rf = a", "rf = None", "das.rf"),
        (
            "cd5m5m_path = /data/cd5m5m",
            "cd5m5m_path = data",
            "path must be absolute",
        ),
        (
            "cd5m5m_path = /data/cd5m5m",
            "cd5m5m_path = None",
            "path must be absolute",
        ),
        (
            "processed_path = /data/processed",
            "processed_path =",
            "path must be absolute",
        ),
        ("log_file = /logs/das.log", "log_file = das.log", "path must be absolute"),
        ("log_level = INFO", "log_level = info", "logging.log_level"),
        (
            "steering_path = /data/steering",
            "steering_path = steering",
            "path must be absolute",
        ),
        (
            "steering_path = /data/steering",
            "steering_path = None",
            "path must be absolute",
        ),
        (
            "clock_config_file = /etc/clocks.yaml",
            "clock_config_file = clocks.yaml",
            "path must be absolute",
        ),
        (
            "clock_config_file = /etc/clocks.yaml",
            "clock_config_file = None",
            "path must be absolute",
        ),
    ],
)
def test_a_bad_required_value_in_the_file_is_refused(
    tmp_path: Path, required_line: str, replacement_line: str, expected_reason: str
) -> None:
    """Refuse a wrong channel, a relative path or None where it is not accepted."""
    ini_file = write_ini_file(tmp_path)
    ini_file.write_text(ini_file.read_text().replace(required_line, replacement_line))
    with pytest.raises(ConfigError, match=expected_reason):
        config.build_config(cli.parse_args(["--config-file", str(ini_file)]))


def test_none_in_the_file_sets_no_value_where_accepted(tmp_path: Path) -> None:
    """Read the literal None as no value in the settings that accept it."""
    app_config = build_from_ini(
        tmp_path,
        LOGGING__backup_count="None",
        PROCESSED__num_workers="None",
    )
    assert app_config.logging.backup_count is None
    assert app_config.processed.num_workers is None


def test_a_number_of_workers_is_read_from_either_source(tmp_path: Path) -> None:
    """Read num_workers from the file, as the command line reads it, which wins."""
    from_file = build_from_ini(tmp_path, PROCESSED__num_workers="+4")
    assert from_file.processed.num_workers == 4
    from_command_line = build_from_ini(
        tmp_path, "--num-workers", "None", PROCESSED__num_workers="4"
    )
    assert from_command_line.processed.num_workers is None


def test_required_settings_given_by_neither_are_named(tmp_path: Path) -> None:
    """Raise MissingSettingsError naming each missing setting, in table order."""
    ini_file = tmp_path / "das.ini"
    ini_file.write_text(REQUIRED_LINES["cd5m5m_path"])
    with pytest.raises(MissingSettingsError) as missing_error:
        config.build_config(cli.parse_args(["--config-file", str(ini_file)]))
    assert str(missing_error.value).endswith(
        "[DAS] rf (--rf), [DAS] steering_path (--steering-path), "
        "[PROCESSED] processed_path (--processed-path), "
        "[PROCESSED] clock_config_file (--clock-config-file), "
        "[LOGGING] log_file (--log-file), [LOGGING] log_level (--log-level)"
    )


@pytest.mark.parametrize("setting_name", ["steering_path", "clock_config_file"])
def test_the_new_required_settings_given_by_neither_are_named(
    tmp_path: Path, setting_name: str
) -> None:
    """Raise MissingSettingsError naming a missing steering or clock setting."""
    ini_file = write_ini_file(tmp_path)
    ini_file.write_text(
        ini_file.read_text().replace(REQUIRED_LINES[setting_name].split("\n")[1], "")
    )
    with pytest.raises(MissingSettingsError, match=f"] {setting_name} "):
        config.build_config(cli.parse_args(["--config-file", str(ini_file)]))


def test_an_entry_the_program_does_not_read_is_refused(tmp_path: Path) -> None:
    """Refuse a file naming an entry no setting reads."""
    with pytest.raises(ConfigError, match=r"\[DAS\] colour"):
        build_from_ini(tmp_path, DAS__colour="red")


def test_a_time_constants_file_is_an_entry_no_setting_reads(tmp_path: Path) -> None:
    """Refuse the time-constants entry, which the clock configuration replaced."""
    with pytest.raises(ConfigError, match=r"\[PROCESSED\] time_constants_file"):
        build_from_ini(tmp_path, PROCESSED__time_constants_file="/etc/tc.yaml")


@pytest.mark.parametrize(
    "config_model", [config.DasConfig, config.ProcessedConfig, config.AppConfig]
)
def test_every_model_refuses_unknown_fields(config_model: type[object]) -> None:
    """Refuse a field the model does not declare."""
    assert isinstance(config_model, type)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config_model.model_validate({"colour": "red"})  # type: ignore[attr-defined]


def test_the_models_are_frozen(tmp_path: Path) -> None:
    """Refuse any change to a built configuration."""
    app_config = build_from_ini(tmp_path)
    with pytest.raises(ValidationError, match="frozen"):
        app_config.das.rf = "b"  # type: ignore[misc]


# ------------------------------------------------- names under processed_path


def test_the_subdirectories_are_named_under_processed_path(tmp_path: Path) -> None:
    """Put meas and ddiff directly under processed_path, and nothing else."""
    processed_config = build_from_ini(tmp_path).processed
    processed_root = Path("/data/processed")
    assert (processed_config.meas_path, processed_config.ddiff_path) == (
        processed_root / "meas",
        processed_root / "ddiff",
    )
    assert not hasattr(config, "FILT_SUBDIRECTORY")


def test_each_channel_has_its_own_lock_file_a_lock_accepts(tmp_path: Path) -> None:
    """Name one plain lock file per channel, beside the subdirectories."""
    lock_names = {config.LOCK_FILE_TEMPLATE.format(rf=rf) for rf in ("a", "b")}
    assert lock_names == {"das_processor_a.lock", "das_processor_b.lock"}
    for lock_name in lock_names:
        assert RunLock(tmp_path, lock_name).path == tmp_path / lock_name
    assert not lock_names & {config.MEAS_SUBDIRECTORY, config.DDIFF_SUBDIRECTORY}


# ------------------------------------------------------------- check_paths


def build_config_with_paths(
    scratch_directory: Path,
    data_directory: Path,
    processed_directory: Path,
    steering_directory: Path | None = None,
    clock_config_file: Path | None = None,
) -> config.AppConfig:
    """Build a configuration naming ``data_directory`` and ``processed_directory``.

    The steering directory is ``data_directory`` unless given, and the clock
    configuration file a readable file made in ``scratch_directory`` unless given.
    """
    if steering_directory is None:
        steering_directory = data_directory
    if clock_config_file is None:
        clock_config_file = scratch_directory / "clocks.yaml"
        clock_config_file.write_text("")
    return config.build_config(
        cli.parse_args(
            [
                "--rf", "a",
                "--cd5m5m-path", str(data_directory),
                "--steering-path", str(steering_directory),
                "--processed-path", str(processed_directory),
                "--clock-config-file", str(clock_config_file),
                "--log-file", "None",
                "--log-level", "None",
            ]
        )
    )  # fmt: skip


def test_usable_paths_pass(tmp_path: Path) -> None:
    """Accept every input path the run can read and a processed path it can write."""
    for directory_name in ("data", "steering", "processed"):
        (tmp_path / directory_name).mkdir()
    config.check_paths(
        build_config_with_paths(
            tmp_path,
            tmp_path / "data",
            tmp_path / "processed",
            steering_directory=tmp_path / "steering",
        )
    )


def test_a_processed_directory_not_yet_made_passes(tmp_path: Path) -> None:
    """Accept a processed_path that is not there yet."""
    config.check_paths(
        build_config_with_paths(tmp_path, tmp_path, tmp_path / "not yet")
    )


@pytest.mark.parametrize("made_as", ["missing", "file"])
def test_a_data_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, made_as: str
) -> None:
    """Refuse a cd5m5m_path that is missing or is a file."""
    data_directory = tmp_path / "data"
    if made_as == "file":
        data_directory.write_text("")
    with pytest.raises(ConfigError, match=r"\[DAS\] cd5m5m_path: .* is not a dir"):
        config.check_paths(
            build_config_with_paths(
                tmp_path, data_directory, tmp_path, steering_directory=tmp_path
            )
        )


@pytest.mark.parametrize("made_as", ["missing", "file"])
def test_a_steering_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, made_as: str
) -> None:
    """Refuse a steering_path that is missing or is a file."""
    steering_directory = tmp_path / "steering"
    if made_as == "file":
        steering_directory.write_text("")
    with pytest.raises(ConfigError, match=r"\[DAS\] steering_path: .* is not a dir"):
        config.check_paths(
            build_config_with_paths(
                tmp_path, tmp_path, tmp_path, steering_directory=steering_directory
            )
        )


def test_a_steering_directory_that_cannot_be_listed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a steering_path whose listing fails, naming the setting."""
    steering_directory = tmp_path / "steering"
    steering_directory.mkdir()
    listed_directories: list[Path] = []
    real_iterdir = Path.iterdir

    def refuse_steering_listing(self: Path) -> object:
        """Fail as listing an unreadable directory fails, for steering only."""
        listed_directories.append(self)
        if self == steering_directory:
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refuse_steering_listing)
    with pytest.raises(
        ConfigError, match=r"\[DAS\] steering_path: .* cannot be listed"
    ):
        config.check_paths(
            build_config_with_paths(
                tmp_path, tmp_path, tmp_path, steering_directory=steering_directory
            )
        )
    assert listed_directories == [tmp_path, steering_directory]


@pytest.mark.parametrize("made_as", ["missing", "directory"])
def test_a_clock_configuration_that_is_not_a_file_is_refused(
    tmp_path: Path, made_as: str
) -> None:
    """Refuse a clock_config_file that is missing or is a directory."""
    clock_config_file = tmp_path / "clocks.yaml"
    if made_as == "directory":
        clock_config_file.mkdir()
    with pytest.raises(
        ConfigError, match=r"\[PROCESSED\] clock_config_file: .* is not a file"
    ):
        config.check_paths(
            build_config_with_paths(
                tmp_path, tmp_path, tmp_path, clock_config_file=clock_config_file
            )
        )


def test_a_clock_configuration_that_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a clock_config_file that cannot be opened for reading."""
    clock_config_file = tmp_path / "clocks.yaml"
    clock_config_file.write_text("")
    opened_files: list[tuple[Path, str]] = []

    def refuse_opening(
        self: Path, open_mode: str = "r", *_args: object, **_kwargs: object
    ) -> None:
        """Fail as opening an unreadable file fails."""
        opened_files.append((self, open_mode))
        raise PermissionError(13, "Permission denied")

    app_config = build_config_with_paths(
        tmp_path, tmp_path, tmp_path, clock_config_file=clock_config_file
    )
    monkeypatch.setattr(Path, "open", refuse_opening)
    with pytest.raises(
        ConfigError,
        match=r"\[PROCESSED\] clock_config_file: .* cannot be read: .*Permission",
    ):
        config.check_paths(app_config)
    assert opened_files == [(clock_config_file, "rb")]


def test_a_data_directory_that_cannot_be_listed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a cd5m5m_path whose listing fails, giving the reason."""

    def refuse_listing(_self: Path) -> object:
        """Fail as listing an unreadable directory fails."""
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", refuse_listing)
    with pytest.raises(
        ConfigError, match=r"cd5m5m_path: .* cannot be listed: .*Permission denied"
    ):
        config.check_paths(
            build_config_with_paths(tmp_path, tmp_path, tmp_path / "not yet")
        )


def test_a_file_in_place_of_the_processed_directory_is_refused(
    tmp_path: Path,
) -> None:
    """Refuse a processed_path that is something other than a directory."""
    (tmp_path / "processed").write_text("")
    with pytest.raises(ConfigError, match="is not a directory to write"):
        config.check_paths(
            build_config_with_paths(tmp_path, tmp_path, tmp_path / "processed")
        )


def test_a_processed_directory_that_cannot_be_written_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a processed directory this process cannot make files in."""
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    access_calls: list[tuple[object, int]] = []

    def deny_access(asked_path: object, access_mode: int) -> bool:
        """Answer as a directory without write permission answers."""
        access_calls.append((asked_path, access_mode))
        return False

    monkeypatch.setattr(os, "access", deny_access)
    with pytest.raises(ConfigError, match="cannot write into"):
        config.check_paths(
            build_config_with_paths(tmp_path, tmp_path, processed_directory)
        )
    assert access_calls == [(processed_directory, os.W_OK | os.X_OK)]


def test_a_processed_directory_refusal_is_word_for_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Say why the processed directory cannot be used, in the program's words."""
    processed_directory = tmp_path / "processed"
    processed_directory.write_text("")
    with pytest.raises(ConfigError) as config_error:
        config.check_paths(
            build_config_with_paths(tmp_path, tmp_path, processed_directory)
        )
    assert str(config_error.value).endswith(
        f"[PROCESSED] processed_path: {processed_directory} is not a directory to write"
        " processed files into"
    )
    processed_directory.unlink()
    processed_directory.mkdir()
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)
    with pytest.raises(ConfigError) as config_error:
        config.check_paths(
            build_config_with_paths(tmp_path, tmp_path, processed_directory)
        )
    assert str(config_error.value).endswith(
        f"[PROCESSED] processed_path: {processed_directory} is a directory this process"
        " cannot write into"
    )


@pytest.mark.parametrize("command_line_entry", ["redo_from_mjd", "steps"])
def test_a_redo_or_a_count_of_epochs_in_the_file_is_refused(
    tmp_path: Path, command_line_entry: str
) -> None:
    """Refuse a file naming either: they are given for one run, on its command line."""
    with pytest.raises(ConfigError, match=command_line_entry):
        build_from_ini(tmp_path, **{f"PROCESSED__{command_line_entry}": "60010"})


def test_the_logging_settings_are_built_alone(tmp_path: Path) -> None:
    """Build the logging settings with other required settings missing."""
    ini_file = tmp_path / "das.ini"
    ini_file.write_text(
        "[LOGGING]\nlog_file = None\nlog_level = INFO\n", encoding="utf-8"
    )
    cli_options = cli.parse_args(["--config-file", str(ini_file)])
    logging_config = config.build_logging_config(cli_options)
    assert (
        logging_config.log_file,
        logging_config.log_level,
        logging_config.backup_count,
    ) == (
        None,
        "INFO",
        None,
    )
    ini_file.write_text(
        "[LOGGING]\nlog_file = run.log\nlog_level = INFO\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"logging: .*path must be absolute"):
        config.build_logging_config(cli_options)
