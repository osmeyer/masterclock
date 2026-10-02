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


def ini(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    """Write an INI file of every required setting plus ``extra`` entries.

    ``extra`` maps ``"SECTION entry"`` to the value written for it.
    """
    sections: dict[str, list[str]] = {}
    for text in REQUIRED_LINES.values():
        header, line = text.splitlines()
        sections.setdefault(header, []).append(line)
    for key, value in (extra or {}).items():
        section, entry = key.split()
        sections.setdefault(f"[{section}]", []).append(f"{entry} = {value}")
    path = tmp_path / "das.ini"
    path.write_text(
        "".join(f"{h}\n" + "".join(f"{e}\n" for e in es) for h, es in sections.items()),
        encoding="utf-8",
    )
    return path


def built(tmp_path: Path, *argv: str, **extra: str) -> config.AppConfig:
    """Build the configuration from an INI file and ``argv``.

    Keyword names are ``SECTION__entry``, written into the file.
    """
    entries = {key.replace("__", " "): value for key, value in extra.items()}
    path = ini(tmp_path, entries)
    return config.build_config(cli.parse_args(["--config-file", str(path), *argv]))


# ------------------------------------------------------------- the table


def test_every_command_line_setting_is_described_once() -> None:
    """Describe every CliOptions field but the three that back no entry."""
    fields = set(cli.CliOptions.model_fields) - {
        "config_file",
        "steps",
        "redo_from_mjd",
    }
    attributes = [setting.options_field for setting in config.SETTINGS]
    assert sorted(attributes) == sorted(fields)


def test_each_setting_is_named_by_a_flag_the_parser_knows() -> None:
    """Name each setting by its flag, its entry and its command-line field alike."""
    flags = cli.build_parser()._option_string_actions
    for setting in config.SETTINGS:
        assert flags[setting.cli_flag].dest == setting.options_field
        assert setting.ini_entry == setting.options_field


def test_the_required_settings_come_first() -> None:
    """List the required settings first, in the order of the help."""
    required = [setting.required for setting in config.SETTINGS]
    assert required == sorted(required, reverse=True)
    assert [entry for _, entry, _ in config.REQUIRED_SETTINGS] == list(REQUIRED_LINES)


def test_none_is_accepted_by_the_settings_whose_options_accept_it() -> None:
    """Accept the literal None in the file exactly where the command line does."""
    accepting = {s.options_field for s in config.SETTINGS if s.allow_none}
    assert accepting == {
        "log_file",
        "log_level",
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
    built_ = built(tmp_path)
    assert built_.das.model_dump() == {
        "rf": "a",
        "cd5m5m_path": Path("/data/cd5m5m"),
        "steering_path": Path("/data/steering"),
    }
    assert built_.processed.model_dump() == {
        "processed_path": Path("/data/processed"),
        "start_from_mjd": cli.START_FROM_MJD,
        "clock_config_file": Path("/etc/clocks.yaml"),
    }
    assert built_.logging.model_dump() == {
        "log_file": Path("/logs/das.log"),
        "log_level": "INFO",
        "backup_count": None,
    }


def test_the_command_line_alone_is_enough() -> None:
    """Build with no file when the command line gives every required setting."""
    built_ = config.build_config(
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
    assert built_.das.rf == "b"
    assert built_.das.steering_path == Path("/s")
    assert built_.processed.clock_config_file == Path("/c.yaml")
    assert built_.logging.log_file is None
    assert built_.logging.log_level is None


def test_the_command_line_wins(tmp_path: Path) -> None:
    """Take the command line's value where both sources give one."""
    built_ = built(
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
    assert built_.das.rf == "b"
    assert built_.processed.start_from_mjd == 60000.0
    assert built_.das.steering_path == Path("/other/steering")
    assert built_.processed.clock_config_file == Path("/other/clocks.yaml")


def test_a_start_given_in_the_file_is_kept(tmp_path: Path) -> None:
    """Use the file's start MJD, read as the command line reads it."""
    built_ = built(tmp_path, PROCESSED__start_from_mjd="6_0010.5")
    assert built_.processed.start_from_mjd == cli.data_mjd("6_0010.5")


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("PROCESSED__start_from_mjd", "None", "invalid positive float value"),
        ("PROCESSED__start_from_mjd", "1", "MJD must be on a day from 50000"),
        ("LOGGING__backup_count", "0", "value must be a positive integer"),
    ],
)
def test_a_bad_value_in_the_file_is_refused(
    tmp_path: Path, key: str, value: str, reason: str
) -> None:
    """Refuse a value the command line would refuse, naming the setting."""
    with pytest.raises(ConfigError, match=reason):
        built(tmp_path, **{key: value})


@pytest.mark.parametrize(
    ("line", "replacement", "reason"),
    [
        ("rf = a", "rf = A", "das.rf"),
        ("rf = a", "rf = None", "das.rf"),
        ("cd5m5m_path = /data/cd5m5m", "cd5m5m_path = data", "path must be absolute"),
        ("cd5m5m_path = /data/cd5m5m", "cd5m5m_path = None", "path must be absolute"),
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
    tmp_path: Path, line: str, replacement: str, reason: str
) -> None:
    """Refuse a wrong channel, a relative path or None where it is not accepted."""
    path = ini(tmp_path)
    path.write_text(path.read_text().replace(line, replacement))
    with pytest.raises(ConfigError, match=reason):
        config.build_config(cli.parse_args(["--config-file", str(path)]))


def test_none_in_the_file_sets_no_value_where_accepted(tmp_path: Path) -> None:
    """Read the literal None as no value in the settings that accept it."""
    built_ = built(
        tmp_path,
        LOGGING__backup_count="None",
    )
    assert built_.logging.backup_count is None


def test_required_settings_given_by_neither_are_named(tmp_path: Path) -> None:
    """Raise MissingSettingsError naming each missing setting, in table order."""
    path = tmp_path / "das.ini"
    path.write_text(REQUIRED_LINES["cd5m5m_path"])
    with pytest.raises(MissingSettingsError) as raised:
        config.build_config(cli.parse_args(["--config-file", str(path)]))
    assert str(raised.value).endswith(
        "[DAS] rf (--rf), [DAS] steering_path (--steering-path), "
        "[PROCESSED] processed_path (--processed-path), "
        "[PROCESSED] clock_config_file (--clock-config-file), "
        "[LOGGING] log_file (--log-file), [LOGGING] log_level (--log-level)"
    )


@pytest.mark.parametrize("name", ["steering_path", "clock_config_file"])
def test_the_new_required_settings_given_by_neither_are_named(
    tmp_path: Path, name: str
) -> None:
    """Raise MissingSettingsError naming a missing steering or clock setting."""
    path = ini(tmp_path)
    path.write_text(path.read_text().replace(REQUIRED_LINES[name].split("\n")[1], ""))
    with pytest.raises(MissingSettingsError, match=f"] {name} "):
        config.build_config(cli.parse_args(["--config-file", str(path)]))


def test_an_entry_the_program_does_not_read_is_refused(tmp_path: Path) -> None:
    """Refuse a file naming an entry no setting reads."""
    with pytest.raises(ConfigError, match=r"\[DAS\] colour"):
        built(tmp_path, DAS__colour="red")


def test_a_time_constants_file_is_an_entry_no_setting_reads(tmp_path: Path) -> None:
    """Refuse the time-constants entry, which the clock configuration replaced."""
    with pytest.raises(ConfigError, match=r"\[PROCESSED\] time_constants_file"):
        built(tmp_path, PROCESSED__time_constants_file="/etc/tc.yaml")


@pytest.mark.parametrize(
    "model", [config.DasConfig, config.ProcessedConfig, config.AppConfig]
)
def test_every_model_refuses_unknown_fields(model: type[object]) -> None:
    """Refuse a field the model does not declare."""
    assert isinstance(model, type)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({"colour": "red"})  # type: ignore[attr-defined]


def test_the_models_are_frozen(tmp_path: Path) -> None:
    """Refuse any change to a built configuration."""
    built_ = built(tmp_path)
    with pytest.raises(ValidationError, match="frozen"):
        built_.das.rf = "b"  # type: ignore[misc]


# ------------------------------------------------- names under processed_path


def test_the_subdirectories_are_named_under_processed_path(tmp_path: Path) -> None:
    """Put meas and ddiff directly under processed_path, and nothing else."""
    processed = built(tmp_path).processed
    root = Path("/data/processed")
    assert (processed.meas_path, processed.ddiff_path) == (
        root / "meas",
        root / "ddiff",
    )
    assert not hasattr(config, "FILT_SUBDIRECTORY")


def test_each_channel_has_its_own_lock_file_a_lock_accepts(tmp_path: Path) -> None:
    """Name one plain lock file per channel, beside the subdirectories."""
    names = {config.LOCK_FILE_TEMPLATE.format(rf=rf) for rf in ("a", "b")}
    assert names == {"das_processor_a.lock", "das_processor_b.lock"}
    for name in names:
        assert RunLock(tmp_path, name).path == tmp_path / name
    assert not names & {config.MEAS_SUBDIRECTORY, config.DDIFF_SUBDIRECTORY}


# ------------------------------------------------------------- check_paths


def configured(
    root: Path,
    data: Path,
    processed: Path,
    steering: Path | None = None,
    clocks: Path | None = None,
) -> config.AppConfig:
    """Build a configuration naming ``data`` and ``processed``.

    The steering directory is ``data`` unless given, and the clock
    configuration file a readable file made in ``root`` unless given.
    """
    if clocks is None:
        clocks = root / "clocks.yaml"
        clocks.write_text("")
    return config.build_config(
        cli.parse_args(
            [
                "--rf", "a",
                "--cd5m5m-path", str(data),
                "--steering-path", str(data if steering is None else steering),
                "--processed-path", str(processed),
                "--clock-config-file", str(clocks),
                "--log-file", "None",
                "--log-level", "None",
            ]
        )
    )  # fmt: skip


def test_usable_paths_pass(tmp_path: Path) -> None:
    """Accept every input path the run can read and a processed path it can write."""
    for name in ("data", "steering", "processed"):
        (tmp_path / name).mkdir()
    config.check_paths(
        configured(
            tmp_path,
            tmp_path / "data",
            tmp_path / "processed",
            steering=tmp_path / "steering",
        )
    )


def test_a_processed_directory_not_yet_made_passes(tmp_path: Path) -> None:
    """Accept a processed_path that is not there yet."""
    config.check_paths(configured(tmp_path, tmp_path, tmp_path / "not yet"))


@pytest.mark.parametrize("make", ["missing", "file"])
def test_a_data_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, make: str
) -> None:
    """Refuse a cd5m5m_path that is missing or is a file."""
    data = tmp_path / "data"
    if make == "file":
        data.write_text("")
    with pytest.raises(ConfigError, match=r"\[DAS\] cd5m5m_path: .* is not a dir"):
        config.check_paths(configured(tmp_path, data, tmp_path, steering=tmp_path))


@pytest.mark.parametrize("make", ["missing", "file"])
def test_a_steering_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, make: str
) -> None:
    """Refuse a steering_path that is missing or is a file."""
    steering = tmp_path / "steering"
    if make == "file":
        steering.write_text("")
    with pytest.raises(ConfigError, match=r"\[DAS\] steering_path: .* is not a dir"):
        config.check_paths(configured(tmp_path, tmp_path, tmp_path, steering=steering))


def test_a_steering_directory_that_cannot_be_listed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a steering_path whose listing fails, naming the setting."""
    steering = tmp_path / "steering"
    steering.mkdir()
    listed: list[Path] = []
    iterdir = Path.iterdir

    def refuse(self: Path) -> object:
        """Fail as listing an unreadable directory fails, for steering only."""
        listed.append(self)
        if self == steering:
            raise PermissionError(13, "Permission denied")
        return iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refuse)
    with pytest.raises(
        ConfigError, match=r"\[DAS\] steering_path: .* cannot be listed"
    ):
        config.check_paths(configured(tmp_path, tmp_path, tmp_path, steering=steering))
    assert listed == [tmp_path, steering]


@pytest.mark.parametrize("make", ["missing", "directory"])
def test_a_clock_configuration_that_is_not_a_file_is_refused(
    tmp_path: Path, make: str
) -> None:
    """Refuse a clock_config_file that is missing or is a directory."""
    clocks = tmp_path / "clocks.yaml"
    if make == "directory":
        clocks.mkdir()
    with pytest.raises(
        ConfigError, match=r"\[PROCESSED\] clock_config_file: .* is not a file"
    ):
        config.check_paths(configured(tmp_path, tmp_path, tmp_path, clocks=clocks))


def test_a_clock_configuration_that_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a clock_config_file that cannot be opened for reading."""
    clocks = tmp_path / "clocks.yaml"
    clocks.write_text("")
    opened: list[tuple[Path, str]] = []

    def refuse(self: Path, mode: str = "r", *_args: object, **_kwargs: object) -> None:
        """Fail as opening an unreadable file fails."""
        opened.append((self, mode))
        raise PermissionError(13, "Permission denied")

    settings = configured(tmp_path, tmp_path, tmp_path, clocks=clocks)
    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(
        ConfigError,
        match=r"\[PROCESSED\] clock_config_file: .* cannot be read: .*Permission",
    ):
        config.check_paths(settings)
    assert opened == [(clocks, "rb")]


def test_a_data_directory_that_cannot_be_listed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a cd5m5m_path whose listing fails, giving the reason."""

    def refuse(_self: Path) -> object:
        """Fail as listing an unreadable directory fails."""
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", refuse)
    with pytest.raises(
        ConfigError, match=r"cd5m5m_path: .* cannot be listed: .*Permission denied"
    ):
        config.check_paths(configured(tmp_path, tmp_path, tmp_path / "not yet"))


def test_a_file_in_place_of_the_processed_directory_is_refused(
    tmp_path: Path,
) -> None:
    """Refuse a processed_path that is something other than a directory."""
    (tmp_path / "processed").write_text("")
    with pytest.raises(ConfigError, match="is not a directory to write"):
        config.check_paths(configured(tmp_path, tmp_path, tmp_path / "processed"))


def test_a_processed_directory_that_cannot_be_written_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a processed directory this process cannot make files in."""
    processed = tmp_path / "processed"
    processed.mkdir()
    asked: list[tuple[object, int]] = []

    def access(path: object, mode: int) -> bool:
        """Answer as a directory without write permission answers."""
        asked.append((path, mode))
        return False

    monkeypatch.setattr(os, "access", access)
    with pytest.raises(ConfigError, match="cannot write into"):
        config.check_paths(configured(tmp_path, tmp_path, processed))
    assert asked == [(processed, os.W_OK | os.X_OK)]


def test_a_processed_directory_refusal_is_word_for_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Say why the processed directory cannot be used, in the program's words."""
    processed = tmp_path / "processed"
    processed.write_text("")
    with pytest.raises(ConfigError) as raised:
        config.check_paths(configured(tmp_path, tmp_path, processed))
    assert str(raised.value).endswith(
        f"[PROCESSED] processed_path: {processed} is not a directory to write"
        " processed files into"
    )
    processed.unlink()
    processed.mkdir()
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)
    with pytest.raises(ConfigError) as raised:
        config.check_paths(configured(tmp_path, tmp_path, processed))
    assert str(raised.value).endswith(
        f"[PROCESSED] processed_path: {processed} is a directory this process"
        " cannot write into"
    )


@pytest.mark.parametrize("entry", ["redo_from_mjd", "steps"])
def test_a_redo_or_a_count_of_epochs_in_the_file_is_refused(
    tmp_path: Path, entry: str
) -> None:
    """Refuse a file naming either: they are given for one run, on its command line."""
    with pytest.raises(ConfigError, match=entry):
        built(tmp_path, **{f"PROCESSED__{entry}": "60010"})


def test_the_logging_settings_are_built_alone(tmp_path: Path) -> None:
    """Build the logging settings with other required settings missing."""
    path = tmp_path / "das.ini"
    path.write_text("[LOGGING]\nlog_file = None\nlog_level = INFO\n", encoding="utf-8")
    options = cli.parse_args(["--config-file", str(path)])
    built_ = config.build_logging_config(options)
    assert (built_.log_file, built_.log_level, built_.backup_count) == (
        None,
        "INFO",
        None,
    )
    path.write_text(
        "[LOGGING]\nlog_file = run.log\nlog_level = INFO\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"logging: .*path must be absolute"):
        config.build_logging_config(options)
