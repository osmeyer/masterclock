"""Tests for src/masterclock/das_processor/config.py.

The rules covered: every setting of the command line is described once,
required ones first, under the flag the parser knows; the merged settings
are validated with the command line winning, text from the file read as the
command line reads it, paths absolute, MJDs on a day a data file covers,
the literal None only where a setting accepts it, and unknown fields
refused; a start MJD given by neither source is 59500; a required setting
given by neither source is a MissingSettingsError; the processed
subdirectories and the lock file are named from processed_path and the
channel; and check_paths refuses a data directory that is not one or cannot
be listed, and a processed directory that is something else or cannot be
written into.
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
    "processed_path": "[PROCESSED]\nprocessed_path = /data/processed\n",
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
    """Describe every CliOptions field but the two that back no entry."""
    fields = set(cli.CliOptions.model_fields) - {"config_file", "steps"}
    attributes = [setting.attribute for setting in config.SETTINGS]
    assert sorted(attributes) == sorted(fields)


def test_each_setting_is_named_by_a_flag_the_parser_knows() -> None:
    """Name each setting by its flag, its entry and its command-line field alike."""
    flags = cli.build_parser()._option_string_actions
    for setting in config.SETTINGS:
        assert flags[setting.flag].dest == setting.attribute
        assert setting.entry == setting.attribute


def test_the_required_settings_come_first() -> None:
    """List the required settings first, in the order of the help."""
    required = [setting.required for setting in config.SETTINGS]
    assert required == sorted(required, reverse=True)
    assert [entry for _, entry, _ in config.REQUIRED_SETTINGS] == list(REQUIRED_LINES)


def test_none_is_accepted_by_the_settings_whose_options_accept_it() -> None:
    """Accept the literal None in the file exactly where the command line does."""
    accepting = {s.attribute for s in config.SETTINGS if s.allow_none}
    assert accepting == {
        "log_file",
        "log_level",
        "redo_from_mjd",
        "clock_config_file",
        "time_constants_file",
        "backup_count",
    }
    for setting in config.SETTINGS:
        if setting.allow_none:
            assert cli.parse_args([setting.flag, "None"]) is not None
        else:
            with pytest.raises(SystemExit):
                cli.parse_args([setting.flag, "None"])


def test_the_sections_and_entries_come_from_the_table() -> None:
    """Read exactly the sections and entries the table names."""
    assert {"DAS", "PROCESSED", "LOGGING"} == config.KNOWN_SECTIONS
    assert config.KNOWN_ENTRIES["DAS"] == {"rf", "cd5m5m_path"}


# ------------------------------------------------------------- building


def test_a_file_of_the_required_settings_builds(tmp_path: Path) -> None:
    """Build from the file alone, the optional settings None and the start 59500."""
    built_ = built(tmp_path)
    assert built_.das.model_dump() == {"rf": "a", "cd5m5m_path": Path("/data/cd5m5m")}
    assert built_.processed.model_dump() == {
        "processed_path": Path("/data/processed"),
        "redo_from_mjd": None,
        "start_from_mjd": cli.START_FROM_MJD,
        "clock_config_file": None,
        "time_constants_file": None,
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
                "--processed-path", "/p",
                "--log-file", "None",
                "--log-level", "None",
            ]
        )
    )  # fmt: skip
    assert built_.das.rf == "b"
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
        PROCESSED__start_from_mjd="61000",
    )
    assert built_.das.rf == "b"
    assert built_.processed.start_from_mjd == 60000.0


def test_a_start_given_in_the_file_is_kept(tmp_path: Path) -> None:
    """Use the file's start MJD, read as the command line reads it."""
    built_ = built(tmp_path, PROCESSED__start_from_mjd="6_0010.5")
    assert built_.processed.start_from_mjd == cli.data_mjd("6_0010.5")


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("PROCESSED__start_from_mjd", "None", "invalid positive float value"),
        ("PROCESSED__start_from_mjd", "1", "MJD must be on a day from 50000"),
        ("PROCESSED__redo_from_mjd", "100000", "MJD must be on a day from 50000"),
        ("PROCESSED__clock_config_file", "clocks.yaml", "path must be absolute"),
        ("PROCESSED__time_constants_file", "tc.yaml", "path must be absolute"),
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
        PROCESSED__redo_from_mjd="None",
        PROCESSED__clock_config_file="None",
        LOGGING__backup_count="None",
    )
    assert built_.processed.redo_from_mjd is None
    assert built_.processed.clock_config_file is None
    assert built_.logging.backup_count is None


def test_required_settings_given_by_neither_are_named(tmp_path: Path) -> None:
    """Raise MissingSettingsError naming each missing setting, in table order."""
    path = tmp_path / "das.ini"
    path.write_text(REQUIRED_LINES["cd5m5m_path"])
    with pytest.raises(MissingSettingsError) as raised:
        config.build_config(cli.parse_args(["--config-file", str(path)]))
    assert str(raised.value).endswith(
        "[DAS] rf (--rf), [PROCESSED] processed_path (--processed-path), "
        "[LOGGING] log_file (--log-file), [LOGGING] log_level (--log-level)"
    )


def test_an_entry_the_program_does_not_read_is_refused(tmp_path: Path) -> None:
    """Refuse a file naming an entry no setting reads."""
    with pytest.raises(ConfigError, match=r"\[DAS\] colour"):
        built(tmp_path, DAS__colour="red")


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
    """Put meas, dd and filt directly under processed_path."""
    processed = built(tmp_path).processed
    root = Path("/data/processed")
    assert (processed.meas_path, processed.dd_path, processed.filt_path) == (
        root / "meas",
        root / "dd",
        root / "filt",
    )


def test_each_channel_has_its_own_lock_file_a_lock_accepts(tmp_path: Path) -> None:
    """Name one plain lock file per channel, beside the subdirectories."""
    names = {config.LOCK_FILE_TEMPLATE.format(rf=rf) for rf in ("a", "b")}
    assert names == {"das_processor_a.lock", "das_processor_b.lock"}
    for name in names:
        assert RunLock(tmp_path, name).path == tmp_path / name
    assert not names & {
        config.MEAS_SUBDIRECTORY,
        config.DD_SUBDIRECTORY,
        config.FILT_SUBDIRECTORY,
    }


# ------------------------------------------------------------- check_paths


def configured(data: Path, processed: Path) -> config.AppConfig:
    """Build a configuration naming ``data`` and ``processed``."""
    return config.build_config(
        cli.parse_args(
            [
                "--rf", "a",
                "--cd5m5m-path", str(data),
                "--processed-path", str(processed),
                "--log-file", "None",
                "--log-level", "None",
            ]
        )
    )  # fmt: skip


def test_usable_paths_pass(tmp_path: Path) -> None:
    """Accept a listable data directory and a writable processed directory."""
    (tmp_path / "data").mkdir()
    (tmp_path / "processed").mkdir()
    config.check_paths(configured(tmp_path / "data", tmp_path / "processed"))


def test_a_processed_directory_not_yet_made_passes(tmp_path: Path) -> None:
    """Accept a processed_path that is not there yet."""
    config.check_paths(configured(tmp_path, tmp_path / "not yet"))


@pytest.mark.parametrize("make", ["missing", "file"])
def test_a_data_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, make: str
) -> None:
    """Refuse a cd5m5m_path that is missing or is a file."""
    data = tmp_path / "data"
    if make == "file":
        data.write_text("")
    with pytest.raises(ConfigError, match=r"\[DAS\] cd5m5m_path: .* is not a dir"):
        config.check_paths(configured(data, tmp_path))


def test_a_data_directory_that_cannot_be_listed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a cd5m5m_path whose listing fails, giving the reason."""

    def refuse(_self: Path) -> object:
        """Fail as listing an unreadable directory fails."""
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", refuse)
    with pytest.raises(ConfigError, match=r"cannot be listed: .*Permission denied"):
        config.check_paths(configured(tmp_path, tmp_path / "not yet"))


def test_a_file_in_place_of_the_processed_directory_is_refused(
    tmp_path: Path,
) -> None:
    """Refuse a processed_path that is something other than a directory."""
    (tmp_path / "processed").write_text("")
    with pytest.raises(ConfigError, match="is not a directory to write"):
        config.check_paths(configured(tmp_path, tmp_path / "processed"))


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
        config.check_paths(configured(tmp_path, processed))
    assert asked == [(processed, os.W_OK | os.X_OK)]
