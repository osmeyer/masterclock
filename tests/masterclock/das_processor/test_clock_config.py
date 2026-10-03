"""Tests for src/masterclock/das_processor/clock_config.py.

The rules covered: the clock configuration is read from YAML with a safe
loader that refuses a key repeated in any mapping, into frozen models that
refuse unknown keys; a clock's entry at a mark starts from the type default
of its first entry, or that entry's own settings when it gives no type and
every setting from the start, and applies, in order of effective_mjd, every
entry in force at the mark, an entry without effective_mjd from the start,
and one
between two marks from the next mark; every check of design 15.2 refuses
the file with ConfigError; the RMS limit of a pair is its own, else its
reference's, else the default; a series takes the settings of its clock
side, a pair the RMS limit too; and the committed example file loads.

Every refusal names the file, then the problem, a merge or repeated key with
its line and column, and is logged as raised; undated entries come first
wherever listed; and rejects_before_restart may equal the gap limit.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.exceptions import ConfigError
from masterclock.das_processor import clock_config
from masterclock.domain.series import SeriesKey, SeriesParams

BASE_YAML: Final = (
    "rejects_before_restart: 36\n"
    "rms_limit:\n"
    "  default: 50\n"
    "  references: {mc2: 40}\n"
    "  pairs: {mc2.nav23: 80}\n"
    "types:\n"
    "  maser: {filter_states: 3, time_constant: 100.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 5.0, gap_limit: 432}\n"
    "  cesium: {filter_states: 2, time_constant: 30.0, scale_time_constant: 50.0,"
    " initial_innovation_scale: 4.0, gap_limit: 432}\n"
    "  mc: {filter_states: 1, scale_time_constant: 50.0,"
    " initial_innovation_scale: 3.0, gap_limit: 432}\n"
    "clocks:\n"
    "  mc1: [{type: mc}]\n"
    "  mc2: [{type: mc}]\n"
    "  cs7: [{type: cesium}]\n"
    "  nav23:\n"
    "    - {type: maser}\n"
    "    - {effective_mjd: 60980.0, time_constant: 150.0}\n"
)
"""An invented clock configuration of the design's shape."""

MARK_BEFORE_MJD_60980: Final = datetime(2025, 10, 31, 23, 50, tzinfo=UTC)
"""The mark just before MJD 60980 begins."""

MJD_60980_START: Final = datetime(2025, 11, 1, 0, 0, tzinfo=UTC)
"""The mark at which MJD 60980 begins."""


def read_config_text(
    tmp_path: Path, yaml_text: str = BASE_YAML
) -> clock_config.ClockConfig:
    """Write ``yaml_text`` as a clock configuration file and read it."""
    config_file = tmp_path / "clock_config.yaml"
    config_file.write_text(yaml_text, encoding="utf-8")
    return clock_config.read_clock_config(config_file)


def assert_refused(tmp_path: Path, yaml_text: str, error_pattern: str) -> None:
    """Check that ``yaml_text`` is refused with a ConfigError matching the pattern."""
    with pytest.raises(ConfigError, match=error_pattern):
        read_config_text(tmp_path, yaml_text)


# ------------------------------------------------------------------- entries


def test_an_entry_starts_from_its_type_default(tmp_path: Path) -> None:
    """Give a clock with one entry its type's values."""
    clock_entry = read_config_text(tmp_path).entry_for("cs7", MJD_60980_START)
    assert clock_entry == clock_config.ClockEntry(
        filter_states=2,
        time_constant=30.0,
        scale_time_constant=50.0,
        initial_innovation_scale=4.0,
        gap_limit=432,
    )


def test_a_later_entry_takes_effect_at_its_mjd(tmp_path: Path) -> None:
    """Apply an entry from the mark its effective_mjd falls on, not before."""
    loaded_config = read_config_text(tmp_path)
    assert (
        loaded_config.entry_for("nav23", MARK_BEFORE_MJD_60980).time_constant == 100.0
    )
    assert loaded_config.entry_for("nav23", MJD_60980_START).time_constant == 150.0
    assert loaded_config.entry_for("nav23", MJD_60980_START).scale_time_constant == 50.0


def test_an_effective_mjd_between_two_marks_takes_effect_at_the_next(
    tmp_path: Path,
) -> None:
    """Apply an entry dated between two marks from the second of them."""
    yaml_text = BASE_YAML.replace("effective_mjd: 60980.0", "effective_mjd: 60980.003")
    loaded_config = read_config_text(tmp_path, yaml_text)
    assert loaded_config.entry_for("nav23", MJD_60980_START).time_constant == 100.0
    assert (
        loaded_config.entry_for(
            "nav23", MJD_60980_START + timedelta(minutes=10)
        ).time_constant
        == 150.0
    )


def test_entries_apply_in_order_of_effective_mjd(tmp_path: Path) -> None:
    """Apply entries by their dates, whatever order the file lists them in."""
    yaml_text = BASE_YAML.replace(
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n",
        "    - {effective_mjd: 60990.0, time_constant: 200.0}\n"
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n",
    )
    loaded_config = read_config_text(tmp_path, yaml_text)
    late_mark = datetime(2025, 11, 20, tzinfo=UTC)
    assert loaded_config.entry_for("nav23", MJD_60980_START).time_constant == 150.0
    assert loaded_config.entry_for("nav23", late_mark).time_constant == 200.0


def test_an_entry_without_a_date_applies_from_the_start(tmp_path: Path) -> None:
    """Apply an undated entry at every mark, after the type default."""
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", "  cs7: [{type: cesium, gap_limit: 300}]"
    )
    assert (
        read_config_text(tmp_path, yaml_text)
        .entry_for("cs7", MARK_BEFORE_MJD_60980)
        .gap_limit
        == 300
    )


def test_a_mark_without_a_timezone_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for a naive mark, which names no one instant."""
    with pytest.raises(ConfigError, match="no timezone"):
        read_config_text(tmp_path).entry_for(
            "nav23", MJD_60980_START.replace(tzinfo=None)
        )


def test_an_unknown_clock_has_no_entry(tmp_path: Path) -> None:
    """Raise ConfigError for a clock the file does not name."""
    with pytest.raises(ConfigError, match="no entry for clock hp9"):
        read_config_text(tmp_path).entry_for("hp9", MJD_60980_START)


# ------------------------------------------------------------------ the checks


@pytest.mark.parametrize(
    ("yaml_old", "yaml_new", "error_pattern"),
    [
        ("  cs7: [{type: cesium}]", "  cs7: []", "cs7 has no entry"),
        ("  mc1: [{type: mc}]", "  mc1: [{type: cesium}]", "mc1 .* type mc"),
        (
            "    - {effective_mjd: 60980.0, time_constant: 150.0}",
            "    - {effective_mjd: 60980.0, filter_states: 2}",
            "changes the filter_states of nav23",
        ),
        ("filter_states: 3,", "filter_states: 4,", "filter_states"),
        ("filter_states: 3,", "filter_states: true,", "filter_states.*not a bool"),
        ("time_constant: 30.0,", "", "cesium.*time constant"),
        ("time_constant: 30.0,", "time_constant: 0.5,", "time_constant"),
        (
            "  mc: {filter_states: 1,",
            "  mc: {filter_states: 1, time_constant: 10.0,",
            "mc.*time constant",
        ),
        (
            "scale_time_constant: 50.0, initial_innovation_scale: 4.0",
            "scale_time_constant: 0.5, initial_innovation_scale: 4.0",
            "scale_time_constant",
        ),
        (
            "initial_innovation_scale: 4.0, gap_limit: 432",
            "initial_innovation_scale: 4.0, gap_limit: 0",
            "gap_limit",
        ),
        (
            "initial_innovation_scale: 4.0",
            "initial_innovation_scale: 0.0",
            "initial_innovation_scale",
        ),
        (
            "initial_innovation_scale: 4.0",
            "initial_innovation_scale: .nan",
            "initial_innovation_scale",
        ),
        (
            "rejects_before_restart: 36",
            "rejects_before_restart: 2",
            "rejects_before_restart",
        ),
        (
            "  cs7: [{type: cesium}]",
            "  cs7: [{type: cesium, gap_limit: 30}]",
            "above the gap_limit 30 of cs7",
        ),
        (
            "  nav23:\n    - {type: maser}\n",
            "  nav23:\n    - {type: maser}\n"
            "    - {effective_mjd: 60970.0, gap_limit: 20}\n",
            "above the gap_limit 20 of nav23",
        ),
        ("default: 50", "default: 0", "default"),
        ("default: 50", "default: 50.0", "default"),
        ("references: {mc2: 40}", "references: {mc2: -4}", "references"),
        ("pairs: {mc2.nav23: 80}", "pairs: {mc2.nav23: 8.5}", "pairs"),
        ("pairs: {mc2.nav23: 80}", "pairs: {nav23: 80}", "pair nav23"),
        ("references: {mc2: 40}", "references: {nav2: 40}", "reference nav2"),
        ("  cs7: [{type: cesium}]", "  cs7: [{type: rubidium}]", "type rubidium"),
        (
            "  cs7: [{type: cesium}]",
            "  cs7: [{gap_limit: 300}]",
            "first entry of cs7 .* type",
        ),
        (
            "    - {effective_mjd: 60980.0, time_constant: 150.0}",
            "    - {effective_mjd: 60980.0, type: cesium}",
            "type of nav23 .* first entry",
        ),
        ("effective_mjd: 60980.0", "effective_mjd: 40000.0", "effective_mjd"),
        (
            "rejects_before_restart: 36",
            "rejects_before_restart: 36\ncolour: red",
            "colour",
        ),
        ("  cs7: [{type: cesium}]", "  cs7: [{type: cesium, colour: red}]", "colour"),
        ("  mc: {filter_states: 1,", "  mc: {colour: red, filter_states: 1,", "colour"),
        ("  default: 50", "  default: 50\n  colour: 3", "colour"),
        ("rejects_before_restart: 36\n", "", "rejects_before_restart"),
    ],
)
def test_every_check_refuses_the_file(
    tmp_path: Path, yaml_old: str, yaml_new: str, error_pattern: str
) -> None:
    """Raise ConfigError for each item of the design's list (15.2, U14 load part)."""
    assert yaml_old in BASE_YAML
    assert_refused(tmp_path, BASE_YAML.replace(yaml_old, yaml_new), error_pattern)


@pytest.mark.parametrize(
    ("yaml_old", "yaml_new"),
    [
        (
            "rejects_before_restart: 36\n",
            "rejects_before_restart: 36\nrejects_before_restart: 37\n",
        ),
        ("  default: 50\n", "  default: 50\n  default: 60\n"),
        ("  mc1: [{type: mc}]\n", "  mc1: [{type: mc}]\n  mc1: [{type: mc}]\n"),
        ("  cs7: [{type: cesium}]", "  cs7: [{type: cesium, type: cesium}]"),
        ("references: {mc2: 40}", "references: {mc2: 40, mc2: 41}"),
    ],
)
def test_a_repeated_key_is_refused_at_any_level(
    tmp_path: Path, yaml_old: str, yaml_new: str
) -> None:
    """Refuse a key that a mapping holds twice, however deep (req 8)."""
    assert_refused(tmp_path, BASE_YAML.replace(yaml_old, yaml_new), "repeated key")


def test_a_merge_key_is_refused(tmp_path: Path) -> None:
    """Refuse a YAML merge key, which would hide where a value came from."""
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", "  cs7: [{<<: {type: cesium}}]"
    )
    assert_refused(tmp_path, yaml_text, "merge key")


@pytest.mark.parametrize(
    "yaml_text",
    ["[1, 2]\n", "rejects_before_restart: [\n", "!!python/object:os.system {}\n", ""],
)
def test_a_file_that_is_not_a_configuration_is_refused(
    tmp_path: Path, yaml_text: str
) -> None:
    """Refuse a file that does not parse, or is not a mapping, safely."""
    assert_refused(tmp_path, yaml_text, "clock configuration")


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for a file that cannot be read."""
    with pytest.raises(ConfigError, match="cannot read"):
        clock_config.read_clock_config(tmp_path / "missing.yaml")


def test_a_file_that_is_not_utf_8_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for bytes that are not UTF-8."""
    config_file = tmp_path / "clock_config.yaml"
    config_file.write_bytes(b"rejects_before_restart: \xff\n")
    with pytest.raises(ConfigError, match="cannot read"):
        clock_config.read_clock_config(config_file)


# ----------------------------------------------------------------- RMS limit


@pytest.mark.parametrize(
    ("pair", "expected_limit"),
    [
        (("mc2", "nav23"), 80),
        (("mc2", "cs7"), 40),
        (("mc1", "cs7"), 50),
        (("mc1", "mc2"), 50),
    ],
)
def test_the_rms_limit_is_the_pair_s_else_the_reference_s_else_the_default(
    tmp_path: Path, pair: tuple[str, str], expected_limit: int
) -> None:
    """Give the pair's own limit, else its reference's, else the default."""
    assert read_config_text(tmp_path).rms_limit(pair) == expected_limit


def test_the_rms_limits_need_only_a_default(tmp_path: Path) -> None:
    """Read an rms_limit section that gives only the default."""
    yaml_text = BASE_YAML.replace(
        "  references: {mc2: 40}\n  pairs: {mc2.nav23: 80}\n", ""
    )
    assert read_config_text(tmp_path, yaml_text).rms_limit(("mc2", "nav23")) == 50


# ---------------------------------------------------------------- series


def test_a_pair_takes_its_second_clock_s_settings(tmp_path: Path) -> None:
    """Give a pair the entry of its clock b and its RMS limit (8.1)."""
    assert read_config_text(tmp_path).params_for(
        ("mc2", "nav23"), MJD_60980_START
    ) == SeriesParams(
        filter_states=3,
        M=150.0,
        M_sigma=50.0,
        sigma0=5.0,
        gmax=432,
        n_break=36,
        rms_max=80,
    )


def test_a_self_or_link_pair_takes_the_reference_s_settings(tmp_path: Path) -> None:
    """Give (r, r) and (r, s) the reference's 1-state entry."""
    series_params = read_config_text(tmp_path).params_for(
        ("mc1", "mc2"), MJD_60980_START
    )
    assert (
        series_params.filter_states,
        series_params.M,
        series_params.sigma0,
        series_params.rms_max,
    ) == (
        1,
        None,
        3.0,
        50,
    )


def test_a_triple_takes_its_clock_s_settings_and_no_rms_limit(tmp_path: Path) -> None:
    """Give (r, s, c) the entry of c and no RMS limit (8.1, 12.5)."""
    series_params = read_config_text(tmp_path).params_for(
        ("mc1", "mc2", "nav23"), MARK_BEFORE_MJD_60980
    )
    assert (series_params.filter_states, series_params.M, series_params.rms_max) == (
        3,
        100.0,
        None,
    )


def test_every_series_gets_its_settings_with_each_clock_looked_up_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give each series what params_for gives it, each clock's entry settled once."""
    loaded_config = read_config_text(tmp_path)
    series_keys: list[SeriesKey] = [
        ("mc1", "mc1"),
        ("mc1", "mc2"),
        ("mc2", "nav23"),
        ("mc1", "mc2", "nav23"),
        ("mc2", "mc2", "nav23"),
    ]
    expected = {
        series_key: loaded_config.params_for(series_key, MJD_60980_START)
        for series_key in series_keys
    }
    looked_up: list[str] = []
    entry_for = clock_config.ClockConfig.entry_for

    def counting_entry_for(
        self: clock_config.ClockConfig, clock: str, epoch_start: datetime
    ) -> clock_config.ClockEntry:
        """Note each clock looked up, then look it up."""
        looked_up.append(clock)
        return entry_for(self, clock, epoch_start)

    monkeypatch.setattr(clock_config.ClockConfig, "entry_for", counting_entry_for)
    assert loaded_config.params_for_series(series_keys, MJD_60980_START) == expected
    assert sorted(looked_up) == ["mc1", "mc2", "nav23"]


# ------------------------------------------------------------------ example


def test_the_example_file_loads() -> None:
    """Read the committed example and find every clock it names usable."""
    example_file = Path(__file__).parents[3] / "etc" / "clock_config.yaml.example"
    loaded_config = clock_config.read_clock_config(example_file)
    epoch_start = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
    for clock_name in loaded_config.clocks:
        assert (
            loaded_config.entry_for(clock_name, epoch_start).gap_limit
            >= loaded_config.rejects_before_restart
        )


# ------------------------------------------------- what a person reads, exactly


def refusal_message(tmp_path: Path, yaml_text: str) -> str:
    """Give the message a file holding ``yaml_text`` is refused with."""
    with pytest.raises(ConfigError) as raised:
        read_config_text(tmp_path, yaml_text)
    return str(raised.value)


def test_a_refusal_names_the_file_and_the_problem(tmp_path: Path) -> None:
    """Start every refusal with the file, then say what is wrong (15.2)."""
    config_file = tmp_path / "clock_config.yaml"
    assert refusal_message(tmp_path, "[1, 2]\n") == (
        f"clock configuration {config_file}: is not a mapping of the sections"
    )
    assert refusal_message(tmp_path, "rejects_before_restart: [\n").startswith(
        f"clock configuration {config_file}: while parsing"
    )
    assert refusal_message(tmp_path, "rejects_before_restart: x\n").startswith(
        f"clock configuration {config_file}: rejects_before_restart: Input should be"
    )
    with pytest.raises(ConfigError) as raised:
        clock_config.read_clock_config(tmp_path / "missing.yaml")
    assert str(raised.value).startswith(
        f"clock configuration {tmp_path / 'missing.yaml'}: cannot read: "
    )


def test_a_merge_or_repeated_key_is_named_where_it_stands(tmp_path: Path) -> None:
    """Say which line and column of the file hold the key refused."""
    merge_refusal = refusal_message(tmp_path, "a: 1\nb: &x {k: 1}\nc: {<<: *x}\n")
    assert ": merge keys are not read\\n" in merge_refusal
    assert "line 3, column 5" in merge_refusal
    repeated_refusal = refusal_message(tmp_path, "a: 1\na: 2\n")
    assert ": repeated key 'a'\\n" in repeated_refusal
    assert "line 2, column 1" in repeated_refusal


def test_every_refusal_is_logged_as_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each ConfigError at ERROR in the words it is raised with."""
    refusal_text = refusal_message(tmp_path, "[1, 2]\n")
    assert [log_record.getMessage() for log_record in caplog.records] == [refusal_text]
    caplog.clear()
    with pytest.raises(ConfigError) as raised:
        read_config_text(tmp_path).entry_for(
            "nav23", MJD_60980_START.replace(tzinfo=None)
        )
    assert [log_record.getMessage() for log_record in caplog.records] == [
        str(raised.value)
    ]


def test_an_undated_entry_listed_last_still_comes_first(tmp_path: Path) -> None:
    """Apply undated entries before dated ones, wherever the file lists them."""
    yaml_text = BASE_YAML.replace(
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n",
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n"
        "    - {time_constant: 120.0}\n",
    )
    loaded_config = read_config_text(tmp_path, yaml_text)
    assert (
        loaded_config.entry_for("nav23", MARK_BEFORE_MJD_60980).time_constant == 120.0
    )
    assert loaded_config.entry_for("nav23", MJD_60980_START).time_constant == 150.0


def test_rejects_before_restart_may_equal_the_gap_limit(tmp_path: Path) -> None:
    """Take a gap limit equal to rejects_before_restart (15.2)."""
    yaml_text = BASE_YAML.replace(
        "rejects_before_restart: 36", "rejects_before_restart: 432"
    )
    assert read_config_text(tmp_path, yaml_text).rejects_before_restart == 432


# ------------------------------------------- a first entry with its own settings

OWN_SETTINGS_ENTRY: Final = (
    "{filter_states: 2, time_constant: 40.0, scale_time_constant: 20.0,"
    " initial_innovation_scale: 6.0, gap_limit: 300}"
)
"""Every setting of a 2-state clock, given in its first entry."""


def test_a_first_entry_may_give_every_setting_instead_of_a_type(
    tmp_path: Path,
) -> None:
    """Take a clock's own settings, from the start, when it fits no type."""
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", f"  rb9: [{OWN_SETTINGS_ENTRY}]"
    )
    loaded_config = read_config_text(tmp_path, yaml_text)
    assert loaded_config.entry_for(
        "rb9", MARK_BEFORE_MJD_60980
    ) == clock_config.ClockEntry(
        filter_states=2,
        time_constant=40.0,
        scale_time_constant=20.0,
        initial_innovation_scale=6.0,
        gap_limit=300,
    )


def test_a_clock_with_its_own_settings_takes_later_entries_too(
    tmp_path: Path,
) -> None:
    """Apply a later entry over a first entry's own settings, from its date."""
    later_entry = "    - {effective_mjd: 60980.0, time_constant: 80.0}\n"
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", f"  rb9:\n    - {OWN_SETTINGS_ENTRY}\n{later_entry}"
    )
    loaded_config = read_config_text(tmp_path, yaml_text)
    assert loaded_config.entry_for("rb9", MARK_BEFORE_MJD_60980).time_constant == 40.0
    assert loaded_config.entry_for("rb9", MJD_60980_START).time_constant == 80.0


@pytest.mark.parametrize(
    ("first_entry", "error_pattern"),
    [
        (
            "{filter_states: 2, time_constant: 40.0, scale_time_constant: 20.0}",
            "rb9 gives no type, so it gives every setting; it leaves out"
            " initial_innovation_scale, gap_limit",
        ),
        (
            OWN_SETTINGS_ENTRY.replace("{", "{effective_mjd: 60980.0, "),
            "rb9 .* no effective_mjd",
        ),
        (OWN_SETTINGS_ENTRY.replace(" time_constant: 40.0,", ""), "time constant"),
    ],
)
def test_a_first_entry_with_its_own_settings_gives_them_all_from_the_start(
    tmp_path: Path, first_entry: str, error_pattern: str
) -> None:
    """Refuse one that leaves a setting out, or holds only from a date."""
    assert_refused(
        tmp_path,
        BASE_YAML.replace("  cs7: [{type: cesium}]", f"  rb9: [{first_entry}]"),
        error_pattern,
    )


def test_a_reference_has_the_reference_type_not_its_own_settings(
    tmp_path: Path,
) -> None:
    """Refuse a reference whose first entry gives settings instead of type mc."""
    yaml_text = BASE_YAML.replace(
        "  mc1: [{type: mc}]", f"  mc1: [{OWN_SETTINGS_ENTRY}]"
    )
    assert_refused(tmp_path, yaml_text, "mc1 is a reference, so of type mc")


def test_a_clock_named_like_a_reference_but_not_one_takes_any_type(
    tmp_path: Path,
) -> None:
    """Accept a type other than mc for a name of mc and more than one digit."""
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", "  cs7: [{type: cesium}]\n  mcq: [{type: cesium}]"
    )
    assert (
        read_config_text(tmp_path, yaml_text)
        .entry_for("mcq", MJD_60980_START)
        .filter_states
        == 2
    )


def test_a_clock_with_its_own_settings_keeps_its_number_of_states(
    tmp_path: Path,
) -> None:
    """Refuse a later entry that changes the states its first entry gave."""
    later_entry = "    - {effective_mjd: 60980.0, filter_states: 3}\n"
    yaml_text = BASE_YAML.replace(
        "  cs7: [{type: cesium}]", f"  rb9:\n    - {OWN_SETTINGS_ENTRY}\n{later_entry}"
    )
    assert_refused(tmp_path, yaml_text, "changes the filter_states of rb9")
