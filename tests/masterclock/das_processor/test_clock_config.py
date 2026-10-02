"""Tests for src/masterclock/das_processor/clock_config.py.

The rules covered: the clock configuration is read from YAML with a safe
loader that refuses a key repeated in any mapping, into frozen models that
refuse unknown keys; a clock's entry at a mark starts from the type default
of its first entry and applies, in order of effective_mjd, every entry in
force at the mark, an entry without effective_mjd from the start, and one
between two marks from the next mark; every check of design 15.2 refuses
the file with ConfigError; the RMS limit of a pair is its own, else its
reference's, else the default; a series takes the settings of its clock
side, a pair the RMS limit too; and the committed example file loads.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.exceptions import ConfigError
from masterclock.das_processor import clock_config
from masterclock.domain.series import SeriesParams

BASE: Final = (
    "rejects_before_restart: 36\n"
    "rms_limit:\n"
    "  default: 50\n"
    "  references: {mc2: 40}\n"
    "  pairs: {mc2.ox23: 80}\n"
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
    "  ox23:\n"
    "    - {type: maser}\n"
    "    - {effective_mjd: 60980.0, time_constant: 150.0}\n"
)
"""An invented clock configuration of the design's shape."""

BEFORE: Final = datetime(2025, 10, 31, 23, 50, tzinfo=UTC)
"""The mark just before MJD 60980 begins."""

AT: Final = datetime(2025, 11, 1, 0, 0, tzinfo=UTC)
"""The mark at which MJD 60980 begins."""


def load(tmp_path: Path, text: str = BASE) -> clock_config.ClockConfig:
    """Write ``text`` as a clock configuration file and read it."""
    path = tmp_path / "clock_config.yaml"
    path.write_text(text, encoding="utf-8")
    return clock_config.read_clock_config(path)


def refused(tmp_path: Path, text: str, match: str) -> None:
    """Check that ``text`` is refused with a ConfigError matching ``match``."""
    with pytest.raises(ConfigError, match=match):
        load(tmp_path, text)


# ------------------------------------------------------------------- entries


def test_an_entry_starts_from_its_type_default(tmp_path: Path) -> None:
    """Give a clock with one entry its type's values."""
    entry = load(tmp_path).entry_for("cs7", AT)
    assert entry == clock_config.ClockEntry(
        filter_states=2,
        time_constant=30.0,
        scale_time_constant=50.0,
        initial_innovation_scale=4.0,
        gap_limit=432,
    )


def test_a_later_entry_takes_effect_at_its_mjd(tmp_path: Path) -> None:
    """Apply an entry from the mark its effective_mjd falls on, not before."""
    config = load(tmp_path)
    assert config.entry_for("ox23", BEFORE).time_constant == 100.0
    assert config.entry_for("ox23", AT).time_constant == 150.0
    assert config.entry_for("ox23", AT).scale_time_constant == 50.0


def test_an_effective_mjd_between_two_marks_takes_effect_at_the_next(
    tmp_path: Path,
) -> None:
    """Apply an entry dated between two marks from the second of them."""
    text = BASE.replace("effective_mjd: 60980.0", "effective_mjd: 60980.003")
    config = load(tmp_path, text)
    assert config.entry_for("ox23", AT).time_constant == 100.0
    assert config.entry_for("ox23", AT + timedelta(minutes=10)).time_constant == 150.0


def test_entries_apply_in_order_of_effective_mjd(tmp_path: Path) -> None:
    """Apply entries by their dates, whatever order the file lists them in."""
    text = BASE.replace(
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n",
        "    - {effective_mjd: 60990.0, time_constant: 200.0}\n"
        "    - {effective_mjd: 60980.0, time_constant: 150.0}\n",
    )
    config = load(tmp_path, text)
    late = datetime(2025, 11, 20, tzinfo=UTC)
    assert config.entry_for("ox23", AT).time_constant == 150.0
    assert config.entry_for("ox23", late).time_constant == 200.0


def test_an_entry_without_a_date_applies_from_the_start(tmp_path: Path) -> None:
    """Apply an undated entry at every mark, after the type default."""
    text = BASE.replace(
        "  cs7: [{type: cesium}]", "  cs7: [{type: cesium, gap_limit: 300}]"
    )
    assert load(tmp_path, text).entry_for("cs7", BEFORE).gap_limit == 300


def test_a_mark_without_a_timezone_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for a naive mark, which names no one instant."""
    with pytest.raises(ConfigError, match="no timezone"):
        load(tmp_path).entry_for("ox23", AT.replace(tzinfo=None))


def test_an_unknown_clock_has_no_entry(tmp_path: Path) -> None:
    """Raise ConfigError for a clock the file does not name."""
    with pytest.raises(ConfigError, match="no entry for clock qx9"):
        load(tmp_path).entry_for("qx9", AT)


# ------------------------------------------------------------------ the checks


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("  cs7: [{type: cesium}]", "  cs7: []", "cs7 has no entry"),
        ("  mc1: [{type: mc}]", "  mc1: [{type: cesium}]", "mc1 .* type mc"),
        (
            "    - {effective_mjd: 60980.0, time_constant: 150.0}",
            "    - {effective_mjd: 60980.0, filter_states: 2}",
            "changes the filter_states of ox23",
        ),
        ("filter_states: 3,", "filter_states: 4,", "filter_states"),
        ("filter_states: 3,", "filter_states: true,", "filter_states"),
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
            "  ox23:\n    - {type: maser}\n",
            "  ox23:\n    - {type: maser}\n"
            "    - {effective_mjd: 60970.0, gap_limit: 20}\n",
            "above the gap_limit 20 of ox23",
        ),
        ("default: 50", "default: 0", "default"),
        ("default: 50", "default: 50.0", "default"),
        ("references: {mc2: 40}", "references: {mc2: -4}", "references"),
        ("pairs: {mc2.ox23: 80}", "pairs: {mc2.ox23: 8.5}", "pairs"),
        ("pairs: {mc2.ox23: 80}", "pairs: {ox23: 80}", "pair ox23"),
        ("references: {mc2: 40}", "references: {ox2: 40}", "reference ox2"),
        ("  cs7: [{type: cesium}]", "  cs7: [{type: rubidium}]", "type rubidium"),
        (
            "  cs7: [{type: cesium}]",
            "  cs7: [{gap_limit: 300}]",
            "first entry of cs7 .* type",
        ),
        (
            "    - {effective_mjd: 60980.0, time_constant: 150.0}",
            "    - {effective_mjd: 60980.0, type: cesium}",
            "type of ox23 .* first entry",
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
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    """Raise ConfigError for each item of the design's list (15.2, U14 load part)."""
    assert old in BASE
    refused(tmp_path, BASE.replace(old, new), match)


@pytest.mark.parametrize(
    ("old", "new"),
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
    tmp_path: Path, old: str, new: str
) -> None:
    """Refuse a key that a mapping holds twice, however deep (req 8)."""
    refused(tmp_path, BASE.replace(old, new), "repeated key")


def test_a_merge_key_is_refused(tmp_path: Path) -> None:
    """Refuse a YAML merge key, which would hide where a value came from."""
    text = BASE.replace("  cs7: [{type: cesium}]", "  cs7: [{<<: {type: cesium}}]")
    refused(tmp_path, text, "merge key")


@pytest.mark.parametrize(
    "text",
    ["[1, 2]\n", "rejects_before_restart: [\n", "!!python/object:os.system {}\n", ""],
)
def test_a_file_that_is_not_a_configuration_is_refused(
    tmp_path: Path, text: str
) -> None:
    """Refuse a file that does not parse, or is not a mapping, safely."""
    refused(tmp_path, text, "clock configuration")


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for a file that cannot be read."""
    with pytest.raises(ConfigError, match="cannot read"):
        clock_config.read_clock_config(tmp_path / "missing.yaml")


def test_a_file_that_is_not_utf_8_is_refused(tmp_path: Path) -> None:
    """Raise ConfigError for bytes that are not UTF-8."""
    path = tmp_path / "clock_config.yaml"
    path.write_bytes(b"rejects_before_restart: \xff\n")
    with pytest.raises(ConfigError, match="cannot read"):
        clock_config.read_clock_config(path)


# ----------------------------------------------------------------- RMS limit


@pytest.mark.parametrize(
    ("pair", "limit"),
    [
        (("mc2", "ox23"), 80),
        (("mc2", "cs7"), 40),
        (("mc1", "cs7"), 50),
        (("mc1", "mc2"), 50),
    ],
)
def test_the_rms_limit_is_the_pair_s_else_the_reference_s_else_the_default(
    tmp_path: Path, pair: tuple[str, str], limit: int
) -> None:
    """Give the pair's own limit, else its reference's, else the default."""
    assert load(tmp_path).rms_limit(pair) == limit


def test_the_rms_limits_need_only_a_default(tmp_path: Path) -> None:
    """Read an rms_limit section that gives only the default."""
    text = BASE.replace("  references: {mc2: 40}\n  pairs: {mc2.ox23: 80}\n", "")
    assert load(tmp_path, text).rms_limit(("mc2", "ox23")) == 50


# ---------------------------------------------------------------- series


def test_a_pair_takes_its_second_clock_s_settings(tmp_path: Path) -> None:
    """Give a pair the entry of its clock b and its RMS limit (8.1)."""
    assert load(tmp_path).params_for(("mc2", "ox23"), AT) == SeriesParams(
        model=3,
        M=150.0,
        M_sigma=50.0,
        sigma0=5.0,
        gmax=432,
        n_break=36,
        rms_max=80,
    )


def test_a_self_or_link_pair_takes_the_reference_s_settings(tmp_path: Path) -> None:
    """Give (r, r) and (r, s) the reference's 1-state entry."""
    params = load(tmp_path).params_for(("mc1", "mc2"), AT)
    assert (params.model, params.M, params.sigma0, params.rms_max) == (1, None, 3.0, 50)


def test_a_triple_takes_its_clock_s_settings_and_no_rms_limit(tmp_path: Path) -> None:
    """Give (r, s, c) the entry of c and no RMS limit (8.1, 12.5)."""
    params = load(tmp_path).params_for(("mc1", "mc2", "ox23"), BEFORE)
    assert (params.model, params.M, params.rms_max) == (3, 100.0, None)


# ------------------------------------------------------------------ example


def test_the_example_file_loads() -> None:
    """Read the committed example and find every clock it names usable."""
    path = Path(__file__).parents[3] / "etc" / "clock_config.yaml.example"
    config = clock_config.read_clock_config(path)
    mark = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
    for clock in config.clocks:
        assert config.entry_for(clock, mark).gap_limit >= config.rejects_before_restart
