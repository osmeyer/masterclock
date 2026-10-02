"""Tests for src/masterclock/das_processor/registry.py.

The rules covered: the references of an epoch are the clocks of its block
whose names start with the reference prefix; a pair exists for every
reference and clock measured together, and is never removed once it
exists; a triple (r, s, c) exists for every clock pair (s, c) and every
reference r whose link with s is measured both ways, the self pair standing
for both ways when r = s, so every clock local to r gets (r, r, c); the
keys come out sorted; a series' file is named for its channel and key, in
the measurement or double-difference directory; and the existing series are
read back from the names of a channel's files, other names ignored.

An archive refusal is logged as raised.
"""

from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Final

import pytest

from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import registry
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import DASData, DASMeasurement
from masterclock.domain.series import SeriesKey

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""An invented epoch start."""

NONE: Final = registry.Existing(pairs=frozenset(), triples=frozenset())
"""No series yet."""


def block(pairs: list[tuple[str, str]]) -> DASData:
    """Give a block measuring each (reference, clock) pair once, in order."""
    start = datetime_to_mjd(E)
    measurements = tuple(
        DASMeasurement(
            measurement_mjd=round(start + (index + 1) * 2e-6, 6),
            measured_phase=1000,
            rms=3,
            switch=f"{reference[-1]}A{index % 100:02d}",
            clock=clock,
        )
        for index, (reference, clock) in enumerate(pairs)
    )
    return DASData(interpolated_datetime=E, measurements=measurements)


REFS: Final = ("mc1", "mc2", "mc3")
"""Three invented references."""


def design_example() -> DASData:
    """Give design 3.3's example: three references, twenty clocks, each local."""
    pairs = [(r, s) for r in REFS for s in REFS]
    pairs += [(REFS[i % 3], f"hm{i}") for i in range(20)]
    return block(pairs)


def test_the_references_are_the_clocks_named_as_references() -> None:
    """Take every clock of the block whose name starts with the prefix."""
    assert registry.refs_of(design_example()) == frozenset(REFS)
    assert registry.refs_of(None) == frozenset()


def test_the_design_example_gives_29_pairs_and_60_triples() -> None:
    """Give 3 self + 6 link + 20 clock pairs and 3 triples per clock (3.3)."""
    data = design_example()
    pairs, triples = registry.build_registry(data, registry.refs_of(data), NONE)
    assert len(pairs) == 29
    assert len(triples) == 60
    assert pairs == tuple(sorted(pairs))
    assert triples == tuple(sorted(triples))


def test_every_clock_local_to_a_reference_gets_its_local_triple() -> None:
    """Give (r, r, c) for every clock c measured against r (3.3)."""
    data = design_example()
    _, triples = registry.build_registry(data, registry.refs_of(data), NONE)
    for i in range(20):
        local = REFS[i % 3]
        assert (local, local, f"hm{i}") in triples
        assert {(r, local, f"hm{i}") for r in REFS} <= set(triples)


def test_a_link_needs_both_directions() -> None:
    """Give no triple through a link measured one way only."""
    data = block([("mc1", "mc1"), ("mc2", "mc2"), ("mc1", "mc2"), ("mc2", "hm7")])
    _, triples = registry.build_registry(data, registry.refs_of(data), NONE)
    assert triples == (("mc2", "mc2", "hm7"),)


def test_a_reference_pair_seeds_no_triple() -> None:
    """Seed triples from clock pairs only, never from a link pair."""
    data = block([(r, s) for r, s in permutations(REFS, 2)] + [(r, r) for r in REFS])
    assert registry.build_registry(data, registry.refs_of(data), NONE)[1] == ()


def test_a_local_triple_needs_the_self_pair() -> None:
    """Give (r, r, c) only when r is measured against itself."""
    data = block([("mc2", "hm7")])
    assert registry.build_registry(data, frozenset({"mc2"}), NONE)[1] == ()


def test_a_series_is_never_removed() -> None:
    """Keep every existing pair and triple, measured this epoch or not (3.4)."""
    existing = registry.Existing(
        pairs=frozenset({("mc1", "hm9"), ("mc1", "mc1")}),
        triples=frozenset({("mc3", "mc1", "hm9")}),
    )
    pairs, triples = registry.build_registry(None, frozenset(), existing)
    assert pairs == (("mc1", "hm9"), ("mc1", "mc1"))
    assert triples == (("mc3", "mc1", "hm9"),)


def test_new_series_join_the_existing_ones() -> None:
    """Add the epoch's new pairs and triples to those that exist."""
    existing = registry.Existing(pairs=frozenset({("mc1", "hm9")}), triples=frozenset())
    data = block([("mc1", "mc1"), ("mc1", "hm7")])
    pairs, triples = registry.build_registry(data, registry.refs_of(data), existing)
    assert pairs == (("mc1", "hm7"), ("mc1", "hm9"), ("mc1", "mc1"))
    assert triples == (("mc1", "mc1", "hm7"), ("mc1", "mc1", "hm9"))


# ---------------------------------------------------------------- file names


@pytest.mark.parametrize(
    ("key", "relative"),
    [
        (("mc2", "ox23"), "meas/das_a.mc2.ox23.dat"),
        (("mc1", "mc2", "ox23"), "ddiff/das_a.mc1.mc2.ox23.dat"),
    ],
)
def test_a_series_file_is_named_for_its_channel_and_key(
    tmp_path: Path, key: SeriesKey, relative: str
) -> None:
    """Name a pair's file in meas/ and a triple's in ddiff/ (5.1)."""
    assert registry.series_file(tmp_path, "a", key) == tmp_path / relative


@pytest.mark.parametrize("clock", ["ox.23", "ox/23", "", ".."])
def test_a_clock_name_that_cannot_name_a_file_is_refused(
    tmp_path: Path, clock: str
) -> None:
    """Raise DataFileError for a name with a dot or slash, or none."""
    with pytest.raises(DataFileError, match="cannot name"):
        registry.series_file(tmp_path, "a", ("mc2", clock))


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("das_a.mc2.ox23.dat", ("mc2", "ox23")),
        ("das_a.mc1.mc2.ox23.dat", ("mc1", "mc2", "ox23")),
        ("das_b.mc2.ox23.dat", None),
        ("das_a.mc2.dat", None),
        ("das_a.mc1.mc2.ox23.x.dat", None),
        ("das_a.mc2..dat", None),
        ("das_a.mc2.ox23.txt", None),
        ("notes.txt", None),
    ],
)
def test_a_key_is_read_from_a_file_name(name: str, key: SeriesKey | None) -> None:
    """Read the key of a channel's file name, and nothing from any other name."""
    assert registry.key_of(name, "a") == key


def test_the_existing_series_are_read_from_the_file_names(tmp_path: Path) -> None:
    """List each archive and take the channel's series files (3.4, 5.1)."""
    (tmp_path / "meas").mkdir()
    (tmp_path / "ddiff").mkdir()
    for name in (
        "das_a.mc2.ox23.dat",
        "das_a.mc1.mc1.dat",
        "das_b.mc2.ox23.dat",
        "README",
    ):
        (tmp_path / "meas" / name).write_text("")
    (tmp_path / "meas" / "das_a.mc1.mc2.ox23.dat").write_text("")
    (tmp_path / "meas" / "das_a.mc3.mc3.dat").mkdir()
    (tmp_path / "ddiff" / "das_a.mc1.mc2.ox23.dat").write_text("")
    (tmp_path / "ddiff" / "das_a.mc2.ox23.dat").write_text("")
    assert registry.existing_series(tmp_path, "a") == registry.Existing(
        pairs=frozenset({("mc2", "ox23"), ("mc1", "mc1")}),
        triples=frozenset({("mc1", "mc2", "ox23")}),
    )


def test_no_archive_directories_mean_no_series(tmp_path: Path) -> None:
    """Give no series before any file has been written."""
    assert registry.existing_series(tmp_path, "a") == NONE


def test_an_archive_that_cannot_be_listed_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for an archive path that is not a directory."""
    (tmp_path / "meas").write_text("")
    with pytest.raises(DataFileError, match="cannot list"):
        registry.existing_series(tmp_path, "a")


def test_an_archive_refusal_is_logged_as_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the DataFileError for an archive that cannot be listed, as raised."""
    (tmp_path / "meas").write_text("")
    with pytest.raises(DataFileError) as raised:
        registry.existing_series(tmp_path, "a")
    assert [r.getMessage() for r in caplog.records] == [str(raised.value)]
    assert str(raised.value).startswith(f"cannot list archive {tmp_path / 'meas'}: ")
