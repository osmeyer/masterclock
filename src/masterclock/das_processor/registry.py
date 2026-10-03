"""The series registry: which pairs and triples exist at an epoch, and their files.

A pair (a, b) exists from the first epoch reference a measured b, and a
triple (r, s, c) from the first epoch at which clock c is measured against
s and the link between r and s is measured both ways; for r = s the self
pair (r, r) stands for both ways, so every clock local to r has its local
triple (r, r, c). A series is never removed: when its measurements stop, it
goes on with predicted and then dormant rows.

Each series has one file, named for the RF channel and the series' key, a
pair's in the measurement archive and a triple's in the double-difference
archive. The series that exist before a run are read back from the names of
the channel's files.
"""

import re
from pathlib import Path
from typing import Final, NoReturn

from pydantic import BaseModel, ConfigDict

from masterclock.app.log import MasterClockLogger, get_logger
from masterclock.das_processor.channels import RfChannel
from masterclock.das_processor.config import DDIFF_SUBDIRECTORY, MEAS_SUBDIRECTORY
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import DASData
from masterclock.domain.references import is_reference
from masterclock.domain.series import PairKey, SeriesKey, TripleKey

_PREFIX: Final[str] = "das_"
"""What every series file's name starts with, before its channel."""

_SUFFIX: Final[str] = ".dat"
"""What every series file's name ends with."""

_SEPARATOR: Final[str] = "."
"""What separates a file name's channel and clock names."""

_NAME: Final[re.Pattern[str]] = re.compile(r"[^./]+")
"""A clock name that can stand in a file name: no dot, no slash."""

_PAIR: Final[int] = 2
"""How many names a pair key holds."""

_TRIPLE: Final[int] = 3
"""How many names a triple key holds."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


class ExistingSeries(BaseModel):
    """The series that exist before an epoch.

    Parameters
    ----------
    pairs : frozenset of (str, str)
        The pairs.
    triples : frozenset of (str, str, str)
        The triples.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    pairs: frozenset[PairKey]
    triples: frozenset[TripleKey]


def _fail(message: str, cause: Exception | None = None) -> NoReturn:
    """Log and raise a data file error.

    Parameters
    ----------
    message : str
        What is wrong.
    cause : Exception or None, optional
        The error it came from, if any.

    Raises
    ------
    DataFileError
        Always.
    """
    _log.error(message)
    raise DataFileError(message) from cause


def refs_of(das_block: DASData | None) -> frozenset[str]:
    """Give an epoch's references: REFS(e) (design 3.1).

    Parameters
    ----------
    das_block : DASData or None
        The epoch's DAS block, or ``None`` when the DAS measured nothing.

    Returns
    -------
    frozenset of str
        Every clock of the block whose name is a reference's, the prefix
        and one digit; none without a block.
    """
    if das_block is None:
        return frozenset()
    return frozenset(
        das_measurement.clock
        for das_measurement in das_block.measurements
        if is_reference(das_measurement.clock)
    )


def build_registry(
    das_block: DASData | None, refs: frozenset[str], earlier_series: ExistingSeries
) -> tuple[tuple[PairKey, ...], tuple[TripleKey, ...]]:
    """Give every pair and triple that exists at an epoch (design 3.4).

    Parameters
    ----------
    das_block : DASData or None
        The epoch's DAS block, or ``None``.
    refs : frozenset of str
        The epoch's references.
    earlier_series : ExistingSeries
        The series that existed before the epoch.

    Returns
    -------
    tuple of (tuple of (str, str), tuple of (str, str, str))
        The pairs and the triples, each sorted: every existing one, every
        (reference, clock) the block measured, and every (r, s, c) for a
        pair (s, c), its clock c a reference or not, and a reference r
        whose link with s is a pair both ways.
    """
    pairs = set(earlier_series.pairs)
    if das_block is not None:
        pairs |= {
            (das_measurement.reference, das_measurement.clock)
            for das_measurement in das_block.measurements
        }
    triples = set(earlier_series.triples)
    for s, c in pairs:
        for r in refs:
            if (r, s) in pairs and (s, r) in pairs:
                triples.add((r, s, c))
    return tuple(sorted(pairs)), tuple(sorted(triples))


def series_file(
    processed_path: Path, channel: RfChannel, series_key: SeriesKey
) -> Path:
    """Give the path of a series' file (design 5.1).

    Parameters
    ----------
    processed_path : Path
        The directory holding the two archives.
    channel : {'a', 'b'}
        The RF channel.
    series_key : (str, str) or (str, str, str)
        The series.

    Returns
    -------
    Path
        ``meas/das_<rf>.<a>.<b>.dat`` for a pair,
        ``ddiff/das_<rf>.<r>.<s>.<c>.dat`` for a triple.

    Raises
    ------
    DataFileError
        If a name in ``series_key`` is empty or holds a dot or a slash, and so
        cannot name a file that reads back as the same key.
    """
    for clock_name in series_key:
        if _NAME.fullmatch(clock_name) is None:
            _fail(
                f"clock name {clock_name!r} of {series_key} cannot name a series file"
            )
    archive_name = MEAS_SUBDIRECTORY if len(series_key) == _PAIR else DDIFF_SUBDIRECTORY
    file_name = f"{_PREFIX}{channel}{_SEPARATOR}{_SEPARATOR.join(series_key)}{_SUFFIX}"
    return processed_path / archive_name / file_name


def series_key_of(file_name: str, channel: RfChannel) -> SeriesKey | None:
    """Read a series' key from its file's name.

    Parameters
    ----------
    file_name : str
        A file name.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    (str, str) or (str, str, str) or None
        The pair or triple the name is for; ``None`` for a name that is not
        one of the channel's series files.

    Examples
    --------
    >>> series_key_of("das_a.mc2.ox23.dat", "a")
    ('mc2', 'ox23')
    >>> series_key_of("das_b.mc2.ox23.dat", "a") is None
    True
    """
    name_start = f"{_PREFIX}{channel}{_SEPARATOR}"
    if not (file_name.startswith(name_start) and file_name.endswith(_SUFFIX)):
        return None
    clock_names = tuple(file_name[len(name_start) : -len(_SUFFIX)].split(_SEPARATOR))
    if not all(clock_names):
        return None
    if len(clock_names) == _PAIR:
        return (clock_names[0], clock_names[1])
    if len(clock_names) == _TRIPLE:
        return (clock_names[0], clock_names[1], clock_names[2])
    return None


def _series_keys(archive: Path, channel: RfChannel) -> list[SeriesKey]:
    """Give the keys of an archive's series files.

    Parameters
    ----------
    archive : Path
        The archive's directory.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    list of keys
        The keys of the regular files named for the channel's series; none
        when the directory does not exist.

    Raises
    ------
    DataFileError
        If the directory is there but cannot be listed.
    """
    if not archive.exists():
        return []
    try:
        archive_entries = sorted(archive.iterdir())
    except OSError as exc:
        _fail(f"cannot list archive {archive}: {exc}", exc)
    series_keys = [
        series_key_of(archive_entry.name, channel)
        for archive_entry in archive_entries
        if archive_entry.is_file()
    ]
    return [series_key for series_key in series_keys if series_key is not None]


def existing_series(processed_path: Path, channel: RfChannel) -> ExistingSeries:
    """Give the series a channel's files hold (design 3.4, 5.1).

    Parameters
    ----------
    processed_path : Path
        The directory holding the two archives.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    ExistingSeries
        The pairs of the measurement archive's files and the triples of the
        double-difference archive's; other channels' files and other names
        are left out.

    Raises
    ------
    DataFileError
        If an archive's directory is there but cannot be listed.
    """
    pairs: set[PairKey] = set()
    for series_key in _series_keys(processed_path / MEAS_SUBDIRECTORY, channel):
        if len(series_key) == _PAIR:
            pairs.add((series_key[0], series_key[1]))
    triples: set[TripleKey] = set()
    for series_key in _series_keys(processed_path / DDIFF_SUBDIRECTORY, channel):
        if len(series_key) == _TRIPLE:
            triples.add((series_key[0], series_key[1], series_key[-1]))
    return ExistingSeries(pairs=frozenset(pairs), triples=frozenset(triples))
