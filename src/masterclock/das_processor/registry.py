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
from masterclock.domain.references import REFERENCE_PREFIX
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


class Existing(BaseModel):
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


def refs_of(block: DASData | None) -> frozenset[str]:
    """Give an epoch's references: REFS(e) (design 3.1).

    Parameters
    ----------
    block : DASData or None
        The epoch's DAS block, or ``None`` when the DAS measured nothing.

    Returns
    -------
    frozenset of str
        Every clock of the block whose name starts with the reference
        prefix; none without a block.
    """
    if block is None:
        return frozenset()
    return frozenset(
        m.clock for m in block.measurements if m.clock.startswith(REFERENCE_PREFIX)
    )


def build_registry(
    block: DASData | None, refs: frozenset[str], existing: Existing
) -> tuple[tuple[PairKey, ...], tuple[TripleKey, ...]]:
    """Give every pair and triple that exists at an epoch (design 3.4).

    Parameters
    ----------
    block : DASData or None
        The epoch's DAS block, or ``None``.
    refs : frozenset of str
        The epoch's references.
    existing : Existing
        The series that existed before the epoch.

    Returns
    -------
    tuple of (tuple of (str, str), tuple of (str, str, str))
        The pairs and the triples, each sorted: every existing one, every
        (reference, clock) the block measured, and every (r, s, c) for a
        clock pair (s, c) and a reference r whose link with s is a pair
        both ways.
    """
    pairs = set(existing.pairs)
    if block is not None:
        pairs |= {(m.reference, m.clock) for m in block.measurements}
    triples = set(existing.triples)
    for s, c in pairs:
        if c in refs:
            continue
        for r in refs:
            if (r, s) in pairs and (s, r) in pairs:
                triples.add((r, s, c))
    return tuple(sorted(pairs)), tuple(sorted(triples))


def series_file(processed_path: Path, channel: RfChannel, key: SeriesKey) -> Path:
    """Give the path of a series' file (design 5.1).

    Parameters
    ----------
    processed_path : Path
        The directory holding the two archives.
    channel : {'a', 'b'}
        The RF channel.
    key : (str, str) or (str, str, str)
        The series.

    Returns
    -------
    Path
        ``meas/das_<rf>.<a>.<b>.dat`` for a pair,
        ``ddiff/das_<rf>.<r>.<s>.<c>.dat`` for a triple.

    Raises
    ------
    DataFileError
        If a name in ``key`` is empty or holds a dot or a slash, and so
        cannot name a file that reads back as the same key.
    """
    for name in key:
        if _NAME.fullmatch(name) is None:
            _fail(f"clock name {name!r} of {key} cannot name a series file")
    directory = MEAS_SUBDIRECTORY if len(key) == _PAIR else DDIFF_SUBDIRECTORY
    name = f"{_PREFIX}{channel}{_SEPARATOR}{_SEPARATOR.join(key)}{_SUFFIX}"
    return processed_path / directory / name


def key_of(name: str, channel: RfChannel) -> SeriesKey | None:
    """Read a series' key from its file's name.

    Parameters
    ----------
    name : str
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
    >>> key_of("das_a.mc2.ox23.dat", "a")
    ('mc2', 'ox23')
    >>> key_of("das_b.mc2.ox23.dat", "a") is None
    True
    """
    start = f"{_PREFIX}{channel}{_SEPARATOR}"
    if not (name.startswith(start) and name.endswith(_SUFFIX)):
        return None
    names = tuple(name[len(start) : -len(_SUFFIX)].split(_SEPARATOR))
    if not all(names):
        return None
    if len(names) == _PAIR:
        return (names[0], names[1])
    if len(names) == _TRIPLE:
        return (names[0], names[1], names[2])
    return None


def _keys(directory: Path, channel: RfChannel) -> list[SeriesKey]:
    """Give the keys of an archive's series files.

    Parameters
    ----------
    directory : Path
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
    if not directory.exists():
        return []
    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        _fail(f"cannot list archive {directory}: {exc}", exc)
    keys = [key_of(entry.name, channel) for entry in entries if entry.is_file()]
    return [key for key in keys if key is not None]


def existing_series(processed_path: Path, channel: RfChannel) -> Existing:
    """Give the series a channel's files hold (design 3.4, 5.1).

    Parameters
    ----------
    processed_path : Path
        The directory holding the two archives.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    Existing
        The pairs of the measurement archive's files and the triples of the
        double-difference archive's; other channels' files and other names
        are left out.

    Raises
    ------
    DataFileError
        If an archive's directory is there but cannot be listed.
    """
    pairs: set[PairKey] = set()
    for key in _keys(processed_path / MEAS_SUBDIRECTORY, channel):
        if len(key) == _PAIR:
            pairs.add((key[0], key[1]))
    triples: set[TripleKey] = set()
    for key in _keys(processed_path / DDIFF_SUBDIRECTORY, channel):
        if len(key) == _TRIPLE:
            triples.add((key[0], key[1], key[-1]))
    return Existing(pairs=frozenset(pairs), triples=frozenset(triples))
