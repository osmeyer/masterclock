"""Choose each clock's estimator settings from a characterization run (design 15.3).

A characterization run is das_processor run with every clock at one state,
into a processed_path of its own. This script reads that run's files and,
for every clock measured against a reference, prepares the z of each local
triple (r, r, c) in the design's steps, fits the clock's noise model and the
measurement's white phase noise to their Allan variances, and works out the
settings: the time constant M, the initial innovation scale sigma0 (the
measurement noise) and the gap limit G_max. The measurement noise comes
from the fit, never from the rms the DAS reports. Days when the phase holds
no clock signal are left out. It prints one line per clock::

    uv run --frozen python scripts/characterize.py RUN --rf a --three-state ox

A clock is measured against every reference that has a local triple for it;
the Allan variances of all of them are combined, so the result is the clock
against the references together. A reference is never characterized against
itself.
"""

import argparse
import math
import statistics
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Final, NamedTuple

from masterclock.das_processor.channels import RF_CHOICES, RfChannel
from masterclock.das_processor.config import DDIFF_SUBDIRECTORY
from masterclock.das_processor.files import (
    DDIFF_COLUMNS,
    DDIFF_HEADER_LINES,
    EMPTY,
    MEAS_COLUMNS,
    MEAS_HEADER_LINES,
    SEPARATOR,
    Column,
)
from masterclock.das_processor.registry import series_file, series_key_of
from masterclock.domain.phase import EPOCH_SECONDS, PHASE_PERIOD
from masterclock.domain.references import is_reference

T: Final[int] = EPOCH_SECONDS
"""One epoch, s."""

EPOCHS_PER_DAY: Final[int] = 86_400 // EPOCH_SECONDS
"""Epochs in a day, to turn an epoch-start MJD into an epoch number."""

NO_SIGNAL_PS: Final[float] = 10_000.0
"""The typical one-epoch change, less the day's median change, above which a
day holds no clock signal, ps (step 3)."""

OUTLIER_SPREADS: Final[float] = 5.0
"""How many robust spreads a change may stray from its day's median (steps 5, 6)."""

MAD_TO_SIGMA: Final[float] = 1.4826
"""The robust spread of normal values per median absolute deviation."""

GATE_SIGMAS: Final[float] = 5.0
"""The decycling margin of the gap limit, in sigma (design 13.2)."""

PS: Final[float] = 1e-12
"""One picosecond, s."""

FIT_TOLERANCE: Final[float] = 1e-9
"""The relative change of every coefficient below which the fit has settled."""

FIT_ROUNDS: Final[int] = 100
"""The most times the fit is repeated with new weights."""

GAP_SEARCH_LIMIT: Final[int] = 10**9
"""The largest gap, in epochs, the gap-limit search looks at."""


def _column_slices(columns: tuple[Column, ...]) -> dict[str, slice]:
    """Give where each column lies in a row.

    Parameters
    ----------
    columns : tuple of Column
        A file kind's columns, in order.

    Returns
    -------
    dict of str to slice
        Each column's characters within a row.
    """
    slices = {}
    position = 0
    for column in columns:
        slices[column.name] = slice(position, position + column.width)
        position += column.width + len(SEPARATOR)
    return slices


DDIFF_SLICES: Final[dict[str, slice]] = _column_slices(DDIFF_COLUMNS)
"""Where each column of a double-difference row lies."""

MEAS_SLICES: Final[dict[str, slice]] = _column_slices(MEAS_COLUMNS)
"""Where each column of a measurement row lies."""


class TripleRow(NamedTuple):
    """What step 1 and 2 need of a local triple's row.

    Parameters
    ----------
    epoch : int
        The epoch number: the epoch start's MJD times the epochs in a day.
    z : int or None
        The double difference, ps; ``None`` on a row without one.
    step_offset : int
        The segment's sum of phase steps, ps.
    flags : str
        The row's flags.
    """

    epoch: int
    z: int | None
    step_offset: int
    flags: str


def epoch_number(mjd_text: str) -> int:
    """Give the epoch number of an epoch start's MJD as a row writes it.

    Parameters
    ----------
    mjd_text : str
        The MJD, six decimals.

    Returns
    -------
    int
        The MJD times the epochs in a day, rounded.

    Examples
    --------
    >>> epoch_number("60941.250000"), epoch_number("60941.256944")
    (8775540, 8775541)
    """
    return round(float(mjd_text) * EPOCHS_PER_DAY)


def _field(row_line: str, slices: dict[str, slice], column_name: str) -> str:
    """Give one column's text from a row, spaces taken off.

    Parameters
    ----------
    row_line : str
        The row.
    slices : dict of str to slice
        Where each column lies.
    column_name : str
        The column.

    Returns
    -------
    str
        The field's text.
    """
    return row_line[slices[column_name]].strip()


def _rows(data_file: Path, header_lines: int) -> Iterator[str]:
    """Yield the rows of a series file, its header skipped.

    Parameters
    ----------
    data_file : Path
        A series file.
    header_lines : int
        How many header lines it has.

    Yields
    ------
    str
        Each row, in file order.
    """
    with data_file.open(encoding="ascii") as open_file:
        for line_number, row_line in enumerate(open_file):
            if line_number >= header_lines:
                yield row_line


def read_triple_rows(data_file: Path) -> list[TripleRow]:
    """Read what steps 1 and 2 need from a local triple's file.

    Parameters
    ----------
    data_file : Path
        The double-difference file of a local triple (r, r, c).

    Returns
    -------
    list of TripleRow
        Every row, in file order.
    """
    triple_rows = []
    for row_line in _rows(data_file, DDIFF_HEADER_LINES):
        z_text = _field(row_line, DDIFF_SLICES, "z")
        triple_rows.append(
            TripleRow(
                epoch=epoch_number(_field(row_line, DDIFF_SLICES, "interpolated_mjd")),
                z=None if z_text == EMPTY else int(z_text),
                step_offset=int(_field(row_line, DDIFF_SLICES, "step_offset")),
                flags=_field(row_line, DDIFF_SLICES, "flags"),
            )
        )
    return triple_rows


def read_pair_deltas(data_file: Path) -> dict[int, float]:
    """Read the measurement times of a local pair's file (step 4).

    Parameters
    ----------
    data_file : Path
        The measurement file of a local pair (r, c).

    Returns
    -------
    dict of int to float
        The measurement time after its epoch start, s, of every row with a
        measurement, by epoch.
    """
    deltas = {}
    for row_line in _rows(data_file, MEAS_HEADER_LINES):
        measured_text = _field(row_line, MEAS_SLICES, "measurement_datetime")
        if measured_text == EMPTY:
            continue
        epoch_start = datetime.fromisoformat(
            _field(row_line, MEAS_SLICES, "interpolated_datetime")
        )
        measured_at = datetime.fromisoformat(measured_text)
        epoch = epoch_number(_field(row_line, MEAS_SLICES, "interpolated_mjd"))
        deltas[epoch] = (measured_at - epoch_start).total_seconds()
    return deltas


type Stretch = dict[int, float]
"""The prepared values of the rows from one cold start up to the next, by epoch."""


def split_at_cold_starts(triple_rows: Sequence[TripleRow]) -> list[Stretch]:
    """Split the accepted rows, z less step_offset, at cold starts (steps 1 and 2).

    Parameters
    ----------
    triple_rows : sequence of TripleRow
        A local triple's rows, in file order.

    Returns
    -------
    list of Stretch
        For the rows from each cold start up to the next, every accepted
        row's z less its step_offset, by epoch. A cold start is a row flagged
        N after a dormant row.

    Examples
    --------
    >>> rows = [TripleRow(1, 5, 0, "A"), TripleRow(2, 9, 2, "A"),
    ...         TripleRow(3, None, 0, "RD"), TripleRow(4, 40, 0, "AN")]
    >>> split_at_cold_starts(rows)
    [{1: 5.0, 2: 7.0}, {4: 40.0}]
    """
    stretches: list[Stretch] = []
    current: Stretch = {}
    previous_flags = ""
    for triple_row in triple_rows:
        if "N" in triple_row.flags and "D" in previous_flags and current:
            stretches.append(current)
            current = {}
        if "A" in triple_row.flags and triple_row.z is not None:
            current[triple_row.epoch] = float(triple_row.z - triple_row.step_offset)
        previous_flags = triple_row.flags
    if current:
        stretches.append(current)
    return stretches


def one_epoch_changes(stretch: Stretch) -> dict[int, float]:
    """Give every change between rows one epoch apart, by the later epoch.

    Parameters
    ----------
    stretch : Stretch
        Values by epoch.

    Returns
    -------
    dict of int to float
        The value at each epoch less the value one epoch before, for every
        epoch whose previous epoch is present.

    Examples
    --------
    >>> one_epoch_changes({1: 0.0, 2: 3.0, 4: 10.0, 5: 11.0})
    {2: 3.0, 5: 1.0}
    """
    return {
        epoch: value - stretch[epoch - 1]
        for epoch, value in stretch.items()
        if epoch - 1 in stretch
    }


def _day(epoch: int) -> int:
    """Give the day an epoch is in, as a day number.

    Parameters
    ----------
    epoch : int
        The epoch number.

    Returns
    -------
    int
        The epoch number divided by the epochs in a day, rounded down.

    Examples
    --------
    >>> _day(143), _day(144)
    (0, 1)
    """
    return epoch // EPOCHS_PER_DAY


def _changes_by_day(changes: dict[int, float]) -> dict[int, list[float]]:
    """Group changes by the day of the epoch each ends at.

    Parameters
    ----------
    changes : dict of int to float
        Changes, by the epoch each ends at.

    Returns
    -------
    dict of int to list of float
        Each day's changes.
    """
    by_day: dict[int, list[float]] = {}
    for epoch, change in changes.items():
        by_day.setdefault(_day(epoch), []).append(change)
    return by_day


def _typical_departure(changes: Sequence[float]) -> tuple[float, float]:
    """Give the median of some changes and their median absolute departure from it.

    Parameters
    ----------
    changes : sequence of float
        At least one change.

    Returns
    -------
    (float, float)
        The median, and the median of the absolute differences from it.

    Examples
    --------
    >>> _typical_departure([1.0, 2.0, 10.0])
    (2.0, 1.0)
    """
    median_change = statistics.median(changes)
    return median_change, statistics.median(abs(c - median_change) for c in changes)


def drop_no_signal_days(stretch: Stretch) -> tuple[Stretch, int]:
    """Drop every day whose phase holds no clock signal (step 3).

    A DAS channel whose clock is off or disconnected still gives readings,
    but their phase is spread over the whole period, so their one-epoch
    changes are tens of nanoseconds; a clock's are far smaller.

    Parameters
    ----------
    stretch : Stretch
        Values from step 2, by epoch.

    Returns
    -------
    (Stretch, int)
        The values less every row of each day whose one-epoch changes, less
        their median, typically depart from it by more than
        :data:`NO_SIGNAL_PS`; and how many days were dropped. A day with no
        two rows one epoch apart cannot be judged, and is kept.

    Examples
    --------
    >>> quiet = {epoch: 5.0 * epoch for epoch in range(144)}
    >>> random_phase = {144 + k: (7_919.0 * k * k) % 200_000 for k in range(144)}
    >>> kept, dropped = drop_no_signal_days({**quiet, **random_phase})
    >>> sorted(kept) == list(range(144)), dropped
    (True, 1)
    """
    no_signal = {
        day
        for day, changes in _changes_by_day(one_epoch_changes(stretch)).items()
        if _typical_departure(changes)[1] > NO_SIGNAL_PS
    }
    kept = {
        epoch: value for epoch, value in stretch.items() if _day(epoch) not in no_signal
    }
    return kept, len(no_signal)


def to_epoch_start(stretch: Stretch, deltas: dict[int, float]) -> Stretch:
    """Move each value back to its epoch start by rate times delta (step 4).

    Parameters
    ----------
    stretch : Stretch
        Values from step 3, by epoch.
    deltas : dict of int to float
        Each epoch's measurement time after its start, s.

    Returns
    -------
    Stretch
        Each value less rate times its delta, the rate being the median
        one-epoch change divided by T; the values unchanged when no two rows
        are one epoch apart.

    Examples
    --------
    >>> to_epoch_start({1: 0.0, 2: 600.0, 3: 1200.0}, {1: 0.0, 2: 10.0, 3: 20.0})
    {1: 0.0, 2: 590.0, 3: 1180.0}
    """
    changes = one_epoch_changes(stretch)
    if not changes:
        return dict(stretch)
    rate = statistics.median(changes.values()) / T
    return {epoch: value - rate * deltas[epoch] for epoch, value in stretch.items()}


class SingularError(ArithmeticError):
    """A linear system with no single solution."""


type Departure = tuple[float, float]
"""A change less its day's median change, and its day's outlier limit, ps."""


def _day_limits(changes: dict[int, float]) -> dict[int, Departure]:
    """Give each day's median change and outlier limit.

    Parameters
    ----------
    changes : dict of int to float
        Changes, by the epoch each ends at.

    Returns
    -------
    dict of int to (float, float)
        For each day, the median of its changes and :data:`OUTLIER_SPREADS`
        robust spreads of them: 1.4826 times their median absolute departure
        from that median.
    """
    limits = {}
    for day, day_changes in _changes_by_day(changes).items():
        median_change, departure = _typical_departure(day_changes)
        limits[day] = (median_change, OUTLIER_SPREADS * MAD_TO_SIGMA * departure)
    return limits


def _is_outlier(before: Departure | None, after: Departure | None) -> bool:
    """Tell whether a row's changes, against their days' limits, mark it as an outlier.

    Parameters
    ----------
    before : (float, float) or None
        The change from the row one epoch before, less its day's median, and
        that day's limit; ``None`` when that row is missing.
    after : (float, float) or None
        The same for the change to the row one epoch after.

    Returns
    -------
    bool
        With both: whether each is past its limit, in opposite directions.
        With one: whether it is past its limit. With neither: false.

    Examples
    --------
    >>> _is_outlier((9.0, 5.0), (-9.0, 5.0)), _is_outlier((9.0, 5.0), (9.0, 5.0))
    (True, False)
    >>> _is_outlier(None, (9.0, 5.0)), _is_outlier(None, None)
    (True, False)
    """
    if before is not None and after is not None:
        return (
            abs(before[0]) > before[1]
            and abs(after[0]) > after[1]
            and before[0] * after[0] < 0
        )
    single = before if before is not None else after
    return single is not None and abs(single[0]) > single[1]


def drop_outliers(stretch: Stretch) -> Stretch:
    """Drop rows whose changes from their neighbours are out of line (step 5).

    Parameters
    ----------
    stretch : Stretch
        Values from step 4, by epoch.

    Returns
    -------
    Stretch
        The values less every row whose changes from both neighbouring rows,
        one epoch either side, each differ from their day's median change by
        more than their day's limit (see :func:`_day_limits`), in opposite
        directions; or, with only one such neighbour present, whose one
        change does. Unchanged when no two rows are one epoch apart.

    Examples
    --------
    >>> values = {epoch: float(epoch) for epoch in range(10)}
    >>> values[5] = 1000.0
    >>> sorted(drop_outliers(values))
    [0, 1, 2, 3, 4, 6, 7, 8, 9]
    """
    changes = one_epoch_changes(stretch)
    limits = _day_limits(changes)
    departures = {
        epoch: (change - limits[_day(epoch)][0], limits[_day(epoch)][1])
        for epoch, change in changes.items()
    }
    return {
        epoch: value
        for epoch, value in stretch.items()
        if not _is_outlier(departures.get(epoch), departures.get(epoch + 1))
    }


def split_at_jumps(stretch: Stretch) -> list[Stretch]:
    """Split the rows wherever the phase jumps and stays (step 6).

    Parameters
    ----------
    stretch : Stretch
        Values from step 5, by epoch, outliers dropped.

    Returns
    -------
    list of Stretch
        The values split before every row whose change from the row before,
        less its day's median change times the epochs between them, is past
        its day's limit times those epochs (see :func:`_day_limits`). A row
        whose day has no two rows one epoch apart cannot be judged, and
        starts no new piece. None for no values.

    Examples
    --------
    >>> values = {epoch: float(epoch) for epoch in range(6)}
    >>> values.update({epoch: epoch + 500.0 for epoch in range(6, 10)})
    >>> [sorted(piece) for piece in split_at_jumps(values)]
    [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]]
    """
    limits = _day_limits(one_epoch_changes(stretch))
    pieces: list[Stretch] = []
    piece: Stretch = {}
    previous: tuple[int, float] | None = None
    for epoch, value in sorted(stretch.items()):
        if previous is not None and _jumps(previous, (epoch, value), limits):
            pieces.append(piece)
            piece = {}
        piece[epoch] = value
        previous = (epoch, value)
    if piece:
        pieces.append(piece)
    return pieces


def _jumps(
    previous: tuple[int, float],
    current: tuple[int, float],
    limits: dict[int, Departure],
) -> bool:
    """Tell whether the phase jumps between two rows.

    Parameters
    ----------
    previous : (int, float)
        The earlier row's epoch and value.
    current : (int, float)
        The later row's epoch and value.
    limits : dict of int to (float, float)
        Each day's median change and limit.

    Returns
    -------
    bool
        Whether the change, less the later row's day's median change times
        the epochs between them, is past that day's limit times those
        epochs; false when that day has no limit.
    """
    day_limit = limits.get(_day(current[0]))
    if day_limit is None:
        return False
    epochs = current[0] - previous[0]
    departure = current[1] - previous[1] - day_limit[0] * epochs
    return abs(departure) > day_limit[1] * epochs


def _eliminate(rows: list[list[float]], column: int) -> None:
    """Clear one column of an augmented matrix below and above its pivot.

    Parameters
    ----------
    rows : list of list of float
        The augmented matrix, changed in place: the row with the largest
        entry in ``column`` at or below the diagonal is swapped to the
        diagonal, then every other row has that column cleared.

    column : int
        The column.

    Raises
    ------
    SingularError
        If no row at or below the diagonal has an entry in the column that
        stands out from rounding.
    """
    size = len(rows)
    pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
    largest = max(abs(rows[row][column]) for row in range(size))
    if abs(rows[pivot][column]) <= 1e-12 * max(largest, 1e-300):
        message = f"no pivot in column {column}"
        raise SingularError(message)
    rows[column], rows[pivot] = rows[pivot], rows[column]
    for row in range(size):
        if row != column:
            factor = rows[row][column] / rows[column][column]
            rows[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(rows[row], rows[column], strict=True)
            ]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small linear system by elimination with partial pivoting.

    Parameters
    ----------
    matrix : list of list of float
        A square matrix.
    vector : list of float
        The right-hand side.

    Returns
    -------
    list of float
        The solution.

    Raises
    ------
    SingularError
        If the matrix is singular.

    Examples
    --------
    >>> _solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0])
    [1.0, 2.0]
    >>> _solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    Traceback (most recent call last):
    ...
    characterize.SingularError: no pivot in column 1
    """
    size = len(vector)
    rows = [[*matrix[row], vector[row]] for row in range(size)]
    for column in range(size):
        _eliminate(rows, column)
    return [rows[row][size] / rows[row][row] for row in range(size)]


def _quadratic(points: dict[float, float]) -> tuple[float, float, float]:
    """Fit c0 + c1 t + c2 t**2 to points by least squares.

    Parameters
    ----------
    points : dict of float to float
        Values by t, at least three distinct t, so the fit has one solution.

    Returns
    -------
    (float, float, float)
        c0, c1, c2.

    Examples
    --------
    >>> [round(c, 9) for c in _quadratic({0.0: 1.0, 0.5: 2.75, 1.0: 6.0})]
    [1.0, 2.0, 3.0]
    """
    sums = [sum(t**power for t in points) for power in range(5)]
    moments = [
        sum(value * t**power for t, value in points.items()) for power in range(3)
    ]
    c0, c1, c2 = _solve(
        [[sums[row + column] for column in range(3)] for row in range(3)], moments
    )
    return c0, c1, c2


def remove_drift(stretch: Stretch) -> Stretch:
    """Take off a quadratic fitted to the values by least squares (step 7).

    Parameters
    ----------
    stretch : Stretch
        Values from step 6, by epoch.

    Returns
    -------
    Stretch
        Each value less the fitted quadratic at its epoch; unchanged with
        fewer than three rows. Epochs are distinct, so three rows always fit.

    Examples
    --------
    >>> drifted = {epoch: 3.0 + 2.0 * epoch + 0.5 * epoch**2 for epoch in range(6)}
    >>> max(abs(value) for value in remove_drift(drifted).values()) < 1e-9
    True
    """
    if len(stretch) < 3:
        return dict(stretch)
    first = min(stretch)
    span = max(stretch) - first
    scaled = {(epoch - first) / span: value for epoch, value in stretch.items()}
    c0, c1, c2 = _quadratic(scaled)
    return {
        epoch: value - (c0 + c1 * t + c2 * t * t)
        for (epoch, value), t in zip(stretch.items(), scaled, strict=True)
    }


class AllanSums(NamedTuple):
    """The sum of squared second differences at one tau, and how many.

    Parameters
    ----------
    squares : float
        The sum of (x[k + 2m] - 2 x[k + m] + x[k])**2, ps**2.
    terms : int
        How many sets of three rows gave a term.
    """

    squares: float
    terms: int


def allan_sums(stretch: Stretch, m: int) -> AllanSums:
    """Sum the squared second differences at tau = mT over every whole set.

    Parameters
    ----------
    stretch : Stretch
        Prepared values, by epoch.
    m : int
        The tau, in epochs.

    Returns
    -------
    AllanSums
        Over every k with rows at k, k + m and k + 2m all present.

    Examples
    --------
    >>> allan_sums({0: 0.0, 1: 1.0, 2: 4.0, 4: 16.0}, 1)
    AllanSums(squares=4.0, terms=1)
    """
    squares = 0.0
    terms = 0
    for epoch, value in stretch.items():
        middle = stretch.get(epoch + m)
        last = stretch.get(epoch + 2 * m)
        if middle is not None and last is not None:
            second_difference = last - 2 * middle + value
            squares += second_difference * second_difference
            terms += 1
    return AllanSums(squares=squares, terms=terms)


def doubling_taus(stretch: Stretch) -> list[int]:
    """Give m = 1, 2, 4, ... up to a third of the time the rows cover.

    Parameters
    ----------
    stretch : Stretch
        Values by epoch.

    Returns
    -------
    list of int
        Every power of two m with 3m at most the epochs from the first row
        to the last.

    Examples
    --------
    >>> doubling_taus({0: 0.0, 13: 0.0}), doubling_taus({0: 0.0, 2: 0.0})
    ([1, 2, 4], [])
    """
    covered = max(stretch) - min(stretch) if stretch else 0
    taus = []
    m = 1
    while 3 * m <= covered:
        taus.append(m)
        m *= 2
    return taus


class TauValue(NamedTuple):
    """An Allan variance at one tau, and the terms behind it.

    Parameters
    ----------
    m : int
        The tau, in epochs.
    variance : float
        The Allan variance, dimensionless.
    terms : int
        How many terms it was averaged over.
    """

    m: int
    variance: float
    terms: int


def reference_variances(stretches: Sequence[Stretch]) -> list[TauValue]:
    """Combine the Allan variances of every stretch at each tau, by terms.

    Parameters
    ----------
    stretches : sequence of Stretch
        One local triple's prepared stretches.

    Returns
    -------
    list of TauValue
        For each tau some stretch reaches, the sum of all squares over the
        sum of all terms, as an Allan variance: the stretches' variances
        weighted by their number of terms.
    """
    totals: dict[int, AllanSums] = {}
    for stretch in stretches:
        for m in doubling_taus(stretch):
            stretch_sums = allan_sums(stretch, m)
            total = totals.get(m, AllanSums(0.0, 0))
            totals[m] = AllanSums(
                total.squares + stretch_sums.squares, total.terms + stretch_sums.terms
            )
    return [
        TauValue(
            m=m,
            variance=sums.squares * PS * PS / (2.0 * (m * T) ** 2 * sums.terms),
            terms=sums.terms,
        )
        for m, sums in sorted(totals.items())
        if sums.terms
    ]


def clock_variances(per_reference: Sequence[list[TauValue]]) -> list[TauValue]:
    """Combine every reference's variances at each tau by their terms.

    Parameters
    ----------
    per_reference : sequence of list of TauValue
        Each local triple's variances.

    Returns
    -------
    list of TauValue
        For each tau, the variances weighted by their number of terms.

    Examples
    --------
    >>> clock_variances([[TauValue(1, 2.0, 10)], [TauValue(1, 4.0, 30)]])
    [TauValue(m=1, variance=3.5, terms=40)]
    """
    totals: dict[int, tuple[float, int]] = {}
    for tau_values in per_reference:
        for tau_value in tau_values:
            weighted, terms = totals.get(tau_value.m, (0.0, 0))
            totals[tau_value.m] = (
                weighted + tau_value.variance * tau_value.terms,
                terms + tau_value.terms,
            )
    return [
        TauValue(m=m, variance=weighted / terms, terms=terms)
        for m, (weighted, terms) in sorted(totals.items())
    ]


type Coefficients = tuple[float, float, float, float]
"""a_-2, a_-1, a_0, a_1 of sigma_y**2(tau) = a_-2 / tau**2 + a_-1 / tau + a_0 + a_1 tau:
the measurement's white phase noise, then the clock's white, flicker and
random-walk frequency noise."""


def model_variance(coefficients: Coefficients, tau: float) -> float:
    """Give the fitted Allan variance at tau, the measurement's noise included.

    Parameters
    ----------
    coefficients : (float, float, float, float)
        a_-2, a_-1, a_0, a_1.
    tau : float
        The averaging time, s.

    Returns
    -------
    float
        a_-2 / tau**2 + a_-1 / tau + a_0 + a_1 tau.

    Examples
    --------
    >>> model_variance((360_000.0, 600.0, 1.0, 1 / 600), 600.0)
    4.0
    """
    return coefficients[0] / (tau * tau) + clock_variance(coefficients, tau)


def clock_variance(coefficients: Coefficients, tau: float) -> float:
    """Give the clock's own Allan variance at tau, sigma_y,c**2.

    Parameters
    ----------
    coefficients : (float, float, float, float)
        a_-2, a_-1, a_0, a_1; a_-2 is not used.
    tau : float
        The averaging time, s.

    Returns
    -------
    float
        a_-1 / tau + a_0 + a_1 tau.

    Examples
    --------
    >>> clock_variance((5.0, 600.0, 1.0, 1 / 600), 600.0)
    3.0
    """
    _, a_minus_1, a_0, a_1 = coefficients
    return a_minus_1 / tau + a_0 + a_1 * tau


def measurement_noise(coefficients: Coefficients) -> float:
    """Give the measurement noise the fit found, sigma_meas, ps.

    Parameters
    ----------
    coefficients : (float, float, float, float)
        a_-2, a_-1, a_0, a_1.

    Returns
    -------
    float
        sqrt(a_-2 / 3) in ps: white phase noise of sigma_meas has an Allan
        variance of 3 sigma_meas**2 / tau**2.

    Examples
    --------
    >>> round(measurement_noise((3e-24 * 100, 0.0, 0.0, 0.0)), 9)
    10.0
    """
    return math.sqrt(coefficients[0] / 3.0) / PS


def _basis(tau: float) -> tuple[float, float, float, float]:
    """Give the model's four terms at tau, each with coefficient one.

    Parameters
    ----------
    tau : float
        The averaging time, s.

    Returns
    -------
    (float, float, float, float)
        1 / tau**2, 1 / tau, 1 and tau.
    """
    return (1.0 / (tau * tau), 1.0 / tau, 1.0, tau)


def _normal_equations(
    columns: list[list[float]], weights: Sequence[float], values: Sequence[float]
) -> tuple[list[list[float]], list[float]]:
    """Build the weighted normal equations of a least-squares fit.

    Parameters
    ----------
    columns : list of list of float
        Each fitted term's value at every point.
    weights : sequence of float
        Each point's weight.
    values : sequence of float
        Each point's value.

    Returns
    -------
    (list of list of float, list of float)
        The matrix of weighted sums of products of the terms, and the
        weighted sums of each term times the values.

    Examples
    --------
    >>> _normal_equations([[1.0, 1.0], [0.0, 1.0]], [1.0, 2.0], [3.0, 5.0])
    ([[3.0, 2.0], [2.0, 2.0]], [13.0, 10.0])
    """
    matrix = [
        [
            sum(w * a * b for w, a, b in zip(weights, first, second, strict=True))
            for second in columns
        ]
        for first in columns
    ]
    vector = [
        sum(w * a * v for w, a, v in zip(weights, column, values, strict=True))
        for column in columns
    ]
    return matrix, vector


def _scaled_columns(
    tau_values: Sequence[TauValue], used: tuple[int, ...]
) -> tuple[list[list[float]], list[float]]:
    """Give each fitted term at every tau, scaled to at most one, and its scale.

    Parameters
    ----------
    tau_values : sequence of TauValue
        The variances.
    used : tuple of int
        Which of the four terms are fitted.

    Returns
    -------
    (list of list of float, list of float)
        Each term's values divided by the largest of them, and that largest.
        The terms differ by many orders of magnitude, so the fit is done on
        the scaled ones.
    """
    columns = [[_basis(tv.m * T)[term] for tv in tau_values] for term in used]
    scales = [max(abs(entry) for entry in column) for column in columns]
    scaled = [
        [entry / scale for entry in column]
        for column, scale in zip(columns, scales, strict=True)
    ]
    return scaled, scales


def _weighted_residual(
    tau_values: Sequence[TauValue], weights: Sequence[float], fitted: Coefficients
) -> float:
    """Give the weighted sum of squared differences from a model.

    Parameters
    ----------
    tau_values : sequence of TauValue
        The variances.
    weights : sequence of float
        Each one's weight.
    fitted : (float, float, float)
        The model.

    Returns
    -------
    float
        The sum of weight times (variance less the model's) squared.
    """
    return sum(
        w * (tv.variance - model_variance(fitted, tv.m * T)) ** 2
        for w, tv in zip(weights, tau_values, strict=True)
    )


def _weighted_fit(
    tau_values: Sequence[TauValue], weights: Sequence[float], used: tuple[int, ...]
) -> tuple[Coefficients, float] | None:
    """Fit only the terms in used by weighted least squares.

    Parameters
    ----------
    tau_values : sequence of TauValue
        The variances.
    weights : sequence of float
        Each one's weight.
    used : tuple of int
        Which of the four terms are fitted; the others are zero.

    Returns
    -------
    ((float, float, float), float) or None
        The coefficients and the weighted sum of squared residuals; ``None``
        when the system is singular or a coefficient comes out negative.
    """
    scaled, scales = _scaled_columns(tau_values, used)
    matrix, vector = _normal_equations(
        scaled, weights, [tv.variance for tv in tau_values]
    )
    try:
        solution = _solve(matrix, vector)
    except SingularError:
        return None
    if any(value < 0 for value in solution):
        return None
    coefficients = [0.0, 0.0, 0.0, 0.0]
    for term, value, scale in zip(used, solution, scales, strict=True):
        coefficients[term] = value / scale
    fitted = (coefficients[0], coefficients[1], coefficients[2], coefficients[3])
    return fitted, _weighted_residual(tau_values, weights, fitted)


def nonnegative_fit(
    tau_values: Sequence[TauValue], weights: Sequence[float]
) -> Coefficients:
    """Fit the model with every coefficient zero or above.

    Parameters
    ----------
    tau_values : sequence of TauValue
        The variances.
    weights : sequence of float
        Each one's weight.

    Returns
    -------
    (float, float, float, float)
        The coefficients with the least weighted squared residual among the
        fits of every set of terms that come out zero or above; all zero
        when none does.

    Examples
    --------
    >>> taus = [TauValue(m, 1.0 / (m * 600) + 1e-3, 10) for m in (1, 2, 4, 8, 16)]
    >>> [round(value, 9) for value in nonnegative_fit(taus, [1.0] * 5)]
    [0.0, 1.0, 0.001, 0.0]
    """
    best: tuple[Coefficients, float] | None = None
    for size in (1, 2, 3, 4):
        for used in combinations(range(4), size):
            fit = _weighted_fit(tau_values, weights, used)
            if fit is not None and (best is None or fit[1] < best[1]):
                best = fit
    return best[0] if best is not None else (0.0, 0.0, 0.0, 0.0)


def fit_noise_model(tau_values: Sequence[TauValue]) -> Coefficients:
    """Fit the model, weighted by the model itself, until it settles (step 8).

    Parameters
    ----------
    tau_values : sequence of TauValue
        The clock's variances, the measurement's noise in them.

    Returns
    -------
    (float, float, float, float)
        The coefficients. Each tau = mT is weighted by (terms / m) over the
        model's value there squared: the measured value at first, then the
        fitted model's, until no coefficient changes by more than
        :data:`FIT_TOLERANCE` of itself or :data:`FIT_ROUNDS` fits are done.
    """
    if not tau_values:
        return (0.0, 0.0, 0.0, 0.0)
    expected = [tv.variance for tv in tau_values]
    coefficients = (0.0, 0.0, 0.0, 0.0)
    for _ in range(FIT_ROUNDS):
        weights = [
            (tv.terms / tv.m) / (value * value)
            for tv, value in zip(tau_values, expected, strict=True)
        ]
        fitted = nonnegative_fit(tau_values, weights)
        settled = all(
            abs(new - old) <= FIT_TOLERANCE * abs(new)
            for new, old in zip(fitted, coefficients, strict=True)
        )
        coefficients = fitted
        if settled:
            break
        expected = [
            model_variance(coefficients, tv.m * T) or tv.variance for tv in tau_values
        ]
    return coefficients


def crossover(coefficients: Coefficients) -> float | None:
    """Find the tau at which measurement noise and clock noise are equal, s.

    Parameters
    ----------
    coefficients : (float, float, float, float)
        The fitted model.

    Returns
    -------
    float or None
        tau_c with a_-2 / tau_c**2 = sigma_y,c**2(tau_c), the smallest tau
        searched when the fit found no measurement noise; ``None`` when the
        clock's own model is zero, and so never reaches it.

    Examples
    --------
    >>> round(crossover((3e-24 * 100, 3e-24 * 100 / 600, 0.0, 0.0)))
    600
    """
    if not any(coefficients[1:]):
        return None

    def excess(tau: float) -> float:
        """Give tau**2 times the clock's variance less the measurement's, at tau."""
        return clock_variance(coefficients, tau) * tau * tau - coefficients[0]

    low, high = 1e-6, 1.0
    while excess(high) < 0:
        high *= 2.0
    for _ in range(200):
        middle = math.sqrt(low * high)
        if excess(middle) < 0:
            low = middle
        else:
            high = middle
    return high


def time_constant(tau_c: float | None) -> int | None:
    """Give the time constant from the crossover, in epochs, at least one.

    Parameters
    ----------
    tau_c : float or None
        The crossover, s; ``None`` when there is none.

    Returns
    -------
    int or None
        max(1, round(tau_c / T)); ``None`` without a crossover.

    Examples
    --------
    >>> time_constant(6100.0), time_constant(100.0), time_constant(None)
    (10, 1, None)
    """
    if tau_c is None:
        return None
    return max(1, round(tau_c / T))


def gap_limit(coefficients: Coefficients, M: int) -> int:
    """Design 13.2's G_max with no cap: the largest gap decycled with margin.

    Parameters
    ----------
    coefficients : (float, float, float, float)
        The fitted model; only the clock's own terms are used.
    M : int
        The time constant, epochs.

    Returns
    -------
    int
        The largest n for which 5 sigma_x,pred((n + 1) T) < P / 2, with
        sigma_x,pred(tau) = 10**12 tau sqrt(sigma_y,c**2(tau) + sigma_y,c**2(MT)),
        up to :data:`GAP_SEARCH_LIMIT`; -1 when not even n = 0 is.
    """
    settled = clock_variance(coefficients, M * T)

    def fits(n: int) -> bool:
        """Tell whether a gap of n held rows is decycled with the margin."""
        tau = (n + 1) * T
        sigma = tau / PS * math.sqrt(clock_variance(coefficients, tau) + settled)
        return GATE_SIGMAS * sigma < PHASE_PERIOD / 2

    if not fits(0):
        return -1
    low, high = 0, 1
    while high < GAP_SEARCH_LIMIT and fits(high):
        low, high = high, high * 2
    if high >= GAP_SEARCH_LIMIT and fits(GAP_SEARCH_LIMIT):
        return GAP_SEARCH_LIMIT
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


class ClockResult(NamedTuple):
    """One clock's characterization.

    Parameters
    ----------
    clock : str
        The clock.
    references : tuple of str
        The references whose local triples were used.
    rows : int
        The prepared rows used, over every reference.
    days_dropped : int
        The days left out for holding no clock signal, over every reference.
    sigma_meas : float
        The measurement noise the fit found, ps: sigma0.
    coefficients : (float, float, float, float)
        The fitted model.
    tau_c : float or None
        The crossover, s.
    M : int or None
        The time constant, epochs; ``None`` without a crossover.
    gap : int or None
        G_max, epochs; ``None`` without a time constant.
    variances : tuple of TauValue
        The clock's combined variances the model was fitted to.
    """

    clock: str
    references: tuple[str, ...]
    rows: int
    days_dropped: int
    sigma_meas: float
    coefficients: Coefficients
    tau_c: float | None
    M: int | None
    gap: int | None
    variances: tuple[TauValue, ...]


def prepare(
    triple_rows: Sequence[TripleRow], deltas: dict[int, float], drift: bool
) -> tuple[list[Stretch], int]:
    """Prepare one local triple's values in steps 1 to 7.

    Parameters
    ----------
    triple_rows : sequence of TripleRow
        The local triple's rows.
    deltas : dict of int to float
        Its local pair's measurement times after their epoch starts, s.
    drift : bool
        Whether the clock runs with three states, so drift is taken off.

    Returns
    -------
    (list of Stretch, int)
        The prepared values of the rows from each cold start up to the
        next, split at every jump; and how many days were left out for
        holding no clock signal.
    """
    prepared = []
    days_dropped = 0
    for stretch in split_at_cold_starts(triple_rows):
        with_signal, dropped = drop_no_signal_days(stretch)
        days_dropped += dropped
        cleaned = drop_outliers(to_epoch_start(with_signal, deltas))
        prepared += [
            remove_drift(piece) if drift else piece for piece in split_at_jumps(cleaned)
        ]
    return prepared, days_dropped


def characterize_clock(
    processed_path: Path,
    channel: RfChannel,
    clock: str,
    references: Sequence[str],
    drift: bool,
) -> ClockResult:
    """Characterize one clock from its local triples against each reference.

    Parameters
    ----------
    processed_path : Path
        The characterization run's processed_path.
    channel : {'a', 'b'}
        The RF channel.
    clock : str
        The clock.
    references : sequence of str
        The references with a local triple for it.
    drift : bool
        Whether its drift is taken off.

    Returns
    -------
    ClockResult
        The clock's settings and the variances they came from.
    """
    per_reference = []
    rows = 0
    days_dropped = 0
    for reference in references:
        triple_rows = read_triple_rows(
            series_file(processed_path, channel, (reference, reference, clock))
        )
        deltas = read_pair_deltas(
            series_file(processed_path, channel, (reference, clock))
        )
        stretches, dropped = prepare(triple_rows, deltas, drift)
        rows += sum(len(stretch) for stretch in stretches)
        days_dropped += dropped
        per_reference.append(reference_variances(stretches))
    variances = clock_variances(per_reference)
    coefficients = fit_noise_model(variances)
    tau_c = crossover(coefficients)
    M = time_constant(tau_c)
    gap = None if M is None else gap_limit(coefficients, M)
    return ClockResult(
        clock=clock,
        references=tuple(references),
        rows=rows,
        days_dropped=days_dropped,
        sigma_meas=measurement_noise(coefficients),
        coefficients=coefficients,
        tau_c=tau_c,
        M=M,
        gap=gap,
        variances=tuple(variances),
    )


def local_triples(processed_path: Path, channel: RfChannel) -> dict[str, list[str]]:
    """Find each clock's local triples (r, r, c), c not a reference.

    Parameters
    ----------
    processed_path : Path
        The characterization run's processed_path.
    channel : {'a', 'b'}
        The RF channel.

    Returns
    -------
    dict of str to list of str
        Each clock with the references r that have a file (r, r, c), the
        references sorted; references themselves and other triples left out.
    """
    found: dict[str, list[str]] = {}
    for data_file in sorted((processed_path / DDIFF_SUBDIRECTORY).iterdir()):
        series_key = series_key_of(data_file.name, channel)
        if series_key is None or len(series_key) != 3:
            continue
        r, s, c = series_key
        if r == s and not is_reference(c):
            found.setdefault(c, []).append(r)
    return found


def format_result(result: ClockResult) -> str:
    """Write one clock's result as one line of the report.

    Parameters
    ----------
    result : ClockResult
        The clock's characterization.

    Returns
    -------
    str
        The clock, its references, rows, days dropped, sigma_meas, the four
        coefficients, tau_c, M and G_max, separated by spaces; '-' for what
        is missing.

    Examples
    --------
    >>> print(format_result(ClockResult("hm1", ("mc1",), 10, 2, 3.0,
    ...                                  (2.7e-23, 1e-22, 0.0, 1e-33),
    ...                                  1200.0, 2, 50, ())))
    hm1 mc1 10 2 3.0 2.700e-23 1.000e-22 0.000e+00 1.000e-33 1200 2 50
    """
    fields = [
        result.clock,
        ",".join(result.references),
        str(result.rows),
        str(result.days_dropped),
        f"{result.sigma_meas:.1f}",
        *(f"{coefficient:.3e}" for coefficient in result.coefficients),
        "-" if result.tau_c is None else f"{result.tau_c:.0f}",
        "-" if result.M is None else str(result.M),
        "-" if result.gap is None else str(result.gap),
    ]
    return " ".join(fields)


REPORT_HEADER: Final[str] = (
    "clock references rows days_dropped sigma_meas_ps a_minus_2 a_minus_1 a_0 a_1"
    " tau_c_s time_constant gap_limit"
)
"""The report's first line: what each field holds."""


def _characterize_one(job: ClockJob) -> ClockResult:
    """Characterize one clock, from one argument, for a pool of processes.

    Parameters
    ----------
    job : (Path, str, str, tuple of str, bool)
        The arguments of :func:`characterize_clock`.

    Returns
    -------
    ClockResult
        The clock's characterization.
    """
    return characterize_clock(*job)


type ClockJob = tuple[Path, RfChannel, str, tuple[str, ...], bool]
"""The arguments of :func:`characterize_clock` for one clock."""


def clock_jobs(
    processed_path: Path, channel: RfChannel, three_state_prefixes: tuple[str, ...]
) -> list[ClockJob]:
    """Give the work for every clock with a local triple, in name order.

    Parameters
    ----------
    processed_path : Path
        The characterization run's processed_path.
    channel : {'a', 'b'}
        The RF channel.
    three_state_prefixes : tuple of str
        Name prefixes of the clocks that run with three states.

    Returns
    -------
    list of ClockJob
        Each clock with its references, its drift taken off exactly when
        its name starts with one of ``three_state_prefixes``.
    """
    return [
        (
            processed_path,
            channel,
            clock,
            tuple(references),
            clock.startswith(three_state_prefixes),
        )
        for clock, references in sorted(local_triples(processed_path, channel).items())
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Characterize every clock of a characterization run and print the report.

    Parameters
    ----------
    argv : sequence of str or None, optional
        The command-line arguments; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("processed_path", type=Path)
    parser.add_argument("--rf", choices=RF_CHOICES, required=True)
    parser.add_argument(
        "--three-state",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="name prefixes of the clocks whose drift is taken off",
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="clocks characterized at once"
    )
    cli_options = parser.parse_args(argv)
    jobs = clock_jobs(
        cli_options.processed_path, cli_options.rf, tuple(cli_options.three_state)
    )
    print(REPORT_HEADER)
    if cli_options.jobs == 1:
        results: Iterator[ClockResult] = map(_characterize_one, jobs)
        for result in results:
            print(format_result(result), flush=True)
        return 0
    with ProcessPoolExecutor(max_workers=cli_options.jobs) as pool:
        for result in pool.map(_characterize_one, jobs):
            print(format_result(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
