"""Tests for src/masterclock/das_processor/files.py.

The rules covered: the column table gives the measurement file a width of
477 and 33 header lines and the double-difference file 455 and 31; every
header line is exactly that wide and starts with '#', in the design's order
with the warning second; rows are right-justified fixed-width columns
separated by ', ', with '-' for an empty field, floats as {:+.16e} and the
estimator's phase as whole femtoseconds written in ps with three decimals;
the design's example rows come out byte for byte; a value too wide for its
column raises DataFileError; parsing a row gives back the record it was
formatted from and refuses a line that is not exactly what formatting gives
(U20); and a record pairs its measurement with its row consistently: a
measurement exactly when the row is not P, of the row's epoch, a pair's
slip correction exactly when the row carries S, and S never on a triple.

The file check: a file whose length is its header plus whole rows, with a
last row that parses, is sound and good through that row, its other rows
not read; otherwise it is good
through the row before its first line that does not parse, holds nothing
good when it has no whole row, and is refused when its first row does not
parse; and the last row of a sound file is read back as its row (U26).

The write: every check is made before a file is opened, and a failed one
changes nothing; files are written one at a time, measurement files first;
with a journal, the earliest buffered epoch is flushed to it before any data
file opens and it is deleted after the last flush, a journal already there
is refused, and one not whole is read as no write stopped.

A row that parses but breaks a record's rules is a damaged line; every
refusal, device fault, roll-back and redo message is word for word, and each
error is logged as raised; free space is counted per device, just enough
being enough; a missing archive is made beside one already there; and a
measurement time on a whole second keeps its microseconds.
"""

import logging
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Final, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import files
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.read_cd5m5m import RMS_MAX
from masterclock.domain.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.domain.phase import PHASE_MAX, exact
from masterclock.domain.series import PairKey, Row, SeriesKey, State, TripleKey

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""The worked epoch's start."""

STEP: Final = timedelta(minutes=10)
"""One epoch."""

RAW: Final[dict[str, float | int]] = {
    "measurement_mjd": 60941.251588,
    "measured_phase": 34579,
    "rms": 3,
}
"""Appendix A's reading: its MJD, phase and rms."""

PAIR: Final = measure_pair(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    prediction=State(x=1_234_567 + exact(0.0123) * 600, y=0.0123),
    w=Fraction(0),
    anchor=None,
)
"""Appendix A's pair measurement."""


def row(**changes: object) -> Row:
    """Build the worked epoch's accepted row, with ``changes`` applied."""
    values: dict[str, object] = {
        "interpolated_datetime": E,
        "innovation": None,
        "x_fs": 1_234_574_457,
        "y": 0.01230129052352643,
        "d": 7.169515400974333e-12,
        "innovation_scale": 3.0,
        "segment": 4,
        "step_offset": 0,
        "epochs_in_segment": 812,
        "epochs_since_accept": 0,
        "consecutive_rejects": 0,
        "rejects": (),
        "filter_states": 3,
        "time_constant": 100.0,
        "scale_time_constant": 50.0,
        "flags": "A",
    }
    values.update(changes)
    return Row.model_validate(values)


MEAS_EXAMPLE: Final = (
    "2025-09-23 06:00:00+00:00,  60941.250000, 2025-09-23 06:02:17.203200+00:00,"
    "  60941.251588,  34579,    3,            6,          1234577,"
    "          1234574.457, +1.2301290523526430e-02, +7.1695154009743332e-12,"
    " +3.0000000000000000e+00,         4,                0,       812,         0,"
    "         0,             -,                       -,             -,"
    "                       -,             -,                       -, 3,"
    " +1.0000000000000000e+02, +5.0000000000000000e+01,        A",
    "2025-09-23 06:10:00+00:00,  60941.256944,                                -,"
    "             -,      -,    -,            -,                -,"
    "          1234581.838, +1.2301294825235671e-02, +7.1695154009743332e-12,"
    " +3.0000000000000000e+00,         4,                0,       813,         1,"
    "         0,             -,                       -,             -,"
    "                       -,             -,                       -, 3,"
    " +1.0000000000000000e+02, +5.0000000000000000e+01,        P",
)
"""The two example rows of design 5.4."""

DDIFF_EXAMPLE: Final = (
    "2025-09-23 06:00:00+00:00,  60941.250000,          6666667,"
    " -2.9999999981373549e-01, +3.3166247903553998e+00, 111,          6666667.291,"
    " +2.0499852230130653e-02, -8.2093687746445554e-13, +3.4650829715892466e+00,"
    "         2,                0,      3107,         0,         0,             -,"
    "                       -,             -,                       -,"
    "             -,                       -, 3, +1.0000000000000000e+02,"
    " +5.0000000000000000e+01,        A",
    "2025-09-23 06:00:00+00:00,  60941.250000,          1234577,"
    " +2.6200000000000001e+00, +3.0000000000000000e+00, 111,          1234574.457,"
    " +1.2301290523526430e-02, +7.1695154009743332e-12, +3.0000000000000000e+00,"
    "         4,                0,       812,         0,         0,             -,"
    "                       -,             -,                       -,"
    "             -,                       -, 3, +1.0000000000000000e+02,"
    " +5.0000000000000000e+01,        A",
)
"""The two example rows of design 5.5: the remote triple, then the local one."""


# ------------------------------------------------------------------ widths


def test_the_column_table_gives_the_widths_and_header_lengths() -> None:
    """Work out W = 477 and 455 and 33 and 31 header lines from the table."""
    assert (files.MEAS_WIDTH, files.MEAS_HEADER_LINES) == (477, 33)
    assert (files.DDIFF_WIDTH, files.DDIFF_HEADER_LINES) == (455, 31)


# ------------------------------------------------------------------- headers

KEYS: Final[tuple[tuple[files.FileKind, SeriesKey], ...]] = (
    ("meas", ("mc2", "nav23")),
    ("meas", ("mc1", "mc1")),
    ("meas", ("mc1", "mc2")),
    ("ddiff", ("mc1", "mc2", "nav23")),
    ("ddiff", ("mc2", "mc2", "nav23")),
)
"""A series of every kind."""


@pytest.mark.parametrize(("kind", "key"), KEYS)
def test_every_header_line_is_w_wide_and_starts_with_a_hash(
    kind: files.FileKind, key: SeriesKey
) -> None:
    """Pad every header line to W, each a comment, the count from the table."""
    lines = files.header(kind, "a", key).split("\n")
    assert lines[-1] == ""
    width, count = files.WIDTHS[kind], files.HEADER_LINES[kind]
    assert len(lines[:-1]) == count
    assert all(len(line) == width and line.startswith("#") for line in lines[:-1])


def test_the_measurement_header_is_the_design_s() -> None:
    """Give the lines of design 5.4 in order, the warning second."""
    lines = [
        line.rstrip()
        for line in files.header("meas", "a", ("mc2", "nav23")).split("\n")
    ]
    assert lines[:6] == [
        "# das_processor measurement file, format 1",
        "# WARNING: do not modify this file. Only das_processor may write it;"
        " any other change damages the archive.",
        "# RF channel a. Pair (mc2, nav23).",
        "# Reference mc2 measured against clock nav23.",
        "# One row per 10-minute epoch; '-' marks an empty field.",
        "# Columns: right-justified, fixed width, separated by ', '.",
    ]
    assert lines[6] == "#   interpolated_datetime   epoch start E, UTC"
    assert lines[32] == (
        "#   flags                   A accepted, R rejected, X excluded, P predicted,"
        " D dormant, S slip corrected, N new segment, U unsettled"
    )


def test_the_double_difference_header_is_the_design_s() -> None:
    """Give the lines of design 5.5 in order, the warning second."""
    text = files.header("ddiff", "b", ("mc1", "mc2", "nav23"))
    lines = [line.rstrip() for line in text.split("\n")]
    assert lines[0] == "# das_processor double-difference file, format 1"
    assert lines[2:4] == [
        "# RF channel b. Triple (mc1, mc2, nav23).",
        "# Clock nav23 against remote reference mc1, through local reference mc2.",
    ]
    assert lines[8] == "#   z                       double difference dd at E, ps"
    assert lines[10] == "#   double_difference_sigma measurement sigma of dd, ps"


@pytest.mark.parametrize(
    ("kind", "key"), [("meas", ("mc1", "mc2", "nav23")), ("ddiff", ("mc1", "nav23"))]
)
def test_a_header_is_for_its_kind_of_series(
    kind: files.FileKind, key: SeriesKey
) -> None:
    """Raise DataFileError for a triple's measurement file or a pair's other file."""
    with pytest.raises(DataFileError, match="is for a"):
        files.header(kind, "a", key)


@pytest.mark.parametrize(
    ("kind", "key", "line"),
    [
        ("meas", ("mc1", "mc1"), "# Reference mc1 measured against itself."),
        ("meas", ("mc1", "mc2"), "# Reference mc1 measured against reference mc2."),
        (
            "ddiff",
            ("mc2", "mc2", "nav23"),
            "# Clock nav23 against its local reference mc2.",
        ),
    ],
)
def test_the_header_says_what_the_series_is(
    kind: files.FileKind, key: SeriesKey, line: str
) -> None:
    """Describe self and link pairs and local triples in words."""
    assert files.header(kind, "a", key).split("\n")[3].rstrip() == line


# ---------------------------------------------------------------- examples


def test_the_measurement_example_rows_come_out_byte_for_byte() -> None:
    """Format the two rows of design 5.4 from their values."""
    accepted = files.MeasRecord(measurement=PAIR, row=row())
    predicted = files.MeasRecord(
        measurement=None,
        row=row(
            interpolated_datetime=E + STEP,
            x_fs=1_234_581_838,
            y=0.012301294825235671,
            epochs_in_segment=813,
            epochs_since_accept=1,
            flags="P",
        ),
    )
    assert files.format_meas_row(accepted) == MEAS_EXAMPLE[0]
    assert files.format_meas_row(predicted) == MEAS_EXAMPLE[1]
    assert all(len(line) == files.MEAS_WIDTH for line in MEAS_EXAMPLE)


def test_the_double_difference_example_rows_come_out_byte_for_byte() -> None:
    """Format the two rows of design 5.5 from their values."""
    remote = files.DdiffRecord(
        measurement=TripleMeasurement(
            z=6_666_667,
            double_difference_sigma=3.3166247903553998,
            components_used="111",
            cold=False,
        ),
        row=row(
            innovation=-0.29999999981373549,
            x_fs=6_666_667_291,
            y=0.020499852230130653,
            d=-8.2093687746445554e-13,
            innovation_scale=3.4650829715892466,
            segment=2,
            epochs_in_segment=3107,
        ),
    )
    local = files.DdiffRecord(
        measurement=TripleMeasurement(
            z=1_234_577, double_difference_sigma=3.0, components_used="111", cold=False
        ),
        row=row(innovation=2.62),
    )
    assert files.format_ddiff_row(remote) == DDIFF_EXAMPLE[0]
    assert files.format_ddiff_row(local) == DDIFF_EXAMPLE[1]
    assert all(len(line) == files.DDIFF_WIDTH for line in DDIFF_EXAMPLE)


def test_the_example_rows_parse_back() -> None:
    """Parse the design's example rows into records that format to them."""
    record = files.parse_meas_row(MEAS_EXAMPLE[0])
    assert record == files.MeasRecord(measurement=PAIR, row=row())
    assert files.parse_meas_row(MEAS_EXAMPLE[1]).measurement is None
    for line in DDIFF_EXAMPLE:
        assert files.format_ddiff_row(files.parse_ddiff_row(line)) == line


# ----------------------------------------------------- empty fields, x, rejects


def test_a_dormant_row_writes_its_state_as_empty_fields() -> None:
    """Write '-' for x, y, d and the scale of a dormant row, right-justified."""
    dormant = row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        innovation=None,
        rejects=((E - STEP, 1_234_570.0),),
    )
    line = files.format_meas_row(files.MeasRecord(measurement=PAIR, row=dormant))
    fields = line.split(", ")
    assert fields[8:12] == ["-".rjust(20), "-".rjust(23), "-".rjust(23), "-".rjust(23)]
    assert fields[17:19] == [" 60941.243056", "+1.2345700000000000e+06"]
    assert files.parse_meas_row(line).row == dormant


@pytest.mark.parametrize(
    ("x_fs", "text"),
    [
        (1_234_574_457, "1234574.457"),
        (-1_500, "-1.500"),
        (7, "0.007"),
        (-7, "-0.007"),
        (0, "0.000"),
    ],
)
def test_the_phase_is_written_in_ps_to_the_femtosecond(x_fs: int, text: str) -> None:
    """Write whole femtoseconds as ps with three decimals, sign and all."""
    line = files.format_meas_row(files.MeasRecord(measurement=PAIR, row=row(x_fs=x_fs)))
    assert line.split(", ")[8] == text.rjust(20)
    assert files.parse_meas_row(line).row.x_fs == x_fs


# ------------------------------------------------------------------ overflow


@pytest.mark.parametrize(
    "changes",
    [
        {"step_offset": 10**16},
        {"segment": 10**9},
        {"x_fs": -(10**19)},
        {"y": 1e-300},
        {"innovation_scale": 1e100},
    ],
)
def test_a_value_too_wide_for_its_column_is_refused(changes: dict[str, object]) -> None:
    """Raise DataFileError naming the column a value does not fit."""
    with pytest.raises(DataFileError, match="does not fit"):
        files.format_meas_row(files.MeasRecord(measurement=PAIR, row=row(**changes)))


def test_a_cycle_count_too_wide_is_refused() -> None:
    """Raise DataFileError for a cycle count wider than its column."""
    pair = PairMeasurement.model_validate({**RAW, "cycle_count": 10**12, "z": 1})
    with pytest.raises(DataFileError, match="cycle_count"):
        files.format_meas_row(files.MeasRecord(measurement=pair, row=row()))


# ---------------------------------------------------------- refused lines


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("        A", "       AX"),
        ("            6,", "           +6,"),
        ("          1234574.457,", "         1234574.4575,"),
        ("          1234574.457,", "           1234574.45,"),
        ("+1.2301290523526430e-02", "+1.230129052352643e-02 "),
        ("+1.2301290523526430e-02", "                    nan"),
        (", 3, +1.0", ", 2, +1.0"),
        ("  60941.250000,", "  60941.250001,"),
        ("2025-09-23 06:00:00+00:00", "2025-09-23 06:05:00+00:00"),
        ("2025-09-23 06:00:00+00:00", "2025-09-23 06:00:00-05:00"),
        (",       812,", ",       81x,"),
        ("  34579,", " 234579,"),
        ("e+00,         4,", "e+00,         -,"),
    ],
)
def test_a_line_not_exactly_as_written_is_refused(old: str, new: str) -> None:
    """Raise DataFileError for any line formatting would not give back (U20)."""
    assert MEAS_EXAMPLE[0].count(old) == 1
    line = MEAS_EXAMPLE[0].replace(old, new)
    with pytest.raises(DataFileError):
        files.parse_meas_row(line)


@pytest.mark.parametrize(
    "line",
    [
        MEAS_EXAMPLE[0][:-1],
        MEAS_EXAMPLE[0] + ", 1",
        MEAS_EXAMPLE[0].replace(", ", ",", 1),
        "",
    ],
)
def test_a_line_of_the_wrong_shape_is_refused(line: str) -> None:
    """Raise DataFileError for a line with the wrong columns."""
    with pytest.raises(DataFileError):
        files.parse_meas_row(line)


def test_a_double_difference_line_is_checked_as_written() -> None:
    """Raise DataFileError for a double-difference line formatting would not give."""
    with pytest.raises(DataFileError):
        files.parse_ddiff_row(DDIFF_EXAMPLE[0].replace(", 111,", ", 011,"))
    with pytest.raises(DataFileError):
        files.parse_ddiff_row(DDIFF_EXAMPLE[0][:-2])


# ------------------------------------------------------------------- records


def test_a_record_has_a_measurement_exactly_when_its_row_is_not_predicted() -> None:
    """Refuse a P row with a measurement and another row without one."""
    with pytest.raises(DataFileError, match="measurement"):
        files.MeasRecord(measurement=PAIR, row=row(flags="P"))
    with pytest.raises(DataFileError, match="measurement"):
        files.MeasRecord(measurement=None, row=row())
    triple = TripleMeasurement(
        z=1, double_difference_sigma=1.0, components_used="111", cold=False
    )
    with pytest.raises(DataFileError, match="measurement"):
        files.DdiffRecord(measurement=triple, row=row(flags="P"))


def test_a_slip_correction_and_the_s_flag_go_together() -> None:
    """Refuse a corrected measurement without S, S without one, and S on a triple."""
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(measurement=None, row=row(flags="PS"))
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(measurement=PAIR.corrected(1), row=row())
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(measurement=PAIR, row=row(flags="AS"))
    triple = TripleMeasurement(
        z=1, double_difference_sigma=1.0, components_used="111", cold=False
    )
    with pytest.raises(DataFileError, match="S"):
        files.DdiffRecord(measurement=triple, row=row(flags="AS"))


def test_a_measurement_belongs_to_its_row_s_epoch() -> None:
    """Refuse a pair measurement taken in another epoch than its row's."""
    with pytest.raises(DataFileError, match="epoch"):
        files.MeasRecord(measurement=PAIR, row=row(interpolated_datetime=E + STEP))


# ------------------------------------------------------------- round trips

MARKS: Final = st.integers(min_value=0, max_value=20 * 365 * 144).map(
    lambda k: datetime(2015, 1, 1, tzinfo=UTC) + k * STEP
)
"""Ten-minute marks over twenty years."""

FLOATS: Final = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-99, max_value=1e99),
    st.floats(min_value=-1e99, max_value=-1e-99),
)
"""Floats that fit a float column."""


@st.composite
def rows(draw: st.DrawFn, *, measured: bool, pair: bool) -> Row:
    """Draw a valid row; with ``measured``, one with a measurement.

    A pair's row may carry S when measured and has no innovation, which its
    file does not hold; a triple's never carries S.
    """
    mark = draw(MARKS)
    states: Literal[1, 2, 3] = draw(st.sampled_from([1, 2, 3]))
    dormant = draw(st.booleans())
    outcomes = ("R", "X") if dormant else ("A", "R", "X")
    outcome = draw(st.sampled_from(outcomes)) if measured else "P"
    extra = draw(st.sets(st.sampled_from("SN" if pair and measured else "N")))
    if not dormant and states > 1 and draw(st.booleans()):
        extra.add("U")
    flags = "".join(
        c for c in "ARXPDSNU" if c in {outcome, *extra, *("D" if dormant else "")}
    )
    count = draw(st.integers(min_value=0, max_value=3))
    rejects = tuple((mark - STEP * (count - i), draw(FLOATS)) for i in range(count))
    tracked = not dormant
    return Row(
        interpolated_datetime=mark,
        innovation=draw(st.one_of(st.none(), FLOATS))
        if measured and not pair
        else None,
        x_fs=draw(st.integers(1 - 10**18, 10**18 - 1)) if tracked else None,
        y=(draw(FLOATS) if states > 1 else 0.0) if tracked else None,
        d=(draw(FLOATS) if states == 3 else 0.0) if tracked else None,
        innovation_scale=abs(draw(FLOATS)) if tracked else None,
        segment=draw(st.integers(0, 10**8)),
        step_offset=draw(st.integers(-(10**14), 10**14)),
        epochs_in_segment=draw(st.integers(0, 10**8)),
        epochs_since_accept=draw(st.integers(0, 10**8)),
        consecutive_rejects=draw(st.integers(0, 10**8)),
        rejects=rejects,
        filter_states=states,
        time_constant=draw(st.floats(1, 1e6)) if states > 1 else None,
        scale_time_constant=draw(st.floats(1, 1e6)),
        flags=flags,
    )


@st.composite
def meas_records(draw: st.DrawFn) -> files.MeasRecord:
    """Draw a valid measurement file record."""
    measured = draw(st.booleans())
    drawn = draw(rows(measured=measured, pair=True))
    if not measured:
        return files.MeasRecord(measurement=None, row=drawn)
    start = datetime_to_mjd(drawn.interpolated_datetime)
    offset = draw(st.integers(min_value=1, max_value=6_900))
    pair = PairMeasurement(
        measurement_mjd=round(start + offset * 1e-6, 6),
        measured_phase=draw(st.integers(0, PHASE_MAX)),
        rms=draw(st.integers(0, RMS_MAX)),
        cycle_count=draw(st.integers(-(10**10), 10**10)),
        z=draw(st.integers(-(10**14), 10**14)),
        slip="S" in drawn.flags,
    )
    return files.MeasRecord(measurement=pair, row=drawn)


@st.composite
def ddiff_records(draw: st.DrawFn) -> files.DdiffRecord:
    """Draw a valid double-difference file record."""
    measured = draw(st.booleans())
    drawn = draw(rows(measured=measured, pair=False))
    if not measured:
        return files.DdiffRecord(measurement=None, row=drawn)
    triple = TripleMeasurement(
        z=draw(st.integers(-(10**14), 10**14)),
        double_difference_sigma=draw(st.floats(min_value=1e-99, max_value=1e99)),
        components_used=draw(st.sampled_from(["111", "110", "101"])),
        cold=False,
    )
    return files.DdiffRecord(measurement=triple, row=drawn)


@given(meas_records())
def test_a_measurement_row_parses_back_to_its_record(record: files.MeasRecord) -> None:
    """Give back the record, and the same text when formatted again (U20)."""
    line = files.format_meas_row(record)
    assert len(line) == files.MEAS_WIDTH
    assert files.parse_meas_row(line) == record
    assert files.format_meas_row(files.parse_meas_row(line)) == line


@given(ddiff_records())
def test_a_double_difference_row_parses_back_to_its_record(
    record: files.DdiffRecord,
) -> None:
    """Give back the record, and the same text when formatted again (U20)."""
    line = files.format_ddiff_row(record)
    assert len(line) == files.DDIFF_WIDTH
    assert files.parse_ddiff_row(line) == record


# --------------------------------------------------- file check, last row

KEY: Final = ("mc2", "nav23")
"""The pair the files below are for."""


def series_rows(count: int) -> list[str]:
    """Give ``count`` rows of the pair, one per epoch from E, each with its newline."""
    lines = []
    for index in range(count):
        record = files.MeasRecord(
            measurement=None,
            row=row(
                interpolated_datetime=E + index * STEP,
                epochs_in_segment=812 + index,
                epochs_since_accept=index + 1,
                flags="P",
            ),
        )
        lines.append(files.format_meas_row(record) + "\n")
    return lines


def meas_file(tmp_path: Path, rows_text: str, *, head: str | None = None) -> Path:
    """Write a measurement file of the pair: its header and ``rows_text``."""
    path = tmp_path / "das_a.mc2.nav23.dat"
    text = files.header("meas", "a", KEY) if head is None else head
    path.write_bytes((text + rows_text).encode("ascii"))
    return path


def test_a_sound_file_is_good_through_its_last_row(tmp_path: Path) -> None:
    """Give the last row's epoch for a file of whole rows (U26)."""
    path = meas_file(tmp_path, "".join(series_rows(4)))
    assert files.good_through(path, "meas") == E + 3 * STEP


def test_a_file_cut_inside_its_last_row_is_good_through_the_row_before(
    tmp_path: Path,
) -> None:
    """Give the epoch of the last whole row of a file with a torn line (U26)."""
    text = "".join(series_rows(4))
    path = meas_file(tmp_path, text[: -files.MEAS_WIDTH // 2])
    assert files.good_through(path, "meas") == E + 2 * STEP


@pytest.mark.parametrize(
    "cut", [0, 1, files.MEAS_WIDTH + 1, 10 * (files.MEAS_WIDTH + 1) + 7]
)
def test_a_file_without_a_whole_row_holds_nothing_good(
    tmp_path: Path, cut: int
) -> None:
    """Give None for a file cut inside its header, or holding only its header (U26)."""
    whole = files.header("meas", "a", KEY)
    path = meas_file(tmp_path, "", head=whole[:cut])
    assert files.good_through(path, "meas") is None
    assert files.good_through(meas_file(tmp_path, "", head=whole), "meas") is None


def test_a_row_of_the_wrong_length_inside_a_file_ends_what_is_good(
    tmp_path: Path,
) -> None:
    """Give the epoch before the first line that does not parse (U26)."""
    lines = series_rows(5)
    lines[2] = lines[2][:100] + lines[2][101:]
    path = meas_file(tmp_path, "".join(lines))
    assert files.good_through(path, "meas") == E + STEP


def predicted_as_accepted(line: str) -> str:
    """Give a predicted row with its flag P made A: a row with no measurement."""
    fields = line.split(", ")
    (at,) = [i for i, field in enumerate(fields) if field.strip() == "P"]
    fields[at] = fields[at].replace("P", "A")
    return ", ".join(fields)


def test_a_row_that_breaks_a_record_rule_is_damaged(tmp_path: Path) -> None:
    """End what is good at a row that parses but breaks a record's rules (U26)."""
    lines = series_rows(4)
    lines[3] = predicted_as_accepted(lines[3])
    path = meas_file(tmp_path, "".join(lines))
    assert files.good_through(path, "meas") == E + 2 * STEP
    files.roll_back(path, "meas", E + 2 * STEP)
    assert files.good_through(path, "meas") == E + 2 * STEP
    assert path.read_text().endswith(series_rows(3)[2])


def test_a_sound_file_is_not_scanned(tmp_path: Path) -> None:
    """Take a file of whole rows whose last row parses as sound, unscanned."""
    lines = series_rows(5)
    lines[3] = lines[3].replace("        P", "        Q")
    path = meas_file(tmp_path, "".join(lines))
    assert files.good_through(path, "meas") == E + 4 * STEP


def test_a_file_whose_last_row_does_not_parse_is_scanned(tmp_path: Path) -> None:
    """Scan a file of whole rows when its last row does not parse."""
    lines = series_rows(5)
    lines[4] = lines[4].replace("        P", "        Q")
    path = meas_file(tmp_path, "".join(lines))
    assert files.good_through(path, "meas") == E + 3 * STEP


def test_a_row_whose_newline_is_lost_is_not_good(tmp_path: Path) -> None:
    """Refuse a slot that does not end in a newline, though its text parses."""
    text = "".join(series_rows(3))
    path = meas_file(tmp_path, text[:-1] + "x")
    assert files.good_through(path, "meas") == E + STEP


def test_the_scan_stops_at_the_first_damaged_line(tmp_path: Path) -> None:
    """Stop at the first damaged line, however good the lines after it."""
    lines = series_rows(5)
    lines[1] = lines[1].replace("        P", "        Q")
    path = meas_file(tmp_path, "".join(lines) + "2025")
    assert files.good_through(path, "meas") == E


def test_a_damaged_first_row_cannot_be_placed_in_time(tmp_path: Path) -> None:
    """Raise DataFileError when no row of the file parses (U26)."""
    lines = series_rows(3)
    lines[0] = lines[0].replace("        P", "        Q")
    path = meas_file(tmp_path, "".join(lines) + "2025")
    with pytest.raises(DataFileError, match="damaged first row"):
        files.good_through(path, "meas")


@pytest.mark.parametrize(
    "line",
    [b"#" + b" " * files.MEAS_WIDTH, b"x" * files.MEAS_WIDTH, "é".encode() * 300],
)
def test_a_line_that_is_not_a_row_has_no_epoch(line: bytes) -> None:
    """Give no epoch for a header line, a line with no newline, or non-ASCII."""
    assert files.row_epoch(line + b"\n", "meas") is None
    assert files.row_epoch(series_rows(1)[0].encode()[:-1], "meas") is None


def test_a_row_gives_its_epoch() -> None:
    """Give a good row's epoch."""
    assert files.row_epoch(series_rows(2)[1].encode(), "meas") == E + STEP


def test_the_last_row_of_a_sound_file_is_read(tmp_path: Path) -> None:
    """Read a sound file's last row back as its row."""
    path = meas_file(tmp_path, "".join(series_rows(3)))
    last = files.read_last_row(path, "meas")
    assert last == row(
        interpolated_datetime=E + 2 * STEP,
        epochs_in_segment=814,
        epochs_since_accept=3,
        flags="P",
    )


def test_the_last_row_of_a_double_difference_file_is_read(tmp_path: Path) -> None:
    """Read a double-difference file's last row."""
    path = tmp_path / "das_a.mc1.mc2.nav23.dat"
    key = ("mc1", "mc2", "nav23")
    text = files.header("ddiff", "a", key) + DDIFF_EXAMPLE[0] + "\n"
    path.write_bytes(text.encode("ascii"))
    assert files.read_last_row(path, "ddiff").x_fs == 6_666_667_291
    assert files.good_through(path, "ddiff") == E


@pytest.mark.parametrize("cut", [1, 2 * (files.MEAS_WIDTH + 1)])
def test_the_last_row_is_read_only_from_a_sound_file(tmp_path: Path, cut: int) -> None:
    """Raise DataFileError for a file that is torn or holds no row."""
    text = "".join(series_rows(2))
    path = meas_file(tmp_path, text[:-cut])
    with pytest.raises(DataFileError, match="not sound"):
        files.read_last_row(path, "meas")


def test_a_last_row_that_is_not_ascii_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for a sound-sized file whose last row is not ASCII."""
    path = meas_file(tmp_path, "".join(series_rows(2)))
    data = bytearray(path.read_bytes())
    data[-3] = 0xE9
    path.write_bytes(bytes(data))
    with pytest.raises(DataFileError, match="not ASCII"):
        files.read_last_row(path, "meas")


def test_a_file_that_cannot_be_opened_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for a file that cannot be read."""
    with pytest.raises(DataFileError, match="cannot read"):
        files.good_through(tmp_path / "missing.dat", "meas")
    with pytest.raises(DataFileError, match="cannot read"):
        files.read_last_row(tmp_path / "missing.dat", "meas")


# ------------------------------------------------------ day buffer and write

TRIPLE: Final = ("mc1", "mc2", "nav23")
"""The triple the double-difference files below are for."""


def directories(tmp_path: Path) -> tuple[Path, Path]:
    """Make the two archive directories."""
    meas, ddiff = tmp_path / "meas", tmp_path / "ddiff"
    meas.mkdir()
    ddiff.mkdir()
    return meas, ddiff


def predicted(index: int) -> files.MeasRecord:
    """Give the pair's predicted record ``index`` epochs after E."""
    return files.MeasRecord(
        measurement=None,
        row=row(
            interpolated_datetime=E + index * STEP,
            epochs_in_segment=812 + index,
            epochs_since_accept=index + 1,
            flags="P",
        ),
    )


def triple_record(index: int) -> files.DdiffRecord:
    """Give the triple's predicted record ``index`` epochs after E."""
    return files.DdiffRecord(measurement=None, row=predicted(index).row)


def filled(tmp_path: Path, epochs: int = 2) -> tuple[files.DayBuffer, Path, Path]:
    """Give a buffer of ``epochs`` rows for the pair and the triple, and their paths."""
    meas, ddiff = directories(tmp_path)
    pair_path, triple_path = (
        meas / "das_a.mc2.nav23.dat",
        ddiff / "das_a.mc1.mc2.nav23.dat",
    )
    buffer = files.DayBuffer("a")
    for index in range(epochs):
        buffer.add(triple_path, TRIPLE, triple_record(index))
        buffer.add(pair_path, KEY, predicted(index))
    return buffer, pair_path, triple_path


def test_a_new_file_gets_its_header_and_rows_in_one_write(tmp_path: Path) -> None:
    """Create each file with its header then its rows, all at once (5.8)."""
    buffer, pair_path, triple_path = filled(tmp_path)
    files.write_buffer(buffer)
    expected = files.header("meas", "a", KEY) + "".join(
        files.format_meas_row(predicted(i)) + "\n" for i in range(2)
    )
    assert pair_path.read_text(encoding="ascii") == expected
    assert files.good_through(triple_path, "ddiff") == E + STEP


def test_a_later_write_appends_to_the_file(tmp_path: Path) -> None:
    """Append the next day's rows after the rows already written."""
    buffer, pair_path, _ = filled(tmp_path)
    files.write_buffer(buffer)
    buffer.add(pair_path, KEY, predicted(2))
    files.write_buffer(buffer)
    assert files.good_through(pair_path, "meas") == E + 2 * STEP
    assert (
        pair_path.read_text(encoding="ascii").count("das_processor measurement file")
        == 1
    )


def test_a_series_first_seen_in_a_day_appears_at_the_day_s_write(
    tmp_path: Path,
) -> None:
    """Create a file for a series first seen mid-day, header first, at the write."""
    buffer, pair_path, _ = filled(tmp_path)
    files.write_buffer(buffer)
    later = pair_path.parent / "das_a.mc2.cs7.dat"
    buffer.add(pair_path, KEY, predicted(2))
    buffer.add(later, ("mc2", "cs7"), predicted(2))
    assert not later.exists()
    files.write_buffer(buffer)
    text = later.read_text(encoding="ascii")
    assert text.startswith("# das_processor measurement file, format 1")
    assert files.read_last_row(later, "meas").interpolated_datetime == E + 2 * STEP


def test_after_a_write_the_text_is_empty_and_the_last_rows_remain(
    tmp_path: Path,
) -> None:
    """Empty the buffer's text and keep each series' newest row (5.8)."""
    buffer, _, _ = filled(tmp_path)
    files.write_buffer(buffer)
    assert buffer.texts == {}
    assert buffer.last == {KEY: predicted(1).row, TRIPLE: triple_record(1).row}


def test_the_newest_row_is_the_one_read_back(tmp_path: Path) -> None:
    """Keep the row parsed back from its line, as a later run would read it (I5)."""
    meas, _ = directories(tmp_path)
    buffer = files.DayBuffer("a")
    record = files.MeasRecord(measurement=PAIR, row=row(innovation=2.62))
    buffer.add(meas / "das_a.mc2.nav23.dat", KEY, record)
    assert buffer.last[KEY].innovation is None
    assert buffer.last[KEY] == row()


def test_a_row_too_wide_is_refused_before_it_is_buffered(tmp_path: Path) -> None:
    """Raise DataFileError at add, before any text is kept."""
    meas, _ = directories(tmp_path)
    buffer = files.DayBuffer("a")
    record = files.MeasRecord(measurement=None, row=row(step_offset=10**16, flags="P"))
    with pytest.raises(DataFileError, match="does not fit"):
        buffer.add(meas / "das_a.mc2.nav23.dat", KEY, record)
    assert buffer.texts == {}


def test_a_file_keeps_one_series(tmp_path: Path) -> None:
    """Raise DataFileError when a path is given rows of two series or kinds."""
    buffer, pair_path, _ = filled(tmp_path)
    with pytest.raises(DataFileError, match="series"):
        buffer.add(pair_path, ("mc2", "cs7"), predicted(2))
    with pytest.raises(DataFileError, match="series"):
        buffer.add(pair_path, KEY, triple_record(2))


class Recorder:
    """Record every open, fsync and close the write makes."""

    def __init__(self) -> None:
        """Start with no events."""
        self.events: list[tuple[str, str]] = []
        self.open_now = 0
        self.most_open = 0


def test_files_are_written_one_at_a_time_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write measurement files, then double-difference files, one open at a time."""
    meas, ddiff = directories(tmp_path)
    buffer = files.DayBuffer("a")
    pairs: list[PairKey] = [("mc2", "nav23"), ("mc1", "mc2"), ("mc1", "mc1")]
    triples: list[TripleKey] = [("mc2", "mc2", "nav23"), ("mc1", "mc2", "nav23")]
    for triple in triples:
        buffer.add(ddiff / f"das_a.{'.'.join(triple)}.dat", triple, triple_record(0))
    for pair in pairs:
        buffer.add(meas / f"das_a.{'.'.join(pair)}.dat", pair, predicted(0))
    recorder = Recorder()
    real_open, real_fsync = Path.open, os.fsync
    fds: dict[int, str] = {}

    class Tracked:
        """A file that records its close."""

        def __init__(self, inner: object, name: str) -> None:
            """Wrap ``inner``."""
            self.inner, self.name = inner, name

        def __enter__(self) -> Tracked:
            """Enter the wrapped file."""
            self.inner.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *details: object) -> None:
            """Close the wrapped file and record it."""
            fds.pop(self.inner.fileno(), None)  # type: ignore[attr-defined]
            self.inner.__exit__(*details)  # type: ignore[attr-defined]
            recorder.open_now -= 1
            recorder.events.append(("close", self.name))

        def write(self, data: bytes) -> int:
            """Write to the wrapped file."""
            return self.inner.write(data)  # type: ignore[attr-defined, no-any-return]

        def flush(self) -> None:
            """Flush the wrapped file."""
            self.inner.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            """Give the wrapped file's descriptor."""
            number: int = self.inner.fileno()  # type: ignore[attr-defined]
            fds[number] = self.name
            return number

    def tracked_open(path: Path, *args: object, **kwargs: object) -> Tracked:
        """Open ``path`` and record it."""
        recorder.open_now += 1
        recorder.most_open = max(recorder.most_open, recorder.open_now)
        recorder.events.append(("open", path.name))
        return Tracked(real_open(path, *args, **kwargs), path.name)  # type: ignore[call-overload]

    def tracked_fsync(fd: int) -> None:
        """Flush ``fd`` and record it."""
        recorder.events.append(("fsync", fds.get(fd, "directory")))
        real_fsync(fd)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(os, "fsync", tracked_fsync)
    files.write_buffer(buffer)
    names = [f"das_a.{'.'.join(key)}.dat" for key in sorted(pairs) + sorted(triples)]
    expected = [
        event
        for name in names
        for event in (("open", name), ("fsync", name), ("close", name))
    ]
    assert recorder.events[: len(expected)] == expected
    assert recorder.events[len(expected) :] == [("fsync", "directory")] * 2
    assert recorder.most_open == 1


@pytest.mark.parametrize(
    "problem",
    [
        "torn",
        "not_regular",
        "symlink",
        "read_only",
        "clash",
        "no_directory",
        "no_space",
        "not_ascii",
    ],
)
def test_a_failed_check_changes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Raise DataFileError in the prepare step with every file as it was (5.8)."""
    buffer, pair_path, triple_path = filled(tmp_path)
    files.write_buffer(buffer)
    buffer.add(pair_path, KEY, predicted(2))
    buffer.add(triple_path, TRIPLE, triple_record(2))
    new_path = pair_path.parent / "das_a.mc2.cs7.dat"
    buffer.add(new_path, ("mc2", "cs7"), predicted(2))
    if problem == "torn":
        with pair_path.open("ab") as file:
            file.write(b"2025")
    elif problem == "not_regular":
        triple_path.unlink()
        triple_path.mkdir()
    elif problem == "symlink":
        target = tmp_path / "copy.dat"
        target.write_bytes(triple_path.read_bytes())
        triple_path.unlink()
        triple_path.symlink_to(target)
    elif problem == "read_only":
        triple_path.chmod(0o444)
    elif problem == "clash":
        new_path.symlink_to(tmp_path / "elsewhere")
    elif problem == "no_directory":
        new_path = tmp_path / "gone" / "das_a.mc2.cs7.dat"
        buffer.add(new_path, ("mc2", "cs7"), predicted(3))
    elif problem == "no_space":
        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10))
    else:
        buffer.texts[pair_path] += "é\n"
    before = {
        path: path.read_bytes() for path in (pair_path, triple_path) if path.is_file()
    }
    with pytest.raises(DataFileError):
        files.write_buffer(buffer)
    after = {
        path: path.read_bytes() for path in (pair_path, triple_path) if path.is_file()
    }
    assert after == before
    assert not (new_path.exists() and not new_path.is_symlink())


def test_an_error_while_writing_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError for a device fault in the write step."""
    buffer, _, _ = filled(tmp_path)

    def failing(_fd: int) -> None:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(DataFileError, match="Input/output error"):
        files.write_buffer(buffer)


def test_an_error_flushing_a_directory_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError when a new file's directory cannot be flushed."""
    buffer, _, _ = filled(tmp_path)

    def failing(_path: object, _flags: int) -> int:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "open", failing)
    with pytest.raises(DataFileError, match="cannot flush directory"):
        files.write_buffer(buffer)


def test_an_empty_buffer_writes_nothing(tmp_path: Path) -> None:
    """Change nothing when no rows are buffered."""
    files.write_buffer(files.DayBuffer("a"))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------- roll-back, redo


def written(path: Path, count: int) -> Path:
    """Write the pair's file with ``count`` rows from E, and give its path."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        files.header("meas", "a", KEY) + "".join(series_rows(count)), encoding="ascii"
    )
    return path


def epochs_in(path: Path) -> list[datetime]:
    """Give the epoch of every row of the pair's file."""
    size = files.MEAS_WIDTH + 1
    data = path.read_bytes()
    rows = [
        data[i : i + size]
        for i in range(files.MEAS_HEADER_LINES * size, len(data), size)
    ]
    return [
        files.parse_meas_row(row.decode()[:-1]).row.interpolated_datetime
        for row in rows
    ]


def test_a_file_is_rolled_back_to_just_after_the_common_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Truncate just after the row for L, with a WARNING naming file and epoch (6.7)."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 5)
    files.roll_back(path, "meas", E + 2 * STEP)
    assert epochs_in(path) == [E, E + STEP, E + 2 * STEP]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert str(path) in warnings[0].getMessage()
    assert str(E + 2 * STEP) in warnings[0].getMessage()


def test_a_torn_line_after_the_common_epoch_goes_too(tmp_path: Path) -> None:
    """Remove a torn line with the rows after L."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 3)
    with path.open("ab") as file:
        file.write(b"2025-09-23 06:3")
    files.roll_back(path, "meas", E + 2 * STEP)
    assert epochs_in(path) == [E, E + STEP, E + 2 * STEP]
    assert files.good_through(path, "meas") == E + 2 * STEP


def test_a_sound_file_ending_at_the_common_epoch_is_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Leave a file that already ends at L as it is, unlogged."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 3)
    before = path.stat()
    files.roll_back(path, "meas", E + 2 * STEP)
    after = path.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    )
    assert not caplog.records


@pytest.mark.parametrize("common", [None, E - STEP])
def test_a_file_with_no_row_at_or_before_the_common_epoch_is_deleted(
    tmp_path: Path, common: datetime | None
) -> None:
    """Delete a file whose first row is after L, and every file when there is no L."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 3)
    files.roll_back(path, "meas", common)
    assert not path.exists()


def test_a_file_without_one_row_per_epoch_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when the row found for L is of another epoch."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 2)
    rows = series_rows(4)
    with path.open("a", encoding="ascii") as file:
        file.write(rows[3])
    with pytest.raises(DataFileError, match="one row per epoch"):
        files.roll_back(path, "meas", E + 2 * STEP)


def test_a_common_epoch_past_the_file_s_rows_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when the file holds no row for L at all."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 2)
    with pytest.raises(DataFileError, match="no row for"):
        files.roll_back(path, "meas", E + 5 * STEP)


def archive(tmp_path: Path) -> list[tuple[Path, files.FileKind]]:
    """Write a measurement file of five rows and a double-difference file of three."""
    pair_path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 5)
    triple_path = tmp_path / "ddiff" / "das_a.mc1.mc2.nav23.dat"
    triple_path.parent.mkdir()
    lines = "".join(files.format_ddiff_row(triple_record(i)) + "\n" for i in range(3))
    triple_path.write_text(files.header("ddiff", "a", TRIPLE) + lines, encoding="ascii")
    return [(pair_path, "meas"), (triple_path, "ddiff")]


def test_a_redo_deletes_every_row_at_or_after_its_epoch(tmp_path: Path) -> None:
    """Truncate every file before its first row at or after the mark (6.5)."""
    series = archive(tmp_path)
    files.redo_from(series, E + 2 * STEP)
    assert [files.good_through(path, kind) for path, kind in series] == [
        E + STEP,
        E + STEP,
    ]


def test_a_redo_deletes_a_file_with_no_earlier_row(tmp_path: Path) -> None:
    """Delete every file when the redo starts at or before its first row."""
    series = archive(tmp_path)
    files.redo_from(series, E)
    assert not any(path.exists() for path, _ in series)


def test_a_redo_past_a_file_s_end_leaves_it(tmp_path: Path) -> None:
    """Keep a file whose rows all come before the redo."""
    series = archive(tmp_path)
    files.redo_from(series, E + 4 * STEP)
    assert [files.good_through(path, kind) for path, kind in series] == [
        E + 3 * STEP,
        E + 2 * STEP,
    ]


def test_an_interrupted_redo_finishes_when_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish the deletion on a second run after the first stopped part way (6.5)."""
    series = archive(tmp_path)
    real_open = Path.open
    calls = {"count": 0}

    def failing_second(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        """Fail the second file opened to be cut."""
        if mode == "r+b":
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError(5, "Input/output error")
        return real_open(path, mode, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", failing_second)
    with pytest.raises(DataFileError, match="Input/output error"):
        files.redo_from(series, E + STEP)
    monkeypatch.undo()
    files.redo_from(series, E + STEP)
    assert [files.good_through(path, kind) for path, kind in series] == [E, E]


def test_a_redo_and_a_roll_back_at_one_start_keep_the_archive_in_step(
    tmp_path: Path,
) -> None:
    """Redo first, then roll back what is left to the common epoch (review focus 5)."""
    series = archive(tmp_path)
    triple_path = series[1][0]
    with triple_path.open("ab") as file:
        file.write(b"2025-09-23 06:3")
    files.redo_from(series, E + 4 * STEP)
    good = [files.good_through(path, kind) for path, kind in series]
    assert good == [E + 3 * STEP, E + 2 * STEP]
    common = min(mark for mark in good if mark is not None)
    for path, kind in series:
        files.roll_back(path, kind, common)
    assert [files.good_through(path, kind) for path, kind in series] == [
        E + 2 * STEP,
        E + 2 * STEP,
    ]
    assert all(
        path.stat().st_size % (files.WIDTHS[kind] + 1) == 0 for path, kind in series
    )


def test_an_error_deleting_a_file_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError when a file cannot be deleted."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 2)

    def failing(_path: Path, *_args: object) -> None:
        """Fail as a device would."""
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", failing)
    with pytest.raises(DataFileError, match="Permission denied"):
        files.roll_back(path, "meas", None)


def test_a_file_that_cannot_be_read_is_not_rolled_back(tmp_path: Path) -> None:
    """Raise DataFileError when the file to roll back cannot be read."""
    with pytest.raises(DataFileError, match="cannot read"):
        files.roll_back(tmp_path / "missing.dat", "meas", E)


def test_the_archive_directories_are_made_once(tmp_path: Path) -> None:
    """Make meas/ and ddiff/ when missing, and leave them when there."""
    files.ensure_archives(tmp_path)
    assert (tmp_path / "meas").is_dir()
    assert (tmp_path / "ddiff").is_dir()
    (tmp_path / "meas" / "kept").write_text("")
    files.ensure_archives(tmp_path)
    assert (tmp_path / "meas" / "kept").exists()


def test_an_archive_directory_that_cannot_be_made_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when an archive's name is taken by a file."""
    (tmp_path / "ddiff").write_text("")
    with pytest.raises(DataFileError, match="cannot make"):
        files.ensure_archives(tmp_path)


def test_a_buffer_takes_another_s_rows(tmp_path: Path) -> None:
    """Move one epoch's rows into the day's buffer, texts and newest rows."""
    day, pair_path, triple_path = filled(tmp_path)
    epoch = files.DayBuffer("a")
    epoch.add(pair_path, KEY, predicted(2))
    epoch.add(triple_path, TRIPLE, triple_record(2))
    day.take(epoch)
    assert day.texts[pair_path].count("\n") == 3
    assert day.last[KEY] == predicted(2).row
    assert day.series_of(triple_path) == ("ddiff", TRIPLE)


def test_a_buffer_takes_nothing_from_a_clashing_one(tmp_path: Path) -> None:
    """Refuse rows of another series for a path, leaving the buffer as it was."""
    day, pair_path, _ = filled(tmp_path)
    before = dict(day.texts), dict(day.last)
    epoch = files.DayBuffer("a")
    epoch.add(pair_path.parent / "das_a.mc2.cs7.dat", ("mc2", "cs7"), predicted(2))
    epoch.add(pair_path, ("mc2", "hm1"), predicted(2))
    with pytest.raises(DataFileError, match="series"):
        day.take(epoch)
    assert (dict(day.texts), dict(day.last)) == before


# ---------------------------------------------------------- the write journal


def journaled(tmp_path: Path) -> tuple[files.DayBuffer, Path, Path]:
    """Give a buffer with a journal and two epochs' rows, its journal and a file."""
    plain, pair_path, _ = filled(tmp_path)
    journal = tmp_path / "das_processor_a.writing"
    buffer = files.DayBuffer("a", journal)
    buffer.take(plain)
    return buffer, journal, pair_path


def test_a_buffer_knows_its_first_epoch(tmp_path: Path) -> None:
    """Keep the earliest epoch buffered since the last write, and forget it after."""
    buffer, _, pair_path = journaled(tmp_path)
    assert buffer.start == E
    later = files.DayBuffer("a")
    later.add(pair_path, KEY, predicted(2))
    buffer.take(later)
    assert buffer.start == E
    files.write_buffer(buffer)
    assert [buffer.start] == [None]
    buffer.add(pair_path, KEY, predicted(3))
    assert buffer.start == E + 3 * STEP


def test_the_journal_is_there_exactly_while_the_files_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flush the first epoch to the journal before any file opens; delete it after."""
    buffer, journal, _ = journaled(tmp_path)
    real_open = Path.open
    seen: list[tuple[str, str | None]] = []

    def watching(path: Path, *args: object, **kwargs: object) -> object:
        """Record, at each data file's opening, what the journal holds."""
        if path != journal:
            seen.append((path.name, journal.read_text() if journal.exists() else None))
        return real_open(path, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", watching)
    files.write_buffer(buffer)
    assert len(seen) == 2
    assert all(text == f"{E.isoformat()}\n" for _, text in seen)
    assert not journal.exists()


def test_a_buffer_without_a_journal_keeps_none(tmp_path: Path) -> None:
    """Write no journal for a buffer given none."""
    buffer, _, _ = filled(tmp_path)
    files.write_buffer(buffer)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ddiff", "meas"]


def test_a_journal_already_there_changes_no_file(tmp_path: Path) -> None:
    """Refuse to write while a journal is there, before any file is opened."""
    buffer, journal, pair_path = journaled(tmp_path)
    journal.write_text(f"{E.isoformat()}\n")
    with pytest.raises(DataFileError, match="still open"):
        files.write_buffer(buffer)
    assert not pair_path.exists()
    assert journal.read_text() == f"{E.isoformat()}\n"


def test_an_empty_buffer_writes_no_journal(tmp_path: Path) -> None:
    """Write no journal when no rows are buffered."""
    journal = tmp_path / "das_processor_a.writing"
    files.write_buffer(files.DayBuffer("a", journal))
    assert not journal.exists()


def test_a_journal_that_cannot_be_written_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError, opening no data file, when the journal cannot be flushed."""
    buffer, _, pair_path = journaled(tmp_path)

    def failing(_fd: int) -> None:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(DataFileError, match="cannot write journal"):
        files.write_buffer(buffer)
    assert not pair_path.exists()


def test_a_journal_gives_the_first_epoch_of_its_write(tmp_path: Path) -> None:
    """Read back the epoch a stopped write started at."""
    journal = tmp_path / "das_processor_a.writing"
    journal.write_text(f"{E.isoformat()}\n")
    assert files.read_journal(journal) == E


def test_no_journal_means_no_write_was_stopped(tmp_path: Path) -> None:
    """Give None when there is no journal."""
    assert files.read_journal(tmp_path / "das_processor_a.writing") is None


@pytest.mark.parametrize(
    "text",
    [
        b"",
        b"2025-09-23T06:0",
        E.isoformat().encode(),
        b"2025-09-23T06:00:00\n",
        b"\xff\n",
    ],
)
def test_a_journal_not_whole_means_no_data_file_was_opened(
    tmp_path: Path, text: bytes
) -> None:
    """Give None for a journal cut short, without its zone, or not ASCII."""
    journal = tmp_path / "das_processor_a.writing"
    journal.write_bytes(text)
    assert files.read_journal(journal) is None


def test_a_journal_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when the journal is there but cannot be read."""
    journal = tmp_path / "das_processor_a.writing"
    journal.mkdir()
    with pytest.raises(DataFileError, match="cannot read journal"):
        files.read_journal(journal)


def test_clearing_deletes_the_journal(tmp_path: Path) -> None:
    """Delete the journal, and do nothing when it is not there."""
    journal = tmp_path / "das_processor_a.writing"
    journal.write_text(f"{E.isoformat()}\n")
    files.clear_journal(journal)
    assert not journal.exists()
    files.clear_journal(journal)
    assert not journal.exists()


# ----------------------------------------------- what a person reads, exactly


def test_an_error_is_logged_with_its_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each data file error at ERROR, in the words it is raised with."""
    with pytest.raises(DataFileError) as raised:
        files.roll_back(tmp_path / "missing.dat", "meas", E)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert [r.getMessage() for r in errors] == [str(raised.value)]
    assert str(raised.value).startswith(f"cannot read data file {tmp_path}")


def test_a_roll_back_says_what_it_did_to_each_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say, at WARNING, which file was cut back to which epoch, or deleted."""
    cut = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 5)
    files.roll_back(cut, "meas", E + 2 * STEP)
    files.roll_back(cut, "meas", None)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", f"data file {cut} rolled back to {E + 2 * STEP}"),
        ("WARNING", f"data file {cut} deleted: no row at or before None"),
    ]


def test_a_redo_says_what_it_did_to_each_file_it_changed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say, at INFO, which files a redo cut or deleted, and nothing of the rest."""
    (pair, _), (triple, _) = series = archive(tmp_path)
    caplog.set_level(logging.INFO)
    files.redo_from(series, E + 3 * STEP)
    files.redo_from(series[1:], E + 3 * STEP)
    files.redo_from(series[1:], E)
    mark = E + 3 * STEP
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("INFO", f"data file {pair} cut for a redo from {mark}"),
        ("INFO", f"data file {triple} deleted for a redo from {E}"),
    ]


def test_a_redo_deletes_a_file_holding_nothing_good(tmp_path: Path) -> None:
    """Delete a file with no whole row, whatever the redo's epoch."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 0)
    with path.open("ab") as file:
        file.write(b"2025-09-23 06:0")
    files.redo_from([(path, "meas")], E + 3 * STEP)
    assert not path.exists()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("torn", "data file {pair} is not sound: {length} bytes"),
        ("not_regular", "data file {triple} is not a regular file"),
        ("read_only", "data file {triple} cannot be written"),
        ("no_directory", "file {gone} cannot be created in {gone_parent}"),
        ("not_ascii", "the rows for {pair} are not ASCII"),
    ],
)
def test_a_failed_check_says_which_file_and_why(
    tmp_path: Path, problem: str, message: str
) -> None:
    """Name the file and what is wrong with it, in the prepare step (5.8)."""
    buffer, pair_path, triple_path = filled(tmp_path)
    files.write_buffer(buffer)
    buffer.add(pair_path, KEY, predicted(2))
    buffer.add(triple_path, TRIPLE, triple_record(2))
    gone = tmp_path / "gone" / "das_a.mc2.cs7.dat"
    if problem == "torn":
        with pair_path.open("ab") as file:
            file.write(b"2025")
    elif problem == "not_regular":
        triple_path.unlink()
        triple_path.mkdir()
    elif problem == "read_only":
        triple_path.chmod(0o444)
    elif problem == "no_directory":
        buffer.add(gone, ("mc2", "cs7"), predicted(3))
    else:
        buffer.texts[pair_path] += "é\n"
    expected = message.format(
        pair=pair_path,
        triple=triple_path,
        length=pair_path.stat().st_size,
        gone=gone,
        gone_parent=gone.parent,
    )
    with pytest.raises(DataFileError) as raised:
        files.write_buffer(buffer)
    assert str(raised.value) == expected


def test_a_new_file_s_directory_that_cannot_be_written_into_is_refused(
    tmp_path: Path,
) -> None:
    """Refuse, before any file opens, a new file in a directory not writable."""
    buffer, pair_path, triple_path = filled(tmp_path)
    triple_path.parent.chmod(0o555)
    try:
        with pytest.raises(DataFileError, match="cannot be created in"):
            files.write_buffer(buffer)
        assert not pair_path.exists()
    finally:
        triple_path.parent.chmod(0o755)


def test_free_space_just_enough_is_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write when the free space is exactly the bytes to write, refuse one less."""
    buffer, pair_path, triple_path = filled(tmp_path)
    data = files.header("meas", "a", KEY) + buffer.texts[pair_path]
    data2 = files.header("ddiff", "a", TRIPLE) + buffer.texts[triple_path]
    total = len(data) + len(data2)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=total - 1))
    with pytest.raises(DataFileError) as raised:
        files.write_buffer(buffer)
    assert str(raised.value) == (
        f"{total} bytes to write in {triple_path.parent}, only {total - 1} free"
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=total))
    files.write_buffer(buffer)
    assert pair_path.exists()


def test_free_space_is_counted_per_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weigh each device's bytes against that device's free space alone."""
    buffer, pair_path, triple_path = filled(tmp_path)
    sizes = {
        pair_path.parent: len(files.header("meas", "a", KEY) + buffer.texts[pair_path]),
        triple_path.parent: len(
            files.header("ddiff", "a", TRIPLE) + buffer.texts[triple_path]
        ),
    }
    devices = {pair_path.parent: 1, triple_path.parent: 2}

    def stat(path: Path, **_kwargs: object) -> object:
        """Put each archive on a device of its own; nothing else is looked at."""
        return SimpleNamespace(st_dev=devices[path])

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda where: SimpleNamespace(free=sizes[where])
    )
    files.write_buffer(buffer)
    monkeypatch.undo()
    assert pair_path.exists()
    assert triple_path.exists()


def test_a_missing_archive_is_made_beside_one_already_there(tmp_path: Path) -> None:
    """Make ddiff/ when only meas/ is there."""
    (tmp_path / "meas").mkdir()
    files.ensure_archives(tmp_path)
    assert (tmp_path / "ddiff").is_dir()


def test_a_measurement_time_on_a_whole_second_keeps_its_microseconds() -> None:
    """Write a measurement time on a whole second with its six zeros."""
    pair = measure_pair(
        measurement_mjd=60941.25,
        measured_phase=34579,
        rms=3,
        prediction=State(x=Fraction(1_234_567), y=0.0),
        w=Fraction(0),
        anchor=None,
    )
    line = files.format_meas_row(files.MeasRecord(measurement=pair, row=row()))
    assert "2025-09-23 06:00:00.000000+00:00" in line


def test_a_value_too_wide_names_its_row_s_epoch() -> None:
    """Say which epoch's row holds a value too wide for its column."""
    record = files.MeasRecord(measurement=None, row=row(flags="P", segment=10**12))
    with pytest.raises(DataFileError, match=r"^row of 2025-09-23 06:00:00\+00:00: "):
        files.format_meas_row(record)
    triple = files.DdiffRecord(measurement=None, row=row(flags="P", segment=10**12))
    with pytest.raises(DataFileError, match=r"^row of 2025-09-23 06:00:00\+00:00: "):
        files.format_ddiff_row(triple)


def test_a_key_of_the_wrong_size_for_its_file_is_named() -> None:
    """Say a measurement file is for a pair, a double-difference file a triple."""
    with pytest.raises(DataFileError) as raised:
        files.header("meas", "a", TRIPLE)
    assert "file is for a pair: " in str(raised.value)
    with pytest.raises(DataFileError) as raised:
        files.header("ddiff", "a", KEY)
    assert "file is for a triple: " in str(raised.value)


def field_changed(line: str, index: int, text: str) -> str:
    """Give a line with one field's text replaced, right-justified as before."""
    fields = line.split(", ")
    fields[index] = text.rjust(len(fields[index]))
    return ", ".join(fields)


@pytest.mark.parametrize(
    ("kind", "change", "reason"),
    [
        ("meas", lambda line: line.rsplit(", ", 1)[0], "26 fields, not 27"),
        ("meas", lambda line: field_changed(line, 9, "nan"), "'nan' is not a finite"),
        ("meas", lambda line: field_changed(line, 8, "1234574.45"), "three decimals"),
        ("meas", lambda line: field_changed(line, 0, "-"), "never empty is empty"),
        (
            "ddiff",
            lambda line: field_changed(field_changed(line, 2, "-"), 5, "-"),
            "never empty is empty",
        ),
        (
            "ddiff",
            lambda line: field_changed(field_changed(line, 4, "-"), 5, "-"),
            "never empty is empty",
        ),
        ("meas", lambda line: field_changed(line, 17, "60941.243056"), "never empty"),
    ],
)
def test_a_line_that_does_not_parse_says_why(
    kind: str, change: Callable[[str], str], reason: str
) -> None:
    """Name the row by its first characters and the reason it does not parse."""
    line = change(MEAS_EXAMPLE[0] if kind == "meas" else DDIFF_EXAMPLE[0])
    with pytest.raises(DataFileError) as raised:
        if kind == "meas":
            files.parse_meas_row(line)
        else:
            files.parse_ddiff_row(line)
    message = str(raised.value)
    assert message.startswith(f"row {line[:25]!r} does not parse: "), message
    assert reason in message


def test_a_line_not_written_so_names_its_row() -> None:
    """Name the row by its first characters when it is not as written."""
    line = MEAS_EXAMPLE[0].replace("+1.2301290523526430e-02", "+12.301290523526430e-03")
    with pytest.raises(DataFileError) as raised:
        files.parse_meas_row(line)
    assert str(raised.value) == (
        f"row {line[:25]!r} is not written as das_processor writes it"
    )


def test_a_field_never_empty_is_refused_in_those_words() -> None:
    """End the refusal of an empty field that is never empty with its reason."""
    line = field_changed(MEAS_EXAMPLE[0], 0, "-")
    with pytest.raises(DataFileError) as raised:
        files.parse_meas_row(line)
    assert str(raised.value).endswith(
        " does not parse: a field that is never empty is empty"
    )


def test_a_deletion_by_roll_back_names_the_common_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say which epoch a deleted file had no row at or before."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 3)
    files.roll_back(path, "meas", E - STEP)
    assert [r.getMessage() for r in caplog.records] == [
        f"data file {path} deleted: no row at or before {E - STEP}"
    ]


def device_error(*_args: object, **_kwargs: object) -> None:
    """Fail as a device would."""
    raise OSError(5, "Input/output error")


def test_a_device_fault_names_the_file_and_what_was_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Say which file could not be deleted, cut or written, and the device's error."""
    path = written(tmp_path / "meas" / "das_a.mc2.nav23.dat", 3)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", device_error)
        with pytest.raises(DataFileError) as raised:
            files.roll_back(path, "meas", None)
    assert str(raised.value) == f"cannot delete {path}: [Errno 5] Input/output error"
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", device_error)
        with pytest.raises(DataFileError) as raised:
            files.roll_back(path, "meas", E)
    assert str(raised.value) == (
        f"cannot cut data file {path}: [Errno 5] Input/output error"
    )
    (tmp_path / "write").mkdir()
    buffer, pair_path, _ = filled(tmp_path / "write")
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", device_error)
        with pytest.raises(DataFileError) as raised:
            files.write_buffer(buffer)
    assert str(raised.value) == (
        f"cannot write data file {pair_path}: [Errno 5] Input/output error"
    )
