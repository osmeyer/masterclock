"""Tests for src/masterclock/das_processor/files.py.

The rules covered: the column table gives the measurement file a width of
485 and 34 header lines and the double-difference file 455 and 31; every
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
"""

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Final, Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app.timeutil import datetime_to_mjd
from masterclock.das_processor import files
from masterclock.das_processor.exceptions import DataFileError
from masterclock.das_processor.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain.phase import PHASE_MAX, exact
from masterclock.domain.series import Row, SeriesKey, State

E: Final = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""The worked epoch's start."""

STEP: Final = timedelta(minutes=10)
"""One epoch."""

RAW: Final = DASMeasurement(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    switch="2B07",
    clock="ox23",
)
"""Appendix A's raw row."""

PAIR: Final = measure_pair(
    RAW, State(x=1_234_567 + exact(0.0123) * 600, y=0.0123), Fraction(0), None
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
    "  60941.251588,  34579,      3, 2B07,            6,          1234577,"
    "          1234574.457, +1.2301290523526430e-02, +7.1695154009743332e-12,"
    " +3.0000000000000000e+00,         4,                0,       812,         0,"
    "         0,             -,                       -,             -,"
    "                       -,             -,                       -, 3,"
    " +1.0000000000000000e+02, +5.0000000000000000e+01,        A",
    "2025-09-23 06:10:00+00:00,  60941.256944,                                -,"
    "             -,      -,      -,    -,            -,                -,"
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
    """Work out W = 485 and 455 and 34 and 31 header lines from the table."""
    assert (files.MEAS_WIDTH, files.MEAS_HEADER_LINES) == (485, 34)
    assert (files.DDIFF_WIDTH, files.DDIFF_HEADER_LINES) == (455, 31)


# ------------------------------------------------------------------- headers

KEYS: Final[tuple[tuple[files.FileKind, SeriesKey], ...]] = (
    ("meas", ("mc2", "ox23")),
    ("meas", ("mc1", "mc1")),
    ("meas", ("mc1", "mc2")),
    ("ddiff", ("mc1", "mc2", "ox23")),
    ("ddiff", ("mc2", "mc2", "ox23")),
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
        line.rstrip() for line in files.header("meas", "a", ("mc2", "ox23")).split("\n")
    ]
    assert lines[:6] == [
        "# das_processor measurement file, format 1",
        "# WARNING: do not modify this file. Only das_processor may write it;"
        " any other change damages the archive.",
        "# RF channel a. Pair (mc2, ox23).",
        "# Reference mc2 measured against clock ox23.",
        "# One row per 10-minute epoch; '-' marks an empty field.",
        "# Columns: right-justified, fixed width, separated by ', '.",
    ]
    assert lines[6] == "#   interpolated_datetime   epoch start E, UTC"
    assert lines[33] == (
        "#   flags                   A accepted, R rejected, X excluded, P predicted,"
        " D dormant, S slip corrected, N new segment, U unsettled"
    )


def test_the_double_difference_header_is_the_design_s() -> None:
    """Give the lines of design 5.5 in order, the warning second."""
    text = files.header("ddiff", "b", ("mc1", "mc2", "ox23"))
    lines = [line.rstrip() for line in text.split("\n")]
    assert lines[0] == "# das_processor double-difference file, format 1"
    assert lines[2:4] == [
        "# RF channel b. Triple (mc1, mc2, ox23).",
        "# Clock ox23 against remote reference mc1, through local reference mc2.",
    ]
    assert lines[8] == "#   z                       double difference dd at E, ps"
    assert lines[10] == "#   double_difference_sigma measurement sigma of dd, ps"


@pytest.mark.parametrize(
    ("kind", "key"), [("meas", ("mc1", "mc2", "ox23")), ("ddiff", ("mc1", "ox23"))]
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
            ("mc2", "mc2", "ox23"),
            "# Clock ox23 against its local reference mc2.",
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
    record = files.parse_meas_row(MEAS_EXAMPLE[0], ("mc2", "ox23"))
    assert record == files.MeasRecord(measurement=PAIR, row=row())
    assert files.parse_meas_row(MEAS_EXAMPLE[1], ("mc2", "ox23")).measurement is None
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
    assert fields[9:13] == ["-".rjust(20), "-".rjust(23), "-".rjust(23), "-".rjust(23)]
    assert fields[18:20] == [" 60941.243056", "+1.2345700000000000e+06"]
    assert files.parse_meas_row(line, ("mc2", "ox23")).row == dormant


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
    assert line.split(", ")[9] == text.rjust(20)
    assert files.parse_meas_row(line, ("mc2", "ox23")).row.x_fs == x_fs


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
    pair = PairMeasurement(measurement=RAW, cycle_count=10**12, z=1)
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
        (" 2B07,", " 2b07,"),
        ("e+00,         4,", "e+00,         -,"),
    ],
)
def test_a_line_not_exactly_as_written_is_refused(old: str, new: str) -> None:
    """Raise DataFileError for any line formatting would not give back (U20)."""
    assert MEAS_EXAMPLE[0].count(old) == 1
    line = MEAS_EXAMPLE[0].replace(old, new)
    with pytest.raises(DataFileError):
        files.parse_meas_row(line, ("mc2", "ox23"))


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
        files.parse_meas_row(line, ("mc2", "ox23"))


def test_a_row_of_another_clock_is_refused() -> None:
    """Raise DataFileError when the row does not parse for this pair's clock."""
    pair = files.parse_meas_row(MEAS_EXAMPLE[0], ("mc2", "ox23"))
    assert pair.measurement is not None
    with pytest.raises(DataFileError, match="mc3"):
        files.parse_meas_row(MEAS_EXAMPLE[0], ("mc3", "ox23"))


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
    raw = DASMeasurement(
        measurement_mjd=round(start + offset * 1e-6, 6),
        measured_phase=draw(st.integers(0, PHASE_MAX)),
        rms=draw(st.integers(0, 999_999)),
        switch=draw(st.from_regex(r"2[A-Z][0-9]{2}", fullmatch=True)),
        clock="ox23",
    )
    pair = PairMeasurement(
        measurement=raw,
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
    assert files.parse_meas_row(line, ("mc2", "ox23")) == record
    assert files.format_meas_row(files.parse_meas_row(line, ("mc2", "ox23"))) == line


@given(ddiff_records())
def test_a_double_difference_row_parses_back_to_its_record(
    record: files.DdiffRecord,
) -> None:
    """Give back the record, and the same text when formatted again (U20)."""
    line = files.format_ddiff_row(record)
    assert len(line) == files.DDIFF_WIDTH
    assert files.parse_ddiff_row(line) == record
