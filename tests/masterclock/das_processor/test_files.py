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
parse, unless a write stopped part way, when it holds nothing good; a
damaged file is explained once at ERROR, where and why; and the last row
of a sound file is read back as its row (U26). A roll-back says what it did
to each file and logs nothing; a redo is logged once at INFO.

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

ONE_EPOCH: Final = timedelta(minutes=10)
"""One epoch."""

APPENDIX_A_READING: Final[dict[str, float | int]] = {
    "measurement_mjd": 60941.251588,
    "measured_phase": 34579,
    "rms": 3,
}
"""Appendix A's reading: its MJD, phase and rms."""

APPENDIX_A_MEASUREMENT: Final = measure_pair(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    prediction=State(x=1_234_567 + exact(0.0123) * 600, y=0.0123),
    w=Fraction(0),
    anchor=None,
)
"""Appendix A's pair measurement."""


def worked_row(**changed_fields: object) -> Row:
    """Build the worked epoch's accepted row, with ``changed_fields`` applied."""
    row_fields: dict[str, object] = {
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
    row_fields.update(changed_fields)
    return Row.model_validate(row_fields)


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

KINDS_AND_SERIES: Final[tuple[tuple[files.FileKind, SeriesKey], ...]] = (
    ("meas", ("mc2", "ox23")),
    ("meas", ("mc1", "mc1")),
    ("meas", ("mc1", "mc2")),
    ("ddiff", ("mc1", "mc2", "ox23")),
    ("ddiff", ("mc2", "mc2", "ox23")),
)
"""A series of every kind."""


@pytest.mark.parametrize(("file_kind", "series_key"), KINDS_AND_SERIES)
def test_every_header_line_is_w_wide_and_starts_with_a_hash(
    file_kind: files.FileKind, series_key: SeriesKey
) -> None:
    """Pad every header line to W, each a comment, the count from the table."""
    header_lines = files.header(file_kind, "a", series_key).split("\n")
    assert header_lines[-1] == ""
    line_width, header_count = files.WIDTHS[file_kind], files.HEADER_LINES[file_kind]
    assert len(header_lines[:-1]) == header_count
    assert all(
        len(header_line) == line_width and header_line.startswith("#")
        for header_line in header_lines[:-1]
    )


def test_the_measurement_header_is_the_design_s() -> None:
    """Give the lines of design 5.4 in order, the warning second."""
    header_lines = [
        header_line.rstrip()
        for header_line in files.header("meas", "a", ("mc2", "ox23")).split("\n")
    ]
    assert header_lines[:6] == [
        "# das_processor measurement file, format 1",
        "# WARNING: do not modify this file. Only das_processor may write it;"
        " any other change damages the archive.",
        "# RF channel a. Pair (mc2, ox23).",
        "# Reference mc2 measured against clock ox23.",
        "# One row per 10-minute epoch; '-' marks an empty field.",
        "# Columns: right-justified, fixed width, separated by ', '.",
    ]
    assert header_lines[6] == "#   interpolated_datetime   epoch start E, UTC"
    assert header_lines[32] == (
        "#   flags                   A accepted, R rejected, X excluded, P predicted,"
        " D dormant, S slip corrected, N new segment, U unsettled"
    )


def test_the_double_difference_header_is_the_design_s() -> None:
    """Give the lines of design 5.5 in order, the warning second."""
    header_text = files.header("ddiff", "b", ("mc1", "mc2", "ox23"))
    header_lines = [header_line.rstrip() for header_line in header_text.split("\n")]
    assert header_lines[0] == "# das_processor double-difference file, format 1"
    assert header_lines[2:4] == [
        "# RF channel b. Triple (mc1, mc2, ox23).",
        "# Clock ox23 against remote reference mc1, through local reference mc2.",
    ]
    assert (
        header_lines[8] == "#   z                       double difference dd at E, ps"
    )
    assert header_lines[10] == "#   double_difference_sigma measurement sigma of dd, ps"


@pytest.mark.parametrize(
    ("file_kind", "series_key"),
    [("meas", ("mc1", "mc2", "ox23")), ("ddiff", ("mc1", "ox23"))],
)
def test_a_header_is_for_its_kind_of_series(
    file_kind: files.FileKind, series_key: SeriesKey
) -> None:
    """Raise DataFileError for a triple's measurement file or a pair's other file."""
    with pytest.raises(DataFileError, match="is for a"):
        files.header(file_kind, "a", series_key)


@pytest.mark.parametrize(
    ("file_kind", "series_key", "header_line"),
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
    file_kind: files.FileKind, series_key: SeriesKey, header_line: str
) -> None:
    """Describe self and link pairs and local triples in words."""
    assert (
        files.header(file_kind, "a", series_key).split("\n")[3].rstrip() == header_line
    )


# ---------------------------------------------------------------- examples


def test_the_measurement_example_rows_come_out_byte_for_byte() -> None:
    """Format the two rows of design 5.4 from their values."""
    accepted_record = files.MeasRecord(
        measurement=APPENDIX_A_MEASUREMENT, row=worked_row()
    )
    predicted_meas_record = files.MeasRecord(
        measurement=None,
        row=worked_row(
            interpolated_datetime=E + ONE_EPOCH,
            x_fs=1_234_581_838,
            y=0.012301294825235671,
            epochs_in_segment=813,
            epochs_since_accept=1,
            flags="P",
        ),
    )
    assert files.format_meas_row(accepted_record) == MEAS_EXAMPLE[0]
    assert files.format_meas_row(predicted_meas_record) == MEAS_EXAMPLE[1]
    assert all(len(row_line) == files.MEAS_WIDTH for row_line in MEAS_EXAMPLE)


def test_the_double_difference_example_rows_come_out_byte_for_byte() -> None:
    """Format the two rows of design 5.5 from their values."""
    remote_record = files.DdiffRecord(
        measurement=TripleMeasurement(
            z=6_666_667,
            double_difference_sigma=3.3166247903553998,
            components_used="111",
            pair_cold_started=False,
        ),
        row=worked_row(
            innovation=-0.29999999981373549,
            x_fs=6_666_667_291,
            y=0.020499852230130653,
            d=-8.2093687746445554e-13,
            innovation_scale=3.4650829715892466,
            segment=2,
            epochs_in_segment=3107,
        ),
    )
    local_record = files.DdiffRecord(
        measurement=TripleMeasurement(
            z=1_234_577,
            double_difference_sigma=3.0,
            components_used="111",
            pair_cold_started=False,
        ),
        row=worked_row(innovation=2.62),
    )
    assert files.format_ddiff_row(remote_record) == DDIFF_EXAMPLE[0]
    assert files.format_ddiff_row(local_record) == DDIFF_EXAMPLE[1]
    assert all(len(row_line) == files.DDIFF_WIDTH for row_line in DDIFF_EXAMPLE)


def test_the_example_rows_parse_back() -> None:
    """Parse the design's example rows into records that format to them."""
    meas_record = files.parse_meas_row(MEAS_EXAMPLE[0])
    assert meas_record == files.MeasRecord(
        measurement=APPENDIX_A_MEASUREMENT, row=worked_row()
    )
    assert files.parse_meas_row(MEAS_EXAMPLE[1]).measurement is None
    for row_line in DDIFF_EXAMPLE:
        assert files.format_ddiff_row(files.parse_ddiff_row(row_line)) == row_line


# ----------------------------------------------------- empty fields, x, rejects


def test_a_dormant_row_writes_its_state_as_empty_fields() -> None:
    """Write '-' for x, y, d and the scale of a dormant row, right-justified."""
    dormant_row = worked_row(
        x_fs=None,
        y=None,
        d=None,
        innovation_scale=None,
        flags="RD",
        innovation=None,
        rejects=((E - ONE_EPOCH, 1_234_570.0),),
    )
    row_line = files.format_meas_row(
        files.MeasRecord(measurement=APPENDIX_A_MEASUREMENT, row=dormant_row)
    )
    row_fields = row_line.split(", ")
    assert row_fields[8:12] == [
        "-".rjust(20),
        "-".rjust(23),
        "-".rjust(23),
        "-".rjust(23),
    ]
    assert row_fields[17:19] == [" 60941.243056", "+1.2345700000000000e+06"]
    assert files.parse_meas_row(row_line).row == dormant_row


@pytest.mark.parametrize(
    ("x_fs", "x_text"),
    [
        (1_234_574_457, "1234574.457"),
        (-1_500, "-1.500"),
        (7, "0.007"),
        (-7, "-0.007"),
        (0, "0.000"),
    ],
)
def test_the_phase_is_written_in_ps_to_the_femtosecond(x_fs: int, x_text: str) -> None:
    """Write whole femtoseconds as ps with three decimals, sign and all."""
    row_line = files.format_meas_row(
        files.MeasRecord(measurement=APPENDIX_A_MEASUREMENT, row=worked_row(x_fs=x_fs))
    )
    assert row_line.split(", ")[8] == x_text.rjust(20)
    assert files.parse_meas_row(row_line).row.x_fs == x_fs


# ------------------------------------------------------------------ overflow


@pytest.mark.parametrize(
    "changed_fields",
    [
        {"step_offset": 10**16},
        {"segment": 10**9},
        {"x_fs": -(10**19)},
        {"y": 1e-300},
        {"innovation_scale": 1e100},
    ],
)
def test_a_value_too_wide_for_its_column_is_refused(
    changed_fields: dict[str, object],
) -> None:
    """Raise DataFileError naming the column a value does not fit."""
    with pytest.raises(DataFileError, match="does not fit"):
        files.format_meas_row(
            files.MeasRecord(
                measurement=APPENDIX_A_MEASUREMENT, row=worked_row(**changed_fields)
            )
        )


def test_a_cycle_count_too_wide_is_refused() -> None:
    """Raise DataFileError for a cycle count wider than its column."""
    wide_measurement = PairMeasurement.model_validate(
        {**APPENDIX_A_READING, "cycle_count": 10**12, "z": 1}
    )
    with pytest.raises(DataFileError, match="cycle_count"):
        files.format_meas_row(
            files.MeasRecord(measurement=wide_measurement, row=worked_row())
        )


# ---------------------------------------------------------- refused lines


@pytest.mark.parametrize(
    ("written_text", "changed_text"),
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
def test_a_line_not_exactly_as_written_is_refused(
    written_text: str, changed_text: str
) -> None:
    """Raise DataFileError for any line formatting would not give back (U20)."""
    assert MEAS_EXAMPLE[0].count(written_text) == 1
    row_line = MEAS_EXAMPLE[0].replace(written_text, changed_text)
    with pytest.raises(DataFileError):
        files.parse_meas_row(row_line)


@pytest.mark.parametrize(
    "row_line",
    [
        MEAS_EXAMPLE[0][:-1],
        MEAS_EXAMPLE[0] + ", 1",
        MEAS_EXAMPLE[0].replace(", ", ",", 1),
        "",
    ],
)
def test_a_line_of_the_wrong_shape_is_refused(row_line: str) -> None:
    """Raise DataFileError for a line with the wrong columns."""
    with pytest.raises(DataFileError):
        files.parse_meas_row(row_line)


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
        files.MeasRecord(measurement=APPENDIX_A_MEASUREMENT, row=worked_row(flags="P"))
    with pytest.raises(DataFileError, match="measurement"):
        files.MeasRecord(measurement=None, row=worked_row())
    triple_measurement = TripleMeasurement(
        z=1, double_difference_sigma=1.0, components_used="111", pair_cold_started=False
    )
    with pytest.raises(DataFileError, match="measurement"):
        files.DdiffRecord(measurement=triple_measurement, row=worked_row(flags="P"))


def test_a_slip_correction_and_the_s_flag_go_together() -> None:
    """Refuse a corrected measurement without S, S without one, and S on a triple."""
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(measurement=None, row=worked_row(flags="PS"))
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(
            measurement=APPENDIX_A_MEASUREMENT.corrected(1), row=worked_row()
        )
    with pytest.raises(DataFileError, match="S"):
        files.MeasRecord(measurement=APPENDIX_A_MEASUREMENT, row=worked_row(flags="AS"))
    triple_measurement = TripleMeasurement(
        z=1, double_difference_sigma=1.0, components_used="111", pair_cold_started=False
    )
    with pytest.raises(DataFileError, match="S"):
        files.DdiffRecord(measurement=triple_measurement, row=worked_row(flags="AS"))


def test_a_measurement_belongs_to_its_row_s_epoch() -> None:
    """Refuse a pair measurement taken in another epoch than its row's."""
    with pytest.raises(DataFileError, match="epoch"):
        files.MeasRecord(
            measurement=APPENDIX_A_MEASUREMENT,
            row=worked_row(interpolated_datetime=E + ONE_EPOCH),
        )


# ------------------------------------------------------------- round trips

EPOCH_STARTS: Final = st.integers(min_value=0, max_value=20 * 365 * 144).map(
    lambda k: datetime(2015, 1, 1, tzinfo=UTC) + k * ONE_EPOCH
)
"""Ten-minute marks over twenty years."""

COLUMN_FLOATS: Final = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-99, max_value=1e99),
    st.floats(min_value=-1e99, max_value=-1e-99),
)
"""Floats that fit a float column."""


@st.composite
def valid_rows(draw: st.DrawFn, *, has_measurement: bool, for_pair: bool) -> Row:
    """Draw a valid row; with ``has_measurement``, one with a measurement.

    A pair's row may carry S when measured and has no innovation, which its
    file does not hold; a triple's never carries S.
    """
    epoch_start = draw(EPOCH_STARTS)
    filter_states: Literal[1, 2, 3] = draw(st.sampled_from([1, 2, 3]))
    is_dormant = draw(st.booleans())
    allowed_outcomes = ("R", "X") if is_dormant else ("A", "R", "X")
    outcome = draw(st.sampled_from(allowed_outcomes)) if has_measurement else "P"
    extra_flags = draw(
        st.sets(st.sampled_from("SN" if for_pair and has_measurement else "N"))
    )
    if not is_dormant and filter_states > 1 and draw(st.booleans()):
        extra_flags.add("U")
    flags = "".join(
        letter
        for letter in "ARXPDSNU"
        if letter in {outcome, *extra_flags, *("D" if is_dormant else "")}
    )
    reject_count = draw(st.integers(min_value=0, max_value=3))
    rejects = tuple(
        (epoch_start - ONE_EPOCH * (reject_count - reject_index), draw(COLUMN_FLOATS))
        for reject_index in range(reject_count)
    )
    is_tracked = not is_dormant
    return Row(
        interpolated_datetime=epoch_start,
        innovation=draw(st.one_of(st.none(), COLUMN_FLOATS))
        if has_measurement and not for_pair
        else None,
        x_fs=draw(st.integers(1 - 10**18, 10**18 - 1)) if is_tracked else None,
        y=(draw(COLUMN_FLOATS) if filter_states > 1 else 0.0) if is_tracked else None,
        d=(draw(COLUMN_FLOATS) if filter_states == 3 else 0.0) if is_tracked else None,
        innovation_scale=abs(draw(COLUMN_FLOATS)) if is_tracked else None,
        segment=draw(st.integers(0, 10**8)),
        step_offset=draw(st.integers(-(10**14), 10**14)),
        epochs_in_segment=draw(st.integers(0, 10**8)),
        epochs_since_accept=draw(st.integers(0, 10**8)),
        consecutive_rejects=draw(st.integers(0, 10**8)),
        rejects=rejects,
        filter_states=filter_states,
        time_constant=draw(st.floats(1, 1e6)) if filter_states > 1 else None,
        scale_time_constant=draw(st.floats(1, 1e6)),
        flags=flags,
    )


@st.composite
def meas_records(draw: st.DrawFn) -> files.MeasRecord:
    """Draw a valid measurement file record."""
    has_measurement = draw(st.booleans())
    drawn_row = draw(valid_rows(has_measurement=has_measurement, for_pair=True))
    if not has_measurement:
        return files.MeasRecord(measurement=None, row=drawn_row)
    epoch_mjd = datetime_to_mjd(drawn_row.interpolated_datetime)
    offset_us = draw(st.integers(min_value=1, max_value=6_900))
    pair_measurement = PairMeasurement(
        measurement_mjd=round(epoch_mjd + offset_us * 1e-6, 6),
        measured_phase=draw(st.integers(0, PHASE_MAX)),
        rms=draw(st.integers(0, RMS_MAX)),
        cycle_count=draw(st.integers(-(10**10), 10**10)),
        z=draw(st.integers(-(10**14), 10**14)),
        slip="S" in drawn_row.flags,
    )
    return files.MeasRecord(measurement=pair_measurement, row=drawn_row)


@st.composite
def ddiff_records(draw: st.DrawFn) -> files.DdiffRecord:
    """Draw a valid double-difference file record."""
    has_measurement = draw(st.booleans())
    drawn_row = draw(valid_rows(has_measurement=has_measurement, for_pair=False))
    if not has_measurement:
        return files.DdiffRecord(measurement=None, row=drawn_row)
    triple_measurement = TripleMeasurement(
        z=draw(st.integers(-(10**14), 10**14)),
        double_difference_sigma=draw(st.floats(min_value=1e-99, max_value=1e99)),
        components_used=draw(st.sampled_from(["111", "110", "101"])),
        pair_cold_started=False,
    )
    return files.DdiffRecord(measurement=triple_measurement, row=drawn_row)


@given(meas_records())
def test_a_measurement_row_parses_back_to_its_record(
    file_record: files.MeasRecord,
) -> None:
    """Give back the record, and the same text when formatted again (U20)."""
    row_line = files.format_meas_row(file_record)
    assert len(row_line) == files.MEAS_WIDTH
    assert files.parse_meas_row(row_line) == file_record
    assert files.format_meas_row(files.parse_meas_row(row_line)) == row_line


@given(ddiff_records())
def test_a_double_difference_row_parses_back_to_its_record(
    file_record: files.DdiffRecord,
) -> None:
    """Give back the record, and the same text when formatted again (U20)."""
    row_line = files.format_ddiff_row(file_record)
    assert len(row_line) == files.DDIFF_WIDTH
    assert files.parse_ddiff_row(row_line) == file_record


# --------------------------------------------------- file check, last row

PAIR_KEY: Final = ("mc2", "ox23")
"""The pair the files below are for."""


def predicted_row_lines(row_count: int) -> list[str]:
    """Give ``row_count`` rows of the pair, one per epoch from E, with newlines."""
    row_lines = []
    for epoch_index in range(row_count):
        meas_record = files.MeasRecord(
            measurement=None,
            row=worked_row(
                interpolated_datetime=E + epoch_index * ONE_EPOCH,
                epochs_in_segment=812 + epoch_index,
                epochs_since_accept=epoch_index + 1,
                flags="P",
            ),
        )
        row_lines.append(files.format_meas_row(meas_record) + "\n")
    return row_lines


def write_meas_file(
    tmp_path: Path, row_text: str, *, header_text: str | None = None
) -> Path:
    """Write a measurement file of the pair: its header and ``row_text``."""
    meas_path = tmp_path / "das_a.mc2.ox23.dat"
    file_header = (
        files.header("meas", "a", PAIR_KEY) if header_text is None else header_text
    )
    meas_path.write_bytes((file_header + row_text).encode("ascii"))
    return meas_path


def test_a_sound_file_is_good_through_its_last_row(tmp_path: Path) -> None:
    """Give the last row's epoch for a file of whole rows (U26)."""
    meas_path = write_meas_file(tmp_path, "".join(predicted_row_lines(4)))
    assert files.good_through(meas_path, "meas") == E + 3 * ONE_EPOCH


def test_a_file_cut_inside_its_last_row_is_good_through_the_row_before(
    tmp_path: Path,
) -> None:
    """Give the epoch of the last whole row of a file with a torn line (U26)."""
    file_text = "".join(predicted_row_lines(4))
    meas_path = write_meas_file(tmp_path, file_text[: -files.MEAS_WIDTH // 2])
    assert files.good_through(meas_path, "meas") == E + 2 * ONE_EPOCH


@pytest.mark.parametrize(
    "cut_bytes", [0, 1, files.MEAS_WIDTH + 1, 10 * (files.MEAS_WIDTH + 1) + 7]
)
def test_a_file_without_a_whole_row_holds_nothing_good(
    tmp_path: Path, cut_bytes: int
) -> None:
    """Give None for a file cut inside its header, or holding only its header (U26)."""
    whole_header = files.header("meas", "a", PAIR_KEY)
    meas_path = write_meas_file(tmp_path, "", header_text=whole_header[:cut_bytes])
    assert files.good_through(meas_path, "meas") is None
    assert (
        files.good_through(
            write_meas_file(tmp_path, "", header_text=whole_header), "meas"
        )
        is None
    )


def test_a_row_of_the_wrong_length_inside_a_file_ends_what_is_good(
    tmp_path: Path,
) -> None:
    """Give the epoch before the first line that does not parse (U26)."""
    row_lines = predicted_row_lines(5)
    row_lines[2] = row_lines[2][:100] + row_lines[2][101:]
    meas_path = write_meas_file(tmp_path, "".join(row_lines))
    assert files.good_through(meas_path, "meas") == E + ONE_EPOCH


def predicted_as_accepted(row_line: str) -> str:
    """Give a predicted row with its flag P made A: a row with no measurement."""
    row_fields = row_line.split(", ")
    (flag_index,) = [
        field_index
        for field_index, row_field in enumerate(row_fields)
        if row_field.strip() == "P"
    ]
    row_fields[flag_index] = row_fields[flag_index].replace("P", "A")
    return ", ".join(row_fields)


def test_a_row_that_breaks_a_record_rule_is_damaged(tmp_path: Path) -> None:
    """End what is good at a row that parses but breaks a record's rules (U26)."""
    row_lines = predicted_row_lines(4)
    row_lines[3] = predicted_as_accepted(row_lines[3])
    meas_path = write_meas_file(tmp_path, "".join(row_lines))
    assert files.good_through(meas_path, "meas") == E + 2 * ONE_EPOCH
    files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH)
    assert files.good_through(meas_path, "meas") == E + 2 * ONE_EPOCH
    assert meas_path.read_text().endswith(predicted_row_lines(3)[2])


def test_a_sound_file_is_not_scanned(tmp_path: Path) -> None:
    """Take a file of whole rows whose last row parses as sound, unscanned."""
    row_lines = predicted_row_lines(5)
    row_lines[3] = row_lines[3].replace("        P", "        Q")
    meas_path = write_meas_file(tmp_path, "".join(row_lines))
    assert files.good_through(meas_path, "meas") == E + 4 * ONE_EPOCH


def test_a_file_whose_last_row_does_not_parse_is_scanned(tmp_path: Path) -> None:
    """Scan a file of whole rows when its last row does not parse."""
    row_lines = predicted_row_lines(5)
    row_lines[4] = row_lines[4].replace("        P", "        Q")
    meas_path = write_meas_file(tmp_path, "".join(row_lines))
    assert files.good_through(meas_path, "meas") == E + 3 * ONE_EPOCH


def test_a_row_whose_newline_is_lost_is_not_good(tmp_path: Path) -> None:
    """Refuse a slot that does not end in a newline, though its text parses."""
    file_text = "".join(predicted_row_lines(3))
    meas_path = write_meas_file(tmp_path, file_text[:-1] + "x")
    assert files.good_through(meas_path, "meas") == E + ONE_EPOCH


def test_the_scan_stops_at_the_first_damaged_line(tmp_path: Path) -> None:
    """Stop at the first damaged line, however good the lines after it."""
    row_lines = predicted_row_lines(5)
    row_lines[1] = row_lines[1].replace("        P", "        Q")
    meas_path = write_meas_file(tmp_path, "".join(row_lines) + "2025")
    assert files.good_through(meas_path, "meas") == E


def test_a_damaged_first_row_cannot_be_placed_in_time(tmp_path: Path) -> None:
    """Raise DataFileError when no row of the file parses (U26)."""
    row_lines = predicted_row_lines(3)
    row_lines[0] = row_lines[0].replace("        P", "        Q")
    meas_path = write_meas_file(tmp_path, "".join(row_lines) + "2025")
    with pytest.raises(DataFileError, match="damaged first row"):
        files.good_through(meas_path, "meas")


@pytest.mark.parametrize(
    "row_line",
    [b"#" + b" " * files.MEAS_WIDTH, b"x" * files.MEAS_WIDTH, "é".encode() * 300],
)
def test_a_line_that_is_not_a_row_has_no_epoch(row_line: bytes) -> None:
    """Give no epoch for a header line, a line with no newline, or non-ASCII."""
    assert files.row_epoch(row_line + b"\n", "meas") is None
    assert files.row_epoch(predicted_row_lines(1)[0].encode()[:-1], "meas") is None


def test_a_row_gives_its_epoch() -> None:
    """Give a good row's epoch."""
    assert files.row_epoch(predicted_row_lines(2)[1].encode(), "meas") == E + ONE_EPOCH


def test_the_last_row_of_a_sound_file_is_read(tmp_path: Path) -> None:
    """Read a sound file's last row back as its row."""
    meas_path = write_meas_file(tmp_path, "".join(predicted_row_lines(3)))
    last_row = files.read_last_row(meas_path, "meas")
    assert last_row == worked_row(
        interpolated_datetime=E + 2 * ONE_EPOCH,
        epochs_in_segment=814,
        epochs_since_accept=3,
        flags="P",
    )


def test_the_last_row_of_a_double_difference_file_is_read(tmp_path: Path) -> None:
    """Read a double-difference file's last row."""
    ddiff_path = tmp_path / "das_a.mc1.mc2.ox23.dat"
    triple_key = ("mc1", "mc2", "ox23")
    file_text = files.header("ddiff", "a", triple_key) + DDIFF_EXAMPLE[0] + "\n"
    ddiff_path.write_bytes(file_text.encode("ascii"))
    assert files.read_last_row(ddiff_path, "ddiff").x_fs == 6_666_667_291
    assert files.good_through(ddiff_path, "ddiff") == E


@pytest.mark.parametrize("cut_bytes", [1, 2 * (files.MEAS_WIDTH + 1)])
def test_the_last_row_is_read_only_from_a_sound_file(
    tmp_path: Path, cut_bytes: int
) -> None:
    """Raise DataFileError for a file that is torn or holds no row."""
    file_text = "".join(predicted_row_lines(2))
    meas_path = write_meas_file(tmp_path, file_text[:-cut_bytes])
    with pytest.raises(DataFileError, match="not sound"):
        files.read_last_row(meas_path, "meas")


def test_a_last_row_that_is_not_ascii_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for a sound-sized file whose last row is not ASCII."""
    meas_path = write_meas_file(tmp_path, "".join(predicted_row_lines(2)))
    file_bytes = bytearray(meas_path.read_bytes())
    file_bytes[-3] = 0xE9
    meas_path.write_bytes(bytes(file_bytes))
    with pytest.raises(DataFileError, match="not ASCII"):
        files.read_last_row(meas_path, "meas")


def test_a_file_that_cannot_be_opened_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError for a file that cannot be read."""
    with pytest.raises(DataFileError, match="cannot read"):
        files.good_through(tmp_path / "missing.dat", "meas")
    with pytest.raises(DataFileError, match="cannot read"):
        files.read_last_row(tmp_path / "missing.dat", "meas")


# ------------------------------------------------------ day buffer and write

TRIPLE_KEY: Final = ("mc1", "mc2", "ox23")
"""The triple the double-difference files below are for."""


def make_archives(tmp_path: Path) -> tuple[Path, Path]:
    """Make the two archive directories."""
    meas_directory, ddiff_directory = tmp_path / "meas", tmp_path / "ddiff"
    meas_directory.mkdir()
    ddiff_directory.mkdir()
    return meas_directory, ddiff_directory


def predicted_record(epoch_index: int) -> files.MeasRecord:
    """Give the pair's predicted record ``epoch_index`` epochs after E."""
    return files.MeasRecord(
        measurement=None,
        row=worked_row(
            interpolated_datetime=E + epoch_index * ONE_EPOCH,
            epochs_in_segment=812 + epoch_index,
            epochs_since_accept=epoch_index + 1,
            flags="P",
        ),
    )


def predicted_triple_record(epoch_index: int) -> files.DdiffRecord:
    """Give the triple's predicted record ``epoch_index`` epochs after E."""
    return files.DdiffRecord(measurement=None, row=predicted_record(epoch_index).row)


def filled_buffer(
    tmp_path: Path, epoch_count: int = 2
) -> tuple[files.DayBuffer, Path, Path]:
    """Give a buffer of ``epoch_count`` rows of the pair and triple, and their paths."""
    meas_directory, ddiff_directory = make_archives(tmp_path)
    pair_path, triple_path = (
        meas_directory / "das_a.mc2.ox23.dat",
        ddiff_directory / "das_a.mc1.mc2.ox23.dat",
    )
    day_buffer = files.DayBuffer("a")
    for epoch_index in range(epoch_count):
        day_buffer.add(triple_path, TRIPLE_KEY, predicted_triple_record(epoch_index))
        day_buffer.add(pair_path, PAIR_KEY, predicted_record(epoch_index))
    return day_buffer, pair_path, triple_path


def test_a_new_file_gets_its_header_and_rows_in_one_write(tmp_path: Path) -> None:
    """Create each file with its header then its rows, all at once (5.8)."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    expected_text = files.header("meas", "a", PAIR_KEY) + "".join(
        files.format_meas_row(predicted_record(i)) + "\n" for i in range(2)
    )
    assert pair_path.read_text(encoding="ascii") == expected_text
    assert files.good_through(triple_path, "ddiff") == E + ONE_EPOCH


def test_a_later_write_appends_to_the_file(tmp_path: Path) -> None:
    """Append the next day's rows after the rows already written."""
    day_buffer, pair_path, _ = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    day_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    files.write_buffer(day_buffer)
    assert files.good_through(pair_path, "meas") == E + 2 * ONE_EPOCH
    assert (
        pair_path.read_text(encoding="ascii").count("das_processor measurement file")
        == 1
    )


def test_a_series_first_seen_in_a_day_appears_at_the_day_s_write(
    tmp_path: Path,
) -> None:
    """Create a file for a series first seen mid-day, header first, at the write."""
    day_buffer, pair_path, _ = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    later_path = pair_path.parent / "das_a.mc2.cs7.dat"
    day_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    day_buffer.add(later_path, ("mc2", "cs7"), predicted_record(2))
    assert not later_path.exists()
    files.write_buffer(day_buffer)
    file_text = later_path.read_text(encoding="ascii")
    assert file_text.startswith("# das_processor measurement file, format 1")
    assert (
        files.read_last_row(later_path, "meas").interpolated_datetime
        == E + 2 * ONE_EPOCH
    )


def test_after_a_write_the_text_is_empty_and_the_last_rows_remain(
    tmp_path: Path,
) -> None:
    """Empty the buffer's text and keep each series' newest row (5.8)."""
    day_buffer, _, _ = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    assert day_buffer.file_texts == {}
    assert day_buffer.last_rows == {
        PAIR_KEY: predicted_record(1).row,
        TRIPLE_KEY: predicted_triple_record(1).row,
    }


def test_the_newest_row_is_the_one_read_back(tmp_path: Path) -> None:
    """Keep the row parsed back from its line, as a later run would read it (I5)."""
    meas, _ = make_archives(tmp_path)
    day_buffer = files.DayBuffer("a")
    file_record = files.MeasRecord(
        measurement=APPENDIX_A_MEASUREMENT, row=worked_row(innovation=2.62)
    )
    day_buffer.add(meas / "das_a.mc2.ox23.dat", PAIR_KEY, file_record)
    assert day_buffer.last_rows[PAIR_KEY].innovation is None
    assert day_buffer.last_rows[PAIR_KEY] == worked_row()


def test_a_row_too_wide_is_refused_before_it_is_buffered(tmp_path: Path) -> None:
    """Raise DataFileError at add, before any text is kept."""
    meas, _ = make_archives(tmp_path)
    day_buffer = files.DayBuffer("a")
    file_record = files.MeasRecord(
        measurement=None, row=worked_row(step_offset=10**16, flags="P")
    )
    with pytest.raises(DataFileError, match="does not fit"):
        day_buffer.add(meas / "das_a.mc2.ox23.dat", PAIR_KEY, file_record)
    assert day_buffer.file_texts == {}


def test_a_file_keeps_one_series(tmp_path: Path) -> None:
    """Raise DataFileError when a path is given rows of two series or kinds."""
    day_buffer, pair_path, _ = filled_buffer(tmp_path)
    with pytest.raises(DataFileError, match="series"):
        day_buffer.add(pair_path, ("mc2", "cs7"), predicted_record(2))
    with pytest.raises(DataFileError, match="series"):
        day_buffer.add(pair_path, PAIR_KEY, predicted_triple_record(2))


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
    meas_directory, ddiff_directory = make_archives(tmp_path)
    day_buffer = files.DayBuffer("a")
    pair_keys: list[PairKey] = [("mc2", "ox23"), ("mc1", "mc2"), ("mc1", "mc1")]
    triple_keys: list[TripleKey] = [("mc2", "mc2", "ox23"), ("mc1", "mc2", "ox23")]
    for triple in triple_keys:
        day_buffer.add(
            ddiff_directory / f"das_a.{'.'.join(triple)}.dat",
            triple,
            predicted_triple_record(0),
        )
    for pair in pair_keys:
        day_buffer.add(
            meas_directory / f"das_a.{'.'.join(pair)}.dat", pair, predicted_record(0)
        )
    recorder = Recorder()
    real_open, real_fsync = Path.open, os.fsync
    fd_names: dict[int, str] = {}

    class Tracked:
        """A file that records its close."""

        def __init__(self, wrapped_file: object, file_name: str) -> None:
            """Wrap ``wrapped_file``."""
            self.inner, self.name = wrapped_file, file_name

        def __enter__(self) -> Tracked:
            """Enter the wrapped file."""
            self.inner.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *exit_details: object) -> None:
            """Close the wrapped file and record it."""
            fd_names.pop(self.inner.fileno(), None)  # type: ignore[attr-defined]
            self.inner.__exit__(*exit_details)  # type: ignore[attr-defined]
            recorder.open_now -= 1
            recorder.events.append(("close", self.name))

        def write(self, chunk: bytes) -> int:
            """Write to the wrapped file."""
            return self.inner.write(chunk)  # type: ignore[attr-defined, no-any-return]

        def flush(self) -> None:
            """Flush the wrapped file."""
            self.inner.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            """Give the wrapped file's descriptor."""
            fd: int = self.inner.fileno()  # type: ignore[attr-defined]
            fd_names[fd] = self.name
            return fd

    def tracked_open(opened_path: Path, *args: object, **kwargs: object) -> Tracked:
        """Open ``opened_path`` and record it."""
        recorder.open_now += 1
        recorder.most_open = max(recorder.most_open, recorder.open_now)
        recorder.events.append(("open", opened_path.name))
        return Tracked(real_open(opened_path, *args, **kwargs), opened_path.name)  # type: ignore[call-overload]

    def tracked_fsync(fd: int) -> None:
        """Flush ``fd`` and record it."""
        recorder.events.append(("fsync", fd_names.get(fd, "directory")))
        real_fsync(fd)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(os, "fsync", tracked_fsync)
    files.write_buffer(day_buffer)
    file_names = [
        f"das_a.{'.'.join(series_key)}.dat"
        for series_key in sorted(pair_keys) + sorted(triple_keys)
    ]
    expected_events = [
        file_event
        for file_name in file_names
        for file_event in (
            ("open", file_name),
            ("fsync", file_name),
            ("close", file_name),
        )
    ]
    assert recorder.events[: len(expected_events)] == expected_events
    assert recorder.events[len(expected_events) :] == [("fsync", "directory")] * 2
    assert recorder.most_open == 1


@pytest.mark.parametrize(
    "check_problem",
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_problem: str
) -> None:
    """Raise DataFileError in the prepare step with every file as it was (5.8)."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    day_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    day_buffer.add(triple_path, TRIPLE_KEY, predicted_triple_record(2))
    new_path = pair_path.parent / "das_a.mc2.cs7.dat"
    day_buffer.add(new_path, ("mc2", "cs7"), predicted_record(2))
    if check_problem == "torn":
        with pair_path.open("ab") as open_file:
            open_file.write(b"2025")
    elif check_problem == "not_regular":
        triple_path.unlink()
        triple_path.mkdir()
    elif check_problem == "symlink":
        link_target = tmp_path / "copy.dat"
        link_target.write_bytes(triple_path.read_bytes())
        triple_path.unlink()
        triple_path.symlink_to(link_target)
    elif check_problem == "read_only":
        triple_path.chmod(0o444)
    elif check_problem == "clash":
        new_path.symlink_to(tmp_path / "elsewhere")
    elif check_problem == "no_directory":
        new_path = tmp_path / "gone" / "das_a.mc2.cs7.dat"
        day_buffer.add(new_path, ("mc2", "cs7"), predicted_record(3))
    elif check_problem == "no_space":
        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10))
    else:
        day_buffer.file_texts[pair_path] += "é\n"
    bytes_before = {
        data_file: data_file.read_bytes()
        for data_file in (pair_path, triple_path)
        if data_file.is_file()
    }
    with pytest.raises(DataFileError):
        files.write_buffer(day_buffer)
    bytes_after = {
        data_file: data_file.read_bytes()
        for data_file in (pair_path, triple_path)
        if data_file.is_file()
    }
    assert bytes_after == bytes_before
    assert not (new_path.exists() and not new_path.is_symlink())


def test_an_error_while_writing_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError for a device fault in the write step."""
    day_buffer, _, _ = filled_buffer(tmp_path)

    def failing(_fd: int) -> None:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(DataFileError, match="Input/output error"):
        files.write_buffer(day_buffer)


def test_an_error_flushing_a_directory_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError when a new file's directory cannot be flushed."""
    day_buffer, _, _ = filled_buffer(tmp_path)

    def failing(_path: object, _flags: int) -> int:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "open", failing)
    with pytest.raises(DataFileError, match="cannot flush directory"):
        files.write_buffer(day_buffer)


def test_an_empty_buffer_writes_nothing(tmp_path: Path) -> None:
    """Change nothing when no rows are buffered."""
    files.write_buffer(files.DayBuffer("a"))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------- roll-back, redo


def written_meas_file(meas_path: Path, row_count: int) -> Path:
    """Write the pair's file with ``row_count`` rows from E, and give its path."""
    meas_path.parent.mkdir(exist_ok=True)
    meas_path.write_text(
        files.header("meas", "a", PAIR_KEY) + "".join(predicted_row_lines(row_count)),
        encoding="ascii",
    )
    return meas_path


def row_epochs_of(meas_path: Path) -> list[datetime]:
    """Give the epoch of every row of the pair's file."""
    line_size = files.MEAS_WIDTH + 1
    file_bytes = meas_path.read_bytes()
    row_slots = [
        file_bytes[slot_start : slot_start + line_size]
        for slot_start in range(
            files.MEAS_HEADER_LINES * line_size, len(file_bytes), line_size
        )
    ]
    return [
        files.parse_meas_row(row_slot.decode()[:-1]).row.interpolated_datetime
        for row_slot in row_slots
    ]


def test_a_file_is_rolled_back_to_just_after_the_common_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Truncate just after the row for L, and say it was cut, logging nothing (6.7)."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 5)
    assert files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH) == "cut"
    assert row_epochs_of(meas_path) == [E, E + ONE_EPOCH, E + 2 * ONE_EPOCH]
    assert not caplog.records


def test_a_torn_line_after_the_common_epoch_goes_too(tmp_path: Path) -> None:
    """Remove a torn line with the rows after L."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 3)
    with meas_path.open("ab") as open_file:
        open_file.write(b"2025-09-23 06:3")
    files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH)
    assert row_epochs_of(meas_path) == [E, E + ONE_EPOCH, E + 2 * ONE_EPOCH]
    assert files.good_through(meas_path, "meas") == E + 2 * ONE_EPOCH


def test_a_sound_file_ending_at_the_common_epoch_is_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Leave a file that already ends at L as it is, unlogged."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 3)
    stat_before = meas_path.stat()
    files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH)
    stat_after = meas_path.stat()
    assert (stat_after.st_ino, stat_after.st_mtime_ns, stat_after.st_size) == (
        stat_before.st_ino,
        stat_before.st_mtime_ns,
        stat_before.st_size,
    )
    assert not caplog.records


@pytest.mark.parametrize("common_epoch", [None, E - ONE_EPOCH])
def test_a_file_with_no_row_at_or_before_the_common_epoch_is_deleted(
    tmp_path: Path, common_epoch: datetime | None
) -> None:
    """Delete a file whose first row is after L, and every file when there is no L."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 3)
    files.roll_back(meas_path, "meas", common_epoch)
    assert not meas_path.exists()


def test_a_file_without_one_row_per_epoch_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when the row found for L is of another epoch."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 2)
    row_lines = predicted_row_lines(4)
    with meas_path.open("a", encoding="ascii") as open_file:
        open_file.write(row_lines[3])
    with pytest.raises(DataFileError, match="one row per epoch"):
        files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH)


def test_a_common_epoch_past_the_file_s_rows_is_refused(tmp_path: Path) -> None:
    """Raise DataFileError when the file holds no row for L at all."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 2)
    with pytest.raises(DataFileError, match="no row for"):
        files.roll_back(meas_path, "meas", E + 5 * ONE_EPOCH)


def write_archive(tmp_path: Path) -> list[tuple[Path, files.FileKind]]:
    """Write a measurement file of five rows and a double-difference file of three."""
    pair_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 5)
    triple_path = tmp_path / "ddiff" / "das_a.mc1.mc2.ox23.dat"
    triple_path.parent.mkdir()
    row_lines = "".join(
        files.format_ddiff_row(predicted_triple_record(i)) + "\n" for i in range(3)
    )
    triple_path.write_text(
        files.header("ddiff", "a", TRIPLE_KEY) + row_lines, encoding="ascii"
    )
    return [(pair_path, "meas"), (triple_path, "ddiff")]


def test_a_redo_deletes_every_row_at_or_after_its_epoch(tmp_path: Path) -> None:
    """Truncate every file before its first row at or after the mark (6.5)."""
    data_files = write_archive(tmp_path)
    files.redo_from(data_files, E + 2 * ONE_EPOCH, "a")
    assert [
        files.good_through(meas_path, file_kind) for meas_path, file_kind in data_files
    ] == [
        E + ONE_EPOCH,
        E + ONE_EPOCH,
    ]


def test_a_redo_deletes_a_file_with_no_earlier_row(tmp_path: Path) -> None:
    """Delete every file when the redo starts at or before its first row."""
    data_files = write_archive(tmp_path)
    files.redo_from(data_files, E, "a")
    assert not any(meas_path.exists() for meas_path, _ in data_files)


def test_a_redo_past_a_file_s_end_leaves_it(tmp_path: Path) -> None:
    """Keep a file whose rows all come before the redo."""
    data_files = write_archive(tmp_path)
    files.redo_from(data_files, E + 4 * ONE_EPOCH, "a")
    assert [
        files.good_through(meas_path, file_kind) for meas_path, file_kind in data_files
    ] == [
        E + 3 * ONE_EPOCH,
        E + 2 * ONE_EPOCH,
    ]


def test_an_interrupted_redo_finishes_when_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish the deletion on a second run after the first stopped part way (6.5)."""
    data_files = write_archive(tmp_path)
    real_open = Path.open
    open_calls = {"count": 0}

    def failing_second(
        opened_path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        """Fail the second file opened to be cut."""
        if mode == "r+b":
            open_calls["count"] += 1
            if open_calls["count"] == 2:
                raise OSError(5, "Input/output error")
        return real_open(opened_path, mode, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", failing_second)
    with pytest.raises(DataFileError, match="Input/output error"):
        files.redo_from(data_files, E + ONE_EPOCH, "a")
    monkeypatch.undo()
    files.redo_from(data_files, E + ONE_EPOCH, "a")
    assert [
        files.good_through(meas_path, file_kind) for meas_path, file_kind in data_files
    ] == [E, E]


def test_a_redo_and_a_roll_back_at_one_start_keep_the_archive_in_step(
    tmp_path: Path,
) -> None:
    """Redo first, then roll back what is left to the common epoch (review focus 5)."""
    data_files = write_archive(tmp_path)
    triple_path = data_files[1][0]
    with triple_path.open("ab") as open_file:
        open_file.write(b"2025-09-23 06:3")
    files.redo_from(data_files, E + 4 * ONE_EPOCH, "a")
    good_epochs = [
        files.good_through(meas_path, file_kind) for meas_path, file_kind in data_files
    ]
    assert good_epochs == [E + 3 * ONE_EPOCH, E + 2 * ONE_EPOCH]
    common_epoch = min(
        good_epoch for good_epoch in good_epochs if good_epoch is not None
    )
    for meas_path, file_kind in data_files:
        files.roll_back(meas_path, file_kind, common_epoch)
    assert [
        files.good_through(meas_path, file_kind) for meas_path, file_kind in data_files
    ] == [
        E + 2 * ONE_EPOCH,
        E + 2 * ONE_EPOCH,
    ]
    assert all(
        meas_path.stat().st_size % (files.WIDTHS[file_kind] + 1) == 0
        for meas_path, file_kind in data_files
    )


def test_an_error_deleting_a_file_is_a_data_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise DataFileError when a file cannot be deleted."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 2)

    def failing(_path: Path, *_args: object) -> None:
        """Fail as a device would."""
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", failing)
    with pytest.raises(DataFileError, match="Permission denied"):
        files.roll_back(meas_path, "meas", None)


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
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    epoch_buffer = files.DayBuffer("a")
    epoch_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    epoch_buffer.add(triple_path, TRIPLE_KEY, predicted_triple_record(2))
    day_buffer.take(epoch_buffer)
    assert day_buffer.file_texts[pair_path].count("\n") == 3
    assert day_buffer.last_rows[PAIR_KEY] == predicted_record(2).row
    assert day_buffer.series_of(triple_path) == ("ddiff", TRIPLE_KEY)


def test_a_buffer_takes_nothing_from_a_clashing_one(tmp_path: Path) -> None:
    """Refuse rows of another series for a path, leaving the buffer as it was."""
    day_buffer, pair_path, _ = filled_buffer(tmp_path)
    buffer_before = dict(day_buffer.file_texts), dict(day_buffer.last_rows)
    epoch_buffer = files.DayBuffer("a")
    epoch_buffer.add(
        pair_path.parent / "das_a.mc2.cs7.dat", ("mc2", "cs7"), predicted_record(2)
    )
    epoch_buffer.add(pair_path, ("mc2", "hm1"), predicted_record(2))
    with pytest.raises(DataFileError, match="series"):
        day_buffer.take(epoch_buffer)
    assert (dict(day_buffer.file_texts), dict(day_buffer.last_rows)) == buffer_before


# ---------------------------------------------------------- the write journal


def journaled_buffer(tmp_path: Path) -> tuple[files.DayBuffer, Path, Path]:
    """Give a buffer with a journal and two epochs' rows, its journal and a file."""
    plain_buffer, pair_path, _ = filled_buffer(tmp_path)
    journal = tmp_path / "das_processor_a.writing"
    day_buffer = files.DayBuffer("a", journal)
    day_buffer.take(plain_buffer)
    return day_buffer, journal, pair_path


def test_a_buffer_knows_its_first_epoch(tmp_path: Path) -> None:
    """Keep the earliest epoch buffered since the last write, and forget it after."""
    day_buffer, _, pair_path = journaled_buffer(tmp_path)
    assert day_buffer.earliest_epoch == E
    later_buffer = files.DayBuffer("a")
    later_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    day_buffer.take(later_buffer)
    assert day_buffer.earliest_epoch == E
    files.write_buffer(day_buffer)
    assert [day_buffer.earliest_epoch] == [None]
    day_buffer.add(pair_path, PAIR_KEY, predicted_record(3))
    assert day_buffer.earliest_epoch == E + 3 * ONE_EPOCH


def test_the_journal_is_there_exactly_while_the_files_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flush the first epoch to the journal before any file opens; delete it after."""
    day_buffer, journal, _ = journaled_buffer(tmp_path)
    real_open = Path.open
    journal_texts: list[tuple[str, str | None]] = []

    def watching(opened_path: Path, *args: object, **kwargs: object) -> object:
        """Record, at each data file's opening, what the journal holds."""
        if opened_path != journal:
            journal_texts.append(
                (opened_path.name, journal.read_text() if journal.exists() else None)
            )
        return real_open(opened_path, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(Path, "open", watching)
    files.write_buffer(day_buffer)
    assert len(journal_texts) == 2
    assert all(
        journal_text == f"{E.isoformat()}\n" for _, journal_text in journal_texts
    )
    assert not journal.exists()


def test_a_buffer_without_a_journal_keeps_none(tmp_path: Path) -> None:
    """Write no journal for a buffer given none."""
    day_buffer, _, _ = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    assert sorted(directory_entry.name for directory_entry in tmp_path.iterdir()) == [
        "ddiff",
        "meas",
    ]


def test_a_journal_already_there_changes_no_file(tmp_path: Path) -> None:
    """Refuse to write while a journal is there, before any file is opened."""
    day_buffer, journal, pair_path = journaled_buffer(tmp_path)
    journal.write_text(f"{E.isoformat()}\n")
    with pytest.raises(DataFileError, match="still open"):
        files.write_buffer(day_buffer)
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
    day_buffer, _, pair_path = journaled_buffer(tmp_path)

    def failing(_fd: int) -> None:
        """Fail as a device would."""
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(DataFileError, match="cannot write journal"):
        files.write_buffer(day_buffer)
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
    "journal_bytes",
    [
        b"",
        b"2025-09-23T06:0",
        E.isoformat().encode(),
        b"2025-09-23T06:00:00\n",
        b"\xff\n",
    ],
)
def test_a_journal_not_whole_means_no_data_file_was_opened(
    tmp_path: Path, journal_bytes: bytes
) -> None:
    """Give None for a journal cut short, without its zone, or not ASCII."""
    journal = tmp_path / "das_processor_a.writing"
    journal.write_bytes(journal_bytes)
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
    with pytest.raises(DataFileError) as refusal:
        files.roll_back(tmp_path / "missing.dat", "meas", E)
    error_records = [
        log_record for log_record in caplog.records if log_record.levelname == "ERROR"
    ]
    assert [log_record.getMessage() for log_record in error_records] == [
        str(refusal.value)
    ]
    assert str(refusal.value).startswith(f"cannot read data file {tmp_path}")


def test_a_roll_back_says_what_it_did_to_each_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Give kept, cut or deleted for each file, the run logging it all once."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 5)
    assert files.roll_back(meas_path, "meas", E + 4 * ONE_EPOCH) == "kept"
    assert files.roll_back(meas_path, "meas", E + 2 * ONE_EPOCH) == "cut"
    assert files.roll_back(meas_path, "meas", E - ONE_EPOCH) == "deleted"
    assert not caplog.records


def test_a_redo_is_logged_once_with_what_it_did(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a redo once, at INFO, with how many files it cut, deleted and left."""
    data_files = write_archive(tmp_path)
    caplog.set_level(logging.INFO)
    files.redo_from(data_files, E + 3 * ONE_EPOCH, "a")
    files.redo_from(data_files, E, "a")
    redo_epoch = E + 3 * ONE_EPOCH
    assert [
        (log_record.levelname, log_record.getMessage()) for log_record in caplog.records
    ] == [
        (
            "INFO",
            f"redo of channel a from {redo_epoch}: 1 files cut, 0 deleted,"
            " 1 with no row at or after it",
        ),
        (
            "INFO",
            f"redo of channel a from {E}: 0 files cut, 2 deleted,"
            " 0 with no row at or after it",
        ),
    ]


def test_a_redo_deletes_a_file_holding_nothing_good(tmp_path: Path) -> None:
    """Delete a file with no whole row, whatever the redo's epoch."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 0)
    with meas_path.open("ab") as open_file:
        open_file.write(b"2025-09-23 06:0")
    files.redo_from([(meas_path, "meas")], E + 3 * ONE_EPOCH, "a")
    assert not meas_path.exists()


@pytest.mark.parametrize(
    ("check_problem", "message_template"),
    [
        ("torn", "data file {pair} is not sound: {length} bytes"),
        ("not_regular", "data file {triple} is not a regular file"),
        ("read_only", "data file {triple} cannot be written"),
        ("no_directory", "file {gone} cannot be created in {gone_parent}"),
        ("not_ascii", "the rows for {pair} are not ASCII"),
    ],
)
def test_a_failed_check_says_which_file_and_why(
    tmp_path: Path, check_problem: str, message_template: str
) -> None:
    """Name the file and what is wrong with it, in the prepare step (5.8)."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    files.write_buffer(day_buffer)
    day_buffer.add(pair_path, PAIR_KEY, predicted_record(2))
    day_buffer.add(triple_path, TRIPLE_KEY, predicted_triple_record(2))
    gone_file = tmp_path / "gone" / "das_a.mc2.cs7.dat"
    if check_problem == "torn":
        with pair_path.open("ab") as open_file:
            open_file.write(b"2025")
    elif check_problem == "not_regular":
        triple_path.unlink()
        triple_path.mkdir()
    elif check_problem == "read_only":
        triple_path.chmod(0o444)
    elif check_problem == "no_directory":
        day_buffer.add(gone_file, ("mc2", "cs7"), predicted_record(3))
    else:
        day_buffer.file_texts[pair_path] += "é\n"
    expected_message = message_template.format(
        pair=pair_path,
        triple=triple_path,
        length=pair_path.stat().st_size,
        gone=gone_file,
        gone_parent=gone_file.parent,
    )
    with pytest.raises(DataFileError) as refusal:
        files.write_buffer(day_buffer)
    assert str(refusal.value) == expected_message


def test_a_new_file_s_directory_that_cannot_be_written_into_is_refused(
    tmp_path: Path,
) -> None:
    """Refuse, before any file opens, a new file in a directory not writable."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    triple_path.parent.chmod(0o555)
    try:
        with pytest.raises(DataFileError, match="cannot be created in"):
            files.write_buffer(day_buffer)
        assert not pair_path.exists()
    finally:
        triple_path.parent.chmod(0o755)


def test_free_space_just_enough_is_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write when the free space is exactly the bytes to write, refuse one less."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    pair_text = files.header("meas", "a", PAIR_KEY) + day_buffer.file_texts[pair_path]
    triple_text = (
        files.header("ddiff", "a", TRIPLE_KEY) + day_buffer.file_texts[triple_path]
    )
    byte_total = len(pair_text) + len(triple_text)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=byte_total - 1)
    )
    with pytest.raises(DataFileError) as refusal:
        files.write_buffer(day_buffer)
    assert str(refusal.value) == (
        f"{byte_total} bytes to write in {triple_path.parent},"
        f" only {byte_total - 1} free"
    )
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=byte_total)
    )
    files.write_buffer(day_buffer)
    assert pair_path.exists()


def test_free_space_is_counted_per_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weigh each device's bytes against that device's free space alone."""
    day_buffer, pair_path, triple_path = filled_buffer(tmp_path)
    bytes_by_directory = {
        pair_path.parent: len(
            files.header("meas", "a", PAIR_KEY) + day_buffer.file_texts[pair_path]
        ),
        triple_path.parent: len(
            files.header("ddiff", "a", TRIPLE_KEY) + day_buffer.file_texts[triple_path]
        ),
    }
    device_by_directory = {pair_path.parent: 1, triple_path.parent: 2}

    def stat(stat_path: Path, **_kwargs: object) -> object:
        """Put each archive on a device of its own; nothing else is looked at."""
        return SimpleNamespace(st_dev=device_by_directory[stat_path])

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda where: SimpleNamespace(free=bytes_by_directory[where]),
    )
    files.write_buffer(day_buffer)
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
    whole_second_measurement = measure_pair(
        measurement_mjd=60941.25,
        measured_phase=34579,
        rms=3,
        prediction=State(x=Fraction(1_234_567), y=0.0),
        w=Fraction(0),
        anchor=None,
    )
    row_line = files.format_meas_row(
        files.MeasRecord(measurement=whole_second_measurement, row=worked_row())
    )
    assert "2025-09-23 06:00:00.000000+00:00" in row_line


def test_a_value_too_wide_names_its_row_s_epoch() -> None:
    """Say which epoch's row holds a value too wide for its column."""
    meas_record = files.MeasRecord(
        measurement=None, row=worked_row(flags="P", segment=10**12)
    )
    with pytest.raises(DataFileError, match=r"^row of 2025-09-23 06:00:00\+00:00: "):
        files.format_meas_row(meas_record)
    ddiff_record = files.DdiffRecord(
        measurement=None, row=worked_row(flags="P", segment=10**12)
    )
    with pytest.raises(DataFileError, match=r"^row of 2025-09-23 06:00:00\+00:00: "):
        files.format_ddiff_row(ddiff_record)


def test_a_key_of_the_wrong_size_for_its_file_is_named() -> None:
    """Say a measurement file is for a pair, a double-difference file a triple."""
    with pytest.raises(DataFileError) as refusal:
        files.header("meas", "a", TRIPLE_KEY)
    assert "file is for a pair: " in str(refusal.value)
    with pytest.raises(DataFileError) as refusal:
        files.header("ddiff", "a", PAIR_KEY)
    assert "file is for a triple: " in str(refusal.value)


def with_field_replaced(row_line: str, field_index: int, field_text: str) -> str:
    """Give a line with one field's text replaced, right-justified as before."""
    row_fields = row_line.split(", ")
    row_fields[field_index] = field_text.rjust(len(row_fields[field_index]))
    return ", ".join(row_fields)


@pytest.mark.parametrize(
    ("file_kind", "line_change", "parse_reason"),
    [
        ("meas", lambda row_line: row_line.rsplit(", ", 1)[0], "26 fields, not 27"),
        (
            "meas",
            lambda row_line: with_field_replaced(row_line, 9, "nan"),
            "'nan' is not a finite",
        ),
        (
            "meas",
            lambda row_line: with_field_replaced(row_line, 8, "1234574.45"),
            "three decimals",
        ),
        (
            "meas",
            lambda row_line: with_field_replaced(row_line, 0, "-"),
            "never empty is empty",
        ),
        (
            "ddiff",
            lambda row_line: with_field_replaced(
                with_field_replaced(row_line, 2, "-"), 5, "-"
            ),
            "never empty is empty",
        ),
        (
            "ddiff",
            lambda row_line: with_field_replaced(
                with_field_replaced(row_line, 4, "-"), 5, "-"
            ),
            "never empty is empty",
        ),
        (
            "meas",
            lambda row_line: with_field_replaced(row_line, 17, "60941.243056"),
            "never empty",
        ),
    ],
)
def test_a_line_that_does_not_parse_says_why(
    file_kind: str, line_change: Callable[[str], str], parse_reason: str
) -> None:
    """Name the row by its first characters and the reason it does not parse."""
    row_line = line_change(MEAS_EXAMPLE[0] if file_kind == "meas" else DDIFF_EXAMPLE[0])
    with pytest.raises(DataFileError) as refusal:
        if file_kind == "meas":
            files.parse_meas_row(row_line)
        else:
            files.parse_ddiff_row(row_line)
    refusal_message = str(refusal.value)
    assert refusal_message.startswith(f"row {row_line[:25]!r} does not parse: "), (
        refusal_message
    )
    assert parse_reason in refusal_message


def test_a_line_not_written_so_names_its_row() -> None:
    """Name the row by its first characters when it is not as written."""
    row_line = MEAS_EXAMPLE[0].replace(
        "+1.2301290523526430e-02", "+12.301290523526430e-03"
    )
    with pytest.raises(DataFileError) as refusal:
        files.parse_meas_row(row_line)
    assert str(refusal.value) == (
        f"row {row_line[:25]!r} is not written as das_processor writes it"
    )


def test_a_field_never_empty_is_refused_in_those_words() -> None:
    """End the refusal of an empty field that is never empty with its reason."""
    row_line = with_field_replaced(MEAS_EXAMPLE[0], 0, "-")
    with pytest.raises(DataFileError) as refusal:
        files.parse_meas_row(row_line)
    assert str(refusal.value).endswith(
        " does not parse: a field that is never empty is empty"
    )


def device_error(*_args: object, **_kwargs: object) -> None:
    """Fail as a device would."""
    raise OSError(5, "Input/output error")


def test_a_device_fault_names_the_file_and_what_was_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Say which file could not be deleted, cut or written, and the device's error."""
    meas_path = written_meas_file(tmp_path / "meas" / "das_a.mc2.ox23.dat", 3)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", device_error)
        with pytest.raises(DataFileError) as refusal:
            files.roll_back(meas_path, "meas", None)
    assert (
        str(refusal.value) == f"cannot delete {meas_path}: [Errno 5] Input/output error"
    )
    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", device_error)
        with pytest.raises(DataFileError) as refusal:
            files.roll_back(meas_path, "meas", E)
    assert str(refusal.value) == (
        f"cannot cut data file {meas_path}: [Errno 5] Input/output error"
    )
    (tmp_path / "write").mkdir()
    day_buffer, pair_path, _ = filled_buffer(tmp_path / "write")
    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", device_error)
        with pytest.raises(DataFileError) as refusal:
            files.write_buffer(day_buffer)
    assert str(refusal.value) == (
        f"cannot write data file {pair_path}: [Errno 5] Input/output error"
    )


def test_after_a_stopped_write_a_damaged_first_row_holds_nothing_good(
    tmp_path: Path,
) -> None:
    """Give None, not a refusal, for a file whose rows never reached the device."""
    meas_path = write_meas_file(tmp_path, "\0" * (files.MEAS_WIDTH + 1) * 2)
    with pytest.raises(DataFileError, match="damaged first row"):
        files.good_through(meas_path, "meas")
    assert files.good_through(meas_path, "meas", stopped_write=True) is None


# ------------------------------------------------ one explanation per damaged file


def logged_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Give the ERROR messages logged."""
    return [
        log_record.getMessage()
        for log_record in caplog.records
        if log_record.levelname == "ERROR"
    ]


def test_a_damaged_line_is_explained_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log one ERROR naming the file, the last good row and what is wrong.

    The file's end is torn too: a sound file is not scanned (design 5.7).
    """
    row_lines = predicted_row_lines(4)
    row_lines[2] = predicted_as_accepted(row_lines[2])
    meas_path = write_meas_file(tmp_path, "".join(row_lines) + "2025")
    assert files.check_file(meas_path, "meas") == files.FileCheck(
        E + ONE_EPOCH, damaged=True
    )
    (error_message,) = logged_errors(caplog)
    assert error_message.startswith(
        f"data file {meas_path} is damaged after its row of {E + ONE_EPOCH}:"
    )
    assert "a row has a measurement exactly when it is not P" in error_message


@pytest.mark.parametrize(
    ("torn_tail", "damage_place", "damage_reason"),
    [
        ("2025", "after its row of {last}", "its last line is cut short"),
        (None, "from its first row", "it holds no whole row"),
    ],
)
def test_a_torn_file_is_explained_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    torn_tail: str | None,
    damage_place: str,
    damage_reason: str,
) -> None:
    """Log a file cut short inside a row, or holding no whole row, once."""
    file_text = (
        "".join(predicted_row_lines(2)) + torn_tail if torn_tail is not None else "2025"
    )
    meas_path = write_meas_file(tmp_path, file_text)
    file_check = files.check_file(meas_path, "meas")
    assert file_check.damaged
    damage_place = damage_place.format(last=E + ONE_EPOCH)
    assert logged_errors(caplog) == [
        f"data file {meas_path} is damaged {damage_place}: {damage_reason}"
    ]


def test_a_damaged_first_row_is_explained_in_its_refusal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse it with the reason, logged once; after a stopped write, explain it."""
    meas_path = write_meas_file(tmp_path, "\0" * (files.MEAS_WIDTH + 1) * 2)
    with pytest.raises(DataFileError) as refusal:
        files.check_file(meas_path, "meas")
    assert logged_errors(caplog) == [str(refusal.value)]
    assert str(refusal.value) == (
        f"{meas_path} has a damaged first row, so its rows cannot be placed in time:"
        " the line is cut short, with no newline"
    )
    caplog.clear()
    assert files.check_file(meas_path, "meas", stopped_write=True).good_through is None
    (error_message,) = logged_errors(caplog)
    assert error_message.startswith(
        f"data file {meas_path} is damaged from its first row: "
    )


def test_a_sound_file_is_not_damaged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say a sound file is not damaged, and log nothing."""
    meas_path = write_meas_file(tmp_path, "".join(predicted_row_lines(3)))
    assert files.check_file(meas_path, "meas") == files.FileCheck(
        E + 2 * ONE_EPOCH, damaged=False
    )
    assert not caplog.records
