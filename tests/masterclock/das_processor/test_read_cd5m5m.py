"""Tests for src/masterclock/das_processor/read_cd5m5m.py.

The rules covered: a line is five columns, its numbers written in their
plain form and each column valid, or it is refused as malformed; a record
works out its reference, instants and epoch from its columns, and none of
them may be passed in; a file's lines that are malformed, on another day,
late in their epoch, backwards in time or a repeated pair are logged under
that reason and skipped, and a skipped line is not remembered; a file whose
last line has no newline, or that cannot be read, ends the read; daily file
names cover days 50000 to 99999 only and read back to the day they were made
from; and the files of a directory are read in day order as one stream of
ten-minute blocks, from a given epoch if one is asked for.

Each refusal's reason, the skipped line as it is, each error and each DEBUG
message are word for word.
"""

import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from masterclock.app.timeutil import datetime_to_mjd, mjd_to_datetime
from masterclock.das_processor import read_cd5m5m as reader
from masterclock.das_processor.epochs import EPOCH_LENGTH
from masterclock.das_processor.exceptions import (
    DataFileError,
    LateLineError,
    MalformedLineError,
)
from masterclock.domain.phase import PHASE_MAX

DAY: Final = 60010
"""An invented day every file here covers unless a test says otherwise."""

SECOND: Final = 1 / 86_400
"""One second, as a fraction of a day."""


def mjd_at(seconds: float, day: int = DAY) -> float:
    """Return the MJD ``seconds`` into ``day``, to the places the DAS writes."""
    return round(day + seconds * SECOND, reader.MJD_DECIMALS)


def exact_mjd(seconds: int, day: int = DAY) -> float:
    """Return the MJD of the instant ``seconds`` into ``day``, unrounded."""
    return datetime_to_mjd(mjd_to_datetime(day) + timedelta(seconds=seconds))


def line(
    mjd: float,
    phase: int = 1000,
    rms: int = 20,
    switch: str = "1A01",
    clock: str = "clka",
) -> str:
    """Write one data line from its columns, ended with a newline."""
    return f"{mjd:.6f} {phase} {rms} {switch} {clock}\n"


def data_file(directory: Path, lines: list[str], day: int = DAY) -> Path:
    """Write ``lines`` as the daily file of ``day`` in ``directory``."""
    path = directory / f"cd5m5m_{day}.dat"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def make(seconds: float, clock: str = "clka", day: int = DAY) -> reader.DASMeasurement:
    """Build a measurement ``seconds`` into ``day``."""
    return reader.DASMeasurement(
        measurement_mjd=mjd_at(seconds, day),
        measured_phase=1000,
        rms=20,
        switch="1A01",
        clock=clock,
    )


def skipped(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the WARNING messages logged, which name each skipped line."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------- parse_line


def test_a_line_is_read_into_its_columns_and_what_follows_from_them() -> None:
    """Read the five columns and work out reference, instants and epoch."""
    measurement = reader.parse_line("60010.004873     12345   21 3B07 clkb\n")
    assert measurement.measurement_mjd == 60010.004873
    assert measurement.measured_phase == 12345
    assert measurement.rms == 21
    assert measurement.switch == "3B07"
    assert measurement.clock == "clkb"
    assert measurement.reference == "mc3"
    assert measurement.measurement_datetime == mjd_to_datetime(60010.004873)
    assert measurement.interpolated_datetime == datetime(2023, 3, 7, tzinfo=UTC)
    assert measurement.interpolated_mjd == 60010.0


@pytest.mark.parametrize(
    ("text", "count"),
    [("", 0), ("60010.000694 1 1 1A01", 4), ("60010.000694 1 1 1A01 clka x", 6)],
)
def test_a_line_without_five_columns_is_malformed(text: str, count: int) -> None:
    """Refuse a line of any other number of columns, saying how many."""
    with pytest.raises(MalformedLineError, match=f"found {count}$"):
        reader.parse_line(text)


@pytest.mark.parametrize(
    ("column", "text"),
    [
        (0, "60010"),
        (0, "6.0010e4"),
        (0, "+60010.000694"),
        (0, "60_010.000694"),
        (0, "inf"),
        (0, "nan"),
        (0, ".5"),
        (1, "+5"),
        (1, "-5"),
        (1, "1_000"),
        (1, "1000.0"),
        (1, "1e3"),
        (1, "\uff11\uff12"),
        (2, "+5"),
        (2, "2_0"),
        (2, "20.0"),
        (2, "\u0662\u0660"),
    ],
)
def test_a_number_not_in_its_plain_form_is_malformed(column: int, text: str) -> None:
    """Refuse a sign, exponent, underscore, point or non-ASCII digit."""
    fields = ["60010.000694", "1000", "20", "1A01", "clka"]
    fields[column] = text
    with pytest.raises(MalformedLineError, match="is not a plain number"):
        reader.parse_line(" ".join(fields))


@pytest.mark.parametrize(
    "text",
    [
        f"60010.000694 {PHASE_MAX + 1} 20 1A01 clka",
        "49999.999999 1000 20 1A01 clka",
        "100000.000000 1000 20 1A01 clka",
        "60010.000694 1000 20 1a01 clka",
        "60010.000694 1000 20 A101 clka",
        "60010.000694 1000 20 1A1 clka",
        "60010.000694 1000 20 1A001 clka",
    ],
)
def test_a_column_outside_its_range_or_spelling_is_malformed(text: str) -> None:
    """Refuse a phase past one period, a day out of range, or a bad switch."""
    with pytest.raises(MalformedLineError, match=r"."):
        reader.parse_line(text)


def test_the_largest_phase_and_both_ends_of_the_days_are_read() -> None:
    """Accept the edge values each column's range allows."""
    assert reader.parse_line(f"60010.000694 {PHASE_MAX} 0 1A01 c").rms == 0
    assert reader.parse_line("50000.000000 0 20 0A00 c").reference == "mc0"
    assert reader.parse_line("99999.999999 0 20 9Z99 c").reference == "mc9"


def test_a_line_that_is_not_utf8_is_malformed() -> None:
    """Refuse a line holding a byte that was not UTF-8, kept as a surrogate."""
    text = b"60010.000694 1000 20 1A01 clk\xff".decode("utf-8", "surrogateescape")
    with pytest.raises(MalformedLineError, match="not UTF-8 text"):
        reader.parse_line(text)


# ---------------------------------------------------------- the two records


@pytest.mark.parametrize("name", reader.DASMeasurement._DERIVED)
@pytest.mark.parametrize("valid", [True, False])
def test_a_worked_out_value_may_not_be_passed_in(name: str, valid: bool) -> None:
    """Refuse any of the four worked-out values, whether it looks valid or not."""
    measurement = make(60)
    value = getattr(measurement, name) if valid else "zz"
    columns = measurement.model_dump(exclude=set(reader.DASMeasurement._DERIVED))
    with pytest.raises(ValidationError, match=f"may not be passed in: {name} "):
        reader.DASMeasurement.model_validate({**columns, name: value})


def test_every_worked_out_value_passed_in_is_named() -> None:
    """Name each worked-out value passed in, not only the first."""
    measurement = make(60)
    expected = ", ".join(reader.DASMeasurement._DERIVED)
    with pytest.raises(ValidationError, match=f"may not be passed in: {expected} "):
        reader.DASMeasurement.model_validate(measurement.model_dump())


def test_the_derived_names_are_every_field_not_a_column() -> None:
    """List as worked out exactly the fields that are not the five columns."""
    columns = {"measurement_mjd", "measured_phase", "rms", "switch", "clock"}
    fields = set(reader.DASMeasurement.model_fields)
    assert set(reader.DASMeasurement._DERIVED) == fields - columns
    assert set(reader.DASData._DERIVED) == set(reader.DASData.model_fields) - {
        "interpolated_datetime",
        "measurements",
    }


@pytest.mark.parametrize("model", [reader.DASMeasurement, reader.DASData])
def test_a_record_refuses_unknown_fields_and_is_frozen(
    model: type[reader.DASMeasurement | reader.DASData],
) -> None:
    """Refuse a field the model does not declare, and any change after."""
    measurement = make(60)
    record = (
        measurement
        if model is reader.DASMeasurement
        else reader.DASData(
            interpolated_datetime=measurement.interpolated_datetime,
            measurements=(measurement,),
        )
    )
    given_fields = record.model_dump(exclude=set(model._DERIVED))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({**given_fields, "other": 1})
    with pytest.raises(ValidationError, match="frozen"):
        record.interpolated_mjd = 1.0  # type: ignore[misc]


def test_input_that_is_not_a_mapping_is_left_for_pydantic_to_refuse() -> None:
    """Let input that cannot name a field through to pydantic's own check."""
    with pytest.raises(ValidationError, match="model_type"):
        reader.DASMeasurement.model_validate("60010.000694 1000 20 1A01 clka")
    with pytest.raises(ValidationError, match="model_type"):
        reader.DASData.model_validate(["not", "a", "mapping"])


def test_a_block_works_out_its_epoch_as_an_mjd() -> None:
    """Give a block the MJD of its mark."""
    measurement = make(60)
    block = reader.DASData(
        interpolated_datetime=measurement.interpolated_datetime,
        measurements=(measurement,),
    )
    assert block.interpolated_mjd == DAY


def test_a_block_may_not_be_passed_its_mjd() -> None:
    """Refuse a block's worked-out MJD, even the right one."""
    measurement = make(60)
    with pytest.raises(
        ValidationError, match="may not be passed in: interpolated_mjd "
    ):
        reader.DASData(
            interpolated_datetime=measurement.interpolated_datetime,
            measurements=(measurement,),
            interpolated_mjd=float(DAY),
        )


def test_a_block_is_never_empty() -> None:
    """Refuse a block with no measurements."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        reader.DASData(
            interpolated_datetime=datetime(2023, 3, 7, tzinfo=UTC), measurements=()
        )


def test_a_block_refuses_a_measurement_of_another_epoch() -> None:
    """Refuse a measurement whose epoch is not the block's, naming both."""
    first, second = make(60), make(660)
    with pytest.raises(ValidationError, match="does not belong in the block"):
        reader.DASData(
            interpolated_datetime=first.interpolated_datetime,
            measurements=(first, second),
        )


def test_a_measurement_is_written_back_in_the_das_layout() -> None:
    """Write each column to its width, with a separator even past it."""
    measurement = reader.DASMeasurement(
        measurement_mjd=60010.000694,
        measured_phase=7,
        rms=123_456,
        switch="2C11",
        clock="clkd",
    )
    assert str(measurement) == "60010.000694         7 123456 2C11 clkd"
    assert len(str(measurement).split()) == reader._FIELD_COUNT


@given(
    st.integers(min_value=reader.FIRST_DAY, max_value=reader.LAST_DAY),
    st.integers(min_value=0, max_value=999_999),
    st.integers(min_value=0, max_value=PHASE_MAX),
    st.integers(min_value=0, max_value=10**7),
    st.from_regex(r"[0-9][A-Z][0-9]{2}", fullmatch=True),
    st.from_regex(r"[a-z0-9]{1,12}", fullmatch=True),
)
def test_a_line_written_back_reads_as_the_same_measurement(
    day: int, micro_day: int, phase: int, rms: int, switch: str, clock: str
) -> None:
    """Read a measurement's own line back as an equal measurement."""
    measurement = reader.DASMeasurement(
        measurement_mjd=round(day + micro_day / 1_000_000, reader.MJD_DECIMALS),
        measured_phase=phase,
        rms=rms,
        switch=switch,
        clock=clock,
    )
    assert reader.parse_line(str(measurement)) == measurement


# ------------------------------------------------------------- file names


@given(st.integers(min_value=reader.FIRST_DAY, max_value=reader.LAST_DAY))
def test_a_file_name_made_from_a_day_reads_back_as_that_day(day: int) -> None:
    """Match the pattern with every name made, and read the same day back."""
    name = reader.data_file_name(day)
    assert reader.DATA_FILE_PATTERN.fullmatch(name) is not None
    assert reader._file_mjd(Path(name)) == day


@pytest.mark.parametrize(
    "day", [reader.FIRST_DAY - 1, reader.LAST_DAY + 1, 0, -1, 123_456]
)
def test_no_file_name_is_made_for_a_day_out_of_range(day: int) -> None:
    """Refuse a day outside 50000 to 99999."""
    with pytest.raises(ValueError, match=f"MJD day {day} is outside"):
        reader.data_file_name(day)


@pytest.mark.parametrize(
    "name",
    [
        "cd5m5m_49999.dat",
        "cd5m5m_01234.dat",
        "cd5m5m_100000.dat",
        "cd5m5m_6001.dat",
        "cd5m5m_\u0666\u0660\u0660\u0661\u0660.dat",
        "cd5m5m_60010.dat.bak",
        "xcd5m5m_60010.dat",
        "cd5m5m_60010xdat",
    ],
)
def test_only_names_of_days_in_range_are_daily_file_names(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Match no other name, and refuse to read a day from one, logging why."""
    assert reader.DATA_FILE_PATTERN.fullmatch(name) is None
    with pytest.raises(DataFileError, match="not a daily data file name"):
        reader._file_mjd(Path(name))
    assert [record.levelname for record in caplog.records] == ["ERROR"]


# --------------------------------------------------------- read_measurements


def test_a_clean_file_is_read_in_file_order(tmp_path: Path) -> None:
    """Yield every line, repeated instants included, in the order written."""
    path = data_file(
        tmp_path,
        [
            line(mjd_at(60), clock="clka"),
            line(mjd_at(60), clock="clkb"),
            line(mjd_at(700), clock="clka"),
        ],
    )
    clocks = [m.clock for m in reader.read_measurements(path)]
    assert clocks == ["clka", "clkb", "clka"]


def test_an_empty_file_yields_nothing(tmp_path: Path) -> None:
    """Yield nothing from a file with no lines, without complaint."""
    assert list(reader.read_measurements(data_file(tmp_path, []))) == []


@pytest.mark.parametrize(
    ("bad", "kind"),
    [
        ("60010.000694 1000 20 1A01\n", "malformed"),
        (line(mjd_at(120, DAY + 1)), "wrong-day"),
        (line(mjd_at(595)), "late"),
        (line(mjd_at(30)), "out-of-order"),
        (line(mjd_at(90)), "duplicate"),
    ],
)
def test_a_refused_line_is_logged_under_its_reason_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad: str, kind: str
) -> None:
    """Skip the line, logging its reason, number, file and text."""
    path = data_file(
        tmp_path,
        [line(mjd_at(60)), bad, line(mjd_at(120), clock="clkz")],
    )
    clocks = [m.clock for m in reader.read_measurements(path)]
    assert clocks == ["clka", "clkz"]
    messages = skipped(caplog)
    assert len(messages) == 1
    assert messages[0].startswith(
        f"skipping {kind} line 2 of {path}: {bad.rstrip()!r} ("
    )


def test_a_line_that_is_not_utf8_is_skipped_and_reading_goes_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Skip a line with a byte that is not UTF-8, and read the lines after."""
    path = tmp_path / f"cd5m5m_{DAY}.dat"
    path.write_bytes(
        line(mjd_at(60)).encode()
        + b"60010.000700 1000 20 1A01 clk\xff\n"
        + line(mjd_at(120), clock="clkz").encode()
    )
    assert [m.clock for m in reader.read_measurements(path)] == ["clka", "clkz"]
    assert skipped(caplog) == [
        f"skipping malformed line 2 of {path}: "
        "'60010.000700 1000 20 1A01 clk\\udcff' (not UTF-8 text)"
    ]


def test_a_skipped_line_is_not_remembered(tmp_path: Path) -> None:
    """Judge the next line against the last line accepted, not the skipped one."""
    path = data_file(
        tmp_path,
        [
            line(mjd_at(60)),
            line(mjd_at(120, DAY + 1), clock="clkb"),
            line(mjd_at(90), clock="clkb"),
            line(mjd_at(595), clock="clkc"),
            line(mjd_at(100), clock="clkc"),
        ],
    )
    clocks = [m.clock for m in reader.read_measurements(path)]
    assert clocks == ["clka", "clkb", "clkc"]


def test_a_pair_may_be_measured_again_in_the_next_epoch(tmp_path: Path) -> None:
    """Refuse a repeated pair only within one epoch."""
    path = data_file(tmp_path, [line(mjd_at(60)), line(mjd_at(660))])
    assert len(list(reader.read_measurements(path))) == 2


def test_the_same_clock_against_another_reference_is_not_a_repeat(
    tmp_path: Path,
) -> None:
    """Treat a pair as reference and clock together."""
    path = data_file(tmp_path, [line(mjd_at(60)), line(mjd_at(60), switch="2A01")])
    assert len(list(reader.read_measurements(path))) == 2


# EPOCH_EDGE before the end of DAY's first epoch: an instant an MJD holds
# exactly, so the test is of the bound and not of rounding around it.
EDGE: Final = mjd_to_datetime(DAY) + EPOCH_LENGTH - reader.EPOCH_EDGE


def at_instant(instant: datetime) -> reader.DASMeasurement:
    """Build a measurement taken at ``instant``."""
    return reader.DASMeasurement(
        measurement_mjd=datetime_to_mjd(instant),
        measured_phase=1000,
        rms=20,
        switch="1A01",
        clock="clka",
    )


def test_a_measurement_exactly_at_the_epoch_edge_is_late() -> None:
    """Refuse a measurement exactly EPOCH_EDGE before the next mark."""
    measurement = at_instant(EDGE)
    assert measurement.measurement_datetime == EDGE
    with pytest.raises(LateLineError, match="before the next epoch"):
        reader._check_early(measurement)


def test_a_measurement_just_before_the_epoch_edge_is_kept() -> None:
    """Accept a measurement a little more than EPOCH_EDGE before the mark."""
    measurement = at_instant(EDGE - timedelta(milliseconds=20))
    assert measurement.measurement_datetime < EDGE
    reader._check_early(measurement)


def test_a_last_line_with_no_newline_ends_the_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse the file at its unended last line, after yielding what precedes it."""
    path = data_file(tmp_path, [line(mjd_at(60)), line(mjd_at(120)).rstrip("\n")])
    measurements = reader.read_measurements(path)
    assert next(measurements).clock == "clka"
    with pytest.raises(DataFileError, match=r"line 2 has no newline$"):
        next(measurements)
    assert [record.levelname for record in caplog.records] == ["ERROR"]


def test_a_missing_file_fails_on_first_iteration_not_at_call(tmp_path: Path) -> None:
    """Raise DataFileError when iteration starts, not when called."""
    measurements = reader.read_measurements(tmp_path / f"cd5m5m_{DAY}.dat")
    with pytest.raises(DataFileError, match="cannot read data file"):
        next(measurements)


def test_a_directory_named_as_a_data_file_cannot_be_read(tmp_path: Path) -> None:
    """Raise DataFileError for an entry that is not a file."""
    path = tmp_path / f"cd5m5m_{DAY}.dat"
    path.mkdir()
    with pytest.raises(DataFileError, match="cannot read data file"):
        list(reader.read_measurements(path))


# -------------------------------------------------------------- the blocks


def test_blocks_group_consecutive_measurements_by_epoch() -> None:
    """Give one block per run of one epoch, measurements in stream order."""
    stream = [make(60, "a"), make(61, "b"), make(660, "c"), make(1300, "d")]
    blocks = list(reader.iter_blocks(stream))
    assert [[m.clock for m in b.measurements] for b in blocks] == [
        ["a", "b"],
        ["c"],
        ["d"],
    ]
    assert [b.interpolated_datetime for b in blocks] == [
        stream[0].interpolated_datetime,
        stream[2].interpolated_datetime,
        stream[3].interpolated_datetime,
    ]


def test_a_stream_that_goes_back_gives_an_epoch_twice() -> None:
    """Neither reorder nor merge: a return to an epoch is a new block."""
    stream = [make(60, "a"), make(660, "b"), make(61, "c")]
    blocks = list(reader.iter_blocks(stream))
    assert [len(b.measurements) for b in blocks] == [1, 1, 1]
    assert blocks[0].interpolated_datetime == blocks[2].interpolated_datetime


def test_no_measurements_give_no_blocks() -> None:
    """Yield nothing from an empty stream."""
    assert list(reader.iter_blocks([])) == []


def test_reading_a_file_as_blocks_is_grouping_its_measurements(
    tmp_path: Path,
) -> None:
    """Give the same blocks as grouping what the file yields."""
    path = data_file(
        tmp_path, [line(mjd_at(60)), line(mjd_at(61), clock="b"), line(mjd_at(700))]
    )
    assert list(reader.read_blocks(path)) == list(
        reader.iter_blocks(reader.read_measurements(path))
    )


# ------------------------------------------------------------- directories


def test_data_files_are_found_in_day_order(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep files and links to files named for a day, ignore the rest."""
    caplog.set_level(logging.DEBUG)
    later = data_file(tmp_path, [], DAY + 1)
    earlier = data_file(tmp_path, [], DAY)
    (tmp_path / f"cd5m5m_{DAY + 2}.dat").mkdir()
    (tmp_path / "cd5m5m_49999.dat").write_text("")
    (tmp_path / "notes.txt").write_text("")
    link = tmp_path / f"cd5m5m_{DAY + 3}.dat"
    link.symlink_to(earlier)
    assert reader.find_data_files(tmp_path) == (earlier, later, link)
    ignored = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(ignored) == 3
    assert sum("not a file" in message for message in ignored) == 1


def test_a_directory_with_no_data_files_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Return nothing, with a WARNING naming the directory."""
    assert reader.find_data_files(tmp_path) == ()
    assert skipped(caplog) == [f"no cd5m5m data files found in {tmp_path}"]


def test_a_directory_that_cannot_be_listed_fails_at_call(tmp_path: Path) -> None:
    """Raise DataFileError when called, before any iteration."""
    with pytest.raises(DataFileError, match="cannot list the DAS directory"):
        reader.read_all_blocks(tmp_path / "missing")


def three_days(directory: Path) -> None:
    """Write files for three days, two epochs each, one line per epoch."""
    for day in (DAY, DAY + 1, DAY + 2):
        data_file(
            directory,
            [
                line(mjd_at(60, day), clock=f"{day}a"),
                line(mjd_at(4000, day), clock=f"{day}b"),
            ],
            day,
        )


def clocks_of(blocks: Iterator[reader.DASData]) -> list[str]:
    """Return the clock of every measurement in ``blocks``, in order."""
    return [m.clock for block in blocks for m in block.measurements]


def test_every_file_is_read_as_one_stream(tmp_path: Path) -> None:
    """Yield the blocks of every file, in day order."""
    three_days(tmp_path)
    assert clocks_of(reader.read_all_blocks(tmp_path)) == [
        f"{day}{part}" for day in (DAY, DAY + 1, DAY + 2) for part in "ab"
    ]


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        (exact_mjd(0, DAY + 1), [f"{DAY + 1}a", f"{DAY + 1}b", f"{DAY + 2}a"]),
        (exact_mjd(599, DAY + 1), [f"{DAY + 1}a", f"{DAY + 1}b", f"{DAY + 2}a"]),
        (exact_mjd(600, DAY + 1), [f"{DAY + 1}b", f"{DAY + 2}a", f"{DAY + 2}b"]),
        (exact_mjd(86_399, DAY + 2), []),
        (float(DAY - 5), [f"{DAY}a", f"{DAY}b", f"{DAY + 1}a"]),
    ],
)
def test_reading_starts_at_the_epoch_holding_the_start(
    tmp_path: Path, start: float, expected: list[str]
) -> None:
    """Begin at the first block whose epoch is at or after the start's epoch."""
    three_days(tmp_path)
    assert clocks_of(reader.read_all_blocks(tmp_path, start))[:3] == expected


def test_files_before_the_start_are_not_read(tmp_path: Path) -> None:
    """Skip a day that ends before the start without reading it."""
    three_days(tmp_path)
    data_file(tmp_path, ["a line with no newline"], DAY - 1)
    with pytest.raises(DataFileError, match="has no newline"):
        list(reader.read_all_blocks(tmp_path))
    assert clocks_of(reader.read_all_blocks(tmp_path, float(DAY)))[0] == f"{DAY}a"


# ------------------------------------------------- what a person reads, exactly


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (
            f"60010.000694 {PHASE_MAX + 1} 20 1A01 clka\n",
            f"measured_phase: Input should be less than or equal to {PHASE_MAX}",
        ),
        (
            line(mjd_at(120, DAY + 1)),
            f"MJD {mjd_at(120, DAY + 1)} is not in day {DAY}",
        ),
        (
            line(mjd_at(30)),
            f"MJD {mjd_at(30)} is earlier than the preceding {mjd_at(60)}",
        ),
        (line(mjd_at(90)), "mc1-clka was already measured in this epoch"),
    ],
)
def test_a_refused_line_gives_its_reason_in_words(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad: str, reason: str
) -> None:
    """End the WARNING for a skipped line with the reason, word for word."""
    path = data_file(tmp_path, [line(mjd_at(60)), bad])
    list(reader.read_measurements(path))
    (message,) = skipped(caplog)
    assert message.endswith(f": {bad.rstrip(chr(10))!r} ({reason})")


def test_a_refused_line_is_shown_with_its_spaces(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Show a skipped line as it is, white space kept, its newline dropped."""
    bad = line(mjd_at(90)).replace("\n", "  \n")
    path = data_file(tmp_path, [line(mjd_at(60)), bad])
    list(reader.read_measurements(path))
    (message,) = skipped(caplog)
    assert f"{bad[:-1]!r}" in message


def test_every_reader_error_is_logged_as_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log each DataFileError at ERROR in the words it is raised with."""
    unended = data_file(tmp_path, [line(mjd_at(60)).rstrip("\n")])
    calls: list[Callable[[], object]] = [
        lambda: list(reader.read_measurements(unended)),
        lambda: list(reader.read_measurements(tmp_path / f"cd5m5m_{DAY + 1}.dat")),
        lambda: reader._file_mjd(Path("notes.txt")),
        lambda: reader.find_data_files(tmp_path / "missing"),
    ]
    for call in calls:
        caplog.clear()
        with pytest.raises(DataFileError) as raised:
            call()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == [str(raised.value)]
    assert str(raised.value).startswith("cannot list the DAS directory")
    caplog.clear()
    with pytest.raises(DataFileError) as raised:
        list(reader.read_measurements(unended))
    assert str(raised.value) == (
        f"malformed data file {unended}: line 1 has no newline"
    )


def test_an_ignored_entry_is_named_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say at DEBUG which entry is passed over, and why, word for word."""
    caplog.set_level(logging.DEBUG)
    notes = tmp_path / "notes.txt"
    notes.write_text("")
    folder = tmp_path / f"cd5m5m_{DAY}.dat"
    folder.mkdir()
    reader.find_data_files(tmp_path)
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug == [
        f"ignoring {folder}, named as a daily data file but not a file",
        f"ignoring {notes}, which is not named as a daily data file",
    ]


def test_a_refused_line_ending_in_any_letter_is_shown_whole(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Drop only the newline from a skipped line, whatever its last letter."""
    bad = line(mjd_at(90), clock="clkX")
    path = data_file(tmp_path, [line(mjd_at(60), clock="clkX"), bad])
    list(reader.read_measurements(path))
    (message,) = skipped(caplog)
    assert f"{bad[:-1]!r}" in message
