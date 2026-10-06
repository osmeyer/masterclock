"""Tests for scripts/document_content.py.

The rules covered: every table, figure, example and chart the documents take
from the code is built from the code itself; every setting and option of the
command line appears in the settings table with its INI entry, whether it is
required and whether it takes None; the help text does not depend on the
terminal it is made in; each output file's columns, widths and header lines
are the program's own; every flag, refusal reason, error class and module is
listed, with what its docstring says; a constant's meaning is read from the
docstring below it, whether it is assigned with an annotation or without;
the gains match the estimator's; the worked epoch is the program's own
arithmetic, its rows formatted by the program; the settling chart and its
figures come from running the program's estimator; and every figure is the
value of a constant, written as the documents write it.
"""

import inspect
import re
from datetime import timedelta

import pytest

import document_content
from masterclock.app.exceptions import MasterClockError
from masterclock.das_processor import config as das_config
from masterclock.das_processor.exceptions import RefusedLineError
from masterclock.das_processor.files import (
    DDIFF_COLUMNS,
    DDIFF_HEADER_LINES,
    DDIFF_WIDTH,
    HEADER_LINES,
    HEADER_SIZES,
    MEAS_COLUMNS,
    MEAS_HEADER_LINES,
    MEAS_WIDTH,
    Column,
    header,
)
from masterclock.domain.filter import gains
from masterclock.domain.phase import EPOCH_SECONDS
from masterclock.domain.series import FLAG_ORDER, FilterStates


def table_rows(markdown_table: str) -> list[list[str]]:
    """Split a Markdown table into its body rows, each cell stripped."""
    table_lines = markdown_table.splitlines()
    assert table_lines[1].startswith("| ---")
    return [
        [cell.strip() for cell in re.split(r"(?<!\\)\|", table_line)[1:-1]]
        for table_line in table_lines[2:]
    ]


@pytest.mark.parametrize("block_name", sorted(document_content.BLOCKS))
def test_every_block_gives_text(block_name: str) -> None:
    """Give some text for every block name, ending without a newline."""
    block_text = document_content.BLOCKS[block_name]()
    assert block_text
    assert not block_text.endswith("\n")


@pytest.mark.parametrize("figure_name", sorted(document_content.FIGURES))
def test_every_figure_gives_one_line(figure_name: str) -> None:
    """Give a figure as text on one line."""
    figure_text = document_content.FIGURES[figure_name]()
    assert figure_text
    assert "\n" not in figure_text


def test_the_settings_table_holds_every_setting_and_option() -> None:
    """List every option once, its INI entry beside it when it has one."""
    rows = table_rows(document_content.settings_table())
    flags = [row[0] for row in rows]
    for setting in das_config.SETTINGS:
        row = rows[flags.index(f"`{setting.cli_flag}`")]
        assert row[1] == f"`[{setting.ini_section}] {setting.ini_entry}`"
        assert row[2] == ("yes" if setting.required else "no")
        assert row[3] == ("yes" if setting.allow_none else "no")
        assert row[4]
    for cli_only in ("--config-file", "--steps", "--redo-from-mjd"):
        assert rows[flags.index(f"`{cli_only}`")][1:4] == [
            "command line only",
            "no",
            "no",
        ]
    assert "`--help`" not in flags
    assert "`--version`" not in flags
    assert len(flags) == len(set(flags))


def test_the_help_text_is_the_same_whatever_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the same help whatever width the terminal says it has."""
    monkeypatch.setenv("COLUMNS", "40")
    narrow = document_content.help_text()
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert document_content.help_text() == narrow
    assert narrow.startswith("```text\nusage: das_processor")
    assert narrow.endswith("\n```")
    assert "\x1b" not in narrow


@pytest.mark.parametrize(
    ("file_kind", "columns"), [("meas", MEAS_COLUMNS), ("ddiff", DDIFF_COLUMNS)]
)
def test_a_column_table_lists_the_file_s_columns(
    file_kind: str, columns: tuple[Column, ...]
) -> None:
    """Give each column its number, name, width and meaning, in file order."""
    rows = table_rows(document_content.column_table(columns))
    assert rows == [
        [str(index), f"`{column.name}`", str(column.width), column.meaning]
        for index, column in enumerate(columns, start=1)
    ]
    assert document_content.BLOCKS[f"{file_kind}-columns"]() == (
        document_content.column_table(columns)
    )


def test_the_row_widths_are_the_program_s() -> None:
    """Give each kind of file its row width, header lines and header size."""
    assert table_rows(document_content.row_widths()) == [
        [
            "Measurement file",
            str(MEAS_WIDTH),
            str(HEADER_LINES["meas"]),
            str(HEADER_SIZES["meas"]),
        ],
        [
            "Double-difference file",
            str(DDIFF_WIDTH),
            str(HEADER_LINES["ddiff"]),
            str(HEADER_SIZES["ddiff"]),
        ],
    ]


def test_the_flags_are_every_flag_in_order() -> None:
    """List every flag letter in the order rows write them, with its word."""
    rows = table_rows(document_content.flag_table())
    assert "".join(row[0].strip("`") for row in rows) == FLAG_ORDER
    assert all(row[1] for row in rows)


def test_flags_out_of_step_with_their_order_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a flags column whose letters are not the row's flag order."""
    monkeypatch.setattr(document_content, "FLAG_ORDER", "RA")
    with pytest.raises(ValueError, match="flag order"):
        document_content.flag_table()


def test_the_refused_lines_are_every_reason() -> None:
    """List every reason a DAS line is refused, under the word the log uses."""
    rows = table_rows(document_content.refused_lines())
    assert [row[0] for row in rows] == [
        f"`{reason.refusal_kind}`" for reason in RefusedLineError.__subclasses__()
    ]


def test_the_errors_are_every_error_class() -> None:
    """List every error class of the project, with its package and base."""
    rows = table_rows(document_content.error_table())
    named = {row[0]: row for row in rows}
    assert named["`MasterClockError`"][1:3] == ["app", "`Exception`"]
    assert named["`DataFileError`"][1:3] == ["das_processor", "`MasterClockError`"]
    assert named["`PhaseError`"][1] == "domain"
    assert named["`MissingSettingsError`"][2] == "`ConfigError`"
    assert all(row[3] for row in rows)


def test_every_error_listed_is_a_masterclock_error() -> None:
    """List only the project's own errors, each once."""
    rows = table_rows(document_content.error_table())
    assert len(rows) == len({row[0] for row in rows})
    for package_name, module in document_content.ERROR_MODULES:
        for class_name, error_class in vars(module).items():
            if (
                inspect.isclass(error_class)
                and issubclass(error_class, MasterClockError)
                and error_class.__module__ == module.__name__
            ):
                assert [f"`{class_name}`", package_name] in [row[:2] for row in rows]


def test_the_modules_are_every_module_of_every_package() -> None:
    """List each package and its modules, each with its docstring's first line."""
    rows = table_rows(document_content.module_table())
    listed = {(row[0], row[1]) for row in rows}
    assert ("das_processor", "(the package)") in listed
    assert ("das_processor", "`run`") in listed
    assert ("domain", "`filter`") in listed
    assert ("app", "`lock`") in listed
    assert all(row[2] for row in rows)


def test_a_constant_s_meaning_is_the_docstring_below_it() -> None:
    """Read the first paragraph of the docstring below an assignment."""
    module_source = (
        "A: int = 1\n"
        '"""The first.\n\nMore about it."""\n'
        "B = 2\n"
        '"""The second,\n    on two lines."""\n'
        "C = 3\n"
        "D = 4\n"
    )
    assert document_content.attribute_docstring(module_source, "A") == "The first."
    assert (
        document_content.attribute_docstring(module_source, "B")
        == "The second, on two lines."
    )
    with pytest.raises(LookupError, match="C"):
        document_content.attribute_docstring(module_source, "C")
    with pytest.raises(LookupError, match="E"):
        document_content.attribute_docstring(module_source, "E")


def test_a_constant_followed_by_other_code_has_no_docstring() -> None:
    """Refuse an assignment followed by a statement that is not a docstring."""
    with pytest.raises(LookupError, match="A"):
        document_content.attribute_docstring("A = 1\nprint(A)\n", "A")


def test_the_constants_table_gives_every_constant() -> None:
    """Give every constant its module, its value and its meaning."""
    rows = table_rows(document_content.constants_table())
    assert len(rows) == len(document_content.CONSTANTS)
    named = {row[0]: row for row in rows}
    assert named["`EPOCH_SECONDS`"][1] == "domain.phase"
    assert named["`EPOCH_SECONDS`"][2] == str(EPOCH_SECONDS)
    assert named["`EPOCH_SECONDS`"][3].startswith("How long one epoch lasts")


@pytest.mark.parametrize(
    ("value_form", "constant_value", "written"),
    [
        ("number", 600, "600"),
        ("number", 5.0, "5"),
        ("number", 59_500.0, "59500"),
        ("grouped", 200_000, "200 000"),
        ("seconds", timedelta(seconds=10), "10 s"),
        ("code", "das_processor_{rf}.lock", "`das_processor_<rf>.lock`"),
        ("names", frozenset({"mc9", "mc8"}), "`mc8`, `mc9`"),
    ],
)
def test_a_value_is_written_in_its_form(
    value_form: str, constant_value: object, written: str
) -> None:
    """Write each kind of constant as the documents write it."""
    assert document_content.written_value(value_form, constant_value) == written


def test_an_unknown_value_form_is_refused() -> None:
    """Refuse a form no constant is written in."""
    with pytest.raises(ValueError, match="no form"):
        document_content.written_value("roman", 5)


def test_a_figure_is_its_constant_s_written_value() -> None:
    """Give each figure as its constant's value, written in its form."""
    assert document_content.FIGURES["EPOCH_SECONDS"]() == str(EPOCH_SECONDS)
    assert document_content.FIGURES["PHASE_PERIOD"]() == "200 000"
    assert document_content.FIGURES["SKIPPED_REFERENCES"]() == "`mc9`"


def test_the_gains_are_the_estimator_s() -> None:
    """Give g, h and k for each time constant as the estimator works them out."""
    rows = table_rows(document_content.gain_table())
    first = rows[0]
    time_constant = float(first[0])
    g3, h3_over_t, two_k3_over_t2 = gains(3, time_constant)
    g2, h2_over_t, _ = gains(2, time_constant)
    seconds = EPOCH_SECONDS
    assert [float(cell) for cell in first[2:]] == pytest.approx(
        [
            g3,
            h3_over_t * seconds,
            two_k3_over_t2 * seconds**2 / 2,
            g2,
            h2_over_t * seconds,
        ],
        rel=1e-5,
    )


@pytest.mark.parametrize(
    ("file_kind", "header_lines"),
    [("meas", MEAS_HEADER_LINES), ("ddiff", DDIFF_HEADER_LINES)],
)
def test_an_example_header_is_the_program_s(file_kind: str, header_lines: int) -> None:
    """Give a kind of file's header exactly as the program writes it."""
    header_block = document_content.BLOCKS[f"{file_kind}-header"]()
    block_lines = header_block.splitlines()
    assert block_lines[0] == "```text"
    assert block_lines[-1] == "```"
    assert len(block_lines) - 2 == header_lines
    assert block_lines[1:-1] == header(file_kind).splitlines()  # type: ignore[arg-type]
    assert block_lines[2].startswith("# WARNING")


def test_the_worked_epoch_decycles_and_accepts() -> None:
    """Work the example measurement through decycling to an accepted row."""
    worked = document_content.worked_epoch()
    assert worked.measurement.cycle_count == 6
    assert worked.measurement.z == 1_234_577
    assert worked.pair_row.flags == "A"
    assert worked.next_row.flags == "P"
    steps = table_rows(document_content.worked_steps())
    assert any("1 234 577" in row[2] for row in steps)
    assert any(row[2] == "n = 6" for row in steps)


def test_the_worked_rows_are_formatted_by_the_program() -> None:
    """Give the measurement rows and the two triple rows as the program writes them."""
    meas_lines = document_content.worked_meas_rows().splitlines()[1:-1]
    assert len(meas_lines) == 2
    assert all(len(line) == MEAS_WIDTH for line in meas_lines)
    assert meas_lines[0].endswith(" A")
    assert meas_lines[1].endswith(" P")
    ddiff_lines = document_content.worked_ddiff_rows().splitlines()[1:-1]
    assert [len(line) for line in ddiff_lines] == [DDIFF_WIDTH, DDIFF_WIDTH]
    assert re.search(r", +6666667, ", ddiff_lines[0])
    assert re.search(r", +1234577, ", ddiff_lines[1])


def test_the_local_triple_s_state_is_its_pair_s() -> None:
    """Give the local triple the same estimate as its pair, from the same row."""
    worked = document_content.worked_epoch()
    assert worked.local_row.x_fs == worked.pair_row.x_fs
    assert worked.local_row.y == worked.pair_row.y
    assert worked.triple_value.z == 6_666_667


def test_the_triple_table_gives_the_double_difference() -> None:
    """Give the remote and local double differences and their sigmas."""
    rows = table_rows(document_content.worked_triples())
    assert rows[0][1] == "6 666 667"
    assert rows[1][1] == "1 234 577"
    assert float(rows[0][2]) == pytest.approx(3.31662, rel=1e-5)


def test_the_raw_line_is_written_as_the_das_writes_it() -> None:
    """Give the example DAS line in the DAS's own layout."""
    assert document_content.worked_raw_line() == (
        "```text\n60941.251588     34579    3 2B07 hm7\n```"
    )


@pytest.mark.parametrize("filter_states", [2, 3])
def test_the_settling_chart_is_the_estimator_s(filter_states: FilterStates) -> None:
    """Draw the rate estimate after a rate step, as the estimator gives it."""
    chart = document_content.settling_chart(filter_states)
    assert chart.startswith("```mermaid\nxychart-beta\n")
    assert chart.endswith("\n```")
    values = re.search(r"line \[([^\]]*)\]", chart)
    assert values is not None
    percents = [float(value) for value in values.group(1).split(", ")]
    assert percents[0] == 0.0
    assert percents[-1] == pytest.approx(100.0, abs=1.0)
    assert percents == [
        round(percent, 1)
        for percent in document_content.settling_percents(filter_states, 2000)[:1001:50]
    ]


def test_the_settling_numbers_come_from_the_same_run() -> None:
    """Give the 3-state overshoot, none for the 2-state, and when each settles."""
    rows = table_rows(document_content.settling_numbers())
    assert [row[0] for row in rows] == ["3-state", "2-state"]
    three_state, two_state = rows
    overshoot, at_m, _, settled_from = (float(cell) for cell in three_state[1:])
    assert overshoot > 0
    assert 0 < at_m < settled_from
    assert two_state[1:3] == ["none", "-"]
    assert float(two_state[4]) > 0


def test_a_settling_run_that_never_settles_is_refused() -> None:
    """Refuse rates that never come within the bound for good."""
    with pytest.raises(ValueError, match="settle"):
        document_content.settled_from([0.0, 50.0, 150.0], 1)


def test_a_pipe_in_a_cell_is_escaped() -> None:
    """Escape a vertical bar so a cell cannot split its row."""
    assert document_content.cell("a | b") == r"a \| b"


def test_docstring_markup_is_written_as_markdown() -> None:
    """Write a role or a double-backquoted span as a Markdown code span."""
    assert (
        document_content.plain_markup("See :mod:`logging`, :func:`~a.b` and ``c``.")
        == "See `logging`, `a.b` and `c`."
    )


@pytest.mark.parametrize("filter_states", [2, 3])
def test_the_settling_run_follows_its_own_model(filter_states: FilterStates) -> None:
    """Give one epoch after the step the rate h times the step, from its model's h.

    The first innovation after a step in rate r is r T, so the rate estimate
    is (h / T) r T = h r: h percent times 100.
    """
    _, h_over_t, _ = gains(filter_states, document_content.SETTLING_TIME_CONSTANT)
    first_epoch = document_content.settling_percents(filter_states, 1)[1]
    assert first_epoch == pytest.approx(100.0 * h_over_t * EPOCH_SECONDS, rel=1e-9)


def test_the_worked_inputs_are_the_example_s() -> None:
    """Give the invented inputs the worked epoch starts from, as the code holds them."""
    rows = {row[0]: row[1] for row in table_rows(document_content.worked_inputs())}
    worked = document_content.worked_epoch()
    assert "x = 1 234 567 ps" in rows["Pair's last row, 2025-09-23 05:50:00 UTC"]
    assert (
        f"y = {worked.last_row.y} ps/s"
        in rows["Pair's last row, 2025-09-23 05:50:00 UTC"]
    )
    assert rows["Estimator"].startswith(
        "3-state, M = 100, M_\N{GREEK SMALL LETTER SIGMA} = 50"
    )
    assert "z(mc1, mc2) = 5 432 100 ps" in rows["Links"]
    assert rows["Steering"] == "none"
