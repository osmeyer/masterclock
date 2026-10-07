"""What the documents say that the code gives: tables, figures, examples and charts.

The project's documents carry tables and numbers that would go out of date
whenever the code changed: the settings, the columns of the output files,
the constants, an example worked through by the estimator. Each one is
written here from the code itself, under a name. ``scripts/documents.py``
puts each in the documents between markers, and its check fails when a
document no longer holds what the code gives.

A block is a whole table, code block or chart, given by :data:`BLOCKS`. A
figure is one value written inside a sentence, given by :data:`FIGURES`:
the value of one of the constants in :data:`CONSTANTS`, in the form the
documents write it.

The worked epoch (:func:`worked_epoch`) is a measurement of one invented pair
taken through the program's own functions: decycling, the gate, the update
and the row formats, with the two triples built on it. The settling chart
(:func:`settling_chart`) runs the program's estimator after a step in rate.
"""

import argparse
import ast
import dataclasses
import functools
import importlib
import inspect
import math
import pkgutil
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from types import ModuleType
from typing import Final, NamedTuple

from gmpy2 import mpq

import masterclock.app
import masterclock.das_processor
import masterclock.domain
from masterclock.app import exceptions as app_exceptions
from masterclock.app.exceptions import MasterClockError
from masterclock.das_processor import exceptions as das_exceptions
from masterclock.das_processor.cli import build_parser
from masterclock.das_processor.config import SETTINGS
from masterclock.das_processor.exceptions import RefusedLineError
from masterclock.das_processor.files import (
    DDIFF_COLUMNS,
    HEADER_LINES,
    HEADER_SIZES,
    MEAS_COLUMNS,
    WIDTHS,
    Column,
    DdiffRecord,
    FileKind,
    MeasRecord,
    format_ddiff_row,
    header,
    record_line,
)
from masterclock.das_processor.read_cd5m5m import DASMeasurement
from masterclock.domain import exceptions as domain_exceptions
from masterclock.domain.double_difference import (
    Component,
    TripleValue,
    double_difference,
)
from masterclock.domain.filter import (
    K_OUT,
    SETTLE_FACTOR,
    exact_gains,
    filter_step,
    gains,
    predict,
    update,
)
from masterclock.domain.measurements import (
    PairMeasurement,
    TripleMeasurement,
    measure_pair,
)
from masterclock.domain.phase import (
    EPOCH_SECONDS,
    PHASE_PERIOD,
    exact,
    from_fs,
    round_even,
    to_fs,
)
from masterclock.domain.series import (
    FLAG_ORDER,
    FilterStates,
    Row,
    SeriesParams,
    State,
)

HELP_WIDTH: Final[int] = 80
"""The width the help text is made at, whatever terminal it is made in."""

ERROR_MODULES: Final[tuple[tuple[str, ModuleType], ...]] = (
    ("app", app_exceptions),
    ("domain", domain_exceptions),
    ("das_processor", das_exceptions),
)
"""Each package with the module that holds its errors."""

PACKAGES: Final[tuple[tuple[str, ModuleType], ...]] = (
    ("app", masterclock.app),
    ("domain", masterclock.domain),
    ("das_processor", masterclock.das_processor),
)
"""Each package of the project whose modules the module table lists."""

CONSTANTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("domain.phase", "PHASE_PERIOD", "grouped"),
    ("domain.phase", "PHASE_MAX", "grouped"),
    ("domain.phase", "FS_PER_PS", "number"),
    ("domain.phase", "EPOCH_SECONDS", "number"),
    ("domain.references", "REFERENCE_PREFIX", "code"),
    ("domain.series", "FLAG_ORDER", "code"),
    ("domain.series", "MAX_REJECTS", "number"),
    ("domain.filter", "K_OUT", "number"),
    ("domain.filter", "K_STEP", "number"),
    ("domain.filter", "SETTLE_FACTOR", "number"),
    ("domain.screening", "K_SHARED", "number"),
    ("das_processor.read_cd5m5m", "FIRST_DAY", "number"),
    ("das_processor.read_cd5m5m", "LAST_DAY", "number"),
    ("das_processor.read_cd5m5m", "RMS_MAX", "number"),
    ("das_processor.read_cd5m5m", "EPOCH_EDGE", "seconds"),
    ("das_processor.read_cd5m5m", "SKIPPED_REFERENCES", "names"),
    ("das_processor.read_cd5m5m", "DATA_FILE_TEMPLATE", "code"),
    ("das_processor.read_steering", "STEERING_FILE_TEMPLATE", "code"),
    ("das_processor.cli", "START_FROM_MJD", "number"),
    ("das_processor.config", "MEAS_SUBDIRECTORY", "code"),
    ("das_processor.config", "DDIFF_SUBDIRECTORY", "code"),
    ("das_processor.config", "LOCK_FILE_TEMPLATE", "code"),
    ("das_processor.config", "JOURNAL_FILE_TEMPLATE", "code"),
    ("das_processor.clock_config", "REFERENCE_TYPE", "code"),
)
"""The constants the documents name: module under masterclock, name, and the
form its value is written in (see :func:`written_value`)."""

GAIN_TIME_CONSTANTS: Final[tuple[float, ...]] = (10.0, 30.0, 100.0, 300.0, 1000.0)
"""The time constants the gain table gives the gains of, epochs."""

SETTLING_TIME_CONSTANT: Final[float] = 100.0
"""The time constant the settling chart runs the estimator at, epochs."""

SETTLING_RATE_STEP: Final[float] = 1.0
"""The step in rate the settling chart follows, ps/s."""

SETTLING_CHART_SPAN: Final[int] = 10
"""How many time constants the settling chart covers after the step."""

SETTLING_RUN_SPAN: Final[int] = 20
"""How many time constants the settling figures are worked out over."""

SHOWN_OVERSHOOT: Final[float] = 0.05
"""The smallest overshoot, in percent of the step, the settling table shows; a
smaller one is no overshoot at the table's one decimal place."""

SETTLED_PERCENT: Final[float] = 1.0
"""How near the step, in percent of it, a rate estimate counts as settled."""

_EXAMPLE_START: Final[datetime] = datetime(2025, 9, 23, 6, 0, tzinfo=UTC)
"""The invented epoch of the worked example."""

_EXAMPLE_PARAMS: Final[SeriesParams] = SeriesParams(
    filter_states=3,
    M=100.0,
    M_sigma=50.0,
    sigma0=5.0,
    gmax=432,
    n_break=36,
    reject_fraction_weight=0.04,
    reject_fraction_limit=0.5,
    rms_max=80,
)
"""The invented settings of the worked example's clock, a 3-state maser."""

_EXAMPLE_READING: Final[DASMeasurement] = DASMeasurement(
    measurement_mjd=60941.251588,
    measured_phase=34579,
    rms=3,
    switch="2B07",
    clock="hm7",
)
"""The invented DAS reading of the worked example: hm7 against mc2."""

_EXAMPLE_LINKS: Final[tuple[Component, Component]] = (
    Component(accepted=True, z=5_432_100, rms=2),
    Component(accepted=True, z=-5_432_080, rms=2),
)
"""The invented accepted measurements of the links (mc1, mc2) and (mc2, mc1)."""

_EXAMPLE_SELF: Final[Component] = Component(accepted=True, z=12, rms=1)
"""The invented accepted measurement of the self pair (mc2, mc2)."""

_NO_STEERING: Final[tuple[mpq, float]] = (mpq(0), 0.0)
"""A steering input of nothing: no reference was steered."""


def cell(cell_text: str) -> str:
    """Make text safe in a Markdown table cell.

    Parameters
    ----------
    cell_text : str
        The text.

    Returns
    -------
    str
        The text with every vertical bar escaped, so it cannot end the cell.
    """
    return cell_text.replace("|", r"\|")


def table(column_names: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Write a Markdown table.

    Parameters
    ----------
    column_names : sequence of str
        The heading of each column.
    rows : sequence of sequence of str
        The cells of each row, already made safe.

    Returns
    -------
    str
        The table, without a final newline.
    """
    table_lines = [
        "| " + " | ".join(column_names) + " |",
        "| " + " | ".join("---" for _ in column_names) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(table_lines)


def fenced(language: str, block_lines: Sequence[str]) -> str:
    """Write lines as a fenced code block.

    Parameters
    ----------
    language : str
        The block's language, such as ``text`` or ``mermaid``.
    block_lines : sequence of str
        The lines.

    Returns
    -------
    str
        The block, without a final newline.
    """
    return "\n".join([f"```{language}", *block_lines, "```"])


def plain_markup(docstring_text: str) -> str:
    """Turn a docstring's reStructuredText code markup into Markdown's.

    Parameters
    ----------
    docstring_text : str
        Text from a docstring.

    Returns
    -------
    str
        The text with every role such as ``:mod:`logging``` and every
        double-backquoted span written as a Markdown code span, a leading
        ``~`` dropped.
    """
    without_roles = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", docstring_text)
    return without_roles.replace("``", "`")


def first_line(documented: object) -> str:
    """Give the first paragraph of an object's docstring, on one line.

    Parameters
    ----------
    documented : object
        A module, class or function with a docstring.

    Returns
    -------
    str
        Its docstring's first paragraph, made safe for a table cell.
    """
    docstring = inspect.getdoc(documented) or ""
    return cell(plain_markup(" ".join(docstring.split("\n\n")[0].split())))


# ------------------------------------------------------------- the settings


def fixed_width_formatter(prog: str) -> argparse.HelpFormatter:
    """Make a help formatter of :data:`HELP_WIDTH` characters.

    Parameters
    ----------
    prog : str
        The program's name.

    Returns
    -------
    argparse.HelpFormatter
        A formatter that ignores the terminal's width.
    """
    return argparse.HelpFormatter(prog, width=HELP_WIDTH)


def help_text() -> str:
    """Give das_processor's ``--help`` text as a code block.

    Returns
    -------
    str
        The help, made at :data:`HELP_WIDTH` characters and without colour,
        so it is the same on every terminal.
    """
    parser = build_parser()
    parser.formatter_class = fixed_width_formatter
    parser.color = False
    return fenced("text", parser.format_help().rstrip("\n").splitlines())


def settings_table() -> str:
    """Give every command-line option with the INI entry it overrides.

    Returns
    -------
    str
        One row per option, in the order the help lists them, but ``--help``
        and ``--version``: its flag, its INI entry or that it has none,
        whether some source must give it, whether it takes ``None``, and its
        help text.
    """
    settings_by_flag = {setting.cli_flag: setting for setting in SETTINGS}
    rows = []
    for action in build_parser()._actions:
        if action.dest in {"help", "version"}:
            continue
        flag = action.option_strings[0]
        setting = settings_by_flag.get(flag)
        rows.append(
            [
                f"`{flag}`",
                "command line only"
                if setting is None
                else f"`[{setting.ini_section}] {setting.ini_entry}`",
                "yes" if setting is not None and setting.required else "no",
                "yes" if setting is not None and setting.allow_none else "no",
                cell(action.help or ""),
            ]
        )
    return table(
        ["Flag", "INI file entry", "Required", "Takes `None`", "What it gives"], rows
    )


# ---------------------------------------------------------- the output files


def column_table(columns: tuple[Column, ...]) -> str:
    """Give a file's columns, in order.

    Parameters
    ----------
    columns : tuple of Column
        The file's columns.

    Returns
    -------
    str
        Each column's number, name, width in characters and meaning.
    """
    return table(
        ["#", "Column", "Width (characters)", "What it holds"],
        [
            [str(index), f"`{column.name}`", str(column.width), cell(column.meaning)]
            for index, column in enumerate(columns, start=1)
        ],
    )


_FILE_TITLES: Final[dict[FileKind, str]] = {
    "meas": "Measurement file",
    "ddiff": "Double-difference file",
}
"""What each kind of file is called in the documents."""


def row_widths() -> str:
    """Give each kind of file's row width, header lines and header size.

    Returns
    -------
    str
        A table of each kind of file's row width W, the newline not
        counted, how many header lines it has, and its header's size H.
    """
    return table(
        [
            "File",
            "Row width W (characters, newline not counted)",
            "Header lines",
            "Header size H (bytes)",
        ],
        [
            [
                title,
                str(WIDTHS[file_kind]),
                str(HEADER_LINES[file_kind]),
                str(HEADER_SIZES[file_kind]),
            ]
            for file_kind, title in _FILE_TITLES.items()
        ],
    )


def flag_table() -> str:
    """Give every flag a row can carry, with its meaning.

    Returns
    -------
    str
        Each flag letter and the word the files' header gives it, in the
        order rows write them.

    Raises
    ------
    ValueError
        If the header's flags are not the flags in the order rows write
        them.
    """
    flags_column = next(column for column in MEAS_COLUMNS if column.name == "flags")
    flag_words = [
        flag_word.split(" ", 1) for flag_word in flags_column.meaning.split(", ")
    ]
    if "".join(letter for letter, _ in flag_words) != FLAG_ORDER:
        message = f"the flags column's letters are not the flag order {FLAG_ORDER}"
        raise ValueError(message)
    return table(
        ["Flag", "Meaning"], [[f"`{letter}`", word] for letter, word in flag_words]
    )


def _header_block(file_kind: FileKind) -> str:
    """Give the header of a kind of file, as the program writes it.

    Parameters
    ----------
    file_kind : {'meas', 'ddiff'}
        The kind of file.

    Returns
    -------
    str
        The header, the same for every file of the kind.
    """
    return fenced("text", header(file_kind).splitlines())


# ------------------------------------------------------- errors and modules


def refused_lines() -> str:
    """Give every reason a DAS line is refused.

    Returns
    -------
    str
        The word the log gives each reason, and what it means.
    """
    return table(
        ["Word in the log", "Why the line is refused"],
        [
            [f"`{reason.refusal_kind}`", first_line(reason)]
            for reason in RefusedLineError.__subclasses__()
        ],
    )


def error_table() -> str:
    """Give every error class of the project.

    Returns
    -------
    str
        Each error's name, the package it belongs to, the class it is
        based on, and when it is raised, package by package in the order
        the classes are written.
    """
    rows = []
    for package_name, module in ERROR_MODULES:
        for class_name, error_class in vars(module).items():
            if (
                inspect.isclass(error_class)
                and issubclass(error_class, MasterClockError)
                and error_class.__module__ == module.__name__
            ):
                rows.append(
                    [
                        f"`{class_name}`",
                        package_name,
                        f"`{error_class.__bases__[0].__name__}`",
                        first_line(error_class),
                    ]
                )
    return table(["Error", "Package", "Based on", "When it is raised"], rows)


def module_table() -> str:
    """Give every module of the project's packages.

    Returns
    -------
    str
        Each package, then its modules in name order, each with the first
        paragraph of its docstring.
    """
    rows = []
    for package_name, package in PACKAGES:
        rows.append([package_name, "(the package)", first_line(package)])
        for module_info in sorted(
            pkgutil.iter_modules(package.__path__), key=lambda info: info.name
        ):
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            rows.append([package_name, f"`{module_info.name}`", first_line(module)])
    return table(["Package", "Module", "What it holds"], rows)


# ---------------------------------------------------------------- constants


def _assigns(statement: ast.stmt, name: str) -> bool:
    """Tell whether a statement assigns a name.

    Parameters
    ----------
    statement : ast.stmt
        A statement of a module.
    name : str
        The name.

    Returns
    -------
    bool
        Whether it is an assignment, with an annotation or without, to the
        name alone or among others.
    """
    if isinstance(statement, ast.AnnAssign):
        targets: list[ast.expr] = [statement.target]
    elif isinstance(statement, ast.Assign):
        targets = statement.targets
    else:
        return False
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def attribute_docstring(module_source: str, name: str) -> str:
    """Give the first paragraph of the docstring below a module's assignment.

    Parameters
    ----------
    module_source : str
        The module's source.
    name : str
        The name assigned at the module's top level.

    Returns
    -------
    str
        The first paragraph of the string standing alone just after the
        assignment, on one line.

    Raises
    ------
    LookupError
        If the module assigns no such name, or no string stands after it.
    """
    for statement, following in pairwise(ast.parse(module_source).body):
        if (
            _assigns(statement, name)
            and isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            paragraph = inspect.cleandoc(following.value.value).split("\n\n")[0]
            return plain_markup(" ".join(paragraph.split()))
    message = f"no docstring follows an assignment to {name}"
    raise LookupError(message)


def written_value(value_form: str, constant_value: object) -> str:
    """Write a constant's value in the form the documents give it.

    Parameters
    ----------
    value_form : str
        ``number``: a whole number or a float as it reads, with no
        decimals it does not need; ``grouped``: a whole number in groups of
        three digits; ``seconds``: a time, in seconds, with its unit; ``code``: text as
        code, any ``{field}`` written ``<field>``; ``names``: names as code,
        sorted.
    constant_value : object
        The value.

    Returns
    -------
    str
        The value, written.

    Raises
    ------
    ValueError
        If ``value_form`` is none of those.
    """
    if value_form == "number":
        return (
            f"{constant_value:g}"
            if isinstance(constant_value, float)
            else str(constant_value)
        )
    if value_form == "grouped":
        return f"{constant_value:_}".replace("_", " ")
    if value_form == "seconds" and isinstance(constant_value, timedelta):
        return f"{constant_value.total_seconds():g} s"
    if value_form == "code":
        return "`" + re.sub(r"\{(\w+)\}", r"<\1>", str(constant_value)) + "`"
    if value_form == "names" and isinstance(constant_value, frozenset):
        return ", ".join(f"`{name}`" for name in sorted(constant_value))
    message = f"no form {value_form!r} for a constant"
    raise ValueError(message)


def _constant(module_name: str, name: str) -> tuple[object, str]:
    """Give a constant's value and meaning.

    Parameters
    ----------
    module_name : str
        Its module, under ``masterclock``.
    name : str
        Its name.

    Returns
    -------
    tuple of (object, str)
        Its value and the first paragraph of its docstring.
    """
    module = importlib.import_module(f"masterclock.{module_name}")
    return getattr(module, name), attribute_docstring(inspect.getsource(module), name)


def constants_table() -> str:
    """Give every constant the documents name.

    Returns
    -------
    str
        Each constant's name, module, value and meaning.
    """
    rows = []
    for module_name, name, value_form in CONSTANTS:
        constant_value, meaning = _constant(module_name, name)
        rows.append(
            [
                f"`{name}`",
                module_name,
                cell(written_value(value_form, constant_value)),
                cell(meaning),
            ]
        )
    return table(["Constant", "Module", "Value", "Meaning"], rows)


def _figure(module_name: str, name: str, value_form: str) -> Callable[[], str]:
    """Make the function that gives a constant as a figure.

    Parameters
    ----------
    module_name : str
        Its module, under ``masterclock``.
    name : str
        Its name.
    value_form : str
        The form it is written in.

    Returns
    -------
    callable
        A function of no arguments giving the constant's written value.
    """

    def figure() -> str:
        """Give the constant's value, written."""
        return written_value(value_form, _constant(module_name, name)[0])

    return figure


# -------------------------------------------------------------------- gains


def gain_table() -> str:
    """Give the estimator's gains at a range of time constants.

    Returns
    -------
    str
        For each time constant M, the pole lambda and the gains g, h and k
        of the 3-state model and g and h of the 2-state model, as the
        estimator works them out.
    """
    rows = []
    for time_constant in GAIN_TIME_CONSTANTS:
        g3, h3_over_t, two_k3_over_t2 = gains(3, time_constant)
        g2, h2_over_t, _ = gains(2, time_constant)
        values = (
            math.exp(-1.0 / time_constant),
            g3,
            h3_over_t * EPOCH_SECONDS,
            two_k3_over_t2 * EPOCH_SECONDS**2 / 2,
            g2,
            h2_over_t * EPOCH_SECONDS,
        )
        rows.append([f"{time_constant:g}", *(f"{value:.6g}" for value in values)])
    return table(
        [
            "M (epochs)",
            LAMBDA,
            "3-state g",
            "3-state h",
            "3-state k",
            "2-state g",
            "2-state h",
        ],
        rows,
    )


# ------------------------------------------------------------- worked epoch


class WorkedEpoch(NamedTuple):
    """The worked example, as the program works it out.

    Parameters
    ----------
    last_row : Row
        The pair's row of the epoch before.
    prediction : State
        Its prediction at the epoch.
    measurement : PairMeasurement
        The reading, decycled and referred to the epoch start.
    pair_row : Row
        The pair's row at the epoch.
    next_row : Row
        Its row at the next epoch, which has no measurement.
    meas_lines : tuple of str
        The two rows as the measurement file holds them.
    triple_last_row : Row
        The remote triple's row of the epoch before.
    triple_value : TripleValue
        The remote triple's double difference.
    local_value : TripleValue
        The local triple's.
    triple_row : Row
        The remote triple's row.
    local_row : Row
        The local triple's row.
    ddiff_lines : tuple of str
        The two triples' rows as their files hold them.
    """

    last_row: Row
    prediction: State
    measurement: PairMeasurement
    pair_row: Row
    next_row: Row
    meas_lines: tuple[str, str]
    triple_last_row: Row
    triple_value: TripleValue
    local_value: TripleValue
    triple_row: Row
    local_row: Row
    ddiff_lines: tuple[str, str]


def _held_row(
    epoch_start: datetime,
    x_ps: int,
    y: float,
    innovation_scale: float,
    epochs_in_segment: int,
) -> Row:
    """Make an invented accepted row of a 3-state series with no drift.

    Parameters
    ----------
    epoch_start : datetime
        The row's epoch.
    x_ps : int
        Its phase, ps.
    y : float
        Its rate, ps/s.
    innovation_scale : float
        Its innovation scale, ps.
    epochs_in_segment : int
        Its rows since the segment started.

    Returns
    -------
    Row
        The row, settled and accepted.
    """
    return Row(
        interpolated_datetime=epoch_start,
        innovation=None,
        x_fs=to_fs(x_ps),
        y=y,
        d=0.0,
        innovation_scale=innovation_scale,
        step_offset=0,
        epochs_in_segment=epochs_in_segment,
        epochs_since_accept=0,
        consecutive_rejects=0,
        reject_fraction=0.0,
        rejects=(),
        filter_states=_EXAMPLE_PARAMS.filter_states,
        time_constant=_EXAMPLE_PARAMS.M,
        scale_time_constant=_EXAMPLE_PARAMS.M_sigma,
        flags="A",
    )


def _triple_row(
    triple: tuple[str, str, str],
    last_row: Row,
    links: tuple[Component, Component],
    pair: Component,
) -> tuple[TripleValue, Row, str]:
    """Work one triple of the example: its double difference, row and line.

    Parameters
    ----------
    triple : tuple of (str, str, str)
        The triple (r, s, c).
    last_row : Row
        Its row of the epoch before.
    links : tuple of (Component, Component)
        The parts of (r, s) and (s, r).
    pair : Component
        The part of (s, c).

    Returns
    -------
    tuple of (TripleValue, Row, str)
        Its double difference, its row and its line.
    """
    triple_value = double_difference(triple, pair, *links)
    if triple_value is None:  # pragma: no cover - every part is accepted
        message = f"the example triple {triple} has no double difference"
        raise ValueError(message)
    measurement = TripleMeasurement.from_triple_value(triple_value)
    step = filter_step(
        _EXAMPLE_START,
        _EXAMPLE_PARAMS,
        last_row,
        predict(last_row, _NO_STEERING),
        measurement.filter_input(),
    )
    line_text = format_ddiff_row(DdiffRecord(measurement=measurement, row=step.row))
    return triple_value, step.row, line_text


@functools.cache
def worked_epoch() -> WorkedEpoch:
    """Work the example epoch through the program's own functions.

    Returns
    -------
    WorkedEpoch
        The pair (mc2, hm7) measured once and then not at all, and the
        remote triple (mc1, mc2, hm7) and local triple (mc2, mc2, hm7)
        built on its first measurement. No reference is steered.
    """
    epoch_before = _EXAMPLE_START - timedelta(seconds=EPOCH_SECONDS)
    last_row = _held_row(epoch_before, 1_234_567, 0.0123, 3.0, 811)
    prediction = predict(last_row, _NO_STEERING)
    if prediction is None:  # pragma: no cover - the last row is accepted
        message = "the example's last row gives no prediction"
        raise ValueError(message)
    measurement = measure_pair(
        measurement_mjd=_EXAMPLE_READING.measurement_mjd,
        measured_phase=_EXAMPLE_READING.measured_phase,
        rms=_EXAMPLE_READING.rms,
        prediction=prediction,
        w=mpq(0),
        anchor=None,
    )
    pair_step = filter_step(
        _EXAMPLE_START,
        _EXAMPLE_PARAMS,
        last_row,
        prediction,
        measurement.filter_input(),
    )
    first_line_text, kept_row = record_line(MeasRecord(measurement, pair_step.row))
    next_start = _EXAMPLE_START + timedelta(seconds=EPOCH_SECONDS)
    next_step = filter_step(
        next_start, _EXAMPLE_PARAMS, kept_row, predict(kept_row, _NO_STEERING), None
    )
    next_line_text, _ = record_line(MeasRecord(None, next_step.row))
    pair_part = Component(
        accepted=True,
        z=measurement.z,
        rms=measurement.rms,
        predicted_phase=prediction.x,
    )
    triple_last_row = _held_row(epoch_before, 6_666_660, 0.0205, 3.5, 3106)
    triple_value, triple_row, triple_line = _triple_row(
        ("mc1", "mc2", "hm7"),
        triple_last_row,
        _EXAMPLE_LINKS,
        pair_part,
    )
    local_value, local_row, local_line = _triple_row(
        ("mc2", "mc2", "hm7"), last_row, (_EXAMPLE_SELF, _EXAMPLE_SELF), pair_part
    )
    return WorkedEpoch(
        last_row=last_row,
        prediction=prediction,
        measurement=measurement,
        pair_row=pair_step.row,
        next_row=next_step.row,
        meas_lines=(first_line_text.rstrip("\n"), next_line_text.rstrip("\n")),
        triple_last_row=triple_last_row,
        triple_value=triple_value,
        local_value=local_value,
        triple_row=triple_row,
        local_row=local_row,
        ddiff_lines=(triple_line, local_line),
    )


def _grouped(phase: mpq | int, places: int = 0) -> str:
    """Write a phase with its digits in groups of three.

    Parameters
    ----------
    phase : mpq or int
        The phase, ps.
    places : int, optional
        Decimal places to give it.

    Returns
    -------
    str
        The phase, rounded to ``places`` decimals for reading.
    """
    return f"{float(phase):_.{places}f}".replace("_", " ")


def _state_text(row: Row) -> str:
    """Write an invented row's state for the inputs table.

    Parameters
    ----------
    row : Row
        A row holding a state.

    Returns
    -------
    str
        Its phase, rate, drift and innovation scale, with their units.
    """
    x_fs, y, d = row.known_state()
    return (
        f"x = {_grouped(from_fs(x_fs))} ps, y = {y} ps/s, d = {d:g} ps/s{SQUARED},"
        f" {SIGMA}_{NU} = {row.innovation_scale:g} ps"
    )


def worked_inputs() -> str:
    """Give the invented values the worked example starts from.

    Returns
    -------
    str
        A table of the example's series, its estimator settings, the last
        rows it starts from, its steering and its other pairs' measurements.
    """
    worked = worked_epoch()
    params = _EXAMPLE_PARAMS
    forward, back = _EXAMPLE_LINKS
    epoch_before = f"{worked.last_row.interpolated_datetime:%Y-%m-%d %H:%M:%S} UTC"
    rows = [
        ("Pair", "(mc2, hm7), RF channel a"),
        (
            "Estimator",
            f"{params.filter_states}-state, M = {params.M:g}, M_{SIGMA} ="
            f" {params.M_sigma:g}, {SIGMA}{SUBSCRIPT_ZERO} = {params.sigma0:g} ps,"
            f" RMS limit {params.rms_max} ps",
        ),
        (f"Pair's last row, {epoch_before}", _state_text(worked.last_row)),
        ("Steering", "none"),
        (
            "Links",
            f"z(mc1, mc2) = {_grouped(forward.z or 0)} ps, z(mc2, mc1) ="
            f" {_grouped(back.z or 0)} ps, rms {forward.rms} ps each, both accepted",
        ),
        (
            "Self pair",
            f"z(mc2, mc2) = {_EXAMPLE_SELF.z} ps, accepted;"
            " it cancels in the local triple",
        ),
        (
            f"Remote triple's last row, {epoch_before}",
            _state_text(worked.triple_last_row),
        ),
        (f"Local triple's last row, {epoch_before}", "the pair's last row"),
    ]
    return table(["Input", "Value"], [[name, cell(value)] for name, value in rows])


def worked_raw_line() -> str:
    """Give the example's DAS line, as the DAS writes it.

    Returns
    -------
    str
        The line, as a code block.
    """
    return fenced("text", [str(_EXAMPLE_READING)])


NU: Final[str] = "\N{GREEK SMALL LETTER NU}"
"""The innovation's symbol."""

SIGMA: Final[str] = "\N{GREEK SMALL LETTER SIGMA}"
"""The symbol of a scale or deviation."""

DELTA: Final[str] = "\N{GREEK SMALL LETTER DELTA}"
"""The symbol of the measurement time after the epoch start."""

LESS_EQUAL: Final[str] = "\N{LESS-THAN OR EQUAL TO}"
"""The less-than-or-equal sign of a formula."""

SQUARED: Final[str] = "\N{SUPERSCRIPT TWO}"
"""The superscript two of a square."""

SUBSCRIPT_ZERO: Final[str] = "\N{SUBSCRIPT ZERO}"
"""The subscript zero of the initial innovation scale's symbol."""

LAMBDA: Final[str] = "\N{GREEK SMALL LETTER LAMDA}"
"""The estimator pole's symbol."""

PHI: Final[str] = "\N{GREEK SMALL LETTER PHI}"
"""The reading's symbol."""

MINUS: Final[str] = "\N{MINUS SIGN}"
"""The minus sign of a formula."""

TIMES: Final[str] = "\N{MULTIPLICATION SIGN}"
"""The multiplication sign of a formula."""

PREDICTED: Final[str] = "x\N{SUPERSCRIPT MINUS}"
"""The predicted phase's symbol, x with a superscript minus."""

RATE_PREDICTED: Final[str] = "y\N{SUPERSCRIPT MINUS}"
"""The predicted rate's symbol."""

DRIFT_PREDICTED: Final[str] = "d\N{SUPERSCRIPT MINUS}"
"""The predicted drift's symbol."""

AT_T: Final[str] = "x\N{COMBINING CIRCUMFLEX ACCENT}"
"""The symbol of the phase predicted at the measurement time, x with a hat."""


def _steps(worked: WorkedEpoch) -> list[tuple[str, str, str]]:
    """Give each step of the worked example, its working and its result.

    Parameters
    ----------
    worked : WorkedEpoch
        The worked example.

    Returns
    -------
    list of (str, str, str)
        The step's name, the sum it makes, and its result.
    """
    prediction, measurement, row = (
        worked.prediction,
        worked.measurement,
        worked.pair_row,
    )
    delta = measurement.delta
    motion = exact(prediction.y) * delta
    decycled = measurement.measured_phase + measurement.cycle_count * PHASE_PERIOD
    innovation = measurement.z - prediction.x
    g, h_over_t, two_k_over_t2 = exact_gains(3, _EXAMPLE_PARAMS.M)
    updated = update(prediction, innovation, 3, _EXAMPLE_PARAMS.M)
    epoch_text = f"{measurement.interpolated_datetime:%Y-%m-%d %H:%M:%S}"
    stored_x = _grouped(from_fs(row.known_state()[0]), 3)
    last_x = _grouped(from_fs(worked.last_row.known_state()[0]))
    return [
        (
            "Epoch start",
            "the measurement time rounded down to ten minutes",
            f"E = {epoch_text} UTC, {DELTA} = {float(delta):g} s",
        ),
        (
            "Predict",
            f"{PREDICTED} = {last_x} + {worked.last_row.y} {TIMES} {EPOCH_SECONDS}",
            f"{PREDICTED} = {_grouped(prediction.x, 4)}",
        ),
        (
            "Prediction at the measurement time",
            f"{AT_T} = {PREDICTED} + {RATE_PREDICTED}{DELTA}",
            f"{AT_T} = {_grouped(prediction.x + motion, 4)}",
        ),
        (
            "Decycle",
            f"n = round_even(({AT_T} {MINUS} {measurement.measured_phase})"
            f" / {PHASE_PERIOD})",
            f"n = {measurement.cycle_count}",
        ),
        ("Unwrapped phase", f"x_u = {PHI} + nP", f"x_u = {_grouped(decycled)}"),
        (
            "Refer back to E",
            f"z_E = round_even(x_u {MINUS} {RATE_PREDICTED}{DELTA})",
            f"z_E = round_even({_grouped(decycled - motion, 4)})"
            f" = {_grouped(measurement.z)}",
        ),
        (
            "Innovation",
            f"{NU} = z_E {MINUS} {PREDICTED}",
            f"{NU} = {float(innovation):.4f}",
        ),
        (
            "Gate",
            f"|{NU}| {LESS_EQUAL} {K_OUT:g} {TIMES} "
            f"{worked.last_row.innovation_scale:g}, rms {measurement.rms}"
            f" {LESS_EQUAL} {_EXAMPLE_PARAMS.rms_max}",
            f"flags {row.flags}",
        ),
        (
            "Gains",
            f"{LAMBDA} = e^({MINUS}1/M)",
            f"g = {float(g):.6g}, h/T = {h_over_t:.6g},"
            f" 2k/T{SQUARED} = {two_k_over_t2:.6g}",
        ),
        (
            "Update",
            f"x = {PREDICTED} + g{NU}, y = {RATE_PREDICTED} + (h/T){NU},"
            f" d = {DRIFT_PREDICTED} + (2k/T{SQUARED}){NU}",
            f"x = {_grouped(updated.x, 4)}, stored as {stored_x};"
            f" y = {row.y!r}; d = {row.d!r}",
        ),
        (
            "Innovation scale",
            f"{SIGMA}_{NU}{SQUARED} = max((1 {MINUS} w){SIGMA}_{NU}{SQUARED}"
            f" + w{NU}{SQUARED}, rms{SQUARED}), w = 1/M_{SIGMA}",
            f"{SIGMA}_{NU} = {row.innovation_scale:g}",
        ),
    ]


def worked_steps() -> str:
    """Give the example's working, step by step, as the program works it.

    Returns
    -------
    str
        A table of each step, the sum it makes and its result, phases shown
        rounded for reading, the program's own values beside them.
    """
    return table(
        ["Step", "What is worked out", "Result"],
        [
            [step, cell(working), cell(result)]
            for step, working, result in _steps(worked_epoch())
        ],
    )


def worked_triples() -> str:
    """Give the example's two double differences.

    Returns
    -------
    str
        The remote and the local triple's double difference and sigma, as
        the program works them out.
    """
    worked = worked_epoch()
    return table(
        ["Triple", "dd (ps)", f"{SIGMA}_dd (ps)", "Components used"],
        [
            [label, _grouped(value.z), f"{value.sigma:.6g}", value.components_used]
            for label, value in (
                ("(mc1, mc2, hm7), remote", worked.triple_value),
                ("(mc2, mc2, hm7), local", worked.local_value),
            )
        ],
    )


def worked_meas_rows() -> str:
    """Give the example pair's two rows, as its measurement file holds them.

    Returns
    -------
    str
        The accepted row, then the next epoch's predicted row, as a code
        block.
    """
    return fenced("text", worked_epoch().meas_lines)


def worked_ddiff_rows() -> str:
    """Give the example triples' rows, as their files hold them.

    Returns
    -------
    str
        The remote triple's row, then the local triple's, as a code block.
    """
    return fenced("text", worked_epoch().ddiff_lines)


# ----------------------------------------------------------------- settling


@functools.cache
def settling_percents(filter_states: FilterStates, epochs: int) -> tuple[float, ...]:
    """Run the estimator after a step in rate, and give its rate each epoch.

    A series at rest, its state zero, is measured with no noise after its
    clock's rate steps by :data:`SETTLING_RATE_STEP`. Each epoch is
    predicted, updated and stored as the program does.

    Parameters
    ----------
    filter_states : int
        2 or 3.
    epochs : int
        How many epochs to run after the step.

    Returns
    -------
    tuple of float
        The rate estimate as a percentage of the step, at the step and at
        each epoch after.
    """
    row = dataclasses.replace(
        _held_row(_EXAMPLE_START, 0, 0.0, 1.0, 0),
        filter_states=filter_states,
        time_constant=SETTLING_TIME_CONSTANT,
    )
    percents = [0.0]
    for epoch in range(1, epochs + 1):
        prediction = predict(row, _NO_STEERING)
        if prediction is None:  # pragma: no cover - every row holds a state
            message = "the settling run lost its state"
            raise ValueError(message)
        true_phase = round_even(exact(SETTLING_RATE_STEP) * EPOCH_SECONDS * epoch)
        updated = update(
            prediction, true_phase - prediction.x, filter_states, SETTLING_TIME_CONSTANT
        )
        row = dataclasses.replace(row, x_fs=to_fs(updated.x), y=updated.y, d=updated.d)
        percents.append(100.0 * updated.y / SETTLING_RATE_STEP)
    return tuple(percents)


def settled_from(percents: Sequence[float], time_constant_epochs: int) -> float:
    """Give when a rate estimate comes within :data:`SETTLED_PERCENT` for good.

    Parameters
    ----------
    percents : sequence of float
        The rate estimate each epoch, percent of the step.
    time_constant_epochs : int
        The time constant, epochs.

    Returns
    -------
    float
        The first epoch from which every estimate is within the bound, in
        time constants.

    Raises
    ------
    ValueError
        If the last estimate is not within the bound, so the run shows no
        such epoch.
    """
    outside = [
        epoch
        for epoch, percent in enumerate(percents)
        if abs(percent - 100.0) >= SETTLED_PERCENT
    ]
    if outside[-1] == len(percents) - 1:
        message = "the rate estimate does not settle within the run"
        raise ValueError(message)
    return (outside[-1] + 1) / time_constant_epochs


def settling_chart(filter_states: FilterStates) -> str:
    """Draw the rate estimate after a step in rate, as a mermaid chart.

    Parameters
    ----------
    filter_states : int
        2 or 3.

    Returns
    -------
    str
        A line chart of the rate estimate, percent of the step, every half
        time constant up to :data:`SETTLING_CHART_SPAN` of them.
    """
    time_constant = round(SETTLING_TIME_CONSTANT)
    percents = settling_percents(filter_states, SETTLING_RUN_SPAN * time_constant)
    half = time_constant // 2
    samples = range(0, SETTLING_CHART_SPAN * time_constant + 1, half)
    labels = ", ".join(f'"{sample / time_constant:g}"' for sample in samples)
    values = ", ".join(f"{percents[sample]:.1f}" for sample in samples)
    top = 10 * math.ceil(max(percents) / 10)
    return fenced(
        "mermaid",
        [
            "xychart-beta",
            f'    title "{filter_states}-state estimator after a rate step"',
            f'    x-axis "time since the step, in time constants M" [{labels}]',
            f'    y-axis "rate estimate, % of the step" 0 --> {top}',
            f"    line [{values}]",
        ],
    )


def settling_numbers() -> str:
    """Give how each model's rate estimate overshoots and settles after a step.

    Returns
    -------
    str
        For the 3- and 2-state models, the largest overshoot and when it
        comes, or none when the estimate never passes the step by
        :data:`SHOWN_OVERSHOOT`, what is left of the error at :data:`SETTLE_FACTOR` time
        constants, where rows stop carrying U, and when the estimate comes
        within :data:`SETTLED_PERCENT` for good.
    """
    time_constant = round(SETTLING_TIME_CONSTANT)
    rows = []
    model_states: tuple[FilterStates, ...] = (3, 2)
    for filter_states in model_states:
        percents = settling_percents(filter_states, SETTLING_RUN_SPAN * time_constant)
        peak = max(percents)
        overshoots = peak - 100.0 >= SHOWN_OVERSHOOT
        rows.append(
            [
                f"{filter_states}-state",
                f"{peak - 100.0:.1f}" if overshoots else "none",
                f"{percents.index(peak) / time_constant:.1f}" if overshoots else "-",
                f"{percents[SETTLE_FACTOR * time_constant] - 100.0:.1f}",
                f"{settled_from(percents, time_constant):.1f}",
            ]
        )
    return table(
        [
            "Model",
            "Largest overshoot (% of the step)",
            "Reached at (time constants)",
            f"Error left at {SETTLE_FACTOR}M (% of the step)",
            f"Within {SETTLED_PERCENT:g}% for good from (time constants)",
        ],
        rows,
    )


# ------------------------------------------------------------- the registry

BLOCKS: Final[dict[str, Callable[[], str]]] = {
    "settings": settings_table,
    "help": help_text,
    "meas-columns": lambda: column_table(MEAS_COLUMNS),
    "ddiff-columns": lambda: column_table(DDIFF_COLUMNS),
    "row-widths": row_widths,
    "flags": flag_table,
    "meas-header": lambda: _header_block("meas"),
    "ddiff-header": lambda: _header_block("ddiff"),
    "refused-lines": refused_lines,
    "errors": error_table,
    "modules": module_table,
    "constants": constants_table,
    "gains": gain_table,
    "worked-inputs": worked_inputs,
    "worked-raw-line": worked_raw_line,
    "worked-steps": worked_steps,
    "worked-triples": worked_triples,
    "worked-meas-rows": worked_meas_rows,
    "worked-ddiff-rows": worked_ddiff_rows,
    "settling-3-state": lambda: settling_chart(3),
    "settling-2-state": lambda: settling_chart(2),
    "settling-numbers": settling_numbers,
}
"""Every block a document may hold, by name, each giving its Markdown."""

FIGURES: Final[dict[str, Callable[[], str]]] = {
    name: _figure(module_name, name, value_form)
    for module_name, name, value_form in CONSTANTS
}
"""Every figure a document may hold, by the name of its constant."""
