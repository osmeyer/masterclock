"""Write the measured block of a report in ``docs/reports/``, on demand.

Each report holds its context, analysis and recommendations around one
measured block, between marker comments that do not show when it is read::

    <!-- measured: NAME -->

    ...

    <!-- end measured -->

and one header line ``**Updated:** YYYY-MM-DD HH:MM:SS UTC``. Give the
project folder and the names of the reports to write; for each, the script
runs its tools through ``uv run --frozen`` in the project folder, writes the
block from what they give, and sets the stamp to the time now. Nothing else
in a report is touched, and a report not named is left alone::

    uv run --frozen python scripts/reports.py . tests lint
    uv run --frozen python scripts/reports.py . performance --timing-folder DIR

The reports, by name:

- ``tests``: the test run, by outcome, every skipped test with its reason,
  and the line and branch coverage of src, scripts and tests apart;
- ``lint``: the files ruff would reformat, ruff's findings by rule, and every
  noqa and nosec comment in the project's Python files;
- ``security``: bandit's findings over scripts, src and tests, every severity;
- ``complexity``: radon's block ranks and maintainability, and xenon's result;
- ``performance``: ``scripts/epoch_timing.py`` on the laboratory's numbers of
  references and clocks, in ``--timing-folder``, a new or empty folder on the
  disk the real runs write to;
- ``mutation``: the counts and surviving mutants of the last mutmut run,
  which is run by hand first; the script runs no mutmut itself.

It prints each report it writes and each it cannot, and why, then how many
were named and written. It exits 0 when it wrote every one named and 1
otherwise; a usage error, such as a folder without ``pyproject.toml``,
exits 2.
"""

import argparse
import io
import json
import re
import subprocess  # nosec B404 - runs the project's own tools
import sys
import tempfile
import tokenize
import xml.etree.ElementTree as ET  # nosec B405 - reads only pytest's own file
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

REPORTS_FOLDER: Final = Path("docs/reports")
"""Where the reports are, under the project folder."""

REPORT_FILES: Final[dict[str, str]] = {
    "tests": "test_coverage.md",
    "lint": "formatting_linting.md",
    "security": "security.md",
    "complexity": "complexity.md",
    "performance": "performance.md",
    "mutation": "mutation.md",
}
"""Each report's name, and its file in the reports folder."""

LABORATORY_DEPLOYMENT: Final = (
    "--references",
    "5",
    "--clocks",
    "150",
    "--every-reference",
)
"""The timing script's options for the laboratory's references and clocks."""

COVERAGE_FOLDERS: Final = ("src", "scripts", "tests")
"""The folders whose coverage is given apart."""

LIMIT_RANK: Final = "B"
"""The highest rank xenon lets a block have."""

MEASURED: Final[re.Pattern[str]] = re.compile(
    r"(?P<start><!-- measured: (?P<name>[\w-]+) -->\n\n)"
    r"(?P<body>.*?)"
    r"(?P<end>\n\n<!-- end measured -->)",
    re.DOTALL,
)
"""A report's measured block between its markers."""

UPDATED: Final[re.Pattern[str]] = re.compile(
    r"^\*\*Updated:\*\* \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$", re.MULTILINE
)
"""A report's Updated stamp."""

FORMATTED: Final[re.Pattern[str]] = re.compile(r"(\d+) files? already formatted")
"""ruff format's count of the files it left as they are."""

MUTANT: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<module>[\w.]+?)\.x(?:_|ǁ)(?P<function>[\wǁ]+?)__mutmut_\d+: survived$",
    re.MULTILINE,
)
"""A surviving mutant in ``mutmut results``: its module and function."""


class ReportError(Exception):
    """A report cannot be written: its measurement failed, or it has no block."""


def run_tool(
    project_folder: Path, command: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    """Run a tool through ``uv run --frozen`` in the project folder.

    Parameters
    ----------
    project_folder : Path
        The project folder.
    command : Sequence of str
        The tool and its arguments.

    Returns
    -------
    subprocess.CompletedProcess of str
        The finished run, its output as text; its status is not checked.
    """
    # The command is a project tool with arguments this script makes.
    return subprocess.run(  # noqa: S603  # nosec B603 B607 - uv, as checks run
        ["uv", "run", "--frozen", *command],  # noqa: S607 - uv, as checks run
        cwd=project_folder,
        capture_output=True,
        text=True,
        check=False,
    )


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Write a Markdown table of ``rows`` under ``header``."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ tests


def tests_block(junit_xml: str, coverage: Mapping[str, object]) -> str:
    """Write the test report's block from pytest's JUnit file and coverage's JSON.

    Parameters
    ----------
    junit_xml : str
        The JUnit XML pytest wrote.
    coverage : Mapping of str to object
        The JSON coverage wrote.

    Returns
    -------
    str
        The tests by outcome, the skipped tests, and coverage by folder.
    """
    suite = ET.fromstring(junit_xml).find("testsuite")  # noqa: S314  # nosec B314 - pytest's own file
    assert suite is not None  # noqa: S101 - pytest always writes one suite
    counts = {
        key: int(suite.get(key, "0"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    outcome_table = table(
        ("Passed", "Failed", "Errors", "Skipped", "Tests", "Time (s)"),
        [(passed, counts["failures"], counts["errors"], counts["skipped"],
          counts["tests"], f"{float(suite.get('time', '0')):.1f}")],
    )  # fmt: skip
    skipped_lines = [
        f"- `{case.get('classname')}::{case.get('name')}`: {skipped.get('message')}"
        for case in suite.iter("testcase")
        if (skipped := case.find("skipped")) is not None
    ]
    skipped_text = "\n".join(skipped_lines) or "No test was skipped."
    return "\n\n".join(
        (outcome_table, "Skipped tests:", skipped_text, _coverage_table(coverage))
    )


def _coverage_table(coverage: Mapping[str, object]) -> str:
    """Write the line and branch coverage of each folder apart."""
    totals = {folder: [0, 0, 0, 0] for folder in COVERAGE_FOLDERS}
    files = coverage["files"]
    assert isinstance(files, dict)  # noqa: S101 - coverage's JSON holds a mapping
    for file_name, file_data in files.items():
        summary = file_data["summary"]
        folder_total = totals[file_name.split("/")[0]]
        for index, key in enumerate(
            ("covered_lines", "num_statements", "covered_branches", "num_branches")
        ):
            folder_total[index] += summary[key]
    rows = [
        (folder, f"{covered} of {lines}", f"{covered_b} of {branches}",
         f"{100 * (covered + covered_b) / max(lines + branches, 1):.2f}%")
        for folder, (covered, lines, covered_b, branches) in totals.items()
    ]  # fmt: skip
    return table(("Folder", "Lines", "Branches", "Coverage"), rows)


def measure_tests(project_folder: Path) -> str:
    """Run the tests with coverage, and write the test report's block."""
    with tempfile.TemporaryDirectory() as work_folder:
        junit_file = Path(work_folder) / "junit.xml"
        coverage_file = Path(work_folder) / "coverage.json"
        run_tool(
            project_folder,
            ["pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit_file}"],
        )
        run_tool(project_folder, ["coverage", "json", "-q", "-o", str(coverage_file)])
        return tests_block(
            junit_file.read_text(encoding="utf-8"),
            json.loads(coverage_file.read_text(encoding="utf-8")),
        )


# ------------------------------------------------------------------- lint


def suppressions(
    project_folder: Path, python_files: Sequence[Path]
) -> list[tuple[str, int, str]]:
    """Give every noqa and nosec comment in the files, with its file and line.

    Parameters
    ----------
    project_folder : Path
        The project folder the files are under.
    python_files : Sequence of Path
        The files, relative to the project folder.

    Returns
    -------
    list of (str, int, str)
        The file, the line and the comment, in file and line order; a noqa
        or nosec inside a string is not a comment and is left out.
    """
    found: list[tuple[str, int, str]] = []
    for python_file in python_files:
        source = (project_folder / python_file).read_text(encoding="utf-8")
        found += [
            (str(python_file), token.start[0], token.string)
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
            and re.search(r"\b(noqa|nosec)\b", token.string)
        ]
    return found


def lint_block(
    format_output: str,
    findings: Sequence[Mapping[str, object]],
    suppression_rows: Sequence[tuple[str, int, str]],
) -> str:
    """Write the formatting and linting report's block.

    Parameters
    ----------
    format_output : str
        What ``ruff format --check`` printed.
    findings : Sequence of Mapping of str to object
        ``ruff check``'s findings, as JSON.
    suppression_rows : Sequence of (str, int, str)
        Every noqa and nosec comment (see :func:`suppressions`).

    Returns
    -------
    str
        The files ruff would reformat, its findings by rule, and every
        suppression.
    """
    to_format = re.findall(r"^Would reformat: (.+)$", format_output, re.MULTILINE)
    formatted = FORMATTED.search(format_output)
    left_alone = int(formatted.group(1)) if formatted else 0
    parts = [
        table(("Files checked", "Files to reformat"),
              [(left_alone + len(to_format), len(to_format))]),
        "\n".join(f"- `{name}`" for name in to_format) or "Every file is formatted.",
    ]  # fmt: skip
    by_rule = Counter(str(finding["code"]) for finding in findings)
    parts.append(
        table(("Rule", "Findings"), sorted(by_rule.items()))
        if by_rule
        else "ruff found nothing."
    )
    parts.append(
        table(
            ("Where", "Comment"),
            [
                (f"`{name}:{line}`", f"`{comment}`")
                for name, line, comment in suppression_rows
            ],
        )
        if suppression_rows
        else "No finding is silenced."
    )
    return "\n\n".join(parts)


def measure_lint(project_folder: Path) -> str:
    """Run ruff and read every Python file, and write the lint report's block."""
    format_run = run_tool(project_folder, ["ruff", "format", "--check", "."])
    check_run = run_tool(
        project_folder, ["ruff", "check", "--output-format", "json", "."]
    )
    # git lists the tracked Python files; the command is fixed.
    listed = subprocess.run(  # nosec B603 B607 - git, as a user runs it
        ["git", "ls-files", "*.py"],  # noqa: S607 - git, as a user runs it
        cwd=project_folder,
        capture_output=True,
        text=True,
        check=False,
    )
    python_files = [Path(name) for name in listed.stdout.split()]
    return lint_block(
        format_run.stdout + format_run.stderr,
        json.loads(check_run.stdout),
        suppressions(project_folder, python_files),
    )


# --------------------------------------------------------------- security


def security_block(bandit: Mapping[str, object]) -> str:
    """Write the security report's block from bandit's JSON.

    Parameters
    ----------
    bandit : Mapping of str to object
        What bandit wrote.

    Returns
    -------
    str
        The findings by severity and by test, every finding above low, and
        how many findings comments silenced.
    """
    results = bandit["results"]
    metrics = bandit["metrics"]
    assert isinstance(results, list)  # noqa: S101 - bandit's JSON holds a list
    assert isinstance(metrics, dict)  # noqa: S101 - and a mapping
    severities = Counter(result["issue_severity"].lower() for result in results)
    by_test = Counter(
        (result["test_id"], result["issue_severity"].lower()) for result in results
    )
    above_low = [
        (f"`{result['filename'].removeprefix('./')}:{result['line_number']}`",
         result["test_id"], result["issue_severity"].lower(), result["issue_text"])
        for result in results
        if result["issue_severity"] != "LOW"
    ]  # fmt: skip
    return "\n\n".join(
        (
            table(("High", "Medium", "Low"),
                  [(severities["high"], severities["medium"], severities["low"])]),
            table(("Test", "Severity", "Findings"),
                  [(*test_severity, count)
                   for test_severity, count in sorted(by_test.items())]),
            table(("Where", "Test", "Severity", "Finding"), above_low)
            if above_low
            else "No finding is above low severity.",
            f"Findings silenced by a comment: {_silenced(metrics)}.",
        )
    )  # fmt: skip


def _silenced(metrics: Mapping[str, object]) -> int:
    """Count the findings comments silenced: bare nosec, and nosec naming a test."""
    totals = metrics["_totals"]
    assert isinstance(totals, dict)  # noqa: S101 - bandit's JSON holds a mapping
    return int(totals["nosec"]) + int(totals["skipped_tests"])


def measure_security(project_folder: Path) -> str:
    """Run bandit over scripts, src and tests, and write the security block."""
    scan = run_tool(
        project_folder,
        ["bandit", "-r", "-f", "json", "-q", "scripts", "src", "tests"],
    )
    return security_block(json.loads(scan.stdout))


# ------------------------------------------------------------- complexity


def complexity_block(
    radon_cc: Mapping[str, Sequence[Mapping[str, object]]],
    radon_mi: Mapping[str, Mapping[str, object]],
    *,
    xenon_passed: bool,
) -> str:
    """Write the complexity report's block from radon's JSON and xenon's result.

    Parameters
    ----------
    radon_cc : Mapping of str to Sequence of Mapping of str to object
        radon's complexity of every block, by file.
    radon_mi : Mapping of str to Mapping of str to object
        radon's maintainability of every module, by file.
    xenon_passed : bool
        Whether xenon passed.

    Returns
    -------
    str
        The blocks by rank, those at the highest rank the limits allow, the
        average complexity, the modules by maintainability rank, and
        xenon's result.
    """
    blocks = sorted(
        (
            (file_name, block)
            for file_name, file_blocks in radon_cc.items()
            for block in file_blocks
        ),
        key=lambda found: (found[0], int(str(found[1]["lineno"]))),
    )
    block_ranks = Counter(str(block["rank"]) for _, block in blocks)
    at_limit = [
        (f"`{file_name}:{block['lineno']}`", f"`{_block_name(block)}`",
         block["complexity"])
        for file_name, block in blocks
        if block["rank"] == LIMIT_RANK
    ]  # fmt: skip
    average = sum(float(str(block["complexity"])) for _, block in blocks) / max(
        len(blocks), 1
    )
    module_ranks = Counter(str(module["rank"]) for module in radon_mi.values())
    return "\n\n".join(
        (
            table(("Block rank", "Blocks"), sorted(block_ranks.items())),
            f"Blocks at rank {LIMIT_RANK}, the highest a block may have:",
            table(("Where", "Block", "Complexity"), at_limit),
            f"Average complexity of a block: {average:.2f}.",
            table(("Module rank", "Modules"), sorted(module_ranks.items())),
            "xenon passes: no block over rank B, no module or average over rank A."
            if xenon_passed
            else "xenon fails: a block, a module or the average is over its limit.",
        )
    )  # fmt: skip


def _block_name(block: Mapping[str, object]) -> str:
    """Name a block as radon gives it: a method with its class."""
    if "classname" in block:
        return f"{block['classname']}.{block['name']}"
    return str(block["name"])


def measure_complexity(project_folder: Path) -> str:
    """Run radon and xenon over scripts and src, and write the complexity block."""
    radon_cc = run_tool(
        project_folder, ["radon", "cc", "-j", "--no-assert", "scripts", "src"]
    )
    radon_mi = run_tool(project_folder, ["radon", "mi", "-j", "scripts", "src"])
    xenon = run_tool(
        project_folder,
        ["xenon", "--no-assert", "--max-absolute", "B", "--max-modules", "A",
         "--max-average", "A", "scripts", "src"],
    )  # fmt: skip
    return complexity_block(
        json.loads(radon_cc.stdout),
        json.loads(radon_mi.stdout),
        xenon_passed=xenon.returncode == 0,
    )


# ------------------------------------------------------------ performance


def performance_block(timing_output: str) -> str:
    """Write the performance report's block: each line the timing script printed."""
    return "\n".join(f"- {line}" for line in timing_output.splitlines() if line)


def measure_performance(project_folder: Path, timing_folder: Path) -> str:
    """Time das_processor on the laboratory's deployment, in ``timing_folder``.

    Raises
    ------
    ReportError
        If the timing script fails, with its error output.
    """
    timing = run_tool(
        project_folder,
        [
            "python",
            "scripts/epoch_timing.py",
            str(timing_folder),
            *LABORATORY_DEPLOYMENT,
        ],
    )
    if timing.returncode != 0:
        message = f"the timing failed: {timing.stderr.strip()}"
        raise ReportError(message)
    return performance_block(timing.stdout)


# --------------------------------------------------------------- mutation


def mutation_block(stats: Mapping[str, int], results_text: str) -> str:
    """Write the mutation report's block from mutmut's counts and results.

    Parameters
    ----------
    stats : Mapping of str to int
        mutmut's counts, as ``mutmut export-cicd-stats`` writes them.
    results_text : str
        What ``mutmut results`` printed.

    Returns
    -------
    str
        The counts, and the surviving mutants by module and function.
    """
    counted = (
        "total",
        "killed",
        "survived",
        "no_tests",
        "timeout",
        "suspicious",
        "skipped",
    )
    counts = table(
        ("Mutants", "Killed", "Survived", "No tests", "Timed out", "Suspicious",
         "Skipped"),
        [tuple(stats[key] for key in counted)],
    )  # fmt: skip
    survivors = Counter(
        (match["module"], match["function"].strip("ǁ").replace("ǁ", "."))
        for match in MUTANT.finditer(results_text)
    )
    survivor_text = (
        table(("Module", "Function", "Survived"),
              [(f"`{module}`", f"`{function}`", count)
               for (module, function), count in sorted(survivors.items())])
        if survivors
        else "No mutant survived."
    )  # fmt: skip
    return "\n\n".join((counts, survivor_text))


def measure_mutation(project_folder: Path) -> str:
    """Read the last mutmut run's counts and results, and write the mutation block.

    Raises
    ------
    ReportError
        If there is no mutmut run to read.
    """
    stats_file = project_folder / "mutants" / "mutmut-cicd-stats.json"
    if not stats_file.is_file():
        message = "no mutmut run: run mutmut run and mutmut export-cicd-stats first"
        raise ReportError(message)
    results = run_tool(project_folder, ["mutmut", "results"])
    return mutation_block(
        json.loads(stats_file.read_text(encoding="utf-8")), results.stdout
    )


# ------------------------------------------------------------------- main

MEASUREMENTS: Final[dict[str, Callable[[Path, Path | None], str]]] = {
    "tests": lambda root, _folder: measure_tests(root),
    "lint": lambda root, _folder: measure_lint(root),
    "security": lambda root, _folder: measure_security(root),
    "complexity": lambda root, _folder: measure_complexity(root),
    "performance": lambda root, folder: measure_performance(root, Path(str(folder))),
    "mutation": lambda root, _folder: measure_mutation(root),
}
"""Each report's measurement: the project folder and the timing folder in."""


def rewrite(report_text: str, block: str, now: datetime) -> str:
    """Give a report with its measured block replaced and its stamp set to ``now``.

    Raises
    ------
    ReportError
        If the report has no measured block or no Updated stamp.
    """
    if MEASURED.search(report_text) is None:
        message = "no measured block"
        raise ReportError(message)
    if UPDATED.search(report_text) is None:
        message = "no Updated stamp"
        raise ReportError(message)
    rewritten = MEASURED.sub(
        lambda found: found["start"] + block + found["end"], report_text, count=1
    )
    return UPDATED.sub(f"**Updated:** {now:%Y-%m-%d %H:%M:%S} UTC", rewritten, count=1)


def _project_root(cli_argument: str) -> Path:
    """Convert a command-line argument to a folder that holds ``pyproject.toml``."""
    project_folder = Path(cli_argument)
    if not (project_folder / "pyproject.toml").is_file():
        refusal = f"not a project folder (no pyproject.toml): {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return project_folder


def _write_report(project_folder: Path, name: str, timing_folder: Path | None) -> None:
    """Measure one report and write its block and stamp.

    Raises
    ------
    ReportError
        If it cannot be measured, or has no block or stamp.
    """
    report_file = project_folder / REPORTS_FOLDER / REPORT_FILES[name]
    block = MEASUREMENTS[name](project_folder, timing_folder)
    report_file.write_text(
        rewrite(report_file.read_text(encoding="utf-8"), block, datetime.now(UTC)),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the named reports and return the exit status."""
    parser = argparse.ArgumentParser(description="Write the named reports.")
    parser.add_argument("root", type=_project_root, help="the project folder")
    parser.add_argument(
        "names", nargs="+", choices=sorted(REPORT_FILES), help="reports"
    )
    parser.add_argument(
        "--timing-folder",
        type=Path,
        help="a new or empty folder, on the real runs' disk, for the performance runs",
    )
    arguments = parser.parse_args(argv)
    if "performance" in arguments.names and arguments.timing_folder is None:
        parser.error("the performance report needs --timing-folder")
    written = 0
    for name in arguments.names:
        try:
            _write_report(arguments.root, name, arguments.timing_folder)
        except ReportError as exc:
            print(f"{name}: {exc}")
            continue
        written += 1
        print(f"wrote {REPORTS_FOLDER / REPORT_FILES[name]}")
    print(f"reports: named: {len(arguments.names)}, written: {written}")
    return 0 if written == len(arguments.names) else 1


if __name__ == "__main__":
    sys.exit(main())
