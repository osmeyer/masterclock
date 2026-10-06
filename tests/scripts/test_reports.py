"""Tests for scripts/reports.py.

The rules covered: each report in docs/reports/ holds one measured block
between marker comments and one Updated stamp; a report named on the command
line has its block written from what its tools give now and its stamp set to
the time, and no other report is touched; a report with no block or no
stamp, or a measurement that cannot be made, is reported and the exit status
is 1; the tools are run through ``uv run --frozen`` in the project folder;
the test report counts the tests by outcome, names every skipped test with
its reason, and gives line and branch coverage for src, scripts and tests
apart; the formatting and linting report gives the files ruff would
reformat, ruff's findings by rule, and every noqa and nosec comment in the
project's Python files, where it is; the security report counts bandit's
findings by severity and test and lists every one above low severity, and
how many findings comments silenced; the complexity report counts the blocks
of each rank, lists those at the most rank the limits allow, in file and
line order, gives the average, the modules' maintainability ranks and
whether xenon passes; the performance report runs the timing script on the
deployment the laboratory has, in the folder given, which is required; the
mutation report gives mutmut's counts and the surviving mutants by function,
and refuses when there is no mutmut run; a folder without pyproject.toml is
refused with status 2; and run as a program, the script exits with the
status main returns.
"""

import json
import runpy
import subprocess  # nosec B404 - for the finished runs the tests give
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

import reports

NOW: Final = datetime(2026, 10, 5, 21, 0, 7, tzinfo=UTC)
"""An invented moment the reports are written."""

REPORT_TEXT: Final = """# A report

**Updated:** 2026-01-01 00:00:00 UTC

## Context

Words.

## Measured

<!-- measured: NAME -->

old block

<!-- end measured -->

## Analysis

More words.
"""
"""A report with its block and stamp, NAME standing for its name."""


def completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Give a finished tool run with ``stdout``."""
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


# ------------------------------------------------------------- writing


def test_a_block_is_written_and_the_report_stamped() -> None:
    """Put the new block between the markers and the time in the stamp."""
    rewritten = reports.rewrite(REPORT_TEXT.replace("NAME", "tests"), "new block", NOW)
    assert "<!-- measured: tests -->\n\nnew block\n\n<!-- end measured -->" in (
        rewritten
    )
    assert "old block" not in rewritten
    assert "**Updated:** 2026-10-05 21:00:07 UTC" in rewritten
    assert rewritten.count("Words.") == 1


@pytest.mark.parametrize(
    ("report_text", "problem"),
    [
        (REPORT_TEXT.replace("<!-- end measured -->", ""), "no measured block"),
        (REPORT_TEXT.replace("**Updated:**", "Updated:"), "no Updated stamp"),
    ],
)
def test_a_report_without_its_block_or_stamp_is_refused(
    report_text: str, problem: str
) -> None:
    """Raise ReportError for a report missing its block or its stamp."""
    with pytest.raises(reports.ReportError, match=problem):
        reports.rewrite(report_text, "new block", NOW)


def test_a_table_has_its_header_and_rows() -> None:
    """Write a Markdown table, every cell as text."""
    assert reports.table(("A", "B"), [(1, "x"), (2.5, "y")]) == (
        "| A | B |\n| --- | --- |\n| 1 | x |\n| 2.5 | y |"
    )


def test_a_tool_is_run_through_uv_in_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run ``uv run --frozen`` with the tool's command, in the project folder."""
    calls: list[tuple[Sequence[str], object]] = []

    def fake_run(command: Sequence[str], **kwargs: object) -> object:
        """Record the command and where it runs."""
        calls.append((command, kwargs["cwd"]))
        return completed("out")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert reports.run_tool(tmp_path, ["ruff", "check"]).stdout == "out"
    assert calls == [(["uv", "run", "--frozen", "ruff", "check"], tmp_path)]


# ------------------------------------------------------------------ tests

JUNIT_XML: Final = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="2" skipped="1"
 tests="10" time="14.25">
<testcase classname="tests.a.test_x" name="test_one" time="0.1"/>
<testcase classname="tests.a.test_x" name="test_two" time="0.1">
<skipped type="pytest.skip" message="no Chrome here">x</skipped></testcase>
</testsuite></testsuites>
"""
"""A JUnit XML file of an invented test run."""


def file_summary(
    lines: int, covered: int, branches: int, covered_branches: int
) -> dict[str, dict[str, int]]:
    """Give one file's coverage summary as coverage's JSON holds it."""
    return {
        "summary": {
            "num_statements": lines,
            "covered_lines": covered,
            "num_branches": branches,
            "covered_branches": covered_branches,
        }
    }


COVERAGE: Final = {
    "files": {
        "src/masterclock/a.py": file_summary(100, 100, 40, 40),
        "src/masterclock/b.py": file_summary(50, 50, 10, 10),
        "scripts/c.py": file_summary(80, 79, 20, 19),
        "tests/test_c.py": file_summary(30, 30, 0, 0),
    }
}
"""Coverage's JSON for an invented run."""


def test_the_test_report_counts_outcomes_and_coverage() -> None:
    """Count each outcome, name skipped tests, and give coverage per folder."""
    block = reports.tests_block(JUNIT_XML, COVERAGE)
    assert "| 6 | 2 | 1 | 1 | 10 | 14.2 |" in block
    assert "- `tests.a.test_x::test_two`: no Chrome here" in block
    assert "| src | 150 of 150 | 50 of 50 | 100.00% |" in block
    assert "| scripts | 79 of 80 | 19 of 20 | 98.00% |" in block
    assert "| tests | 30 of 30 | 0 of 0 | 100.00% |" in block


def test_a_run_with_nothing_skipped_says_so() -> None:
    """Say no test was skipped, rather than give an empty list."""
    junit_xml = JUNIT_XML.replace('skipped="1"', 'skipped="0"').replace(
        '<skipped type="pytest.skip" message="no Chrome here">x</skipped>', ""
    )
    assert "No test was skipped." in reports.tests_block(junit_xml, COVERAGE)


def test_the_tests_are_run_with_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run pytest for its JUnit file, then coverage for its JSON, and read both."""
    commands: list[list[str]] = []

    def fake_tool(_root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Write what each tool would write, and record the command."""
        commands.append(command)
        for argument in command:
            if argument.startswith("--junitxml="):
                Path(argument.removeprefix("--junitxml=")).write_text(JUNIT_XML)
            if argument.endswith(".json") and command[0] == "coverage":
                Path(argument).write_text(json.dumps(COVERAGE))
        return completed()

    monkeypatch.setattr(reports, "run_tool", fake_tool)
    assert "| 6 | 2 | 1 | 1 | 10 |" in reports.measure_tests(tmp_path)
    assert [command[0] for command in commands] == ["pytest", "coverage"]


# ------------------------------------------------------------------- lint


def test_suppressions_are_found_in_comments_only(tmp_path: Path) -> None:
    """List every noqa and nosec comment, with its file and line, never a string."""
    (tmp_path / "a.py").write_text(
        "import subprocess  # nosec B404\n"
        'x = "# noqa: E501 is only text here"\n'
        "y = 1  # noqa: S603  # nosec B603 - the reason\n",
        encoding="utf-8",
    )
    assert reports.suppressions(tmp_path, [Path("a.py")]) == [
        ("a.py", 1, "# nosec B404"),
        ("a.py", 3, "# noqa: S603  # nosec B603 - the reason"),
    ]


def test_the_lint_report_gives_format_findings_and_suppressions() -> None:
    """Give the files to reformat, the findings by rule and every suppression."""
    block = reports.lint_block(
        "Would reformat: scripts/x.py\n"
        "1 file would be reformatted, 94 files already formatted\n",
        [{"code": "E501"}, {"code": "E501"}, {"code": "F401"}],
        [("a.py", 3, "# noqa: S603")],
    )
    assert "| 95 | 1 |" in block
    assert "- `scripts/x.py`" in block
    assert "| E501 | 2 |\n| F401 | 1 |" in block
    assert "| `a.py:3` | `# noqa: S603` |" in block


def test_a_clean_lint_report_says_so() -> None:
    """Say no file needs formatting and ruff found nothing."""
    block = reports.lint_block("95 files already formatted\n", [], [])
    assert "| 95 | 0 |" in block
    assert "Every file is formatted." in block
    assert "ruff found nothing." in block
    assert "No finding is silenced." in block


def test_the_lint_tools_are_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run ruff format, ruff check and git, and read every Python file listed."""
    (tmp_path / "a.py").write_text("x = 1  # noqa: E501\n", encoding="utf-8")
    outputs = {
        "ruff": [completed("1 file already formatted\n"), completed("[]")],
        "git": completed("a.py\n"),
    }

    def fake_run(command: Sequence[str], **_kwargs: object) -> object:
        """Give each tool's output."""
        tool = command[3] if command[0] == "uv" else command[0]
        output = outputs[tool]
        return output.pop(0) if isinstance(output, list) else output

    monkeypatch.setattr(subprocess, "run", fake_run)
    block = reports.measure_lint(tmp_path)
    assert "| `a.py:1` | `# noqa: E501` |" in block


# --------------------------------------------------------------- security

BANDIT_RESULTS: Final[list[dict[str, object]]] = [
    {"test_id": "B101", "issue_severity": "LOW", "filename": "./tests/t.py",
     "line_number": 3, "issue_text": "Use of assert detected."},
    {"test_id": "B101", "issue_severity": "LOW", "filename": "./tests/t.py",
     "line_number": 4, "issue_text": "Use of assert detected."},
    {"test_id": "B603", "issue_severity": "MEDIUM", "filename": "./src/m.py",
     "line_number": 9, "issue_text": "subprocess call."},
]  # fmt: skip
"""bandit's findings in an invented scan."""

BANDIT: Final = {
    "metrics": {"_totals": {"nosec": 2, "skipped_tests": 5}},
    "results": BANDIT_RESULTS,
}
"""bandit's JSON for an invented scan."""


def test_the_security_report_counts_and_lists_findings() -> None:
    """Count by severity and test, and list every finding above low."""
    block = reports.security_block(BANDIT)
    assert "| 0 | 1 | 2 |" in block
    assert "| B101 | low | 2 |\n| B603 | medium | 1 |" in block
    assert "| `src/m.py:9` | B603 | medium | subprocess call. |" in block
    assert "Findings silenced by a comment: 7." in block


def test_a_scan_with_nothing_above_low_says_so() -> None:
    """Say no finding is above low severity."""
    low_only = {**BANDIT, "results": BANDIT_RESULTS[:2]}
    assert "No finding is above low severity." in reports.security_block(low_only)


def test_bandit_scans_every_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run bandit over scripts, src and tests, every severity, as JSON."""
    commands: list[list[str]] = []

    def fake_tool(_root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Record the command and give the scan."""
        commands.append(command)
        return completed(json.dumps(BANDIT), returncode=1)

    monkeypatch.setattr(reports, "run_tool", fake_tool)
    assert "| 0 | 1 | 2 |" in reports.measure_security(tmp_path)
    assert commands == [["bandit", "-r", "-f", "json", "-q", "scripts", "src", "tests"]]


# ------------------------------------------------------------- complexity

RADON_CC: Final = {
    "scripts/a.py": [
        {"type": "function", "rank": "B", "complexity": 7, "name": "f", "lineno": 3},
        {"type": "class", "rank": "A", "complexity": 2, "name": "K", "lineno": 9},
        {"type": "method", "rank": "A", "complexity": 1, "name": "m",
         "classname": "K", "lineno": 10},
    ],
    "src/b.py": [
        {"type": "function", "rank": "A", "complexity": 3, "name": "g", "lineno": 1},
    ],
}  # fmt: skip
"""radon's block complexities for an invented project."""

RADON_MI: Final = {
    "scripts/a.py": {"mi": 61.0, "rank": "A"},
    "src/b.py": {"mi": 15.0, "rank": "B"},
}
"""radon's maintainability for an invented project."""


def test_the_complexity_report_counts_ranks_and_lists_the_highest() -> None:
    """Count blocks by rank, list those at B, give the average and xenon's result."""
    block = reports.complexity_block(RADON_CC, RADON_MI, xenon_passed=True)
    assert "| A | 3 |\n| B | 1 |" in block
    assert "| `scripts/a.py:3` | `f` | 7 |" in block
    assert "Average complexity of a block: 3.25." in block
    assert "| A | 1 |\n| B | 1 |" in block
    assert "xenon passes" in block


def test_a_failing_xenon_is_reported() -> None:
    """Say xenon fails."""
    block = reports.complexity_block(RADON_CC, RADON_MI, xenon_passed=False)
    assert "xenon fails" in block


def test_a_method_is_named_with_its_class() -> None:
    """Name a method at rank B with its class."""
    radon_cc = {
        "src/c.py": [
            {"type": "method", "rank": "B", "complexity": 6, "name": "m",
             "classname": "K", "lineno": 4},
        ]
    }  # fmt: skip
    assert "| `src/c.py:4` | `K.m` | 6 |" in reports.complexity_block(
        radon_cc, {}, xenon_passed=True
    )


def test_the_blocks_at_rank_b_are_listed_by_file_and_line() -> None:
    """List rank B blocks in file and line order, whatever order radon gives."""

    def rank_b(name: str, lineno: int) -> dict[str, object]:
        """Give a function block of rank B."""
        return {"type": "function", "rank": "B", "complexity": 6, "name": name,
                "lineno": lineno}  # fmt: skip

    radon_cc = {
        "src/z.py": [rank_b("z", 9)],
        "src/a.py": [rank_b("b", 30), rank_b("a", 4)],
    }
    block = reports.complexity_block(radon_cc, {}, xenon_passed=True)
    assert block.index("`src/a.py:4`") < block.index("`src/a.py:30`")
    assert block.index("`src/a.py:30`") < block.index("`src/z.py:9`")


def test_the_complexity_tools_are_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run radon cc, radon mi and xenon over scripts and src."""
    outputs = {
        "cc": completed(json.dumps(RADON_CC)),
        "mi": completed(json.dumps(RADON_MI)),
        "xenon": completed(returncode=1),
    }

    def fake_tool(_root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Give each tool's output."""
        return outputs[command[0] if command[0] == "xenon" else command[1]]

    monkeypatch.setattr(reports, "run_tool", fake_tool)
    assert "xenon fails" in reports.measure_complexity(tmp_path)


# ------------------------------------------------------------ performance


def test_the_performance_report_holds_the_timing_lines() -> None:
    """Give each line the timing script printed as an item."""
    assert reports.performance_block("a: 1 s\nb: 2 s\n") == "- a: 1 s\n- b: 2 s"


def test_the_timing_runs_on_the_laboratory_s_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time the laboratory's numbers of references and clocks, in the folder given."""
    commands: list[list[str]] = []

    def fake_tool(_root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
        """Record the command and give the timing."""
        commands.append(command)
        return completed("run: 1 s\n")

    monkeypatch.setattr(reports, "run_tool", fake_tool)
    assert reports.measure_performance(tmp_path, tmp_path / "timing") == "- run: 1 s"
    assert commands == [
        [
            "python",
            "scripts/epoch_timing.py",
            str(tmp_path / "timing"),
            *reports.LABORATORY_DEPLOYMENT,
        ]
    ]


def test_a_failed_timing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise ReportError with the timing script's error output."""
    monkeypatch.setattr(
        reports,
        "run_tool",
        lambda _root, command: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="a run failed"
        ),
    )
    with pytest.raises(reports.ReportError, match="a run failed"):
        reports.measure_performance(tmp_path, tmp_path / "timing")


# --------------------------------------------------------------- mutation

MUTMUT_STATS: Final = {
    "killed": 6000,
    "survived": 3,
    "total": 6010,
    "no_tests": 5,
    "skipped": 0,
    "suspicious": 1,
    "timeout": 1,
    "check_was_interrupted_by_user": 0,
    "segfault": 0,
}
"""mutmut's counts for an invented run."""

MUTMUT_RESULTS: Final = """    masterclock.domain.filter.x_gate__mutmut_3: survived
    masterclock.domain.filter.x_gate__mutmut_7: survived
    masterclock.app.lock.xǁRunLockǁacquire__mutmut_2: survived
    masterclock.app.log.x_get_logger__mutmut_1: no tests
"""
"""What ``mutmut results`` prints for an invented run."""


def test_the_mutation_report_counts_and_names_the_survivors() -> None:
    """Give mutmut's counts, and the surviving mutants by function."""
    block = reports.mutation_block(MUTMUT_STATS, MUTMUT_RESULTS)
    assert "| 6010 | 6000 | 3 | 5 | 1 | 1 | 0 |" in block
    assert "| `masterclock.app.lock` | `RunLock.acquire` | 1 |" in block
    assert "| `masterclock.domain.filter` | `gate` | 2 |" in block


def test_a_run_with_no_survivor_says_so() -> None:
    """Say no mutant survived."""
    block = reports.mutation_block({**MUTMUT_STATS, "survived": 0}, "")
    assert "No mutant survived." in block


def test_the_mutation_report_reads_the_last_mutmut_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read mutmut's counts file and its results, running no mutmut."""
    (tmp_path / "mutants").mkdir()
    (tmp_path / "mutants" / "mutmut-cicd-stats.json").write_text(
        json.dumps(MUTMUT_STATS)
    )
    monkeypatch.setattr(
        reports, "run_tool", lambda _root, _command: completed(MUTMUT_RESULTS)
    )
    assert "| 6010 | 6000 | 3 |" in reports.measure_mutation(tmp_path)


def test_no_mutmut_run_is_refused(tmp_path: Path) -> None:
    """Raise ReportError when mutmut has not been run."""
    with pytest.raises(reports.ReportError, match="mutmut run"):
        reports.measure_mutation(tmp_path)


# ------------------------------------------------------------------- main


def make_project(project_root: Path) -> None:
    """Make a project with every report."""
    (project_root / "pyproject.toml").write_text("", encoding="utf-8")
    (project_root / "docs" / "reports").mkdir(parents=True)
    for name, file_name in reports.REPORT_FILES.items():
        (project_root / "docs" / "reports" / file_name).write_text(
            REPORT_TEXT.replace("NAME", name), encoding="utf-8"
        )


def test_only_the_reports_named_are_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the named reports' blocks and stamps, and leave the others alone."""
    make_project(tmp_path)
    monkeypatch.setitem(reports.MEASUREMENTS, "security", lambda _root, _folder: "new")
    monkeypatch.setitem(
        reports.MEASUREMENTS, "performance", lambda _root, folder: str(folder)
    )
    reports_folder = tmp_path / "docs" / "reports"
    before = (reports_folder / "complexity.md").read_text()
    assert (
        reports.main(
            [str(tmp_path), "security", "performance", "--timing-folder", "/t"]
        )
        == 0
    )
    assert "\n\nnew\n\n" in (reports_folder / "security.md").read_text()
    assert "\n\n/t\n\n" in (reports_folder / "performance.md").read_text()
    assert (reports_folder / "complexity.md").read_text() == before
    assert capsys.readouterr().out.endswith("reports: named: 2, written: 2\n")


def test_a_report_that_cannot_be_written_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a measurement that fails, and a report with no block, and exit 1."""
    make_project(tmp_path)
    (tmp_path / "docs" / "reports" / "security.md").write_text("# Security\n")

    def failing(_root: Path, _folder: Path | None) -> str:
        """Fail as a measurement that cannot be made."""
        raise reports.ReportError("no mutmut run")

    monkeypatch.setitem(reports.MEASUREMENTS, "mutation", failing)
    monkeypatch.setitem(reports.MEASUREMENTS, "security", lambda _root, _folder: "x")
    assert reports.main([str(tmp_path), "mutation", "security"]) == 1
    printed = capsys.readouterr().out
    assert "mutation: no mutmut run" in printed
    assert "security: no measured block" in printed
    assert printed.endswith("reports: named: 2, written: 0\n")


def test_the_performance_report_needs_its_timing_folder(tmp_path: Path) -> None:
    """Exit 2 for the performance report without --timing-folder."""
    make_project(tmp_path)
    with pytest.raises(SystemExit) as system_exit:
        reports.main([str(tmp_path), "performance"])
    assert system_exit.value.code == 2


def test_a_folder_without_pyproject_is_refused(tmp_path: Path) -> None:
    """Exit 2 for a folder that is not a project."""
    with pytest.raises(SystemExit) as system_exit:
        reports.main([str(tmp_path), "tests"])
    assert system_exit.value.code == 2


def test_the_measurements_are_the_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure each report with its own function."""
    names: list[str] = []

    def recorder(name: str) -> Callable[..., str]:
        """Give a measurement that records its name."""

        def record(*_arguments: object) -> str:
            """Record the name."""
            names.append(name)
            return ""

        return record

    for name in ("tests", "lint", "security", "complexity", "mutation"):
        monkeypatch.setattr(reports, f"measure_{name}", recorder(name))
    monkeypatch.setattr(reports, "measure_performance", recorder("p"))
    for measurement in reports.MEASUREMENTS.values():
        measurement(tmp_path, tmp_path)
    assert names == ["tests", "lint", "security", "complexity", "p", "mutation"]


def test_run_as_a_program_it_exits_with_main_s_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a script."""
    monkeypatch.setattr(sys, "argv", ["reports.py", str(tmp_path), "tests"])
    with pytest.raises(SystemExit) as system_exit:
        runpy.run_path(str(Path(reports.__file__)), run_name="__main__")
    assert system_exit.value.code == 2
