"""Tests for scripts/documents.py.

The rules covered: the documents are the README at the root and every
Markdown file under docs/, the reports left out; a block or figure between
markers is written from the code, a block or figure the code does not give
is a problem, and a document whose blocks or figures differ from what the
code gives fails the check, each one named; a document holds exactly one
Date header; while a document differs from what is committed, its date must
be today's UTC date, and once committed and unchanged, the UTC date of the
last commit that touched it, whatever the committer's timezone; run without
--check, the script writes the blocks and figures and stamps every changed
document with the time now; the check prints how many documents, blocks and
figures it examined, and fails when it found no document; a folder git
cannot read fails; a missing folder is a usage error; and run as a program,
the script exits with the status main returns.
"""

import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

import document_content
import documents

NOW: Final = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
"""The time the tests take to be now."""

TODAY_HEADER: Final = "**Date:** 2026-03-04 05:06:07 UTC"
"""A Date header of today."""

COMMIT_DATE: Final = "2026-01-02T03:04:05+00:00"
"""The committer date the tests commit with."""

COMMIT_HEADER: Final = "**Date:** 2026-01-02 09:09:09 UTC"
"""A Date header of the commit's day; its time is never checked."""

GIT: Final = shutil.which("git")
"""The git command."""

BLOCKS: Final[dict[str, Callable[[], str]]] = {"demo": lambda: "| a |\n| --- |"}
"""The blocks the tests' documents hold."""

FIGURES: Final[dict[str, Callable[[], str]]] = {"answer": lambda: "42"}
"""The figures the tests' documents hold."""

REAL_UTC_NOW: Final = documents.utc_now
"""The script's own clock, before the tests replace it."""


def git(repository: Path, *git_arguments: str) -> None:
    """Run git in a test repository, its committer date fixed."""
    assert GIT is not None
    git_environment = {**os.environ, "GIT_COMMITTER_DATE": COMMIT_DATE}
    # The command is git with arguments written in the tests.
    subprocess.run(  # noqa: S603  # nosec B603
        [GIT, "-C", str(repository), *git_arguments],
        check=True,
        capture_output=True,
        env=git_environment,
    )


def document(*body_lines: str, date_header: str = TODAY_HEADER) -> str:
    """Write a document's text: a title, its Date header, then its lines."""
    return "\n".join(["# Title", "", date_header, "", *body_lines]) + "\n"


@pytest.fixture(autouse=True)
def fixed_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the tests' blocks and figures, and the tests' time now."""
    monkeypatch.setattr(document_content, "BLOCKS", BLOCKS)
    monkeypatch.setattr(document_content, "FIGURES", FIGURES)
    monkeypatch.setattr(documents, "utc_now", lambda: NOW)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Make an empty git repository to hold the tests' documents."""
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "user.email", "tester@example.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


def commit(repository: Path, *document_paths: str) -> None:
    """Commit the documents, with the fixed committer date."""
    git(repository, "add", *document_paths)
    git(repository, "commit", "--quiet", "-m", "Add documents")


def write(repository: Path, document_path: str, document_text: str) -> Path:
    """Write a document in the repository."""
    written = repository / document_path
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(document_text, encoding="utf-8")
    return written


def test_the_documents_are_the_readme_and_docs_but_reports(tmp_path: Path) -> None:
    """Find the README and every Markdown file under docs, but the reports."""
    for document_path in (
        "README.md",
        "docs/a/manual.md",
        "docs/b.md",
        "docs/reports/tests.md",
        "docs/notes.txt",
        "other.md",
    ):
        write(tmp_path, document_path, "")
    assert documents.document_paths(tmp_path) == [
        tmp_path / "README.md",
        tmp_path / "docs/a/manual.md",
        tmp_path / "docs/b.md",
    ]


def test_no_readme_is_no_document(tmp_path: Path) -> None:
    """Leave out a README that is not there."""
    write(tmp_path, "docs/b.md", "")
    assert documents.document_paths(tmp_path) == [tmp_path / "docs/b.md"]


def test_blocks_and_figures_are_written_from_the_code() -> None:
    """Replace each block's and figure's old text with the code's."""
    filled = documents.fill(
        "Before.\n\n<!-- generated: demo -->\n\nold\ntable\n\n<!-- end generated -->\n"
        "The answer is <!-- figure: answer -->7<!-- end figure -->.\n"
    )
    assert filled.text == (
        "Before.\n\n<!-- generated: demo -->\n\n| a |\n| --- |\n\n"
        "<!-- end generated -->\n"
        "The answer is <!-- figure: answer -->42<!-- end figure -->.\n"
    )
    assert (filled.blocks, filled.figures) == (1, 1)
    assert filled.stale == ["block demo", "figure answer"]
    assert filled.problems == []


def test_blocks_and_figures_that_match_are_not_stale() -> None:
    """Name nothing as stale when every block and figure is the code's."""
    current_text = (
        "<!-- generated: demo -->\n\n| a |\n| --- |\n\n<!-- end generated -->\n"
        "<!-- figure: answer -->42<!-- end figure -->\n"
    )
    filled = documents.fill(current_text)
    assert filled.text == current_text
    assert filled.stale == []


def test_a_name_the_code_does_not_give_is_a_problem() -> None:
    """Report a block or figure no code gives, and leave its text alone."""
    unknown_text = (
        "<!-- generated: nothing -->\n\nkept\n\n<!-- end generated -->\n"
        "<!-- figure: nobody -->kept<!-- end figure -->\n"
    )
    filled = documents.fill(unknown_text)
    assert filled.text == unknown_text
    assert filled.problems == [
        "no code gives the block 'nothing'",
        "no code gives the figure 'nobody'",
    ]


def test_a_date_header_is_read() -> None:
    """Read the date of the one Date header."""
    assert documents.header_dates(document()) == ["2026-03-04"]
    assert documents.header_dates("no header\n") == []


def test_stamping_writes_the_time_now() -> None:
    """Replace the Date header's date and time with the time now."""
    assert documents.stamped(document(date_header=COMMIT_HEADER)) == document()


def check(repository: Path) -> int:
    """Run the check on the repository."""
    return documents.main(["--check", str(repository)])


def test_a_new_document_dated_today_passes(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass a document not yet committed whose header is today's date."""
    write(
        repository,
        "README.md",
        document("<!-- figure: answer -->42<!-- end figure -->"),
    )
    assert check(repository) == 0
    assert (
        "documents examined: 1, generated blocks: 0, figures: 1, problems: 0"
        in capsys.readouterr().out
    )


def test_a_changed_document_not_dated_today_fails(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a changed document whose header is not today's date."""
    write(repository, "README.md", document(date_header=COMMIT_HEADER))
    assert check(repository) == 1
    assert (
        "README.md: changed, so its Date header must be 2026-03-04, not 2026-01-02"
        in (capsys.readouterr().out)
    )


def test_a_committed_document_dated_as_its_commit_passes(repository: Path) -> None:
    """Pass a committed, unchanged document dated as the commit that last touched it."""
    write(repository, "README.md", document(date_header=COMMIT_HEADER))
    commit(repository, "README.md")
    assert check(repository) == 0


def test_a_committed_document_dated_otherwise_fails(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a committed, unchanged document whose date is not its commit's."""
    write(repository, "README.md", document())
    commit(repository, "README.md")
    assert check(repository) == 1
    assert (
        "README.md: its Date header must be 2026-01-02, the date of the last commit"
        " that changed it, not 2026-03-04" in capsys.readouterr().out
    )


def test_the_commit_date_is_taken_in_utc(repository: Path) -> None:
    """Read a commit made late in the evening west of UTC as the next UTC day."""
    write(
        repository,
        "README.md",
        document(date_header="**Date:** 2026-01-03 01:00:00 UTC"),
    )
    git(repository, "add", "README.md")
    assert GIT is not None
    # The command is git with arguments written in the test.
    subprocess.run(  # noqa: S603  # nosec B603
        [GIT, "-C", str(repository), "commit", "--quiet", "-m", "Add"],
        check=True,
        env={**os.environ, "GIT_COMMITTER_DATE": "2026-01-02T22:00:00-05:00"},
    )
    assert check(repository) == 0


@pytest.mark.parametrize(
    ("date_headers", "reported"),
    [(0, "has no Date header"), (2, "has 2 Date headers, not one")],
)
def test_a_document_needs_one_date_header(
    repository: Path,
    capsys: pytest.CaptureFixture[str],
    date_headers: int,
    reported: str,
) -> None:
    """Fail a document with no Date header or with more than one."""
    write(repository, "README.md", "# Title\n" + f"\n{TODAY_HEADER}\n" * date_headers)
    assert check(repository) == 1
    assert f"README.md: {reported}" in capsys.readouterr().out


def test_a_stale_block_fails_the_check(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a document whose block is not what the code gives, and name it."""
    stale_text = document(
        "<!-- generated: demo -->", "", "old", "", "<!-- end generated -->"
    )
    write(repository, "README.md", stale_text)
    assert check(repository) == 1
    assert (
        "README.md: block demo is not what the code gives; run scripts/documents.py"
        in capsys.readouterr().out
    )
    assert (repository / "README.md").read_text(encoding="utf-8") == stale_text


def test_an_unknown_block_fails_the_check(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a document naming a block no code gives."""
    write(
        repository, "docs/x.md", document("<!-- figure: nobody -->1<!-- end figure -->")
    )
    assert check(repository) == 1
    assert "docs/x.md: no code gives the figure 'nobody'" in capsys.readouterr().out


def test_no_document_fails_the_check(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a folder holding no document, since nothing was checked."""
    assert check(repository) == 1
    output = capsys.readouterr().out
    assert "documents examined: 0" in output
    assert "no documents found, so nothing was checked" in output


def test_a_folder_git_cannot_read_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail when git cannot say which documents changed."""
    write(tmp_path, "README.md", document())
    assert check(tmp_path) == 1
    assert "documents: git rev-parse failed: " in capsys.readouterr().out


def test_no_git_command_fails(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail when there is no git command to ask."""
    write(repository, "README.md", document())
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert check(repository) == 1
    assert "documents: no git command found" in capsys.readouterr().out


def test_writing_fills_blocks_and_stamps_changed_documents(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write the blocks and figures, and stamp only the documents that changed."""
    write(repository, "docs/kept.md", document(date_header=COMMIT_HEADER))
    commit(repository, "docs/kept.md")
    write(
        repository,
        "README.md",
        document(
            "<!-- figure: answer -->7<!-- end figure -->", date_header=COMMIT_HEADER
        ),
    )
    assert documents.main([str(repository)]) == 0
    assert (repository / "README.md").read_text(encoding="utf-8") == document(
        "<!-- figure: answer -->42<!-- end figure -->"
    )
    assert (repository / "docs/kept.md").read_text(encoding="utf-8") == document(
        date_header=COMMIT_HEADER
    )
    assert check(repository) == 0
    assert "problems: 0" in capsys.readouterr().out


def test_writing_a_committed_document_into_a_change_stamps_it(repository: Path) -> None:
    """Stamp a committed document that filling its blocks changes."""
    write(
        repository,
        "README.md",
        document(
            "<!-- figure: answer -->7<!-- end figure -->", date_header=COMMIT_HEADER
        ),
    )
    commit(repository, "README.md")
    assert documents.main([str(repository)]) == 0
    assert TODAY_HEADER in (repository / "README.md").read_text(encoding="utf-8")


def test_writing_reports_a_document_it_cannot_stamp(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail to stamp a changed document with no Date header, and say so."""
    write(repository, "README.md", "# Title\n")
    assert documents.main([str(repository)]) == 1
    assert "README.md: has no Date header" in capsys.readouterr().out


def test_writing_reports_an_unknown_block(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail to write a block no code gives, and say so."""
    write(
        repository, "README.md", document("<!-- figure: nobody -->1<!-- end figure -->")
    )
    assert documents.main([str(repository)]) == 1
    assert "README.md: no code gives the figure 'nobody'" in capsys.readouterr().out


def test_writing_fails_when_git_cannot_read_the_folder(tmp_path: Path) -> None:
    """Fail to stamp when git cannot say which documents changed."""
    write(tmp_path, "README.md", document())
    assert documents.main([str(tmp_path)]) == 1


def test_a_missing_folder_is_a_usage_error(tmp_path: Path) -> None:
    """Refuse a folder that is not there, with status 2."""
    with pytest.raises(SystemExit) as exit_info:
        documents.main(["--check", str(tmp_path / "missing")])
    assert exit_info.value.code == 2


def test_the_time_now_is_in_utc() -> None:
    """Give the time now with the UTC timezone."""
    assert REAL_UTC_NOW().tzinfo is UTC


def test_a_folder_below_the_top_of_its_repository_fails(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a folder inside a repository, where git gives other paths."""
    write(repository, "inner/README.md", document())
    assert check(repository / "inner") == 1
    assert "is not the top of a git repository" in capsys.readouterr().out


def test_a_document_git_does_not_track_fails(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a document git ignores, since it has no commit to be dated by."""
    write(repository, ".gitignore", "README.md\n")
    commit(repository, ".gitignore")
    write(repository, "README.md", document())
    assert check(repository) == 1
    assert "README.md: is unchanged and in no commit" in capsys.readouterr().out


def test_a_document_with_two_headers_is_not_stamped(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Leave a changed document with two Date headers as it is, and say so."""
    two_headers = document(COMMIT_HEADER)
    write(repository, "README.md", two_headers)
    assert documents.main([str(repository)]) == 1
    assert "README.md: has 2 Date headers, not one" in capsys.readouterr().out
    assert (repository / "README.md").read_text(encoding="utf-8") == two_headers


def test_running_the_file_as_a_script_exits_with_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exit with main's status when run as a program."""
    monkeypatch.setattr(sys, "argv", ["documents.py", "--check", str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(documents.__file__, run_name="__main__")
    assert exit_info.value.code == 1
