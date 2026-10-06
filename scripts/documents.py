"""Keep the documents true to the code, and each one's Date header true to git.

The documents are the README at the top of the project and every Markdown
file under ``docs/``, the reports in ``docs/reports/`` apart, which carry a
stamp of their own.

Between marker comments, which do not show when the document is read, a
document holds text the code gives (see ``scripts/document_content.py``): a
block, a whole table, code block or chart, on lines of its own::

    <!-- generated: NAME -->

    ...

    <!-- end generated -->

and a figure, one value inside a sentence::

    <!-- figure: NAME -->VALUE<!-- end figure -->

Each document has one header line ``**Date:** YYYY-MM-DD HH:MM:SS UTC``.
While the document differs from what is committed, as git shows it new,
staged or modified, its date must be today's UTC date. Once it is committed
and unchanged, its date must be the UTC date of the last commit that
changed it, by the committer date. The time is written, never checked.

Give it the project folder, the top of its git repository::

    uv run --frozen python scripts/documents.py .
    uv run --frozen python scripts/documents.py --check .

Without ``--check`` it writes every block and figure from the code, then
stamps the Date header of every document that differs from what is
committed with the time now. With ``--check`` it changes nothing: it fails
when a block or figure is not what the code gives, a name is one the code
does not give, or a Date header is missing, repeated or wrong. Either way
it prints each problem and how many documents, blocks and figures it
examined, and exits 0 with no problem and 1 with any, or when it found no
document. A usage error exits 2.
"""

import argparse
import re
import shutil
import subprocess  # nosec B404 - runs git, to read each document's date
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import document_content

if TYPE_CHECKING:
    from collections.abc import Sequence

PASSED: Final = 0
FAILED: Final = 1

BLOCK: Final[re.Pattern[str]] = re.compile(
    r"(?P<start><!-- generated: (?P<name>[\w-]+) -->\n\n)"
    r"(?P<body>.*?)"
    r"(?P<end>\n\n<!-- end generated -->)",
    re.DOTALL,
)
"""A block between its markers, a blank line inside each."""

FIGURE: Final[re.Pattern[str]] = re.compile(
    r"(?P<start><!-- figure: (?P<name>[\w-]+) -->)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- end figure -->)"
)
"""A figure between its markers, on one line."""

DATE_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^\*\*Date:\*\* (?P<day>\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2} UTC$", re.MULTILINE
)
"""A document's Date header line."""

REPORTS: Final = Path("docs/reports")
"""The folder of the reports, which are not documents here."""


class Filled(NamedTuple):
    """A document with every block and figure written from the code.

    Parameters
    ----------
    text : str
        The document's text, each block and figure the code's.
    blocks : int
        How many blocks it holds.
    figures : int
        How many figures it holds.
    stale : list of str
        Each block or figure whose old text was not the code's, named.
    problems : list of str
        Each block or figure no code gives, left as it was.
    """

    text: str
    blocks: int
    figures: int
    stale: list[str]
    problems: list[str]


class GitError(Exception):
    """Git could not be asked, or could not answer."""


def document_paths(project_folder: Path) -> list[Path]:
    """Give the project's documents.

    Parameters
    ----------
    project_folder : Path
        The top of the project.

    Returns
    -------
    list of Path
        The README when there is one, then every Markdown file under
        ``docs/`` but the reports, sorted.
    """
    readme = project_folder / "README.md"
    docs_files = sorted(
        markdown_file
        for markdown_file in (project_folder / "docs").rglob("*.md")
        if not markdown_file.is_relative_to(project_folder / REPORTS)
    )
    return [readme, *docs_files] if readme.is_file() else docs_files


def fill(document_text: str) -> Filled:
    """Write every block and figure of a document from the code.

    Parameters
    ----------
    document_text : str
        The document.

    Returns
    -------
    Filled
        The document as the code gives it, and what was found on the way.
    """
    stale: list[str] = []
    problems: list[str] = []
    counts = {"block": 0, "figure": 0}

    def replace(marked: re.Match[str], marker_kind: str) -> str:
        """Give one block or figure as the code gives it."""
        counts[marker_kind] += 1
        name = marked.group("name")
        sources = (
            document_content.BLOCKS
            if marker_kind == "block"
            else document_content.FIGURES
        )
        source = sources.get(name)
        if source is None:
            problems.append(f"no code gives the {marker_kind} {name!r}")
            return marked.group(0)
        code_text = source()
        if code_text != marked.group("body"):
            stale.append(f"{marker_kind} {name}")
        return marked.group("start") + code_text + marked.group("end")

    with_blocks = BLOCK.sub(lambda marked: replace(marked, "block"), document_text)
    with_figures = FIGURE.sub(lambda marked: replace(marked, "figure"), with_blocks)
    return Filled(with_figures, counts["block"], counts["figure"], stale, problems)


def header_dates(document_text: str) -> list[str]:
    """Give the date of every Date header of a document.

    Parameters
    ----------
    document_text : str
        The document.

    Returns
    -------
    list of str
        Each header's date, ``YYYY-MM-DD``, in order.
    """
    return [found.group("day") for found in DATE_HEADER.finditer(document_text)]


def utc_now() -> datetime:
    """Give the time now, in UTC.

    Returns
    -------
    datetime
        The time now, with the UTC timezone.
    """
    return datetime.now(UTC)


def stamped(document_text: str) -> str:
    """Write the time now into a document's Date header.

    Parameters
    ----------
    document_text : str
        The document, with one Date header.

    Returns
    -------
    str
        The document, its Date header giving the time now, to the second.
    """
    header_line = f"**Date:** {utc_now():%Y-%m-%d %H:%M:%S} UTC"
    return DATE_HEADER.sub(header_line, document_text)


def _git(project_folder: Path, *git_arguments: str) -> str:
    """Run git in the project folder and give what it prints.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.
    *git_arguments : str
        The git command and its arguments.

    Returns
    -------
    str
        What git printed.

    Raises
    ------
    GitError
        If there is no git command, or git fails.
    """
    git_command = shutil.which("git")
    if git_command is None:
        message = "no git command found"
        raise GitError(message)
    # The command is git, found on the path, with arguments this script makes.
    finished = subprocess.run(  # noqa: S603  # nosec B603
        [git_command, "-C", str(project_folder), *git_arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        message = f"git {git_arguments[0]} failed: {' '.join(finished.stderr.split())}"
        raise GitError(message)
    return finished.stdout


def changed_paths(project_folder: Path, relative_paths: Sequence[str]) -> set[str]:
    """Give the documents that differ from what is committed.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.
    relative_paths : sequence of str
        The documents, relative to it.

    Returns
    -------
    set of str
        Those git shows as new, staged or modified.

    Raises
    ------
    GitError
        If git cannot be asked, or the folder is not the top of its
        repository, where git gives paths from.
    """
    if _git(project_folder, "rev-parse", "--show-prefix").strip():
        message = f"{project_folder} is not the top of a git repository"
        raise GitError(message)
    status_text = _git(
        project_folder,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *relative_paths,
    )
    return {entry[3:] for entry in status_text.split("\0") if entry}


def commit_day(project_folder: Path, relative_path: str) -> str | None:
    """Give the UTC date of the last commit that changed a document.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.
    relative_path : str
        The document, relative to it.

    Returns
    -------
    str or None
        The committer date's day in UTC, ``YYYY-MM-DD``; ``None`` when no
        commit holds the document.

    Raises
    ------
    GitError
        If git cannot be asked.
    """
    committed = _git(project_folder, "log", "-1", "--format=%cI", "--", relative_path)
    if not committed.strip():
        return None
    return f"{datetime.fromisoformat(committed.strip()).astimezone(UTC):%Y-%m-%d}"


def header_count_problem(header_count: int) -> str:
    """Say what is wrong with a document that has other than one Date header.

    Parameters
    ----------
    header_count : int
        How many Date headers it has.

    Returns
    -------
    str
        That it has none, or how many it has.
    """
    if not header_count:
        return "has no Date header"
    return f"has {header_count} Date headers, not one"


def date_problem(
    project_folder: Path, relative_path: str, document_text: str, *, changed: bool
) -> str | None:
    """Say what is wrong with a document's Date header, if anything.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.
    relative_path : str
        The document, relative to it.
    document_text : str
        The document.
    changed : bool
        Whether it differs from what is committed.

    Returns
    -------
    str or None
        What is wrong; ``None`` when nothing is.

    Raises
    ------
    GitError
        If git cannot be asked.
    """
    dates = header_dates(document_text)
    if len(dates) != 1:
        return header_count_problem(len(dates))
    if changed:
        today = f"{utc_now():%Y-%m-%d}"
        if dates[0] != today:
            return f"changed, so its Date header must be {today}, not {dates[0]}"
        return None
    committed_day = commit_day(project_folder, relative_path)
    if committed_day is None:
        return "is unchanged and in no commit: git does not track it"
    if dates[0] != committed_day:
        return (
            f"its Date header must be {committed_day}, the date of the last commit"
            f" that changed it, not {dates[0]}"
        )
    return None


class Examined(NamedTuple):
    """What a pass over the documents examined and found.

    Parameters
    ----------
    documents : int
        How many documents.
    blocks : int
        How many blocks they hold.
    figures : int
        How many figures they hold.
    problems : list of str
        Each problem, naming its document.
    """

    documents: int
    blocks: int
    figures: int
    problems: list[str]


def check_documents(project_folder: Path) -> Examined:
    """Check every document, changing nothing.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.

    Returns
    -------
    Examined
        What was examined, and every problem: a stale or unknown block or
        figure, or a Date header missing, repeated or wrong.

    Raises
    ------
    GitError
        If git cannot be asked.
    """
    paths = document_paths(project_folder)
    relative_paths = [path.relative_to(project_folder).as_posix() for path in paths]
    changed = changed_paths(project_folder, relative_paths)
    blocks = figures = 0
    problems: list[str] = []
    for relative_path in relative_paths:
        document_text = (project_folder / relative_path).read_text(encoding="utf-8")
        filled = fill(document_text)
        blocks += filled.blocks
        figures += filled.figures
        found = [
            *filled.problems,
            *(
                f"{stale_name} is not what the code gives; run scripts/documents.py"
                for stale_name in filled.stale
            ),
            date_problem(
                project_folder,
                relative_path,
                document_text,
                changed=relative_path in changed,
            ),
        ]
        problems += [f"{relative_path}: {problem}" for problem in found if problem]
    return Examined(len(paths), blocks, figures, problems)


def write_documents(project_folder: Path) -> Examined:
    """Write every block and figure, then stamp every changed document.

    Parameters
    ----------
    project_folder : Path
        The top of the project's git repository.

    Returns
    -------
    Examined
        What was examined, and every problem: an unknown block or figure,
        or a changed document without exactly one Date header, which is
        then not stamped.

    Raises
    ------
    GitError
        If git cannot be asked.
    """
    paths = document_paths(project_folder)
    blocks = figures = 0
    problems: list[str] = []
    for path in paths:
        filled = fill(path.read_text(encoding="utf-8"))
        blocks += filled.blocks
        figures += filled.figures
        relative_path = path.relative_to(project_folder).as_posix()
        problems += [f"{relative_path}: {problem}" for problem in filled.problems]
        if filled.stale:
            path.write_text(filled.text, encoding="utf-8")
    relative_paths = [path.relative_to(project_folder).as_posix() for path in paths]
    for relative_path in sorted(changed_paths(project_folder, relative_paths)):
        path = project_folder / relative_path
        document_text = path.read_text(encoding="utf-8")
        header_count = len(header_dates(document_text))
        if header_count != 1:
            problems.append(f"{relative_path}: {header_count_problem(header_count)}")
            continue
        path.write_text(stamped(document_text), encoding="utf-8")
    return Examined(len(paths), blocks, figures, problems)


def _project_folder(cli_argument: str) -> Path:
    """Convert a command-line argument to the path of an existing folder."""
    folder = Path(cli_argument)
    if not folder.is_dir():
        refusal = f"not a directory: {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return folder


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the documents of the project folder, and give the exit status.

    Parameters
    ----------
    argv : sequence of str or None, optional
        The arguments, without the program name; ``None`` for
        ``sys.argv[1:]``.

    Returns
    -------
    int
        0 with no problem; 1 with any, when git cannot be asked, or when
        there is no document.
    """
    parser = argparse.ArgumentParser(
        description="Write or check the documents' generated text and Date headers."
    )
    parser.add_argument(
        "--check", action="store_true", help="check only; change nothing"
    )
    parser.add_argument("folder", type=_project_folder, help="the project folder")
    cli_options = parser.parse_args(argv)
    try:
        examined = (
            check_documents(cli_options.folder)
            if cli_options.check
            else write_documents(cli_options.folder)
        )
    except GitError as exc:
        print(f"documents: {exc}")
        return FAILED
    for problem in examined.problems:
        print(problem)
    print(
        f"documents: documents examined: {examined.documents}, generated blocks:"
        f" {examined.blocks}, figures: {examined.figures}, problems:"
        f" {len(examined.problems)}"
    )
    if not examined.documents:
        print("documents: no documents found, so nothing was checked")
        return FAILED
    return FAILED if examined.problems else PASSED


if __name__ == "__main__":
    sys.exit(main())
