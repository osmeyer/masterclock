"""Report every module, class and function that has no docstring.

ruff's D rules skip some definitions: with the numpy convention they don't ask
for a docstring on ``__init__``, and they never ask for one on private or nested
functions or on the methods of private classes. This script reads each file's
syntax tree and checks every module, class, method and function, whatever its
name, nested ones included. An empty docstring counts as missing.

Give it the folders to search::

    uv run --frozen python scripts/check_docstrings.py scripts src tests

It prints how many files and definitions it examined. It exits 0 when every
definition has a docstring, and 1 when any is missing, when a file can't be
read or parsed, or when no Python files were found. A usage error exits 2.
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

PASSED: Final = 0
FAILED: Final = 1

type Definition = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _directory(cli_argument: str) -> Path:
    """Convert a command-line argument to the path of an existing folder."""
    folder = Path(cli_argument)
    if not folder.is_dir():
        refusal = f"not a directory: {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return folder


def python_files(directories: Sequence[Path]) -> list[Path]:
    """Return every ``.py`` file under the given folders, sorted.

    A folder whose name ends in ``.py`` is not a file and is left out.
    """
    return sorted(
        python_file
        for folder in directories
        for python_file in folder.rglob("*.py")
        if python_file.is_file()
    )


def _definitions(tree: ast.Module) -> Iterator[Definition]:
    """Yield the module itself and every class and function in it."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def _describe(node: Definition) -> str:
    """Give the line number and name of a definition for the report."""
    if isinstance(node, ast.Module):
        return "1: module"
    definition_kind = "class" if isinstance(node, ast.ClassDef) else "function"
    return f"{node.lineno}: {definition_kind} {node.name}"


def check_files(checked_files: Sequence[Path]) -> tuple[int, list[str]]:
    """Check each file and return the definitions examined and the problems found."""
    examined = 0
    problems: list[str] = []
    for python_file in checked_files:
        try:
            source = python_file.read_bytes()
        except OSError as read_error:
            problems.append(f"{python_file}: cannot be read: {read_error}")
            continue
        try:
            tree = ast.parse(source, filename=str(python_file))
        except SyntaxError as parse_error:
            problems.append(f"{python_file}: cannot be parsed: {parse_error}")
            continue
        for node in _definitions(tree):
            examined += 1
            if not ast.get_docstring(node):
                problems.append(f"{python_file}:{_describe(node)} has no docstring")
    return examined, problems


def main(argv: Sequence[str] | None = None) -> int:
    """Check the folders named on the command line and return the exit status."""
    parser = argparse.ArgumentParser(
        description="Report modules, classes and functions without a docstring."
    )
    parser.add_argument(
        "directories", nargs="+", type=_directory, help="folders to search"
    )
    directories: list[Path] = parser.parse_args(argv).directories
    found_files = python_files(directories)
    examined, problems = check_files(found_files)
    for problem_line in problems:
        print(problem_line)
    print(
        f"check_docstrings: files examined: {len(found_files)}, "
        f"definitions examined: {examined}, problems: {len(problems)}"
    )
    if not found_files:
        print("check_docstrings: no Python files found, so nothing was checked")
        return FAILED
    return FAILED if problems else PASSED


if __name__ == "__main__":
    sys.exit(main())
