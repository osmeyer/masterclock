"""Run each test module on its own and report any that fail.

A test module can pass in the full run only because another module set
something up first. This script finds every ``test_*.py`` file under the folder
it is given, and runs pytest on each one separately, without coverage. A module
that collects no tests counts as a failure.

Give it the tests folder::

    uv run --frozen python scripts/check_test_modules_alone.py tests

It prints each failing module and how many modules it ran. It exits 0 when
every module passes alone, and 1 when any fails or no test modules were found.
A usage error exits 2.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

PASSED: Final = 0
FAILED: Final = 1


def _directory(text: str) -> Path:
    """Convert a command-line argument to the path of an existing folder."""
    path = Path(text)
    if not path.is_dir():
        message = f"not a directory: {text}"
        raise argparse.ArgumentTypeError(message)
    return path


def find_test_modules(directory: Path) -> list[Path]:
    """Return every test module under ``directory``, sorted."""
    return sorted(directory.rglob("test_*.py"))


def run_alone(module: Path) -> int:
    """Run pytest on one module by itself and return pytest's exit status."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(module),
        "--no-cov",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    # The command is a fixed list plus a path found under the tests folder.
    completed = subprocess.run(command, check=False)  # noqa: S603  # nosec B603
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run every test module in the folder on its own and return the exit status."""
    parser = argparse.ArgumentParser(
        description="Run each test module on its own and report any that fail."
    )
    parser.add_argument("directory", type=_directory, help="the tests folder")
    directory: Path = parser.parse_args(argv).directory
    modules = find_test_modules(directory)
    failures = 0
    for module in modules:
        status = run_alone(module)
        if status != PASSED:
            failures += 1
            print(f"FAILED {module} (pytest exit status {status})")
    print(f"check_test_modules_alone: modules run: {len(modules)}, failed: {failures}")
    if not modules:
        print("check_test_modules_alone: no test modules found, so nothing was checked")
        return FAILED
    return FAILED if failures else PASSED


if __name__ == "__main__":
    sys.exit(main())
