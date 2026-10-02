"""Remove the files and folders that the development tools generate.

Only the names in the constants below are removed. The script never works from
``.gitignore``, which also covers deployment configuration and data.
``__pycache__`` folders are removed wherever they are, except inside the folders
in ``NEVER_ENTERED``, and inside the folders in ``KEPT_AT_TOP`` at the top of the
project, which hold configuration, data and logs. The other names are removed
only at the top of the project. A symbolic link is removed as a link; its target
is left alone.

Give it the project folder::

    uv run --frozen python scripts/clean.py /path/to/masterclock

It prints each path it removes and how many there were. A folder without
``pyproject.toml`` is refused with exit status 2.
"""

import argparse
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

CACHE_FOLDER: Final = "__pycache__"
NEVER_ENTERED: Final = frozenset({".git", ".venv"})
KEPT_AT_TOP: Final = frozenset({"etc", "data", "logs"})
TOP_LEVEL_NAMES: Final = (
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "htmlcov",
    "mutants",
    "build",
    "dist",
    ".coverage",
    ".coverage.*",
)


def _project_root(cli_argument: str) -> Path:
    """Convert a command-line argument to a folder that holds ``pyproject.toml``."""
    project_folder = Path(cli_argument)
    if not (project_folder / "pyproject.toml").is_file():
        refusal = f"not a project folder (no pyproject.toml): {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return project_folder


def _cache_folders(folder: Path, skipped: frozenset[str]) -> list[Path]:
    """Return the ``__pycache__`` folders under ``folder``, not entering ``skipped``."""
    cache_folders_found: list[Path] = []
    for folder_entry in folder.iterdir():
        if (
            not folder_entry.is_dir()
            or folder_entry.is_symlink()
            or folder_entry.name in skipped
        ):
            continue
        if folder_entry.name == CACHE_FOLDER:
            cache_folders_found.append(folder_entry)
        else:
            cache_folders_found.extend(_cache_folders(folder_entry, NEVER_ENTERED))
    return cache_folders_found


def generated_paths(project_folder: Path) -> list[Path]:
    """Return every generated path under ``project_folder`` that the script removes.

    A ``__pycache__`` folder inside a folder that is removed whole is left out,
    since removing the outer folder removes it too.
    """
    top_level_paths = [
        folder_entry
        for folder_entry in project_folder.iterdir()
        if any(
            fnmatch(folder_entry.name, name_pattern) for name_pattern in TOP_LEVEL_NAMES
        )
    ]
    cache_folders = [
        cache_folder
        for cache_folder in _cache_folders(project_folder, NEVER_ENTERED | KEPT_AT_TOP)
        if not any(
            cache_folder.is_relative_to(removed_folder)
            for removed_folder in top_level_paths
        )
    ]
    return sorted(top_level_paths + cache_folders)


def remove(removed_path: Path) -> None:
    """Remove a file, a folder, or a symbolic link without following it."""
    if removed_path.is_dir() and not removed_path.is_symlink():
        shutil.rmtree(removed_path)
    else:
        removed_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Remove the generated paths in the project folder and return the exit status."""
    parser = argparse.ArgumentParser(
        description="Remove the files and folders the development tools generate."
    )
    parser.add_argument("root", type=_project_root, help="the project folder")
    project_folder: Path = parser.parse_args(argv).root
    removed_paths = generated_paths(project_folder)
    for removed_path in removed_paths:
        remove(removed_path)
        print(f"removed {removed_path}")
    print(f"clean: paths removed: {len(removed_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
