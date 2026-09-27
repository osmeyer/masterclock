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


def _project_root(text: str) -> Path:
    """Convert a command-line argument to a folder that holds ``pyproject.toml``."""
    path = Path(text)
    if not (path / "pyproject.toml").is_file():
        message = f"not a project folder (no pyproject.toml): {text}"
        raise argparse.ArgumentTypeError(message)
    return path


def _cache_folders(folder: Path, skipped: frozenset[str]) -> list[Path]:
    """Return the ``__pycache__`` folders under ``folder``, not entering ``skipped``."""
    found: list[Path] = []
    for child in folder.iterdir():
        if not child.is_dir() or child.is_symlink() or child.name in skipped:
            continue
        if child.name == CACHE_FOLDER:
            found.append(child)
        else:
            found.extend(_cache_folders(child, NEVER_ENTERED))
    return found


def generated_paths(root: Path) -> list[Path]:
    """Return every generated path under ``root`` that the script removes.

    A ``__pycache__`` folder inside a folder that is removed whole is left out,
    since removing the outer folder removes it too.
    """
    top = [
        child
        for child in root.iterdir()
        if any(fnmatch(child.name, pattern) for pattern in TOP_LEVEL_NAMES)
    ]
    caches = [
        cache
        for cache in _cache_folders(root, NEVER_ENTERED | KEPT_AT_TOP)
        if not any(cache.is_relative_to(folder) for folder in top)
    ]
    return sorted(top + caches)


def remove(path: Path) -> None:
    """Remove a file, a folder, or a symbolic link without following it."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Remove the generated paths in the project folder and return the exit status."""
    parser = argparse.ArgumentParser(
        description="Remove the files and folders the development tools generate."
    )
    parser.add_argument("root", type=_project_root, help="the project folder")
    root: Path = parser.parse_args(argv).root
    paths = generated_paths(root)
    for path in paths:
        remove(path)
        print(f"removed {path}")
    print(f"clean: paths removed: {len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
