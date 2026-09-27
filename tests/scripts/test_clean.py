"""Tests for scripts/clean.py.

The rule covered: only the generated paths the script names are removed, never
configuration, data, logs, the virtual environment or git's own files.
"""

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import clean
import pytest

GENERATED: Final = (
    "src/pkg/__pycache__/m.cpython-314.pyc",
    "tests/__pycache__/t.cpython-314.pyc",
    "src/pkg/data/__pycache__/m.cpython-314.pyc",
    ".mypy_cache/x",
    ".ruff_cache/x",
    ".pytest_cache/x",
    ".hypothesis/x",
    "htmlcov/index.html",
    "mutants/x.py",
    "build/x",
    "dist/x",
    ".coverage",
    ".coverage.host.1.2",
)
KEPT: Final = (
    "pyproject.toml",
    "etc/site.ini",
    "etc/__pycache__/x",
    "data/run.dat",
    "data/run/__pycache__/x",
    "logs/run.log",
    "logs/__pycache__/x",
    ".venv/lib/__pycache__/m.cpython-314.pyc",
    ".git/__pycache__/x",
    "src/pkg/.coverage_notes",
    "src/pkg/build/x",
)


def make(root: Path, names: Sequence[str]) -> None:
    """Create each named file under ``root``, with any folders it needs."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


def test_only_generated_paths_are_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Remove every generated path and keep everything else."""
    make(tmp_path, GENERATED + KEPT)
    assert clean.main([str(tmp_path)]) == 0
    for name in GENERATED:
        assert not (tmp_path / name).exists(), name
    for name in KEPT:
        assert (tmp_path / name).exists(), name
    assert "paths removed: 13" in capsys.readouterr().out


def test_a_cache_folder_inside_a_generated_folder_is_removed_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Remove a generated folder whole, without listing the caches inside it too."""
    make(tmp_path, ("pyproject.toml", "mutants/scripts/__pycache__/m.pyc"))
    assert clean.main([str(tmp_path)]) == 0
    assert not (tmp_path / "mutants").exists()
    assert "paths removed: 1" in capsys.readouterr().out


def test_a_linked_cache_folder_is_unlinked_not_emptied(tmp_path: Path) -> None:
    """Remove a symbolic link named like a generated folder, keeping its target."""
    project = tmp_path / "project"
    target = tmp_path / "elsewhere"
    make(project, ("pyproject.toml",))
    make(target, ("keep.txt",))
    (project / ".mypy_cache").symlink_to(target, target_is_directory=True)
    assert clean.main([str(project)]) == 0
    assert not (project / ".mypy_cache").exists()
    assert (target / "keep.txt").exists()


def test_a_clean_project_removes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Remove nothing when nothing is generated, and report zero."""
    make(tmp_path, KEPT)
    assert clean.main([str(tmp_path)]) == 0
    assert "paths removed: 0" in capsys.readouterr().out


def test_a_folder_without_pyproject_is_refused(tmp_path: Path) -> None:
    """Exit with status 2, removing nothing, when the folder isn't a project."""
    make(tmp_path, (".mypy_cache/x",))
    with pytest.raises(SystemExit) as stopped:
        clean.main([str(tmp_path)])
    assert stopped.value.code == 2
    assert (tmp_path / ".mypy_cache/x").exists()


def test_a_missing_folder_is_a_usage_error(tmp_path: Path) -> None:
    """Exit with status 2 when the folder doesn't exist."""
    with pytest.raises(SystemExit) as stopped:
        clean.main([str(tmp_path / "absent")])
    assert stopped.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    make(tmp_path, ("pyproject.toml",))
    monkeypatch.setattr(sys, "argv", ["clean.py", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(clean.__file__, run_name="__main__")
    assert stopped.value.code == 0
