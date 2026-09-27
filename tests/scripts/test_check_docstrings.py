"""Tests for scripts/check_docstrings.py.

The rule covered: every module, class, method and function has a docstring,
public or private, nested or not, including ``__init__`` and the magic methods.
"""

import runpy
import sys
from pathlib import Path
from typing import Final

import check_docstrings
import pytest

COMPLETE: Final = '''"""A module."""


class Thing:
    """A class."""

    def __init__(self) -> None:
        """Make a thing."""

    def __eq__(self, other: object) -> bool:
        """Compare two things."""
        return True

    def _private(self) -> None:
        """Do something private."""


def outer() -> None:
    """Hold a nested function."""

    def inner() -> None:
        """Be nested."""


async def waiting() -> None:
    """Wait."""
'''


def write(directory: Path, name: str, text: str) -> Path:
    """Write ``text`` to a file in ``directory`` and return its path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_every_definition_with_a_docstring_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass a file where every definition has a docstring, and print the count."""
    write(tmp_path, "complete.py", COMPLETE)
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert (
        "files examined: 1, definitions examined: 8, problems: 0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("old", "new", "reported"),
    [
        ('"""A module."""\n', "", "1: module has no docstring"),
        ('    """A class."""\n', "    pass\n", "class Thing has no docstring"),
        (
            '        """Make a thing."""\n',
            "        pass\n",
            "function __init__ has no docstring",
        ),
        ('        """Compare two things."""\n', "", "function __eq__ has no docstring"),
        (
            '        """Do something private."""\n',
            "        pass\n",
            "function _private has no docstring",
        ),
        (
            '        """Be nested."""\n',
            "        pass\n",
            "function inner has no docstring",
        ),
        ('    """Wait."""\n', "    pass\n", "function waiting has no docstring"),
        ('"""A module."""', '""""""', "1: module has no docstring"),
    ],
    ids=[
        "module",
        "class",
        "__init__",
        "magic method",
        "private method",
        "nested function",
        "async function",
        "empty docstring",
    ],
)
def test_each_kind_of_missing_docstring_is_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    old: str,
    new: str,
    reported: str,
) -> None:
    """Fail when any one docstring is removed or empty, and name the definition."""
    assert old in COMPLETE
    path = write(tmp_path, "gap.py", COMPLETE.replace(old, new, 1))
    assert check_docstrings.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert f"{path}:" in output
    assert reported in output
    assert "problems: 1" in output


@pytest.mark.parametrize(
    "content",
    [b"def (:\n", b"x = '\xff'\n", b"x = 1\x00\n"],
    ids=["bad syntax", "not utf-8", "null byte"],
)
def test_a_file_that_cannot_be_parsed_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    """Fail on a file the parser rejects, rather than skipping it as clean."""
    path = tmp_path / "broken.py"
    path.write_bytes(content)
    assert check_docstrings.main([str(tmp_path)]) == 1
    assert f"{path}: cannot be parsed:" in capsys.readouterr().out


def test_a_file_that_cannot_be_read_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a file that can't be read, rather than stopping with a traceback."""
    path = write(tmp_path, "locked.py", COMPLETE)
    path.chmod(0)
    try:
        assert check_docstrings.main([str(tmp_path)]) == 1
    finally:
        path.chmod(0o644)
    assert f"{path}: cannot be read:" in capsys.readouterr().out


def test_a_folder_named_like_a_python_file_is_not_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Leave out a folder whose name ends in .py, since it holds no source itself."""
    write(tmp_path, "real.py", COMPLETE)
    (tmp_path / "folder.py").mkdir()
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert "files examined: 1," in capsys.readouterr().out


def test_files_in_subdirectories_are_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Examine Python files in nested folders."""
    write(tmp_path, "a/b/deep.py", COMPLETE)
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert "files examined: 1," in capsys.readouterr().out


def test_no_python_files_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail on a folder with no Python files, because nothing was checked."""
    write(tmp_path, "notes.txt", "not python")
    assert check_docstrings.main([str(tmp_path)]) == 1
    assert "no Python files found" in capsys.readouterr().out


def test_no_directory_given_is_a_usage_error() -> None:
    """Exit with status 2 when no folder is given."""
    with pytest.raises(SystemExit) as stopped:
        check_docstrings.main([])
    assert stopped.value.code == 2


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    """Exit with status 2 when the folder doesn't exist."""
    with pytest.raises(SystemExit) as stopped:
        check_docstrings.main([str(tmp_path / "absent")])
    assert stopped.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    write(tmp_path, "complete.py", COMPLETE)
    monkeypatch.setattr(sys, "argv", ["check_docstrings.py", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(check_docstrings.__file__, run_name="__main__")
    assert stopped.value.code == 0
