"""Tests for scripts/check_docstrings.py.

The rules covered: every module, class, method and function has a docstring,
public or private, nested or not, including ``__init__`` and the magic methods,
and an empty docstring counts as missing; Python files are found in nested
folders, and a folder named like one is left out; a file that cannot be read
or parsed fails, and so does finding no Python files; no folder, or a missing
one, is a usage error with status 2; and run as a program, the script exits
with the status main returns.
"""

import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

import check_docstrings

COMPLETE_MODULE: Final = '''"""A module."""


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


def write_source(source_directory: Path, file_name: str, source_text: str) -> Path:
    """Write ``source_text`` to a file in ``source_directory`` and return its path."""
    source_file = source_directory / file_name
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(source_text, encoding="utf-8")
    return source_file


def test_every_definition_with_a_docstring_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass a file where every definition has a docstring, and print the count."""
    write_source(tmp_path, "complete.py", COMPLETE_MODULE)
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert (
        "files examined: 1, definitions examined: 8, problems: 0"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("docstring_text", "replacement_text", "reported_problem"),
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
    docstring_text: str,
    replacement_text: str,
    reported_problem: str,
) -> None:
    """Fail when any one docstring is removed or empty, and name the definition."""
    assert docstring_text in COMPLETE_MODULE
    source_file = write_source(
        tmp_path, "gap.py", COMPLETE_MODULE.replace(docstring_text, replacement_text, 1)
    )
    assert check_docstrings.main([str(tmp_path)]) == 1
    printed_output = capsys.readouterr().out
    assert f"{source_file}:" in printed_output
    assert reported_problem in printed_output
    assert "problems: 1" in printed_output


@pytest.mark.parametrize(
    "file_bytes",
    [b"def (:\n", b"x = '\xff'\n", b"x = 1\x00\n"],
    ids=["bad syntax", "not utf-8", "null byte"],
)
def test_a_file_that_cannot_be_parsed_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], file_bytes: bytes
) -> None:
    """Fail on a file the parser rejects, rather than skipping it as clean."""
    source_file = tmp_path / "broken.py"
    source_file.write_bytes(file_bytes)
    assert check_docstrings.main([str(tmp_path)]) == 1
    assert f"{source_file}: cannot be parsed:" in capsys.readouterr().out


def test_a_file_that_cannot_be_read_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a file that can't be read, rather than stopping with a traceback."""
    source_file = write_source(tmp_path, "locked.py", COMPLETE_MODULE)
    source_file.chmod(0)
    try:
        assert check_docstrings.main([str(tmp_path)]) == 1
    finally:
        source_file.chmod(0o644)
    assert f"{source_file}: cannot be read:" in capsys.readouterr().out


def test_a_folder_named_like_a_python_file_is_not_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Leave out a folder whose name ends in .py, since it holds no source itself."""
    write_source(tmp_path, "real.py", COMPLETE_MODULE)
    (tmp_path / "folder.py").mkdir()
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert "files examined: 1," in capsys.readouterr().out


def test_files_in_subdirectories_are_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Examine Python files in nested folders."""
    write_source(tmp_path, "a/b/deep.py", COMPLETE_MODULE)
    assert check_docstrings.main([str(tmp_path)]) == 0
    assert "files examined: 1," in capsys.readouterr().out


def test_no_python_files_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail on a folder with no Python files, because nothing was checked."""
    write_source(tmp_path, "notes.txt", "not python")
    assert check_docstrings.main([str(tmp_path)]) == 1
    assert "no Python files found" in capsys.readouterr().out


def test_no_directory_given_is_a_usage_error() -> None:
    """Exit with status 2 when no folder is given."""
    with pytest.raises(SystemExit) as program_exit:
        check_docstrings.main([])
    assert program_exit.value.code == 2


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    """Exit with status 2 when the folder doesn't exist."""
    with pytest.raises(SystemExit) as program_exit:
        check_docstrings.main([str(tmp_path / "absent")])
    assert program_exit.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    write_source(tmp_path, "complete.py", COMPLETE_MODULE)
    monkeypatch.setattr(sys, "argv", ["check_docstrings.py", str(tmp_path)])
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(check_docstrings.__file__, run_name="__main__")
    assert program_exit.value.code == 0
