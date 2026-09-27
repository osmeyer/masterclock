"""Tests for scripts/check_test_modules_alone.py.

The rule covered: every test module passes when it is run on its own.
"""

import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

import check_test_modules_alone

PASSING: Final = '"""Passes."""\n\n\ndef test_passes() -> None:\n    """Pass."""\n'
SETS: Final = (
    '"""Sets a value other modules rely on."""\n\nimport builtins\n\n\n'
    'def test_sets() -> None:\n    """Set it."""\n    builtins.SHARED_FOR_TEST = 1\n'
)
USES: Final = (
    '"""Relies on another module having run first."""\n\nimport builtins\n\n\n'
    'def test_uses() -> None:\n    """Use it."""\n'
    "    assert builtins.SHARED_FOR_TEST == 1\n"
)
EMPTY: Final = '"""Has no tests."""\n'


def write(directory: Path, name: str, text: str) -> None:
    """Write ``text`` to a file in ``directory``."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_modules_that_pass_alone_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass when every module passes on its own, and print the count."""
    write(tmp_path, "test_one.py", PASSING)
    write(tmp_path, "sub/test_two.py", PASSING)
    assert check_test_modules_alone.main([str(tmp_path)]) == 0
    assert "modules run: 2, failed: 0" in capsys.readouterr().out


def test_a_module_that_needs_another_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a module that passes only after another module has run."""
    write(tmp_path, "test_a_sets.py", SETS)
    write(tmp_path, "test_b_uses.py", USES)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert f"FAILED {tmp_path / 'test_b_uses.py'}" in output
    assert "modules run: 2, failed: 1" in output


def test_a_module_with_no_tests_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Count a test module that collects no tests as a failure."""
    write(tmp_path, "test_empty.py", EMPTY)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    assert "exit status 5" in capsys.readouterr().out


def test_no_test_modules_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail on a folder with no test modules, because nothing was checked."""
    write(tmp_path, "helper.py", PASSING)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    assert "no test modules found" in capsys.readouterr().out


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    """Exit with status 2 when the folder doesn't exist."""
    with pytest.raises(SystemExit) as stopped:
        check_test_modules_alone.main([str(tmp_path / "absent")])
    assert stopped.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    write(tmp_path, "test_one.py", PASSING)
    monkeypatch.setattr(sys, "argv", ["check_test_modules_alone.py", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(check_test_modules_alone.__file__, run_name="__main__")
    assert stopped.value.code == 0
