"""Tests for scripts/check_test_modules_alone.py.

The rule covered: every test module passes when it is run on its own.
"""

import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

import check_test_modules_alone

PASSING_MODULE: Final = (
    '"""Passes."""\n\n\ndef test_passes() -> None:\n    """Pass."""\n'
)
SETTING_MODULE: Final = (
    '"""Sets a value other modules rely on."""\n\nimport builtins\n\n\n'
    'def test_sets() -> None:\n    """Set it."""\n    builtins.SHARED_FOR_TEST = 1\n'
)
USING_MODULE: Final = (
    '"""Relies on another module having run first."""\n\nimport builtins\n\n\n'
    'def test_uses() -> None:\n    """Use it."""\n'
    "    assert builtins.SHARED_FOR_TEST == 1\n"
)
EMPTY_MODULE: Final = '"""Has no tests."""\n'


def write_module(test_directory: Path, file_name: str, module_text: str) -> None:
    """Write ``module_text`` to a file in ``test_directory``."""
    module_file = test_directory / file_name
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text(module_text, encoding="utf-8")


def test_modules_that_pass_alone_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pass when every module passes on its own, and print the count."""
    write_module(tmp_path, "test_one.py", PASSING_MODULE)
    write_module(tmp_path, "sub/test_two.py", PASSING_MODULE)
    assert check_test_modules_alone.main([str(tmp_path)]) == 0
    assert "modules run: 2, failed: 0" in capsys.readouterr().out


def test_a_module_that_needs_another_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail a module that passes only after another module has run."""
    write_module(tmp_path, "test_a_sets.py", SETTING_MODULE)
    write_module(tmp_path, "test_b_uses.py", USING_MODULE)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    printed_output = capsys.readouterr().out
    assert f"FAILED {tmp_path / 'test_b_uses.py'}" in printed_output
    assert "modules run: 2, failed: 1" in printed_output


def test_a_module_with_no_tests_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Count a test module that collects no tests as a failure."""
    write_module(tmp_path, "test_empty.py", EMPTY_MODULE)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    assert "exit status 5" in capsys.readouterr().out


def test_no_test_modules_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail on a folder with no test modules, because nothing was checked."""
    write_module(tmp_path, "helper.py", PASSING_MODULE)
    assert check_test_modules_alone.main([str(tmp_path)]) == 1
    assert "no test modules found" in capsys.readouterr().out


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    """Exit with status 2 when the folder doesn't exist."""
    with pytest.raises(SystemExit) as program_exit:
        check_test_modules_alone.main([str(tmp_path / "absent")])
    assert program_exit.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    write_module(tmp_path, "test_one.py", PASSING_MODULE)
    monkeypatch.setattr(sys, "argv", ["check_test_modules_alone.py", str(tmp_path)])
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(check_test_modules_alone.__file__, run_name="__main__")
    assert program_exit.value.code == 0
