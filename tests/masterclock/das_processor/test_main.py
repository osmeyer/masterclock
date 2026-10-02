"""Tests for src/masterclock/das_processor/__main__.py.

The rule covered: ``python -m masterclock.das_processor`` calls the same
``main()`` as the declared ``das_processor`` command.
"""

import importlib
import runpy
import sys
from importlib.metadata import entry_points

import pytest

from masterclock import das_processor


def test_the_das_processor_command_is_declared_once() -> None:
    """Declare exactly one command named das_processor."""
    declared = entry_points(group="console_scripts", name="das_processor")
    assert len(declared) == 1


def test_the_module_calls_the_declared_main() -> None:
    """Call, from __main__, the same function the das_processor command calls."""
    (declared,) = entry_points(group="console_scripts", name="das_processor")
    module = importlib.import_module("masterclock.das_processor.__main__")
    assert module.main is declared.load()
    assert module.main is das_processor.main


def test_running_the_module_exits_with_the_status_main_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit with main's status when run with ``python -m``."""
    # Forget any earlier import, so the module runs fresh as a program.
    monkeypatch.delitem(
        sys.modules, "masterclock.das_processor.__main__", raising=False
    )
    monkeypatch.setattr(das_processor, "main", lambda: 7)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("masterclock.das_processor", run_name="__main__")
    assert stopped.value.code == 7
