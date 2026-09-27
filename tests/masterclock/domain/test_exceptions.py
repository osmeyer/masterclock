"""Tests for src/masterclock/domain/exceptions.py.

The rule covered: every exception the module defines descends from
MasterClockError, so one clause catches all of them.
"""

import inspect

import pytest

from masterclock.app.exceptions import MasterClockError
from masterclock.domain import exceptions


def defined_here() -> dict[str, type[BaseException]]:
    """Return every exception class the module itself defines, by name."""
    return {
        name: value
        for name, value in vars(exceptions).items()
        if inspect.isclass(value)
        and issubclass(value, BaseException)
        and value.__module__ == exceptions.__name__
    }


def test_every_exception_descends_from_the_root() -> None:
    """Make every exception defined here a MasterClockError."""
    classes = defined_here()
    assert {"PhaseError", "FilterError"} <= classes.keys()
    outside = [
        name
        for name, value in classes.items()
        if not issubclass(value, MasterClockError)
    ]
    assert outside == []


def test_one_clause_catches_every_one_of_them() -> None:
    """Catch every exception defined here with one MasterClockError clause."""
    for name, value in defined_here().items():
        with pytest.raises(MasterClockError, match=name):
            raise value(name)
