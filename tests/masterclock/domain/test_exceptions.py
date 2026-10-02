"""Tests for src/masterclock/domain/exceptions.py.

The rule covered: every exception the module defines descends from
MasterClockError, so one clause catches all of them.
"""

import inspect

import pytest

from masterclock.app.exceptions import MasterClockError
from masterclock.domain import exceptions


def exception_classes_defined_here() -> dict[str, type[BaseException]]:
    """Return every exception class the module itself defines, by name."""
    return {
        class_name: exception_class
        for class_name, exception_class in vars(exceptions).items()
        if inspect.isclass(exception_class)
        and issubclass(exception_class, BaseException)
        and exception_class.__module__ == exceptions.__name__
    }


def test_every_exception_descends_from_the_root() -> None:
    """Make every exception defined here a MasterClockError."""
    exception_classes = exception_classes_defined_here()
    assert {"PhaseError", "FilterError"} <= exception_classes.keys()
    not_descended = [
        class_name
        for class_name, exception_class in exception_classes.items()
        if not issubclass(exception_class, MasterClockError)
    ]
    assert not_descended == []


def test_one_clause_catches_every_one_of_them() -> None:
    """Catch every exception defined here with one MasterClockError clause."""
    for class_name, exception_class in exception_classes_defined_here().items():
        with pytest.raises(MasterClockError, match=class_name):
            raise exception_class(class_name)
