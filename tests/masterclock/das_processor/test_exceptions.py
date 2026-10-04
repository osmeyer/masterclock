"""Tests for src/masterclock/das_processor/exceptions.py.

The rules covered: every exception the module defines descends from
MasterClockError, so one clause catches all of them; every reason for
refusing a line names its own kind, distinct from every other, and the base
for them names none; and each reason's word is fixed, since logs are
searched by it.
"""

import inspect

import pytest

from masterclock.app.exceptions import MasterClockError
from masterclock.das_processor import exceptions


def exceptions_defined_here() -> dict[str, type[BaseException]]:
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
    exception_classes = exceptions_defined_here()
    assert {"DataFileError", "LateLineError"} <= exception_classes.keys()
    not_master_clock_errors = [
        class_name
        for class_name, exception_class in exception_classes.items()
        if not issubclass(exception_class, MasterClockError)
    ]
    assert not_master_clock_errors == []


def test_one_clause_catches_every_one_of_them() -> None:
    """Catch every exception defined here with one MasterClockError clause."""
    for class_name, exception_class in exceptions_defined_here().items():
        with pytest.raises(MasterClockError, match=class_name):
            raise exception_class(class_name)


def refusal_reasons() -> dict[str, type[exceptions.RefusedLineError]]:
    """Return every class below RefusedLineError that the module defines."""
    return {
        class_name: exception_class
        for class_name, exception_class in exceptions_defined_here().items()
        if issubclass(exception_class, exceptions.RefusedLineError)
        and exception_class is not exceptions.RefusedLineError
    }


def test_every_refusal_reason_names_its_own_kind() -> None:
    """Give each reason a non-empty kind of its own, unlike any other's."""
    refusal_classes = refusal_reasons()
    assert "MalformedLineError" in refusal_classes
    refusal_kinds = {
        class_name: refusal_class.__dict__.get("refusal_kind")
        for class_name, refusal_class in refusal_classes.items()
    }
    without_kind = [
        class_name
        for class_name, refusal_kind in refusal_kinds.items()
        if not refusal_kind
    ]
    assert without_kind == []
    assert len(set(refusal_kinds.values())) == len(refusal_kinds)


@pytest.mark.parametrize(
    ("refusal_class", "refusal_kind"),
    [
        (exceptions.MalformedLineError, "malformed"),
        (exceptions.InconsistentLineError, "inconsistent"),
        (exceptions.WrongDayError, "wrong-day"),
        (exceptions.OutOfOrderError, "out-of-order"),
        (exceptions.DuplicatePairError, "duplicate"),
        (exceptions.LateLineError, "late"),
    ],
)
def test_each_refusal_reason_is_logged_by_its_word(
    refusal_class: type[exceptions.RefusedLineError], refusal_kind: str
) -> None:
    """Keep the word each reason is logged by, since logs are read by it."""
    assert refusal_class.refusal_kind == refusal_kind


def test_the_base_for_refusals_names_no_kind() -> None:
    """Fail on reading the kind of the base itself, so no reason can borrow one."""
    with pytest.raises(AttributeError, match="refusal_kind"):
        _ = exceptions.RefusedLineError.refusal_kind
