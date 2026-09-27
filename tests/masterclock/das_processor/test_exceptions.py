"""Tests for src/masterclock/das_processor/exceptions.py.

The rules covered: every exception the module defines descends from
MasterClockError, so one clause catches all of them; and every reason for
refusing a line names its own kind, distinct from every other, and the base
for them names none.
"""

import inspect

import pytest

from masterclock.app.exceptions import MasterClockError
from masterclock.das_processor import exceptions


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
    assert {"DataFileError", "LateLineError"} <= classes.keys()
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


def refusal_reasons() -> dict[str, type[exceptions.RefusedLineError]]:
    """Return every class below RefusedLineError that the module defines."""
    return {
        name: value
        for name, value in defined_here().items()
        if issubclass(value, exceptions.RefusedLineError)
        and value is not exceptions.RefusedLineError
    }


def test_every_refusal_reason_names_its_own_kind() -> None:
    """Give each reason a non-empty kind of its own, unlike any other's."""
    reasons = refusal_reasons()
    assert "MalformedLineError" in reasons
    kinds = {name: value.__dict__.get("kind") for name, value in reasons.items()}
    missing = [name for name, kind in kinds.items() if not kind]
    assert missing == []
    assert len(set(kinds.values())) == len(kinds)


@pytest.mark.parametrize(
    ("reason", "kind"),
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
    reason: type[exceptions.RefusedLineError], kind: str
) -> None:
    """Keep the word each reason is logged by, since logs are read by it."""
    assert reason.kind == kind


def test_the_base_for_refusals_names_no_kind() -> None:
    """Fail on reading the kind of the base itself, so no reason can borrow one."""
    with pytest.raises(AttributeError, match="kind"):
        _ = exceptions.RefusedLineError.kind
