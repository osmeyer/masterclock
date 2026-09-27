"""Tests for src/masterclock/app/exceptions.py.

The rules covered: every exception the module defines descends from
MasterClockError, which descends from Exception alone, so one clause catches
all of them and nothing else; and describe_error writes a validation error as
each field's dotted location and message, and any other error as its text,
always on one line and never empty.
"""

import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from masterclock.app import exceptions


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
    assert "MasterClockError" in classes
    assert "RunLockError" in classes
    outside = [
        name
        for name, value in classes.items()
        if not issubclass(value, exceptions.MasterClockError)
    ]
    assert outside == []


def test_the_root_descends_from_exception_alone() -> None:
    """Derive the root from Exception only, apart from built-in errors."""
    assert exceptions.MasterClockError.__bases__ == (Exception,)


def test_one_clause_catches_every_one_of_them() -> None:
    """Catch every exception defined here with one MasterClockError clause."""
    for name, value in defined_here().items():
        with pytest.raises(exceptions.MasterClockError, match=name):
            raise value(name)


def test_a_missing_setting_is_a_configuration_error() -> None:
    """Catch MissingSettingsError as a ConfigError."""
    assert issubclass(exceptions.MissingSettingsError, exceptions.ConfigError)


class Reading(BaseModel):
    """A small invented record, only for making validation errors."""

    value: int


class Record(BaseModel):
    """An invented record with a nested field and a list, for error locations."""

    count: int
    readings: list[Reading]

    @field_validator("count")
    @classmethod
    def count_is_not_thirteen(cls, count: int) -> int:
        """Refuse 13, with a message that runs to two lines."""
        if count == 13:
            raise ValueError("thirteen\nis refused")
        return count

    @model_validator(mode="after")
    def some_readings_when_counted(self) -> Record:
        """Refuse a positive count with no readings, an error of no one field."""
        if self.count > 0 and not self.readings:
            raise ValueError("a count with no readings")
        return self


def validation_error(data: object) -> ValidationError:
    """Return the ValidationError that validating ``data`` as a Record raises."""
    with pytest.raises(ValidationError) as raised:
        Record.model_validate(data)
    return raised.value


def test_a_validation_error_is_written_field_by_field() -> None:
    """Write each field's dotted location and message, joined by semicolons."""
    error = validation_error({"count": "many", "readings": [{"value": 1}, {}]})
    assert exceptions.describe_error(error) == (
        "count: Input should be a valid integer, unable to parse string as an"
        " integer; readings.1.value: Field required"
    )


def test_an_error_of_no_one_field_is_written_as_its_message() -> None:
    """Write a validation error with no location as its message alone."""
    # A record the validator accepts, so the error below is its refusal.
    assert Record.model_validate({"count": 0, "readings": []}).count == 0
    error = validation_error({"count": 2, "readings": []})
    assert exceptions.describe_error(error) == "Value error, a count with no readings"


def test_line_breaks_are_written_as_escapes() -> None:
    """Keep a message with line breaks on one line, its breaks written as escapes."""
    error = validation_error({"count": 13, "readings": []})
    assert (
        exceptions.describe_error(error) == r"count: Value error, thirteen\nis refused"
    )
    assert exceptions.describe_error(ValueError("one\r\ntwo\rthree")) == (
        r"one\r\ntwo\rthree"
    )


@pytest.mark.parametrize(
    "error", [ValueError(), ValueError(""), exceptions.LoggingError()]
)
def test_an_error_with_no_message_is_written_as_its_class_name(
    error: Exception,
) -> None:
    """Write the class name when the error carries no text."""
    assert exceptions.describe_error(error) == type(error).__name__


@given(st.text())
def test_any_other_error_is_written_as_its_text_on_one_line(message: str) -> None:
    """Write an error's text, on one line and never empty."""
    written = exceptions.describe_error(ValueError(message))
    assert "\n" not in written
    assert "\r" not in written
    assert written
    if message and "\n" not in message and "\r" not in message:
        assert written == message
