"""Tests for src/masterclock/app/exceptions.py.

The rules covered: every exception the module defines descends from
MasterClockError, which descends from Exception alone, so one clause catches
all of them and nothing else; a missing setting is a ConfigError; and
describe_error writes a validation error as each field's dotted location and
message, and any other error as its text, always on one line and never
empty.
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
        class_name: module_value
        for class_name, module_value in vars(exceptions).items()
        if inspect.isclass(module_value)
        and issubclass(module_value, BaseException)
        and module_value.__module__ == exceptions.__name__
    }


def test_every_exception_descends_from_the_root() -> None:
    """Make every exception defined here a MasterClockError."""
    defined_classes = defined_here()
    assert "MasterClockError" in defined_classes
    assert "RunLockError" in defined_classes
    non_root_names = [
        class_name
        for class_name, exception_class in defined_classes.items()
        if not issubclass(exception_class, exceptions.MasterClockError)
    ]
    assert non_root_names == []


def test_the_root_descends_from_exception_alone() -> None:
    """Derive the root from Exception alone, so it stays apart from built-in errors."""
    assert exceptions.MasterClockError.__bases__ == (Exception,)


def test_one_clause_catches_every_one_of_them() -> None:
    """Catch every exception defined here with one MasterClockError clause."""
    for class_name, exception_class in defined_here().items():
        with pytest.raises(exceptions.MasterClockError, match=class_name):
            raise exception_class(class_name)


def test_a_missing_setting_is_a_configuration_error() -> None:
    """Catch MissingSettingsError as a ConfigError."""
    assert issubclass(exceptions.MissingSettingsError, exceptions.ConfigError)


class Reading(BaseModel):
    """A small invented record, only for making validation errors."""

    phase_ps: int


class Record(BaseModel):
    """An invented record with a nested field and a list, for error locations."""

    reading_count: int
    readings: list[Reading]

    @field_validator("reading_count")
    @classmethod
    def count_is_not_thirteen(cls, reading_count: int) -> int:
        """Refuse 13, with a message that runs to two lines."""
        if reading_count == 13:
            raise ValueError("thirteen\nis refused")
        return reading_count

    @model_validator(mode="after")
    def some_readings_when_counted(self) -> Record:
        """Refuse a positive count with no readings, an error of no one field."""
        if self.reading_count > 0 and not self.readings:
            raise ValueError("a count with no readings")
        return self


def validation_error(record_input: object) -> ValidationError:
    """Return the ValidationError that validating ``record_input`` raises."""
    with pytest.raises(ValidationError) as raised:
        Record.model_validate(record_input)
    return raised.value


def test_a_validation_error_is_written_field_by_field() -> None:
    """Write each field's dotted location and message, joined by semicolons."""
    raised_error = validation_error(
        {"reading_count": "many", "readings": [{"phase_ps": 1}, {}]}
    )
    assert exceptions.describe_error(raised_error) == (
        "reading_count: Input should be a valid integer, unable to parse string as an"
        " integer; readings.1.phase_ps: Field required"
    )


def test_an_error_of_no_one_field_is_written_as_its_message() -> None:
    """Write a validation error with no location as its message alone."""
    # A record the validator accepts, so the error below is its refusal.
    assert (
        Record.model_validate({"reading_count": 0, "readings": []}).reading_count == 0
    )
    raised_error = validation_error({"reading_count": 2, "readings": []})
    assert (
        exceptions.describe_error(raised_error)
        == "Value error, a count with no readings"
    )


def test_line_breaks_are_written_as_escapes() -> None:
    """Keep a message with line breaks on one line, its breaks written as escapes."""
    raised_error = validation_error({"reading_count": 13, "readings": []})
    assert (
        exceptions.describe_error(raised_error)
        == r"reading_count: Value error, thirteen\nis refused"
    )
    assert exceptions.describe_error(ValueError("one\r\ntwo\rthree")) == (
        r"one\r\ntwo\rthree"
    )


@pytest.mark.parametrize(
    "textless_error", [ValueError(), ValueError(""), exceptions.LoggingError()]
)
def test_an_error_with_no_message_is_written_as_its_class_name(
    textless_error: Exception,
) -> None:
    """Write the class name when the error carries no text."""
    assert exceptions.describe_error(textless_error) == type(textless_error).__name__


@given(st.text())
def test_any_other_error_is_written_as_its_text_on_one_line(error_text: str) -> None:
    """Write an error's text, on one line and never empty."""
    description = exceptions.describe_error(ValueError(error_text))
    assert "\n" not in description
    assert "\r" not in description
    assert description
    if error_text and "\n" not in error_text and "\r" not in error_text:
        assert description == error_text
