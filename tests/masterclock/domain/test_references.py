"""Tests for src/masterclock/domain/references.py.

The rule covered: a reference clock's name is the prefix and one ASCII digit,
and nothing else, as pydantic checks the pattern and as is_reference tells.
"""

from typing import Annotated, Final

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from masterclock.domain import references

REFERENCE_NAME: Final[TypeAdapter[str]] = TypeAdapter(
    Annotated[str, Field(pattern=references.REFERENCE_PATTERN)]
)
"""A reference name checked as a model field checks it."""


@pytest.mark.parametrize("digit", "0123456789")
def test_the_prefix_and_a_digit_name_a_reference(digit: str) -> None:
    """Accept the prefix followed by any one digit."""
    reference_name = f"{references.REFERENCE_PREFIX}{digit}"
    assert REFERENCE_NAME.validate_python(reference_name) == reference_name
    assert references.is_reference(reference_name)


@pytest.mark.parametrize(
    "candidate_name", ["mc", "mc10", "mca", "Mc1", "xmc1", "mc1 ", "mc\u0661", "mc1\n"]
)
def test_nothing_else_names_a_reference(candidate_name: str) -> None:
    """Refuse no digit, two digits, other letters or digits, or anything more."""
    with pytest.raises(ValidationError, match="should match pattern"):
        REFERENCE_NAME.validate_python(candidate_name)
    assert not references.is_reference(candidate_name)
