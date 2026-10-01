"""Tests for src/masterclock/das_processor/channels.py.

The rule covered: the RF channels are a and b, the same names the type
allows.
"""

from typing import get_args

from masterclock.das_processor import channels


def test_the_channels_are_a_and_b() -> None:
    """Name the two channels, lowercase."""
    assert channels.RF_CHOICES == ("a", "b")


def test_the_choices_are_the_names_the_type_allows() -> None:
    """Offer exactly the names RfChannel admits, in its order."""
    assert get_args(channels.RfChannel.__value__) == channels.RF_CHOICES
