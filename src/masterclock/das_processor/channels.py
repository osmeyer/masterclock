"""The RF channels of the DAS, one of which a run processes."""

from typing import Final, Literal

type RfChannel = Literal["a", "b"]
"""Name of an RF channel."""

RF_CHOICES: Final[tuple[RfChannel, ...]] = ("a", "b")
"""Every RF channel there is."""
