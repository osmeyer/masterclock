"""How the reference clocks are named.

A reference clock is named by a fixed prefix followed by one digit.
"""

from typing import Final

REFERENCE_PREFIX: Final[str] = "mc"
"""What every reference clock's name begins with."""

REFERENCE_PATTERN: Final[str] = rf"^{REFERENCE_PREFIX}[0-9]$"
"""How a reference clock's name is spelled: the prefix and one digit.

Written for pydantic's ``Field(pattern=...)``, where ``$`` is the end of the
text. Python's :mod:`re` would also let ``$`` match before a final newline.
"""
