"""How the reference clocks are named.

A reference clock is named by a fixed prefix followed by one digit.
"""

import re
from typing import Final

REFERENCE_PREFIX: Final[str] = "mc"
"""What every reference clock's name begins with."""

REFERENCE_PATTERN: Final[str] = rf"^{REFERENCE_PREFIX}[0-9]$"
"""How a reference clock's name is spelled: the prefix and one digit.

Written for pydantic's ``Field(pattern=...)``, where ``$`` is the end of the
text. Python's :mod:`re` would also let ``$`` match before a final newline.
"""


def is_reference(clock_name: str) -> bool:
    """Tell whether a clock's name is a reference's: the prefix and one digit.

    Parameters
    ----------
    clock_name : str
        A clock name.

    Returns
    -------
    bool
        Whether the whole name matches :data:`REFERENCE_PATTERN`. A name that
        only starts with the prefix is not a reference's.

    Examples
    --------
    >>> is_reference("mc2"), is_reference("mc12"), is_reference("mcq")
    (True, False, False)
    """
    return re.fullmatch(REFERENCE_PATTERN, clock_name) is not None
