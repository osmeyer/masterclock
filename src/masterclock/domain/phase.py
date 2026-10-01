"""The phase of a 5 MHz signal, as the measurements give it.

A phase is a whole number of picoseconds. It wraps at one period of the
signal, so a reading can only say where in the period it fell.
"""

from typing import Final

PHASE_PERIOD: Final[int] = 200_000
"""One period of a 5 MHz signal, in picoseconds."""

PHASE_MAX: Final[int] = PHASE_PERIOD - 1
"""The largest phase a reading can give, in picoseconds.

A whole period would be indistinguishable from zero.
"""
