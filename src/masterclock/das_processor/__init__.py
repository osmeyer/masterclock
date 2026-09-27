"""das_processor: the masterclock program.

It has no behaviour yet. Until it does, running it prints a message saying so
on standard error and exits with status 1, so it can't be mistaken for a
working program.
"""

import sys
from typing import Final

NOT_IMPLEMENTED: Final = 1


def main() -> int:
    """Run das_processor and return its exit status."""
    print("das_processor: not implemented yet", file=sys.stderr)
    return NOT_IMPLEMENTED
