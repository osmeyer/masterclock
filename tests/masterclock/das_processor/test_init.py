"""Tests for src/masterclock/das_processor/__init__.py.

The rule covered: until das_processor has any behaviour, running it fails with
a clear message, so it can't be mistaken for a working program.
"""

import pytest

from masterclock import das_processor


def test_main_says_it_is_not_implemented_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return a non-zero status and say on standard error that nothing is done."""
    assert das_processor.main() != 0
    captured = capsys.readouterr()
    assert "das_processor: not implemented yet" in captured.err
    assert captured.out == ""
