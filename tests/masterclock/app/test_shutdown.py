"""Tests for src/masterclock/app/shutdown.py.

The rules covered: a shutdown handler starts with nothing requested and
nothing installed; installing replaces the handlers of the configured signals
and no others, uninstalling puts back exactly what was there, and an install
that fails puts back whatever it had replaced; a configured signal, sent for
real, sets the shutdown flag and records which signal it was; a shutdown can
also be requested in code; waiting returns as soon as a shutdown is
requested, or when its timeout runs out.
"""

import os
import signal
import threading
import time
from collections.abc import Iterator
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import shutdown

WATCHED: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)


@pytest.fixture(autouse=True)
def restore_handlers() -> Iterator[None]:
    """Put back every watched signal's handler after each test."""
    before = {signum: signal.getsignal(signum) for signum in WATCHED}
    yield
    for signum, handler in before.items():
        signal.signal(signum, handler)


def send(handler: shutdown.ShutdownHandler, signum: signal.Signals) -> None:
    """Send ``signum`` to this process, once ``handler`` is known to catch it.

    Checked first, so a handler that is not installed fails the test instead
    of letting the signal stop the test run.
    """
    assert signal.getsignal(signum) == handler._handle_signal
    os.kill(os.getpid(), signum)


def test_the_default_signals_are_interrupt_terminate_and_hangup() -> None:
    """Treat SIGINT, SIGTERM and SIGHUP as shutdown requests by default."""
    assert shutdown.DEFAULT_SIGNALS == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    assert shutdown.ShutdownHandler().signals == shutdown.DEFAULT_SIGNALS


def test_a_new_handler_has_requested_and_installed_nothing() -> None:
    """Start with no request, no signal received, and no handler replaced."""
    before = {signum: signal.getsignal(signum) for signum in WATCHED}
    handler = shutdown.ShutdownHandler()
    assert not handler.shutdown_requested
    assert handler.received_signal is None
    assert not handler.installed
    assert {signum: signal.getsignal(signum) for signum in WATCHED} == before


@given(st.lists(st.sampled_from(WATCHED)))
def test_signals_are_kept_once_each_in_the_order_given(
    signals: list[signal.Signals],
) -> None:
    """Drop repeated signals, keeping each where it first appears."""
    expected = tuple(
        signum for index, signum in enumerate(signals) if signum not in signals[:index]
    )
    assert shutdown.ShutdownHandler(signals).signals == expected


def test_request_shutdown_sets_the_flag_without_a_signal() -> None:
    """Set the flag from code, leave the received signal empty, and repeat safely."""
    handler = shutdown.ShutdownHandler()
    handler.request_shutdown()
    handler.request_shutdown()
    assert handler.shutdown_requested
    assert handler.received_signal is None
    assert handler.wait(0)


def test_wait_returns_false_when_the_timeout_runs_out() -> None:
    """Return False once the timeout runs out with nothing requested."""
    handler = shutdown.ShutdownHandler()
    started = time.monotonic()
    assert not handler.wait(0.05)
    assert time.monotonic() - started >= 0.05


def test_install_replaces_only_the_configured_signals() -> None:
    """Replace the handlers of the configured signals and leave others alone."""
    untouched = signal.getsignal(signal.SIGINT)
    handler = shutdown.ShutdownHandler([signal.SIGTERM, signal.SIGUSR1])
    handler.install()
    assert handler.installed
    assert signal.getsignal(signal.SIGTERM) == handler._handle_signal
    assert signal.getsignal(signal.SIGUSR1) == handler._handle_signal
    assert signal.getsignal(signal.SIGINT) is untouched


def test_uninstall_puts_back_exactly_the_earlier_handlers() -> None:
    """Restore each earlier handler, even after installing twice."""

    def earlier(_signum: int, _frame: object) -> None:
        """Stand in for a handler installed before this one."""

    signal.signal(signal.SIGUSR1, earlier)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    handler = shutdown.ShutdownHandler([signal.SIGUSR1, signal.SIGTERM])
    handler.install()
    handler.install()
    handler.uninstall()
    assert not handler.installed
    assert signal.getsignal(signal.SIGUSR1) is earlier
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN


def test_uninstall_when_not_installed_changes_nothing() -> None:
    """Leave every handler as it was when there is nothing to restore."""
    before = {signum: signal.getsignal(signum) for signum in WATCHED}
    shutdown.ShutdownHandler().uninstall()
    assert {signum: signal.getsignal(signum) for signum in WATCHED} == before


@pytest.mark.parametrize("signum", shutdown.DEFAULT_SIGNALS)
def test_a_signal_sent_for_real_requests_a_shutdown(signum: signal.Signals) -> None:
    """Set the flag and record the signal when a configured signal arrives."""
    handler = shutdown.ShutdownHandler()
    handler.install()
    send(handler, signum)
    assert handler.shutdown_requested
    assert handler.received_signal is signum
    assert handler.wait(0)


def test_the_last_signal_received_is_the_one_recorded() -> None:
    """Record the most recent signal, and keep it through a request from code."""
    handler = shutdown.ShutdownHandler()
    handler.install()
    send(handler, signal.SIGTERM)
    send(handler, signal.SIGHUP)
    handler.request_shutdown()
    assert handler.received_signal is signal.SIGHUP


def test_uninstall_leaves_the_request_in_place() -> None:
    """Keep the flag and the signal set after the handlers are restored."""
    handler = shutdown.ShutdownHandler()
    handler.install()
    send(handler, signal.SIGTERM)
    handler.uninstall()
    assert handler.shutdown_requested
    assert handler.received_signal is signal.SIGTERM


def test_the_with_block_installs_and_restores() -> None:
    """Install on entering, give back the handler, and restore on leaving."""
    before = {signum: signal.getsignal(signum) for signum in WATCHED}
    handler = shutdown.ShutdownHandler()
    with handler as entered:
        assert entered is handler
        send(handler, signal.SIGHUP)
    assert not handler.installed
    assert handler.received_signal is signal.SIGHUP
    assert {signum: signal.getsignal(signum) for signum in WATCHED} == before


def test_the_with_block_restores_when_it_raises() -> None:
    """Restore the handlers and let the exception through when the block fails."""
    before = {signum: signal.getsignal(signum) for signum in WATCHED}
    handler = shutdown.ShutdownHandler()
    with pytest.raises(RuntimeError, match="stop here"), handler:
        raise RuntimeError("stop here")
    assert not handler.installed
    assert {signum: signal.getsignal(signum) for signum in WATCHED} == before


def test_nested_handlers_restore_in_turn() -> None:
    """Put each earlier handler back when nested blocks end."""
    before = signal.getsignal(signal.SIGTERM)
    outer = shutdown.ShutdownHandler([signal.SIGTERM])
    inner = shutdown.ShutdownHandler([signal.SIGTERM])
    with outer:
        with inner:
            send(inner, signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == outer._handle_signal
    assert signal.getsignal(signal.SIGTERM) is before
    assert inner.shutdown_requested
    assert not outer.shutdown_requested


def test_wait_wakes_when_a_signal_arrives() -> None:
    """Return True from a long wait soon after a signal is sent."""
    handler = shutdown.ShutdownHandler()
    handler.install()
    assert signal.getsignal(signal.SIGTERM) == handler._handle_signal
    sender = threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGTERM))
    started = time.monotonic()
    sender.start()
    woken = handler.wait(10)
    waited = time.monotonic() - started
    # The signal must arrive while this test's handler still catches it,
    # whatever the result, so wait for the sender before any assert.
    sender.join()
    assert woken
    assert 0.1 <= waited < 5
    assert handler.received_signal is signal.SIGTERM


def test_install_from_another_thread_is_refused_and_changes_nothing() -> None:
    """Refuse to install off the main thread, and leave a later install working."""
    earlier = signal.getsignal(signal.SIGUSR1)
    handler = shutdown.ShutdownHandler([signal.SIGUSR1])
    raised: list[BaseException] = []

    def install() -> None:
        """Try to install from this thread, keeping what it raises."""
        with pytest.raises(ValueError, match="main thread") as refused:
            handler.install()
        raised.append(refused.value)

    worker = threading.Thread(target=install)
    worker.start()
    worker.join()
    assert len(raised) == 1
    assert not handler.installed
    assert signal.getsignal(signal.SIGUSR1) is earlier
    handler.install()
    send(handler, signal.SIGUSR1)
    assert handler.received_signal is signal.SIGUSR1


def test_a_failed_install_puts_back_what_it_replaced() -> None:
    """Restore the signals replaced before one that cannot be caught."""

    def earlier(_signum: int, _frame: object) -> None:
        """Stand in for a handler installed before this one."""

    signal.signal(signal.SIGUSR1, earlier)
    handler = shutdown.ShutdownHandler([signal.SIGUSR1, signal.SIGKILL])
    with pytest.raises(OSError, match="Invalid argument"):
        handler.install()
    assert not handler.installed
    assert signal.getsignal(signal.SIGUSR1) is earlier
    handler.uninstall()
    assert signal.getsignal(signal.SIGUSR1) is earlier
