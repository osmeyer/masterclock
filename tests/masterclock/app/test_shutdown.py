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

WATCHED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)


@pytest.fixture(autouse=True)
def restore_handlers() -> Iterator[None]:
    """Put back every watched signal's handler after each test."""
    handlers_before = {signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS}
    yield
    for signum, saved_handler in handlers_before.items():
        signal.signal(signum, saved_handler)


def send(shutdown_handler: shutdown.ShutdownHandler, signum: signal.Signals) -> None:
    """Send ``signum`` to this process, once ``shutdown_handler`` is known to catch it.

    Checked first, so a handler that is not installed fails the test instead
    of letting the signal stop the test run.
    """
    assert signal.getsignal(signum) == shutdown_handler._handle_signal
    os.kill(os.getpid(), signum)


def test_the_default_signals_are_interrupt_terminate_and_hangup() -> None:
    """Treat SIGINT, SIGTERM and SIGHUP as shutdown requests by default."""
    assert shutdown.DEFAULT_SIGNALS == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    assert shutdown.ShutdownHandler().signals == shutdown.DEFAULT_SIGNALS


def test_a_new_handler_has_requested_and_installed_nothing() -> None:
    """Start with no request, no signal received, and no handler replaced."""
    handlers_before = {signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS}
    shutdown_handler = shutdown.ShutdownHandler()
    assert not shutdown_handler.shutdown_requested
    assert shutdown_handler.received_signal is None
    assert not shutdown_handler.installed
    assert {
        signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS
    } == handlers_before


@given(st.lists(st.sampled_from(WATCHED_SIGNALS)))
def test_signals_are_kept_once_each_in_the_order_given(
    given_signals: list[signal.Signals],
) -> None:
    """Drop repeated signals, keeping each where it first appears."""
    kept_signals = tuple(
        signum
        for signal_index, signum in enumerate(given_signals)
        if signum not in given_signals[:signal_index]
    )
    assert shutdown.ShutdownHandler(given_signals).signals == kept_signals


def test_request_shutdown_sets_the_flag_without_a_signal() -> None:
    """Set the flag from code, leave the received signal empty, and repeat safely."""
    shutdown_handler = shutdown.ShutdownHandler()
    shutdown_handler.request_shutdown()
    shutdown_handler.request_shutdown()
    assert shutdown_handler.shutdown_requested
    assert shutdown_handler.received_signal is None
    assert shutdown_handler.wait(0)


def test_wait_returns_false_when_the_timeout_runs_out() -> None:
    """Return False once the timeout runs out with nothing requested."""
    shutdown_handler = shutdown.ShutdownHandler()
    wait_started = time.monotonic()
    assert not shutdown_handler.wait(0.05)
    assert time.monotonic() - wait_started >= 0.05


def test_install_replaces_only_the_configured_signals() -> None:
    """Replace the handlers of the configured signals and leave others alone."""
    sigint_handler = signal.getsignal(signal.SIGINT)
    shutdown_handler = shutdown.ShutdownHandler([signal.SIGTERM, signal.SIGUSR1])
    shutdown_handler.install()
    assert shutdown_handler.installed
    assert signal.getsignal(signal.SIGTERM) == shutdown_handler._handle_signal
    assert signal.getsignal(signal.SIGUSR1) == shutdown_handler._handle_signal
    assert signal.getsignal(signal.SIGINT) is sigint_handler


def test_uninstall_puts_back_exactly_the_earlier_handlers() -> None:
    """Restore each earlier handler, even after installing twice."""

    def earlier_handler(_signum: int, _frame: object) -> None:
        """Stand in for a handler installed before this one."""

    signal.signal(signal.SIGUSR1, earlier_handler)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    shutdown_handler = shutdown.ShutdownHandler([signal.SIGUSR1, signal.SIGTERM])
    shutdown_handler.install()
    shutdown_handler.install()
    shutdown_handler.uninstall()
    assert not shutdown_handler.installed
    assert signal.getsignal(signal.SIGUSR1) is earlier_handler
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN


def test_uninstall_when_not_installed_changes_nothing() -> None:
    """Leave every handler as it was when there is nothing to restore."""
    handlers_before = {signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS}
    shutdown.ShutdownHandler().uninstall()
    assert {
        signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS
    } == handlers_before


@pytest.mark.parametrize("signum", shutdown.DEFAULT_SIGNALS)
def test_a_signal_sent_for_real_requests_a_shutdown(signum: signal.Signals) -> None:
    """Set the flag and record the signal when a configured signal arrives."""
    shutdown_handler = shutdown.ShutdownHandler()
    shutdown_handler.install()
    send(shutdown_handler, signum)
    assert shutdown_handler.shutdown_requested
    assert shutdown_handler.received_signal is signum
    assert shutdown_handler.wait(0)


def test_the_last_signal_received_is_the_one_recorded() -> None:
    """Record the most recent signal, and keep it through a request from code."""
    shutdown_handler = shutdown.ShutdownHandler()
    shutdown_handler.install()
    send(shutdown_handler, signal.SIGTERM)
    send(shutdown_handler, signal.SIGHUP)
    shutdown_handler.request_shutdown()
    assert shutdown_handler.received_signal is signal.SIGHUP


def test_uninstall_leaves_the_request_in_place() -> None:
    """Keep the flag and the signal set after the handlers are restored."""
    shutdown_handler = shutdown.ShutdownHandler()
    shutdown_handler.install()
    send(shutdown_handler, signal.SIGTERM)
    shutdown_handler.uninstall()
    assert shutdown_handler.shutdown_requested
    assert shutdown_handler.received_signal is signal.SIGTERM


def test_the_with_block_installs_and_restores() -> None:
    """Install on entering, give back the handler, and restore on leaving."""
    handlers_before = {signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS}
    shutdown_handler = shutdown.ShutdownHandler()
    with shutdown_handler as entered_handler:
        assert entered_handler is shutdown_handler
        send(shutdown_handler, signal.SIGHUP)
    assert not shutdown_handler.installed
    assert shutdown_handler.received_signal is signal.SIGHUP
    assert {
        signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS
    } == handlers_before


def test_the_with_block_restores_when_it_raises() -> None:
    """Restore the handlers and let the exception through when the block fails."""
    handlers_before = {signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS}
    shutdown_handler = shutdown.ShutdownHandler()
    with pytest.raises(RuntimeError, match="stop here"), shutdown_handler:
        raise RuntimeError("stop here")
    assert not shutdown_handler.installed
    assert {
        signum: signal.getsignal(signum) for signum in WATCHED_SIGNALS
    } == handlers_before


def test_nested_handlers_restore_in_turn() -> None:
    """Put each earlier handler back when nested blocks end."""
    sigterm_handler_before = signal.getsignal(signal.SIGTERM)
    outer_handler = shutdown.ShutdownHandler([signal.SIGTERM])
    inner_handler = shutdown.ShutdownHandler([signal.SIGTERM])
    with outer_handler:
        with inner_handler:
            send(inner_handler, signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == outer_handler._handle_signal
    assert signal.getsignal(signal.SIGTERM) is sigterm_handler_before
    assert inner_handler.shutdown_requested
    assert not outer_handler.shutdown_requested


def test_wait_wakes_when_a_signal_arrives() -> None:
    """Return True from a long wait soon after a signal is sent."""
    shutdown_handler = shutdown.ShutdownHandler()
    shutdown_handler.install()
    assert signal.getsignal(signal.SIGTERM) == shutdown_handler._handle_signal
    signal_sender = threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGTERM))
    wait_started = time.monotonic()
    signal_sender.start()
    woken = shutdown_handler.wait(10)
    waited_seconds = time.monotonic() - wait_started
    # The signal must arrive while this test's handler still catches it,
    # whatever the result, so wait for the sender before any assert.
    signal_sender.join()
    assert woken
    assert 0.1 <= waited_seconds < 5
    assert shutdown_handler.received_signal is signal.SIGTERM


def test_install_from_another_thread_is_refused_and_changes_nothing() -> None:
    """Refuse to install off the main thread, and leave a later install working."""
    sigusr1_handler_before = signal.getsignal(signal.SIGUSR1)
    shutdown_handler = shutdown.ShutdownHandler([signal.SIGUSR1])
    raised_errors: list[BaseException] = []

    def install_off_main_thread() -> None:
        """Try to install from this thread, keeping what it raises."""
        with pytest.raises(ValueError, match="main thread") as refused:
            shutdown_handler.install()
        raised_errors.append(refused.value)

    install_thread = threading.Thread(target=install_off_main_thread)
    install_thread.start()
    install_thread.join()
    assert len(raised_errors) == 1
    assert not shutdown_handler.installed
    assert signal.getsignal(signal.SIGUSR1) is sigusr1_handler_before
    shutdown_handler.install()
    send(shutdown_handler, signal.SIGUSR1)
    assert shutdown_handler.received_signal is signal.SIGUSR1


def test_a_failed_install_puts_back_what_it_replaced() -> None:
    """Restore the signals replaced before one that cannot be caught."""

    def earlier_handler(_signum: int, _frame: object) -> None:
        """Stand in for a handler installed before this one."""

    signal.signal(signal.SIGUSR1, earlier_handler)
    shutdown_handler = shutdown.ShutdownHandler([signal.SIGUSR1, signal.SIGKILL])
    with pytest.raises(OSError, match="Invalid argument"):
        shutdown_handler.install()
    assert not shutdown_handler.installed
    assert signal.getsignal(signal.SIGUSR1) is earlier_handler
    shutdown_handler.uninstall()
    assert signal.getsignal(signal.SIGUSR1) is earlier_handler
