"""Graceful-shutdown signal handling.

:class:`ShutdownHandler` captures termination signals from the operating
system (see :data:`DEFAULT_SIGNALS`) and records the request in a thread-safe
flag. The rest of the program polls
:attr:`ShutdownHandler.shutdown_requested`, or blocks on
:meth:`ShutdownHandler.wait`, and stops cleanly at a safe point instead of
being interrupted mid-operation.

Signal handlers can only be installed from the main thread of the main
interpreter, so :meth:`ShutdownHandler.install` must be called there. The
shutdown flag itself may be queried or waited on from any thread.
"""

import signal
import threading
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType, TracebackType

type SignalHandler = (
    Callable[[int, FrameType | None], object] | int | signal.Handlers | None
)
"""Any value :func:`signal.getsignal` may return for a previously installed handler."""

DEFAULT_SIGNALS: Final[tuple[signal.Signals, ...]] = (
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGHUP,
)
"""Signals treated as shutdown requests when none are specified explicitly.

Every one of them means the same thing here - stop at the next safe point -
and none is handled differently from the others. A hangup is included because
by default it ends the process at once, so closing the terminal a run was
started from by hand would abandon the work in progress. Nothing reloads its
configuration on a hangup; a run's configuration does not change while it
runs.
"""


class ShutdownHandler:
    """Capture OS termination signals and expose a graceful-shutdown flag.

    When installed, replaces the process's handlers for the configured signals
    with one that sets an internal :class:`threading.Event`. The previous
    handlers are remembered and restored on :meth:`uninstall`. The class is
    also a context manager: entering installs the handlers and exiting
    restores them.

    Handlers that share a signal must be uninstalled in the reverse of the
    order they were installed in, as nested ``with`` blocks do. Otherwise
    the handler uninstalled first is put back by the other, and stays
    installed.

    Parameters
    ----------
    signals : Sequence[signal.Signals], optional
        The signals to treat as shutdown requests. Duplicates are removed
        while preserving order. Defaults to :data:`DEFAULT_SIGNALS`.

    Examples
    --------
    >>> handler = ShutdownHandler()
    >>> handler.shutdown_requested
    False
    >>> handler.request_shutdown()
    >>> handler.shutdown_requested
    True

    Typical use in an application loop::

        with ShutdownHandler() as handler:
            while not handler.shutdown_requested:
                do_one_unit_of_work()
    """

    def __init__(self, signals: Sequence[signal.Signals] = DEFAULT_SIGNALS) -> None:
        """Initialize the handler without installing it.

        The class docstring describes the argument. A handler built here has
        requested nothing and installed nothing.
        """
        self._signals: Final[tuple[signal.Signals, ...]] = tuple(dict.fromkeys(signals))
        self._event: Final[threading.Event] = threading.Event()
        self._previous_handlers: Final[dict[signal.Signals, SignalHandler]] = {}
        self._received_signal: signal.Signals | None = None

    @property
    def signals(self) -> tuple[signal.Signals, ...]:
        """tuple[signal.Signals, ...]: The signals this handler listens for.

        The configured signals, deduplicated, in the order given.
        """
        return self._signals

    @property
    def shutdown_requested(self) -> bool:
        """bool: Whether a shutdown has been requested.

        ``True`` once a configured signal has been received or
        :meth:`request_shutdown` has been called; ``False`` otherwise.
        """
        return self._event.is_set()

    @property
    def received_signal(self) -> signal.Signals | None:
        """signal.Signals | None: The signal that triggered the shutdown request.

        The most recently received configured signal, or ``None`` if no signal
        has been received, which includes a shutdown requested in code through
        :meth:`request_shutdown`.
        """
        return self._received_signal

    @property
    def installed(self) -> bool:
        """bool: Whether this handler's signal handlers are currently installed.

        ``True`` between a call to :meth:`install` and the matching
        :meth:`uninstall`; ``False`` otherwise.
        """
        return bool(self._previous_handlers)

    def install(self) -> None:
        """Install signal handlers for the configured signals.

        The handlers previously registered for those signals are remembered so
        that :meth:`uninstall` can restore them. Calling this method while
        already installed is a no-op.

        Installing is all or nothing: if replacing any handler fails, the
        handlers already replaced are put back before the error is raised,
        so the handler is left not installed.

        Raises
        ------
        ValueError
            If called from a thread other than the main thread of the main
            interpreter (raised by :func:`signal.signal`).
        OSError
            If a configured signal cannot be caught, such as ``SIGKILL`` or
            ``SIGSTOP`` (raised by :func:`signal.signal`).
        """
        if self._previous_handlers:
            return
        try:
            for signum in self._signals:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
                # Recorded only once replaced, so a failure leaves nothing
                # recorded that was not changed.
                self._previous_handlers[signum] = previous
        except BaseException:
            self.uninstall()
            raise

    def uninstall(self) -> None:
        """Restore the signal handlers that were replaced by :meth:`install`.

        Calling this method while not installed is a no-op. The shutdown flag
        is left untouched.

        Raises
        ------
        ValueError
            If called from a thread other than the main thread of the main
            interpreter (raised by :func:`signal.signal`).
        TypeError
            If a replaced handler had been set outside Python, so that
            :func:`signal.getsignal` gave ``None`` for it.
        """
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def request_shutdown(self) -> None:
        """Request a shutdown programmatically.

        Sets the shutdown flag exactly as an OS signal would, but leaves
        :attr:`received_signal` unchanged. Safe to call from any thread and
        idempotent.
        """
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a shutdown is requested or the timeout elapses.

        Parameters
        ----------
        timeout : float or None, optional
            Maximum number of seconds to block. ``None`` (the default) blocks
            indefinitely.

        Returns
        -------
        bool
            ``True`` if a shutdown has been requested, ``False`` if the
            timeout elapsed first.
        """
        return self._event.wait(timeout)

    def __enter__(self) -> Self:
        """Install the signal handlers and return this handler.

        Returns
        -------
        Self
            This handler, with its signal handlers installed.
        """
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the previously registered signal handlers.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            The type of the exception raised in the ``with`` block, if any.
        exc_value : BaseException or None
            The exception raised in the ``with`` block, if any.
        traceback : TracebackType or None
            The traceback of the exception raised in the ``with`` block, if any.
        """
        self.uninstall()

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        """Record the received signal and set the shutdown flag.

        Registered via :func:`signal.signal` for each configured signal.

        Parameters
        ----------
        signum : int
            The number of the received signal.
        _frame : FrameType or None
            The stack frame interrupted by the signal. Required by the
            :func:`signal.signal` calling convention and not used here; the
            leading underscore is what marks it as deliberately unused.
        """
        self._received_signal = signal.Signals(signum)
        self._event.set()
