"""Keeping two runs from writing the same output at once.

:class:`RunLock` is an advisory inter-process lock on one named file. Two runs
that would write the same output must not proceed at once: each would work out
the same starting point and both would write from it, so everything after it
would be written twice. So a run holds its lock for its whole duration, and a second run
aborts at once with a :class:`~masterclock.app.exceptions.RunLockError`
naming the lock and its holder.

The name comes from the caller, and it is the name that decides who excludes
whom: runs that would write the same files ask for the same one, and runs
whose outputs are disjoint ask for different ones. Nothing here knows which
program is running or how that program divides its work.

The lock is an :func:`fcntl.flock` exclusive lock on that file, which sits
directly in the directory it is given rather than in any subdirectory of it,
so the scans that read the output files never encounter it. The holder's PID
is written to the file for diagnostics. The kernel releases the lock when
its file descriptor closes - on :meth:`RunLock.release`, or when the process
ends however abruptly - so a crashed run can never leave a stale lock behind;
the lock file itself is left in place and reused by later runs.
"""

import errno
import fcntl
import os
from typing import TYPE_CHECKING, Final, NoReturn, Self

from masterclock.app.exceptions import RunLockError
from masterclock.app.log import MasterClockLogger, get_logger

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

_PID_READ_LIMIT: Final[int] = 64
"""Bytes read from a held lock file when looking for the holder's PID."""

_log: Final[MasterClockLogger] = get_logger(__name__)
"""Logger for this module."""


class RunLock:
    """An exclusive inter-process lock on one named file.

    Locks the named file in the given directory with :func:`fcntl.flock`,
    recording the holder's PID in it. The lock is advisory and covers exactly
    what the name covers: two :class:`RunLock` instances for the same
    directory and name exclude each other, across processes and across
    instances within one process, while two different names do not. The class
    is also a context manager: entering acquires the lock and exiting
    releases it.

    The operating system ties the lock to the open file descriptor, so it
    is released however the holding process ends - a crash cannot leave a
    stale lock. The lock file itself persists between runs; only the lock
    on it comes and goes.

    Parameters
    ----------
    lock_directory : Path
        The directory the lock file lives in, which is the parent of the
        output subdirectories rather than one of them.
    lock_file_name : str
        What the lock file is called: a plain file name, so the file sits
        directly in ``lock_directory``. Two runs exclude each other exactly when
        they ask for the same name in the same directory.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     with RunLock(Path(directory), "run.lock") as lock:
    ...         lock.locked
    True

    Typical use in the application::

        with RunLock(processed_path, lock_file_name):
            run_the_processing()
    """

    def __init__(self, lock_directory: Path, lock_file_name: str) -> None:
        """Initialize the lock without acquiring it.

        The class docstring describes each argument. A lock built here holds
        nothing and has touched nothing on disk.

        Raises
        ------
        ValueError
            If ``lock_file_name`` is not a plain file name: empty, ``.`` or ``..``, or
            containing a path separator or a NUL character.
        """
        if (
            lock_file_name in {"", ".", ".."}
            or os.sep in lock_file_name
            or (os.altsep is not None and os.altsep in lock_file_name)
            or "\0" in lock_file_name
        ):
            msg = f"run lock name {lock_file_name!r} is not a plain file name"
            raise ValueError(msg)
        self._path: Final[Path] = lock_directory / lock_file_name
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        """Path: Full path of the lock file.

        The named file in the lock's directory, created by :meth:`acquire`
        when missing.
        """
        return self._path

    @property
    def locked(self) -> bool:
        """bool: Whether this instance currently holds the lock.

        ``True`` between a successful :meth:`acquire` and the matching
        :meth:`release`; ``False`` otherwise.
        """
        return self._fd is not None

    def acquire(self) -> None:
        """Acquire the lock, failing immediately if another run holds it.

        Opens (creating if missing) the lock file, takes a non-blocking
        exclusive :func:`fcntl.flock` on it, and records the holder's PID
        in the file. Calling this method while already holding the lock is
        a no-op.

        Raises
        ------
        RunLockError
            If another run already holds the lock (the error names the
            lock file and, when readable, the holder's PID), or if the
            lock file cannot be opened, locked for any other reason, or
            written. Nothing is left held when it is raised.
        """
        if self._fd is not None:
            return
        fd = self._open()
        self._lock(fd)
        self._write_pid(fd)
        self._fd = fd
        _log.debug("acquired run lock %s", self._path)

    def _refuse(self, message: str, cause: OSError) -> NoReturn:
        """Log ``message`` as an error and raise it as a RunLockError.

        Parameters
        ----------
        message : str
            What went wrong, naming the lock file.
        cause : OSError
            The error that made the lock unusable.

        Raises
        ------
        RunLockError
            Always, with ``message``, raised from ``cause``.
        """
        _log.error(message)
        raise RunLockError(message) from cause

    def _open(self) -> int:
        """Open the lock file, creating it when missing.

        Returns
        -------
        int
            A descriptor of the lock file, open for reading and writing.

        Raises
        ------
        RunLockError
            If the file cannot be opened.
        """
        try:
            return os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            self._refuse(f"cannot open run lock file {self._path}: {exc}", exc)

    def _lock(self, fd: int) -> None:
        """Take the exclusive lock on ``fd`` without waiting for it.

        Parameters
        ----------
        fd : int
            An open descriptor of the lock file; closed if the lock is not
            taken.

        Raises
        ------
        RunLockError
            If another run holds the lock, or the lock cannot be taken for
            another reason, which the error then names instead.
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno != errno.EWOULDBLOCK:
                os.close(fd)
                self._refuse(f"cannot lock run lock file {self._path}: {exc}", exc)
            holder = self._holder_pid(fd)
            os.close(fd)
            run = "another run" if holder is None else f"another run (pid {holder})"
            self._refuse(
                f"{run} already holds the run lock {self._path}; wait for it to finish",
                exc,
            )

    def _write_pid(self, fd: int) -> None:
        """Replace the lock file's content with this process's PID.

        Parameters
        ----------
        fd : int
            A descriptor of the lock file, holding the lock; closed, which
            releases the lock, if the PID cannot be written.

        Raises
        ------
        RunLockError
            If the file cannot be truncated or written.
        """
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        except OSError as exc:
            os.close(fd)
            self._refuse(f"cannot write run lock file {self._path}: {exc}", exc)

    def release(self) -> None:
        """Release the lock by closing its file descriptor.

        The lock file is left in place - only the lock on it is dropped.
        Calling this method while not holding the lock is a no-op.
        """
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        _log.debug("released run lock %s", self._path)

    def __enter__(self) -> Self:
        """Acquire the lock and return this instance.

        Returns
        -------
        Self
            This lock, held.

        Raises
        ------
        RunLockError
            If this lock is already held, since the block's end would
            release it early for whoever holds it now; or for any of the
            reasons :meth:`acquire` gives.
        """
        if self._fd is not None:
            message = (
                f"run lock {self._path} is already held by this lock;"
                " a with block cannot enter it again"
            )
            _log.error(message)
            raise RunLockError(message)
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            The type of the exception raised in the ``with`` block, if any.
        exc_value : BaseException or None
            The exception raised in the ``with`` block, if any.
        traceback : TracebackType or None
            The traceback of the exception raised in the ``with`` block, if any.
        """
        self.release()

    @staticmethod
    def _holder_pid(fd: int) -> str | None:
        """Read the holder's PID from a lock file that refused the lock.

        Best-effort diagnostics for the conflict error message: the holder
        wrote its PID on acquiring, but the content is read without any
        synchronization and is not trusted beyond formatting. Read in the
        moment after a new holder takes the lock and before it writes its
        PID, it gives the previous holder's.

        Parameters
        ----------
        fd : int
            An open descriptor of the lock file; its position is moved.

        Returns
        -------
        str or None
            The PID recorded in the file, or ``None`` when the file cannot
            be read or does not hold a plain number.
        """
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            content = (
                os.read(fd, _PID_READ_LIMIT).decode("ascii", errors="replace").strip()
            )
        except OSError:
            return None
        return content if content.isdigit() else None
