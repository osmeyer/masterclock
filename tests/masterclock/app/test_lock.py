"""Tests for src/masterclock/app/lock.py.

The rules covered: a run lock on a name in a directory is held by one lock
at a time, across processes and within one, and records its holder's PID;
names that differ do not exclude each other; releasing frees it and keeps
the file; a refusal names the lock and, when it can be read, the holder; a
failure that is not another holder is reported as what it is and leaves
nothing held; the name must be a plain file name in the directory; and a
with block cannot enter a lock that is already held.

A holder file that is not ASCII digits names no PID; the lock file is made
0o644; taking and giving back the lock are logged at DEBUG, and entering a
held lock is refused, word for word.
"""

import errno
import fcntl
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from masterclock.app import lock
from masterclock.app.exceptions import RunLockError

NAME: Final = "run.lock"

# Tries the lock from a separate process, and prints what happened.
OTHER_PROCESS: Final = """
import sys
from pathlib import Path
from masterclock.app.exceptions import RunLockError
from masterclock.app.lock import RunLock
held = RunLock(Path(sys.argv[1]), sys.argv[2])
try:
    held.acquire()
except RunLockError as refused:
    print("refused:", refused)
else:
    print("acquired")
"""


def try_from_another_process(directory: Path, name: str = NAME) -> str:
    """Return what another process printed on trying the lock."""
    finished = subprocess.run(  # noqa: S603 - fixed arguments, this interpreter
        [sys.executable, "-c", OTHER_PROCESS, str(directory), name],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return finished.stdout.strip()


def holding(run_lock: lock.RunLock) -> bool:
    """Say whether ``run_lock`` holds its lock, read afresh at each call."""
    return run_lock.locked


def test_acquire_holds_the_lock_and_records_the_pid(tmp_path: Path) -> None:
    """Create the lock file, hold it, and write this process's PID in it."""
    held = lock.RunLock(tmp_path, NAME)
    assert held.path == tmp_path / NAME
    assert not holding(held)
    assert not held.path.exists()
    held.acquire()
    assert holding(held)
    assert held.path.read_text(encoding="ascii") == f"{os.getpid()}\n"
    held.release()


def test_acquire_replaces_what_an_earlier_run_left(tmp_path: Path) -> None:
    """Leave only this process's PID, however long the earlier content was."""
    (tmp_path / NAME).write_text("99999999999999999999\nmore\n", encoding="ascii")
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    assert held.path.read_text(encoding="ascii") == f"{os.getpid()}\n"
    held.release()


def test_acquire_and_release_twice_change_nothing(tmp_path: Path) -> None:
    """Keep holding on a second acquire; do nothing on a second release."""
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    held.acquire()
    assert holding(held)
    held.release()
    held.release()
    assert not holding(held)
    assert held.path.exists()


def test_another_process_is_refused_while_held(tmp_path: Path) -> None:
    """Refuse another process, naming the lock file and this process's PID."""
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    answer = try_from_another_process(tmp_path)
    held.release()
    assert answer == (
        f"refused: another run (pid {os.getpid()}) already holds the run lock"
        f" {tmp_path / NAME}; wait for it to finish"
    )
    assert try_from_another_process(tmp_path) == "acquired"


def test_another_lock_in_this_process_is_refused_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse a second lock on the same name here too, and log the refusal."""
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    second = lock.RunLock(tmp_path, NAME)
    with pytest.raises(RunLockError, match=f"pid {os.getpid()}"):
        second.acquire()
    assert not holding(second)
    assert holding(held)
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    held.release()
    second.acquire()
    assert holding(second)
    second.release()


def test_different_names_do_not_exclude_each_other(tmp_path: Path) -> None:
    """Hold two locks with different names in one directory at once."""
    first = lock.RunLock(tmp_path, "first.lock")
    second = lock.RunLock(tmp_path, "second.lock")
    first.acquire()
    second.acquire()
    assert holding(first)
    assert holding(second)
    first.release()
    second.release()


def test_the_descriptor_is_not_inherited(tmp_path: Path) -> None:
    """Keep the lock out of programs this process starts."""
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    assert held._fd is not None
    assert not os.get_inheritable(held._fd)
    held.release()


def test_the_with_block_holds_and_releases(tmp_path: Path) -> None:
    """Hold the lock inside the block, give back the lock, and free it after."""
    held = lock.RunLock(tmp_path, NAME)
    with held as entered:
        assert entered is held
        assert holding(held)
    assert not holding(held)
    with pytest.raises(RuntimeError, match="stop here"), held:
        raise RuntimeError("stop here")
    assert try_from_another_process(tmp_path) == "acquired"


def test_a_with_block_cannot_enter_a_held_lock(tmp_path: Path) -> None:
    """Refuse to enter a held lock again, and leave it held."""
    held = lock.RunLock(tmp_path, NAME)
    with held:
        with pytest.raises(RunLockError, match="already held by this lock"):
            held.__enter__()
        assert holding(held)
        assert try_from_another_process(tmp_path).startswith("refused")
    assert not holding(held)


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    """Refuse a lock whose directory does not exist."""
    held = lock.RunLock(tmp_path / "missing", NAME)
    with pytest.raises(RunLockError, match="cannot open run lock file"):
        held.acquire()
    assert not holding(held)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "sub/run.lock", "../run.lock", "/absolute/run.lock", "a\0b"]
)
def test_a_name_that_is_not_a_plain_file_name_is_refused(
    tmp_path: Path, name: str
) -> None:
    """Refuse a name that is empty, a directory, a path, or has a NUL in it."""
    with pytest.raises(ValueError, match="is not a plain file name"):
        lock.RunLock(tmp_path, name)


@given(
    st.text(min_size=1).filter(
        lambda name: "/" not in name and "\0" not in name and name not in {".", ".."}
    )
)
def test_any_plain_file_name_is_accepted(name: str) -> None:
    """Accept any other name, and place its file directly in the directory."""
    directory = Path("/invented/directory")
    assert lock.RunLock(directory, name).path.parent == directory


def failing_flock(code: int) -> Callable[[int, int], None]:
    """Return a stand-in for flock that fails with the given error number."""

    def flock(_fd: int, _operation: int) -> None:
        """Fail as flock would with ``code``."""
        raise OSError(code, os.strerror(code))

    return flock


def test_a_lock_that_cannot_be_taken_is_not_called_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a flock failure other than another holder as what it is."""
    monkeypatch.setattr(fcntl, "flock", failing_flock(errno.ENOLCK))
    held = lock.RunLock(tmp_path, NAME)
    with pytest.raises(RunLockError, match=r"cannot lock run lock file .*No locks"):
        held.acquire()
    assert not holding(held)


def test_a_holder_that_left_no_pid_is_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave out the PID when the file holds no plain number."""
    (tmp_path / NAME).write_text("not a number\n", encoding="ascii")
    monkeypatch.setattr(fcntl, "flock", failing_flock(errno.EWOULDBLOCK))
    with pytest.raises(RunLockError, match=r"^another run already holds"):
        lock.RunLock(tmp_path, NAME).acquire()


def test_a_holder_file_that_cannot_be_read_is_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave out the PID when reading the file fails."""

    def read(_fd: int, size: int) -> bytes:
        """Fail as the read of the holder's PID, the only read expected."""
        assert size == lock._PID_READ_LIMIT
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(fcntl, "flock", failing_flock(errno.EWOULDBLOCK))
    monkeypatch.setattr(os, "read", read)
    with pytest.raises(RunLockError, match=r"^another run already holds"):
        lock.RunLock(tmp_path, NAME).acquire()


def test_a_pid_that_cannot_be_written_leaves_nothing_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release the lock and report it when the PID cannot be written."""
    pid_line = f"{os.getpid()}\n".encode("ascii")

    def write(_fd: int, data: bytes) -> int:
        """Fail as the write of the PID, the only write expected."""
        assert data == pid_line
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "write", write)
    held = lock.RunLock(tmp_path, NAME)
    with pytest.raises(RunLockError, match=r"cannot write run lock file .*No space"):
        held.acquire()
    assert not holding(held)
    monkeypatch.undo()
    assert try_from_another_process(tmp_path) == "acquired"


@pytest.mark.parametrize("content", ["١٢٣\n".encode(), b"\xff\xfe\n"])
def test_a_holder_file_that_is_not_ascii_is_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """Leave out the PID when the file holds anything but ASCII digits."""
    (tmp_path / NAME).write_bytes(content)
    monkeypatch.setattr(fcntl, "flock", failing_flock(errno.EWOULDBLOCK))
    with pytest.raises(RunLockError, match=r"^another run already holds"):
        lock.RunLock(tmp_path, NAME).acquire()


def test_the_lock_file_is_made_readable_by_all_writable_by_its_owner(
    tmp_path: Path,
) -> None:
    """Create the lock file with mode 0o644, under the usual umask."""
    old = os.umask(0o022)
    try:
        held = lock.RunLock(tmp_path, NAME)
        held.acquire()
        held.release()
    finally:
        os.umask(old)
    assert (tmp_path / NAME).stat().st_mode & 0o777 == 0o644


def test_taking_and_giving_back_the_lock_is_logged_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say at DEBUG which lock was taken and given back, word for word."""
    caplog.set_level(logging.DEBUG)
    held = lock.RunLock(tmp_path, NAME)
    held.acquire()
    held.release()
    assert [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"] == [
        f"acquired run lock {tmp_path / NAME}",
        f"released run lock {tmp_path / NAME}",
    ]


def test_entering_a_held_lock_is_refused_in_words(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Say why a with block cannot enter a held lock, and log it as raised."""
    held = lock.RunLock(tmp_path, NAME)
    with held, pytest.raises(RunLockError) as raised:
        held.__enter__()
    assert str(raised.value) == (
        f"run lock {tmp_path / NAME} is already held by this lock;"
        " a with block cannot enter it again"
    )
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == [str(raised.value)]
