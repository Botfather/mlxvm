from __future__ import annotations

import errno
import fcntl
import os
import time
from pathlib import Path
from typing import IO, Optional

from mlxvm.errors import LockTimeoutError


class FileLock:
    """Advisory process lock for mutation operations on macOS/POSIX."""

    def __init__(self, path: Path, *, timeout: float = 60.0) -> None:
        self.path = path
        self.timeout = timeout
        self._handle: Optional[IO[str]] = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    handle.close()
                    raise
                if time.monotonic() >= deadline:
                    handle.seek(0)
                    owner = handle.read().strip() or "unknown process"
                    handle.close()
                    raise LockTimeoutError(
                        f"timed out waiting for lock {self.path.name}",
                        hint=f"operation appears to be held by {owner}",
                    )
                time.sleep(0.1)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired={time.time():.6f}\n")
        handle.flush()
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
