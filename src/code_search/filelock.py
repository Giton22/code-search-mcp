"""Tiny cross-process lock, so Codex and Claude Code running the server at the same time
never index the same repo concurrently."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_thread_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


class FileLock:
    def __init__(self, path: Path, timeout: float = 600):
        self.path = Path(path)
        self.timeout = timeout
        self._fh = None
        with _guard:
            self._tlock = _thread_locks.setdefault(str(self.path), threading.Lock())

    def __enter__(self):
        if not self._tlock.acquire(timeout=self.timeout):
            raise TimeoutError(f"timed out waiting for {self.path}")
        try:
            self._fh = open(self.path, "a+")
            deadline = time.time() + self.timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        self._fh.seek(0)
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except OSError:
                    if time.time() > deadline:
                        raise TimeoutError(f"another process holds {self.path}")
                    time.sleep(0.2)
        except BaseException:
            if self._fh:
                self._fh.close()
            self._tlock.release()
            raise

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._tlock.release()
