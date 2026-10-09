"""Background re-indexing on file changes.

Instead of one recursive watch (which on Linux would also watch node_modules, build output, etc.
and can exhaust inotify limits), we put a non-recursive watch on each directory that actually holds
indexable files, plus .git for branch switches and new commits. Newly created directories are added
on the fly. Events are debounced, then an incremental index runs in a worker thread.
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import config
from .indexer import SKIP_DIRS, index, list_files, log, wanted

MAX_DIRS = 8000


class _RepoWatch:
    def __init__(self, root: Path):
        from watchdog.observers import Observer

        self.root = root
        self.observer = Observer()
        self.observer.daemon = True
        self.watched: set[str] = set()
        self.timer: threading.Timer | None = None
        self.history_due = False
        self.lock = threading.Lock()
        self.running = threading.Lock()
        self.ok = True

    # --- watches
    def start(self) -> bool:
        from watchdog.events import FileSystemEventHandler

        outer = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                outer.on_event(event)

        self.handler = Handler()
        dirs = {self.root}
        for rel in list_files(self.root):
            parent = (self.root / rel).parent
            while parent != self.root and parent not in dirs:
                dirs.add(parent)
                parent = parent.parent
        if len(dirs) > MAX_DIRS:
            log(f"watch: {self.root.name} has {len(dirs)} source directories; falling back to refresh-on-search")
            return False
        for d in dirs:
            self._watch(d)
        git_dir = self.root / ".git"
        if git_dir.is_dir():
            self._watch(git_dir)  # HEAD / ORIG_HEAD changes on checkout, commit, rebase
        try:
            self.observer.start()
        except Exception as e:  # e.g. inotify limit reached
            log(f"watch: could not start for {self.root.name}: {e}")
            return False
        log(f"watching {len(self.watched)} directories in {self.root.name}")
        return True

    def _watch(self, d: Path):
        key = str(d)
        if key in self.watched:
            return
        try:
            self.observer.schedule(self.handler, key, recursive=False)
            self.watched.add(key)
        except Exception:
            pass

    def stop(self):
        try:
            self.observer.stop()
        except Exception:
            pass

    # --- events
    def on_event(self, event):
        src = getattr(event, "dest_path", None) or event.src_path
        if isinstance(src, bytes):
            src = src.decode(errors="replace")
        p = Path(src)
        try:
            rel = p.relative_to(self.root).as_posix()
        except ValueError:
            return
        if rel.startswith(".git/"):
            name = rel[5:]
            if name in ("HEAD", "ORIG_HEAD", "packed-refs") or name.startswith("refs/heads"):
                self.history_due = True
                self.schedule()
            return
        if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
            return
        if event.is_directory:
            if event.event_type in ("created", "moved") and p.name not in SKIP_DIRS and not p.name.startswith("."):
                self._watch(p)
            return
        if wanted(rel):
            self.schedule()

    def schedule(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(config.CFG.watch_debounce, self._run)
            self.timer.daemon = True
            self.timer.start()

    def _run(self):
        if not self.running.acquire(blocking=False):
            self.schedule()  # an index is already running; try again after it
            return
        try:
            index(self.root)
            if self.history_due:
                self.history_due = False
                from . import history

                history.update(self.root)
        except Exception as e:
            log(f"watch: re-index of {self.root.name} failed: {type(e).__name__}: {e}")
        finally:
            self.running.release()


class WatchManager:
    def __init__(self):
        self._repos: dict[str, _RepoWatch | None] = {}
        self._lock = threading.Lock()

    def ensure(self, root: Path) -> bool:
        """Start watching `root` if enabled. Returns True if a watcher is active."""
        if not config.CFG.watch:
            return False
        key = str(root)
        with self._lock:
            if key in self._repos:
                return self._repos[key] is not None
            try:
                w = _RepoWatch(root)
                ok = w.start()
            except Exception as e:
                log(f"watch disabled for {root.name}: {e}")
                ok, w = False, None
            self._repos[key] = w if ok else None
            return ok

    def active(self, root: Path) -> bool:
        return self._repos.get(str(root)) is not None

    def stop_all(self):
        for w in self._repos.values():
            if w:
                w.stop()


WATCHERS = WatchManager()
