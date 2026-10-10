"""One shared background service per user and configuration.

Every MCP client (Codex, Claude Code, Claude Desktop, the hook, the web UI) talks to the same
process, so the model is loaded once, searches are fast from the first query, and one process owns
the index writes and file watchers. Clients start it on demand; it exits after sitting idle.

This module stays import-light (no numpy, no torch): the prompt hook imports it on every prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config

PREFERRED_PORT = 7341
_last_request = time.time()


def touch() -> None:
    global _last_request
    _last_request = time.time()


def fingerprint() -> str:
    """Clients share a daemon only when they'd get the same results: same Python env and settings."""
    env = sorted((k, v) for k, v in os.environ.items()
                 if k.startswith("CODE_EMBED_") and not k.startswith("CODE_EMBED_DAEMON"))
    return hashlib.sha1(json.dumps([sys.executable, env]).encode()).hexdigest()[:12]


def state_path() -> Path:
    return config.CFG.cache_dir / f"daemon-{fingerprint()}.json"


def enabled() -> bool:
    return os.environ.get("CODE_EMBED_DAEMON", "1").strip().lower() not in {"0", "false", "no", "off"}


def _request(info: dict, path: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{info['port']}{path}", data=data,
                                 headers={"X-Token": info["token"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error") or str(e)
        except ValueError:
            msg = str(e)
        raise RuntimeError(msg) from None


def info(timeout: float = 1.5) -> dict | None:
    """The running daemon's {port, token, pid}, or None when there isn't a healthy one."""
    try:
        st = json.loads(state_path().read_text())
        if _request(st, "/api/ping", timeout=timeout).get("ok"):
            return st
    except (OSError, ValueError, RuntimeError, KeyError):
        pass
    return None


def spawn() -> None:
    """Start the daemon in the background, detached from this process (and its console)."""
    log = config.CFG.cache_dir / "daemon.log"
    config.CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    with open(log, "ab") as fh:
        subprocess.Popen([sys.executable, "-m", "code_search", "daemon"], stdin=subprocess.DEVNULL,
                         stdout=fh, stderr=fh, env=os.environ.copy(), **kw)


def ensure(wait: float = 60) -> dict | None:
    """A running daemon, starting one if needed. None if it doesn't come up within `wait` seconds."""
    st = info()
    if st:
        return st
    spawn()
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(0.25)
        st = info(timeout=1)
        if st:
            return st
    return None


def call(tool: str, args: dict, timeout: float = 900):
    """Run an MCP tool inside the daemon. Raises ConnectionError when no daemon is reachable."""
    st = ensure()
    if not st:
        raise ConnectionError("code-search daemon did not start; see daemon.log in the cache directory")
    try:
        res = _request(st, "/api/call", {"tool": tool, "args": args}, timeout=timeout)
    except urllib.error.URLError as e:
        raise ConnectionError(str(e)) from e
    return res["result"]


def run(idle_minutes: float | None = None) -> None:
    """Serve until idle. Exits at once if another daemon for this configuration already runs."""
    import socket
    from http.server import ThreadingHTTPServer

    from .filelock import FileLock
    from . import embed, indexer, server, ui

    idle = 60 * (idle_minutes if idle_minutes is not None
                 else float(os.environ.get("CODE_EMBED_DAEMON_IDLE_MINUTES", "30")))
    state = state_path()
    state.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(state.with_suffix(".lock"), timeout=0.5)
    try:
        lock.__enter__()
    except TimeoutError:
        print("[code-search] a daemon for this configuration is already running", file=sys.stderr)
        return
    class Server(ThreadingHTTPServer):
        # Windows lets two sockets share a port when SO_REUSEADDR is set, and then requests go to
        # either daemon at random; make the port ours alone
        allow_reuse_address = os.name != "nt"
        daemon_threads = True

        def server_bind(self):
            if os.name == "nt":
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    try:
        try:
            httpd = Server(("127.0.0.1", PREFERRED_PORT), ui.Handler)
        except OSError:
            httpd = Server(("127.0.0.1", 0), ui.Handler)
        token = ui.new_token()
        ui.Handler.token, ui.Handler.port = token, httpd.server_address[1]
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "port": ui.Handler.port, "token": token,
                                   "started": time.time()}))
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        tmp.replace(state)
        indexer.log(f"daemon on http://127.0.0.1:{ui.Handler.port}/ (pid {os.getpid()}, idle exit "
                    f"after {idle / 60:.0f} min)")
        if config.CFG.backend == "st":
            threading.Thread(target=embed.get, daemon=True).start()  # load the model before the first search
        threading.Thread(target=server.warm_configured_repos, daemon=True).start()

        def reaper():
            while True:
                time.sleep(15)
                busy = any(t.is_alive() for t, _ in list(server._jobs.values()))
                if not busy and time.time() - _last_request > idle:
                    indexer.log("daemon idle; exiting")
                    httpd.shutdown()
                    return

        threading.Thread(target=reaper, daemon=True).start()
        touch()
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if json.loads(state.read_text()).get("pid") == os.getpid():
                state.unlink()
        except (OSError, ValueError):
            pass
        from .watcher import WATCHERS

        WATCHERS.stop_all()
        lock.__exit__(None, None, None)
