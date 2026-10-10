"""The web UI and the daemon's HTTP API: a local web app over the same functions the MCP tools use.

Standard library only (no extra packages, no frontend build). Listens on 127.0.0.1, checks the Host
header, and requires a per-run token on every API call, so other web pages can't reach it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, daemon, embed, hardware, indexer, search, server
from .store import all_stores

PAGE = Path(__file__).with_name("ui.html")
PROJECT = Path(__file__).resolve().parents[2]
SNIPPET_LINES = 60


def _hit(h: dict) -> dict:
    row = h["row"]
    lines = (row.get("text") or "").splitlines()
    return {"repo": h["repo"], "root": h["root"], "path": row["path"], "start": row["start"], "end": row["end"],
            "symbol": row.get("symbol"), "kind": row.get("kind"), "lang": row.get("lang"),
            "score": round(float(h["score"]), 3), "text": "\n".join(lines[:SNIPPET_LINES]),
            "more": max(0, len(lines) - SNIPPET_LINES)}


def _known_root(root: str) -> Path:
    """A repo root the UI may act on: one that is already indexed."""
    p = Path(root).resolve()
    if str(p) not in {str(s.root.resolve()) for s in all_stores()}:
        raise ValueError(f"{root} is not an indexed repo")
    return p


def _open_in_editor(root: str, rel: str, line: int) -> str:
    base = _known_root(root)
    target = (base / rel).resolve()
    if base not in target.parents or not target.is_file():
        raise ValueError("file is outside the repo")
    # editor launchers on Windows are often .cmd scripts, and cmd.exe would interpret these characters
    unsafe = os.name == "nt" and any(ch in str(target) for ch in '&|<>^%"')
    for editor in ("code", "cursor", "zed"):
        exe = shutil.which(editor)
        if exe and not (unsafe and exe.lower().endswith((".cmd", ".bat"))):
            args = [exe, f"{target}:{line}"] if editor == "zed" else [exe, "-g", f"{target}:{line}"]
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return editor
    if os.name == "nt":
        os.startfile(target)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(target)])
    return "default app"


def _connect_snippets() -> dict:
    uv = os.environ.get("UV") or shutil.which("uv") or "uv"
    args = ["run", "--project", str(PROJECT), "code-search", "serve"]
    cmd = " ".join(["uv", *args])
    return {
        "claude_code": f"claude mcp add --scope user --transport stdio code-search -- {cmd}",
        "codex": f"codex mcp add code-search -- {cmd}",
        "claude_desktop": json.dumps({"mcpServers": {"code-search": {"command": uv, "args": args}}}, indent=2),
    }


def _system() -> dict:
    gpus = hardware.gpus()
    gpu = hardware.best_gpu(gpus)
    desc, _ = hardware.install_plan(gpu)
    loaded = getattr(embed._instance, "device_desc", None)
    no_torch = importlib.util.find_spec("torch") is None
    unused_gpu = loaded == "CPU" and gpu is not None and not config.CFG.device
    return {"gpus": [g["name"] + (f" ({g['gfx']})" if g["gfx"] else "") for g in gpus],
            "device": loaded, "plan": desc, "setup_needed": config.CFG.backend == "st" and (no_torch or unused_gpu),
            "setup_command": "uv run code-search setup", "model": config.CFG.model_tag,
            "backend": config.CFG.backend, "reranker": config.CFG.rerank_model or None,
            "project": str(PROJECT), "connect": _connect_snippets()}


def _repos() -> list[dict]:
    out = server.index_status(all_repos=True)
    for r in out:
        job = server._jobs.get(r["root"])
        if job and job[0].is_alive():
            r["progress"] = {k: job[1].get(k) for k in ("done", "total")}
    return out


class Handler(BaseHTTPRequestHandler):
    token = ""
    port = 0

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode())

    def _allowed(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def do_GET(self):
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")
        url = urlparse(self.path)
        if url.path == "/":
            html = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", self.token)
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        self._api(url.path, {k: v[-1] for k, v in parse_qs(url.query).items()})

    def do_POST(self):
        if not self._allowed():
            return self._send(403, b"forbidden", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        self._api(urlparse(self.path).path, body)

    def _api(self, path: str, q: dict):
        if not secrets.compare_digest(self.headers.get("X-Token", ""), self.token):
            return self._json({"error": "missing or wrong token"}, 403)
        daemon.touch()
        try:
            self._json(self._route(path, q))
        except (ValueError, FileNotFoundError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # surfaced in the UI as a toast, or re-raised by the MCP client
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _route(self, path: str, q: dict):
        repo = q.get("repo") or None
        if path == "/api/ping":
            return {"ok": True, "pid": os.getpid()}
        if path == "/api/call":  # an MCP tool, run here for a thin client
            fn = server.TOOLS.get(q.get("tool", ""))
            if fn is None:
                raise FileNotFoundError(f"unknown tool {q.get('tool')!r}")
            return {"result": fn(**(q.get("args") or {}))}
        if path == "/api/system":
            return _system()
        if path == "/api/repos":
            return _repos()
        if path == "/api/search":
            if not (q.get("q") or "").strip():
                return {"hits": [], "note": ""}
            repos = ["all"] if repo == "all" else None
            stores, note = server._stores(None if repo == "all" else repo, repos)
            hits = search.search(stores, q["q"], k=int(q.get("k") or 10), path_glob=q.get("glob") or None)
            return {"hits": [_hit(h) for h in hits], "note": note.strip("_ ")}
        if path == "/api/related":
            store, note = server._prepare(repo)
            return {"items": search.related_files(store, q["file"]), "note": note.strip("_ ")}
        if path == "/api/tests":
            store, note = server._prepare(repo)
            return {"items": search.find_tests(store, q["target"]), "note": note.strip("_ ")}
        if path == "/api/history":
            store, note = server._prepare(repo)
            return {"markdown": search.search_history(store, q["q"], path=q.get("file") or None),
                    "note": note.strip("_ ")}
        if path == "/api/explain":
            store, note = server._prepare(repo)
            return {"markdown": search.explain_area(store, q.get("glob") or None), "note": note.strip("_ ")}
        if path == "/api/index":
            root = indexer.resolve_root(q.get("path") or repo)
            if not root.is_dir():
                raise ValueError(f"{root} is not a folder")
            server._start_index(root)
            return {"root": str(root), "started": True}
        if path == "/api/open":
            return {"editor": _open_in_editor(q["root"], q["path"], int(q.get("line") or 1))}
        raise FileNotFoundError(f"unknown endpoint {path}")


def new_token() -> str:
    return secrets.token_urlsafe(24)


def open_ui(open_browser: bool = True) -> str:
    """Make sure the shared daemon runs (it serves the UI) and open it in the browser."""
    st = daemon.ensure()
    if not st:
        raise SystemExit("code-search daemon did not start; see daemon.log in the cache directory")
    url = f"http://127.0.0.1:{st['port']}/"
    print(f"code-search UI at {url}", file=sys.stderr, flush=True)
    if open_browser:
        webbrowser.open(url)
    return url
