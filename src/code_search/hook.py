"""Claude Code `UserPromptSubmit` hook: attach likely-relevant code locations to each prompt.

Agents forget to call search tools; this makes the first look automatic. It runs on every prompt,
so it is built to be cheap and quiet: import-light, never waits for a model or index to load
(it starts the daemon and skips that prompt instead), and adds nothing unless a match is confident.

Install with `code-search hook install`; remove with `code-search hook uninstall`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

from . import daemon

PROJECT = Path(__file__).resolve().parents[2]
TIMEOUT = 8.0  # seconds the hook may spend in the daemon; Claude Code's own limit is set at install
MARKER = "code-search hook"


def worth_searching(prompt: str) -> bool:
    """Skip slash commands, very short messages and plain acknowledgements."""
    p = prompt.strip()
    if len(p) < 15 or p.startswith("/") or "\n" not in p and len(p.split()) < 3:
        return False
    return not re.fullmatch(r"(?i)(ok(ay)?|yes|no|thanks?( you)?|go( ahead)?|continue|do it|sounds good)[.! ]*", p)


def run(stdin=sys.stdin, stdout=sys.stdout) -> None:
    try:
        event = json.load(stdin)
    except ValueError:
        return
    prompt, cwd = event.get("prompt") or "", event.get("cwd") or os.getcwd()
    if not worth_searching(prompt):
        return
    st = daemon.info(timeout=1)
    if not st:
        daemon.spawn()  # ready for the next prompt; never make this one wait for a model to load
        return
    try:
        text = daemon._request(st, "/api/call", {"tool": "auto_context", "args": {"prompt": prompt[:4000], "path": cwd}},
                               timeout=TIMEOUT)["result"]
    except Exception:  # a hook must never break the prompt
        return
    if text:
        json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}}, stdout)


# ---------------------------------------------------------------- install / uninstall

def _settings_path(scope: str) -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return Path.home() / ".claude" / "settings.json"


def command() -> str:
    uv = shutil.which("uv") or "uv"
    return f'"{uv}" run --project "{PROJECT}" code-search hook'  # contains MARKER, which is how we find it again


def _is_ours(entry: dict) -> bool:
    return any(MARKER in (h.get("command") or "") for h in entry.get("hooks", []))


def install(scope: str = "user") -> str:
    path = _settings_path(scope)
    settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    hooks = settings.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])
    hooks[:] = [e for e in hooks if not _is_ours(e)]
    hooks.append({"hooks": [{"type": "command", "command": command(), "timeout": int(TIMEOUT) + 4}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return str(path)


def uninstall(scope: str = "user") -> str:
    path = _settings_path(scope)
    if not path.exists():
        return str(path)
    settings = json.loads(path.read_text(encoding="utf-8"))
    hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
    hooks[:] = [e for e in hooks if not _is_ours(e)]
    if not hooks:
        settings.get("hooks", {}).pop("UserPromptSubmit", None)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return str(path)
