"""Git history indexing: commit message + changed files + names of changed functions.

Function names come from the hunk headers git already computes (`@@ ... @@ def foo(`), so this
is cheap and doesn't embed whole diffs. The same table drives co-change analysis.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from . import config, embed
from .filelock import FileLock
from .indexer import git, is_git, log
from .store import Store, blob_to_vec, vec_to_blob

_SYM = re.compile(
    r"(?:def|class|fn|func|function|struct|enum|trait|interface|impl|module|type|record|object)\s+"
    r"(?:\([^)]*\)\s*)?([A-Za-z_][\w$]*)|([A-Za-z_][\w$]*)\s*\("
)
_KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "sizeof", "elif", "else", "with", "new"}
MAX_FILES = 80
MAX_SYMBOLS = 30
FORMAT = "%x1e%H%x1f%at%x1f%an%x1f%s%x1f%b%x1d"


def _hunk_symbol(header: str) -> str | None:
    m = _SYM.search(header)
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    return None if not name or name in _KEYWORDS else name


def _parse(stream) -> list[dict]:
    commits, cur, in_body, body = [], None, False, []
    last_minus = None
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith("\x1e"):
            if cur:
                commits.append(cur)
            head = line[1:]
            parts = head.split("\x1f")
            cur = dict(sha=parts[0], ts=int(parts[1] or 0), author=parts[2], subject=parts[3],
                       body="", files=[], symbols=Counter())
            rest = parts[4] if len(parts) > 4 else ""
            if "\x1d" in rest:
                cur["body"] = rest.split("\x1d")[0].strip()
                in_body = False
            else:
                body, in_body = [rest], True
            continue
        if cur is None:
            continue
        if in_body:
            if "\x1d" in line:
                body.append(line.split("\x1d")[0])
                cur["body"] = "\n".join(body).strip()[:2000]
                in_body = False
            else:
                body.append(line)
            continue
        if line.startswith("--- a/"):
            last_minus = line[6:]
        elif line.startswith("+++ "):
            f = line[6:] if line.startswith("+++ b/") else last_minus  # deletions: +++ /dev/null
            if f and f not in cur["files"] and len(cur["files"]) < MAX_FILES:
                cur["files"].append(f)
            last_minus = None
        elif line.startswith("@@"):
            i = line.find("@@", 2)
            if i > 0:
                s = _hunk_symbol(line[i + 2:])
                if s:
                    cur["symbols"][s] += 1
    if cur:
        commits.append(cur)
    for c in commits:
        c["symbols"] = [s for s, _ in c["symbols"].most_common(MAX_SYMBOLS)]
    return commits


def _doc(c: dict) -> tuple[str, str]:
    parts = [c["subject"]]
    if c["body"]:
        parts.append(c["body"][:1200])
    if c["symbols"]:
        parts.append("Changed: " + ", ".join(c["symbols"][:20]))
    if c["files"]:
        parts.append("Files: " + ", ".join(c["files"][:25]))
    return "commit", "\n".join(parts)


def update(root: Path, store: Store | None = None) -> dict:
    """Index any commits reachable from HEAD (up to CODE_EMBED_HISTORY) not seen before."""
    if not is_git(root):
        return {"commits_added": 0, "note": "not a git repository"}
    store = store or Store(root)
    t0 = time.time()
    with FileLock(store.path.with_suffix(".history.lock")):
        revs = git(root, "rev-list", "--no-merges", f"--max-count={config.CFG.history_limit}", "HEAD", timeout=120)
        if not revs:
            return {"commits_added": 0}
        shas = revs.split()
        with store.connect() as con:
            known = {r[0] for r in con.execute("SELECT sha FROM commits")}
        missing = [s for s in shas if s not in known]
        if not missing:
            return {"commits_added": 0, "total_commits": len(known)}
        emb = embed.get()
        added = 0
        for i in range(0, len(missing), 300):
            part = missing[i:i + 300]
            try:
                p = subprocess.Popen(
                    ["git", "-C", str(root), "log", "--no-walk=unsorted", "--stdin", "--no-color", "-U0",
                     "--no-ext-diff", "--diff-filter=AMDR", f"--format={FORMAT}"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace")
                p.stdin.write("\n".join(part) + "\n")
                p.stdin.close()
                commits = _parse(p.stdout)
                p.wait(timeout=600)
            except (OSError, subprocess.SubprocessError) as e:
                log(f"history: git log failed: {e}")
                break
            if not commits:
                continue
            vecs = emb.embed_docs([_doc(c) for c in commits])
            with store.connect() as con:
                for c, v in zip(commits, vecs):
                    con.execute("INSERT OR REPLACE INTO commits VALUES(?,?,?,?,?,?,?,?)",
                                (c["sha"], c["ts"], c["author"], c["subject"], c["body"],
                                 json.dumps(c["files"]), json.dumps(c["symbols"]), vec_to_blob(v)))
                    if store.fts:
                        con.execute("INSERT INTO commits_fts(rowid, subject, body, files, symbols) "
                                    "VALUES((SELECT rowid FROM commits WHERE sha=?),?,?,?,?)",
                                    (c["sha"], c["subject"], c["body"], " ".join(c["files"]), " ".join(c["symbols"])))
            added += len(commits)
            log(f"history: {root.name} {added}/{len(missing)} commits")
        with store.connect() as con:
            total = con.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    return {"commits_added": added, "total_commits": total, "seconds": round(time.time() - t0, 1)}


def commit_rows(store: Store, with_vec=False):
    cols = "sha, ts, author, subject, body, files, symbols" + (", vec" if with_vec else "")
    with store.connect() as con:
        rows = con.execute(f"SELECT {cols} FROM commits ORDER BY ts DESC").fetchall()
    out = []
    for r in rows:
        d = dict(sha=r[0], ts=r[1], author=r[2], subject=r[3], body=r[4],
                 files=json.loads(r[5] or "[]"), symbols=json.loads(r[6] or "[]"))
        if with_vec:
            d["vec"] = blob_to_vec(r[7])
        out.append(d)
    return out


def cochange(store: Store, file: str, max_commit_files: int = 40) -> Counter:
    """How often each other file changed in the same commit as `file` (big sweeping commits ignored)."""
    counts: Counter = Counter()
    with store.connect() as con:
        rows = con.execute("SELECT files FROM commits WHERE files LIKE ?", (f'%"{file}"%',)).fetchall()
    for (files,) in rows:
        fs = json.loads(files or "[]")
        if file not in fs or len(fs) > max_commit_files:
            continue
        w = 1.0 / max(1, len(fs) - 1) ** 0.5  # small focused commits count more
        for f in fs:
            if f != file:
                counts[f] += w
    return counts
