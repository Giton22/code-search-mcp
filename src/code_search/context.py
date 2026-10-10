"""`prepare_context`: one call that hands an agent what it needs for a task, inside a token budget.

Instead of search -> read -> search -> open five files, the agent gets the most relevant code in
full, the next most relevant as signatures, what that code calls and who calls it, the tests that
cover it and the commits that last touched it. Everything carries a handle for `expand`.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import graph, search
from .store import Store

CHARS_PER_TOKEN = 3.6
PRIMARY = 3  # chunks shown in full
CANDIDATES = 16


def tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def handle(r: dict) -> str:
    return f"{r['path']}:{r['start']}-{r['end']}"


def _code(text: str, start: int, max_tokens: int) -> tuple[str, bool]:
    """Numbered code lines that fit `max_tokens`; True when lines were cut."""
    out, used = [], 0
    lines = text.splitlines()
    for i, line in enumerate(lines):
        row = f"{start + i:>5}  {line}"
        used += tokens(row) + 1
        if used > max_tokens and out:
            return "\n".join(out), True
        out.append(row)
    return "\n".join(out), False


def build(stores: list[Store], task: str, budget: int = 6000, path_glob: str | None = None) -> dict:
    """Context for `task`: {"markdown", "files" (ranked), "tokens"}."""
    t0 = time.time()
    hits = search.search(stores, task, k=CANDIDATES, path_glob=path_glob)
    if not hits:
        return {"markdown": "Nothing indexed matches this task.", "files": [], "tokens": 0}
    files = list(dict.fromkeys(h["row"]["path"] for h in hits))
    multi = len(stores) > 1

    def where(h):
        return f"[{h['repo']}] " if multi else ""

    # the best chunks in full, at most two per file so one big file can't take the whole budget
    primary, per_file = [], {}
    for h in hits:
        if len(primary) >= PRIMARY:
            break
        p = h["row"]["path"]
        if per_file.get(p, 0) < 2 and h["row"].get("kind") != "image":
            primary.append(h)
            per_file[p] = per_file.get(p, 0) + 1
    secondary = [h for h in hits if h not in primary]

    parts = [f"# Context for: {task.strip().splitlines()[0][:120]}"]
    used = tokens(parts[0])
    code_budget = int(budget * 0.55) // max(1, len(primary))

    parts.append("\n## Most relevant code")
    for h in primary:
        r, store = h["row"], stores[h["store"]]
        kind = f" · {r['kind'] or 'symbol'} `{r['symbol']}`" if r.get("symbol") else ""
        body, cut = _code(r["text"], r["start"], code_budget)
        block = [f"\n### {where(h)}{handle(r)}{kind}", f"```{r.get('lang') or ''}", body, "```"]
        if cut:
            block.append(f"(cut to fit; `expand(\"{handle(r)}\")` for the rest)")
        # what this code calls, and who calls it
        calls = [c for c in graph.callees(store, r["id"], r["path"]) if c["path"] != r["path"]][:6]
        if calls:
            block.append("Calls: " + "; ".join(
                f"`{c['symbol']}` {c['path']}:{c['start']}" for c in calls))
        name = graph.def_name(r.get("symbol"))
        if name and r.get("kind") in graph.DEF_KINDS and len(name) > 2 and name.lower() not in search.GENERIC:
            refs = graph.references(store, name, k=5)
            if refs["total"]:
                uses = [f"{f}:{u[0]['lines'][0][0]}" for f, u in refs["files"].items()]
                more = refs["total"] - len(uses)
                block.append(f"Used by: {', '.join(uses)}" + (f" (+{more} more; `find_references(\"{name}\")`)"
                                                               if more > 0 else ""))
        text = "\n".join(block)
        if used + tokens(text) > budget * 0.8 and len(parts) > 2:
            secondary.insert(0, h)
            continue
        parts.append(text)
        used += tokens(text)

    if secondary:
        lines = ["\n## Also relevant (signature only; `expand(handle)` for the code)"]
        for h in secondary:
            r = h["row"]
            sig = (r.get("signature") or "").strip() or (r["text"].strip().splitlines() or [""])[0].strip()
            line = f"- {where(h)}`{handle(r)}`" + (f" `{r['symbol']}`" if r.get("symbol") else "") + \
                   (f" — {sig[:140]}" if sig else "")
            if used + tokens(line) > budget * 0.9:
                break
            lines.append(line)
            used += tokens(line)
        if len(lines) > 1:
            parts.append("\n".join(lines))

    top_store, top_file = stores[primary[0]["store"]] if primary else stores[0], files[0]
    try:
        tests = search.find_tests(top_store, top_file, k=3)
    except ValueError:
        tests = []
    if tests:
        text = "\n## Tests\n" + "\n".join(
            f"- {t['path']}" + (f" (best match: {t['best_test']})" if t["best_test"] else "") for t in tests)
        if used + tokens(text) <= budget * 0.95:
            parts.append(text)
            used += tokens(text)

    try:
        commits, _ = search.history_hits(top_store, task, k=3)
    except Exception:  # history is a bonus; never fail the whole call over it
        commits = []
    commits = [c for c in commits if c["score"] > 0.35]
    if commits:
        text = "\n## Related commits\n" + "\n".join(
            f"- {c['sha'][:10]} {time.strftime('%Y-%m-%d', time.localtime(c['ts']))} {c['subject'][:100]}"
            for c in commits)
        if used + tokens(text) <= budget:
            parts.append(text)
            used += tokens(text)

    parts.append(f"\n_{len(files)} files considered in {time.time() - t0:.1f}s; "
                 "`expand(handle)` shows full code, `find_references(name)` every use._")
    md = "\n".join(parts)
    return {"markdown": md, "files": files, "tokens": tokens(md)}


_RANGE = re.compile(r"^(?P<path>.+?):(?P<a>\d+)(?:-(?P<b>\d+))?$")


def expand(store: Store, ref: str, max_lines: int = 400) -> str:
    """Full code for a handle: "path:12-40", "path:12", "path::Symbol", "Symbol" or "path"."""
    ref = ref.strip().strip("`")
    m = _RANGE.match(ref)
    if "::" in ref:
        path, sym = ref.split("::", 1)
        rows = [r for r in graph.definitions(store, sym) if r["path"] == path] or graph.definitions(store, sym)
        return _show_defs(store, rows, ref)
    if m and (store.root / m["path"]).is_file():
        a = int(m["a"])
        b = int(m["b"] or a)
        if not m["b"]:  # a single line: the chunk around it
            with store.connect() as con:
                row = con.execute("SELECT start, end FROM chunks WHERE path=? AND start<=? AND end>=? "
                                  "ORDER BY end-start LIMIT 1", (m["path"], a, a)).fetchone()
            a, b = row if row else (max(1, a - 20), a + 20)
        return _read(store.root, m["path"], a, b, max_lines)
    if (store.root / ref).is_file():
        return _read(store.root, ref, 1, max_lines, max_lines)
    return _show_defs(store, graph.definitions(store, ref), ref)


def _show_defs(store: Store, rows: list[dict], ref: str) -> str:
    if not rows:
        return f"Nothing found for {ref!r}. Use a handle like path:12-40, path::Symbol, a symbol name or a file."
    out = []
    for r in rows[:3]:
        out.append(_read(store.root, r["path"], r["start"], r["end"], 400))
    if len(rows) > 3:
        out.append(f"… {len(rows) - 3} more definitions: " + ", ".join(handle(r) for r in rows[3:10]))
    return "\n\n".join(out)


def _read(root: Path, rel: str, a: int, b: int, max_lines: int) -> str:
    """Lines a..b of a file as it is on disk now (fresher than the index)."""
    fp = (root / rel).resolve()
    if root.resolve() not in fp.parents:
        raise ValueError("path is outside the repo")
    lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    b = min(b, len(lines), a + max_lines - 1)
    body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(a, b + 1))
    more = f"\n… file continues to line {len(lines)}" if b < len(lines) and b - a + 1 >= max_lines else ""
    return f"{rel}:{a}-{b}\n```\n{body}\n```{more}"
