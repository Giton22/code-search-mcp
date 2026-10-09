"""Retrieval evaluation.

Eval file: JSONL, one object per line
  {"query": "where are JWTs validated", "expected": ["src/auth/jwt.py"]}
  expected entries may be "path", "path:LINE" (hit if the result range covers the line) or
  "path::Symbol" (hit if the result's symbol ends with Symbol). Any expected entry counts as a hit.

`generate` builds a set automatically from git history: commit subject -> files it changed,
for small focused commits. Real-world queries, no labelling needed.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import history, indexer, search
from .store import Store

SKIP_SUBJECT = re.compile(r"^(merge|bump|release|version|revert|wip|fixup|squash|format|lint|typo|update \S+$)",
                          re.I)


def generate(root: Path, out: Path, n: int = 100, seed: int = 0) -> int:
    store = Store(root)
    history.update(root, store)
    with store.connect() as con:
        indexed = {r[0] for r in con.execute("SELECT path FROM files")}
    rows = []
    for c in history.commit_rows(store):
        files = [f for f in c["files"] if f in indexed and not search.is_test(f)]
        subj = c["subject"].strip()
        if not (1 <= len(files) <= 3) or len(subj.split()) < 4 or SKIP_SUBJECT.match(subj):
            continue
        rows.append({"query": subj, "expected": files, "commit": c["sha"][:10]})
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def _hit(row: dict, exp: str) -> bool:
    if "::" in exp:
        p, sym = exp.split("::", 1)
        return row["path"] == p and (row["symbol"] or "").endswith(sym)
    m = re.match(r"^(.*):(\d+)$", exp)
    if m:
        return row["path"] == m.group(1) and row["start"] <= int(m.group(2)) <= row["end"]
    return row["path"] == exp


def run(root: Path, eval_file: Path, k: int = 10, verbose: bool = False) -> dict:
    store = Store(root)
    indexer.index(root, quiet=True)
    items = [json.loads(x) for x in eval_file.read_text().splitlines() if x.strip()]
    ks = sorted({1, 3, 5, k})
    hits = {kk: 0 for kk in ks}
    rr = 0.0
    misses = []
    for it in items:
        res = search.search([store], it["query"], k=k)
        rank = next((i for i, h in enumerate(res, 1) if any(_hit(h["row"], e) for e in it["expected"])), None)
        for kk in ks:
            hits[kk] += int(rank is not None and rank <= kk)
        rr += 1.0 / rank if rank else 0.0
        if verbose and (rank is None or rank > 3):
            misses.append({"query": it["query"], "expected": it["expected"], "rank": rank,
                           "got": [f"{h['row']['path']}:{h['row']['start']}" for h in res[:3]]})
    n = max(len(items), 1)
    out = {"queries": len(items), **{f"recall@{kk}": round(hits[kk] / n, 3) for kk in ks},
           f"mrr@{k}": round(rr / n, 3)}
    if verbose:
        out["misses"] = misses[:20]
    return out
