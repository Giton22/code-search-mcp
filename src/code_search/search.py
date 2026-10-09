"""Search and analysis over one or more indexed repos."""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from . import config, embed, history
from .embed import tokens
from .indexer import mean_vector
from .store import Store

TEST_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec|specs|testing|test_?utils?)/|(^|/)test_[^/]+$|_test\.[^/.]+$|"
    r"\.(test|spec)\.[^/]+$|Tests?\.[^/.]+$|_spec\.rb$|(^|/)conftest\.py$"
)
DEF_KINDS = {"function", "method", "class", "interface", "trait", "impl", "struct", "enum", "module",
             "type", "record", "constructor", "macro", "namespace", "protocol", "test", "block"}
GENERIC = {"main", "index", "init", "__init__", "utils", "util", "helpers", "common", "test", "tests",
           "config", "types", "constants", "mod", "lib", "app", "self", "this", "new", "get", "set",
           "run", "data", "value", "name", "id", "len", "str", "int", "list", "dict", "map", "string"}
HISTORY_PREFIX = "task: search result | query: "
RRF_K = 60


# ---------------------------------------------------------------- helpers


def is_test(path: str) -> bool:
    return bool(TEST_RE.search(path))


def fts_query(q: str, limit: int = 24) -> str:
    seen, out = set(), []
    for t in tokens(q):
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        out.append('"' + t.replace('"', "") + '"')
        if len(out) >= limit:
            break
    return " OR ".join(out)


def glob_match(path: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    p = PurePosixPath(path)
    pats = [x.strip() for x in pattern.split(",") if x.strip()]
    for pat in pats:
        neg = pat.startswith("!")
        pat = pat[1:] if neg else pat
        hit = p.match(pat) or p.match(f"**/{pat}") or path.startswith(pat.rstrip("*/") + "/")
        if neg and hit:
            return False
        if not neg and hit:
            return True
    return all(x.startswith("!") for x in pats)


def _sim_lookup(ids: np.ndarray, sims: np.ndarray, cid: int) -> float:
    i = int(np.searchsorted(ids, cid))
    return float(sims[i]) if i < len(ids) and ids[i] == cid else 0.0


def _fmt_header(n, row, score, repo=None):
    h = f"### {n}. {row['path']}:{row['start']}-{row['end']}"
    if row.get("symbol") and row.get("kind") != "image":
        h += f" · {row['kind'] or 'symbol'} `{row['symbol']}`"
    if row.get("kind") == "image":
        h += " · image"
    if repo:
        h += f" · [{repo}]"
    return h + f"  (score {score:.3f})"


def format_hits(hits: list[dict], snippet_lines: int = 25, multi: bool = False) -> str:
    if not hits:
        return "No matches."
    out = []
    for n, h in enumerate(hits, 1):
        row = h["row"]
        head = _fmt_header(n, row, h["score"], h["repo"] if multi else None)
        if row.get("kind") == "image":
            out.append(head)
            continue
        lines = row["text"].splitlines()
        snippet = "\n".join(lines[:snippet_lines]) + ("\n…" if len(lines) > snippet_lines else "")
        out.append(f"{head}\n```{row.get('lang') or ''}\n{snippet}\n```")
    return "\n\n".join(out)


# ---------------------------------------------------------------- core search


def search(stores: list[Store], query: str | None = None, k: int = 8, path_glob: str | None = None,
           kinds: list[str] | None = None, qvec: np.ndarray | None = None,
           include_tests: bool = True) -> list[dict]:
    """Hybrid semantic + BM25 search with reciprocal-rank fusion, symbol boost and optional reranking."""
    emb = embed.get()
    qv = qvec if qvec is not None else emb.embed_query(query or "")
    fq = fts_query(query) if query else ""
    qtoks = set(tokens(query)) if query else set()
    qlow = (query or "").strip().lower()
    n_sem = max(k * 10, 100)

    cand: dict[tuple[int, int], dict] = {}
    sem_all: list[tuple[float, int, int]] = []
    per_store = []
    for si, st in enumerate(stores):
        ids, mat = st.matrix()
        if not len(ids):
            per_store.append(None)
            continue
        sims = mat @ qv
        n = min(n_sem, len(ids))
        top = np.argpartition(-sims, n - 1)[:n]
        sem_all += [(float(sims[i]), si, int(ids[i])) for i in top]
        kw = []
        if fq and st.fts:
            try:
                with st.connect() as con:
                    kw = [r[0] for r in con.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                        "ORDER BY bm25(chunks_fts, 1.0, 4.0, 0.5) LIMIT 100", (fq,))]
            except sqlite3.OperationalError:
                kw = []
        per_store.append((ids, sims, kw))

    sem_all.sort(reverse=True)
    for r, (s, si, cid) in enumerate(sem_all):
        e = cand.setdefault((si, cid), {"sim": s, "fused": 0.0})
        e["fused"] += 1.0 / (RRF_K + r)
    for si, ps in enumerate(per_store):
        if not ps:
            continue
        ids, sims, kw = ps
        for r, cid in enumerate(kw):
            e = cand.setdefault((si, cid), {"sim": _sim_lookup(ids, sims, cid), "fused": 0.0})
            e["fused"] += 0.6 / (RRF_K + r)
    if not cand:
        return []

    # fetch rows for candidates, grouped per store
    rows: dict[tuple[int, int], dict] = {}
    by_store: dict[int, list[int]] = defaultdict(list)
    for si, cid in cand:
        by_store[si].append(cid)
    for si, cids in by_store.items():
        with stores[si].connect() as con:
            for cid, row in stores[si].chunk_rows(con, cids).items():
                rows[(si, cid)] = row

    kinds_set = {x.lower() for x in kinds} if kinds else None
    for key, e in cand.items():
        row = rows.get(key)
        if not row:
            e["fused"] = -1
            continue
        sym = (row["symbol"] or "").lower()
        last = sym.rsplit(".", 1)[-1]
        if sym and (sym == qlow or last == qlow):
            e["fused"] += 2.0 / RRF_K  # exact symbol name typed as the query
        elif last and last in qtoks and len(last) > 3:
            e["fused"] += 0.5 / RRF_K

    ordered = sorted(cand.items(), key=lambda kv: kv[1]["fused"], reverse=True)
    rr_n = max(k, config.CFG.rerank_top) if config.CFG.rerank_model and query else k
    picked, spans = [], []
    for (si, cid), e in ordered:
        row = rows.get((si, cid))
        if not row or e["fused"] < 0:
            continue
        if not glob_match(row["path"], path_glob):
            continue
        if kinds_set and (row["kind"] or "").lower() not in kinds_set:
            continue
        if not include_tests and is_test(row["path"]):
            continue
        if any(s == si and p == row["path"] and row["start"] <= b and row["end"] >= a for s, p, a, b in spans):
            continue
        spans.append((si, row["path"], row["start"], row["end"]))
        picked.append({"store": si, "repo": stores[si].root.name, "root": str(stores[si].root),
                       "id": cid, "score": e["sim"], "row": row})
        if len(picked) >= rr_n:
            break

    if config.CFG.rerank_model and query and len(picked) > 1:
        docs = [f"{h['row']['path']} {h['row']['symbol'] or ''}\n{h['row']['text']}" for h in picked]
        scores = embed.rerank(query, docs)
        if scores is not None:
            for h, s in zip(picked, scores):
                h["rerank"] = float(s)
            picked.sort(key=lambda h: h["rerank"], reverse=True)
            for h in picked:
                h["score"] = h["rerank"]
    return picked[:k]


# ---------------------------------------------------------------- similar code


def find_similar(store: Store, file: str, line: int, k: int = 6, cross_file_only: bool = False) -> list[dict]:
    with store.connect() as con:
        row = con.execute(
            "SELECT id FROM chunks WHERE path=? AND start<=? AND end>=? ORDER BY end-start LIMIT 1",
            (file, line, line)).fetchone()
    if not row:
        return []
    ids, mat = store.matrix()
    src = int(np.searchsorted(ids, row[0]))
    sims = mat @ mat[src]
    order = np.argsort(-sims)[: max(k * 6, 60)]
    with store.connect() as con:
        meta = store.chunk_rows(con, [int(ids[i]) for i in order])
    out = []
    for i in order:
        cid = int(ids[i])
        r = meta.get(cid)
        if cid == row[0] or not r:
            continue
        if r["path"] == file and (cross_file_only or r["start"] <= line <= r["end"]):
            continue
        out.append({"store": 0, "repo": store.root.name, "id": cid, "score": float(sims[i]), "row": r})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- related files


def _symbols_by_file(store: Store) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    defs: dict[str, set[str]] = defaultdict(set)  # file -> names it defines
    where: dict[str, set[str]] = defaultdict(set)  # name -> files defining it
    with store.connect() as con:
        for path, sym, kind in con.execute("SELECT path, symbol, kind FROM chunks WHERE symbol != ''"):
            if kind not in DEF_KINDS:
                continue
            name = sym.rsplit(".", 1)[-1]
            if len(name) < 3 or name.lower() in GENERIC:
                continue
            defs[path].add(name)
            where[name].add(path)
    return defs, where


def _mentions(store: Store, names: list[str], limit: int = 300) -> Counter:
    """Files whose chunks mention any of `names` (BM25 keyword index), counted per file."""
    counts: Counter = Counter()
    if not names or not store.fts:
        return counts
    q = " OR ".join('"' + n.replace('"', "") + '"' for n in names[:40])
    try:
        with store.connect() as con:
            for (path,) in con.execute(
                    "SELECT path FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?", (f"text : ({q})", limit)):
                counts[path] += 1
    except sqlite3.OperationalError:
        pass
    return counts


def related_files(store: Store, file: str, k: int = 10) -> list[dict]:
    with store.connect() as con:
        if not con.execute("SELECT 1 FROM files WHERE path=?", (file,)).fetchone():
            raise ValueError(f"{file} is not indexed (use a path relative to the repo root)")
        text = "\n".join(r[0] for r in con.execute("SELECT text FROM chunks WHERE path=? ORDER BY start", (file,)))
        n_commits = con.execute("SELECT COUNT(*) FROM commits").fetchone()[0]

    scores: dict[str, dict] = defaultdict(lambda: {"cochange": 0.0, "refs": 0.0, "sem": 0.0, "why": []})

    # 1) changed together in git history
    if n_commits == 0:
        history.update(store.root, store)
    co = history.cochange(store, file)
    if co:
        mx = max(co.values())
        for f, v in co.most_common(50):
            scores[f]["cochange"] = v / mx
            scores[f]["why"].append(f"changed together ({v:.1f} weighted commits)")

    # 2) references in both directions
    defs, where = _symbols_by_file(store)
    mine = sorted(defs.get(file, ()), key=len, reverse=True)
    stem = Path(file).stem
    used_by = _mentions(store, mine + ([stem] if len(stem) > 3 and stem.lower() not in GENERIC else []))
    used_by.pop(file, None)
    ident = {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)}
    uses: Counter = Counter()
    uses_names: dict[str, list[str]] = defaultdict(list)
    for name in ident:
        fs = where.get(name)
        if fs and len(fs) <= 5:
            for f in fs:
                if f != file:
                    uses[f] += 1
                    uses_names[f].append(name)
    ref = Counter()
    ref.update(used_by)
    ref.update({f: v * 1.5 for f, v in uses.items()})
    if ref:
        mx = max(ref.values())
        for f, v in ref.most_common(60):
            scores[f]["refs"] = v / mx
            if f in uses_names:
                scores[f]["why"].append("uses " + ", ".join(sorted(uses_names[f])[:4]))
            if used_by.get(f):
                scores[f]["why"].append(f"references this file ({used_by[f]}×)")

    # 3) semantic similarity of whole files
    cents = mean_vector(store, [])
    if file in cents:
        me = cents[file]
        sims = sorted(((float(v @ me), f) for f, v in cents.items() if f != file), reverse=True)[:40]
        for s, f in sims:
            scores[f]["sem"] = max(0.0, s)
            if s > 0.5:
                scores[f]["why"].append(f"similar content ({s:.2f})")

    out = []
    for f, s in scores.items():
        total = 0.4 * s["cochange"] + 0.35 * s["refs"] + 0.25 * s["sem"]
        if total <= 0:
            continue
        out.append({"path": f, "score": round(total, 3), "test": is_test(f), "why": s["why"][:3]})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:k]


# ---------------------------------------------------------------- tests


def _test_name_match(src: str, test: str) -> bool:
    s = Path(src).stem.lower()
    t = Path(test).name.lower()
    t_stem = t.split(".")[0]
    return (t_stem in (f"test_{s}", f"{s}_test", f"{s}test", f"{s}tests", f"{s}_spec", f"test{s}")
            or t.startswith(f"{s}.test.") or t.startswith(f"{s}.spec."))


def find_tests(store: Store, target: str, k: int = 8) -> list[dict]:
    """target: 'path/to/file.py', 'path/to/file.py:42' or a symbol name."""
    target = target.strip()
    file, line = target, None
    m = re.match(r"^(.*?):(\d+)$", target)
    if m:
        file, line = m.group(1), int(m.group(2))
    with store.connect() as con:
        is_file = con.execute("SELECT 1 FROM files WHERE path=?", (file,)).fetchone() is not None
        if is_file:
            q = "SELECT id, symbol FROM chunks WHERE path=?" + (" AND start<=? AND end>=?" if line else "")
            rows = con.execute(q, (file, line, line) if line else (file,)).fetchall()
            files = [file]
        else:
            rows = con.execute("SELECT id, symbol FROM chunks WHERE symbol=? OR symbol LIKE ?",
                               (target, f"%.{target}")).fetchall()
            files = sorted({r[0] for r in con.execute(
                "SELECT path FROM chunks WHERE symbol=? OR symbol LIKE ?", (target, f"%.{target}"))})
        all_files = [r[0] for r in con.execute("SELECT path FROM files")]
    if not rows:
        raise ValueError(f"Nothing indexed matches {target!r}")
    names = {r[1].rsplit(".", 1)[-1] for r in rows if r[1]}
    names = {n for n in names if len(n) > 2 and n.lower() not in GENERIC}
    if not is_file:
        names.add(target.rsplit(".", 1)[-1])
    stems = {Path(f).stem for f in files if len(Path(f).stem) > 3 and Path(f).stem.lower() not in GENERIC}
    tests = [f for f in all_files if is_test(f) and f not in files]
    if not tests:
        return []

    ids, mat = store.matrix()
    target_ids = [r[0] for r in rows]
    tv = mat[np.searchsorted(ids, target_ids)].mean(axis=0)
    tv /= max(np.linalg.norm(tv), 1e-12)

    score: dict[str, float] = defaultdict(float)
    why: dict[str, list[str]] = defaultdict(list)
    for t in tests:
        if any(_test_name_match(f, t) for f in files):
            score[t] += 3.0
            why[t].append("naming convention")
    mentions = _mentions(store, sorted(names | stems), limit=1000)
    for t, n in mentions.items():
        if t in tests:
            score[t] += min(3.0, 1.0 + 0.5 * n)
            why[t].append(f"mentions {', '.join(sorted(names)[:3]) or 'module'} ({n} chunks)")

    best_chunk: dict[str, tuple[float, int]] = {}
    with store.connect() as con:
        q = ",".join("?" * len(tests))
        trows = con.execute(f"SELECT id, path FROM chunks WHERE path IN ({q})", tests).fetchall()
    for cid, p in trows:
        i = int(np.searchsorted(ids, cid))
        if i < len(ids) and ids[i] == cid:
            s = float(mat[i] @ tv)
            if p not in best_chunk or s > best_chunk[p][0]:
                best_chunk[p] = (s, cid)
    for p, (s, _) in best_chunk.items():
        score[p] += 2.0 * max(0.0, s)

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    with store.connect() as con:
        meta = store.chunk_rows(con, [best_chunk[p][1] for p, _ in ranked if p in best_chunk])
    out = []
    for p, s in ranked:
        bc = meta.get(best_chunk[p][1]) if p in best_chunk else None
        out.append({"path": p, "score": round(s, 3), "why": why[p][:2],
                    "best_test": f"{bc['symbol'] or 'chunk'} (lines {bc['start']}-{bc['end']})" if bc else None})
    return out


# ---------------------------------------------------------------- explain / cluster


def _kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means with k-means++ init. x rows are unit vectors."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = 1 - np.max(np.stack([x @ c for c in centers], axis=1), axis=1)
        d = np.clip(d, 0, None) ** 2
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1 / n)
        centers.append(x[rng.choice(n, p=p)])
    c = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = x[labels == j]
            if len(m):
                v = m.mean(axis=0)
                c[j] = v / max(np.linalg.norm(v), 1e-12)
    return labels, c


def explain_area(store: Store, path_glob: str | None = None, clusters: int | None = None,
                 include_tests: bool = False) -> str:
    cents = mean_vector(store, [])
    files = sorted(f for f in cents if glob_match(f, path_glob) and (include_tests or not is_test(f)))
    if not files:
        return "No indexed files match."
    if len(files) < 4:
        return "Too few files to cluster:\n" + "\n".join(f"- {f}" for f in files)
    x = np.stack([cents[f] for f in files])
    k = clusters or int(np.clip(round((len(files) / 3) ** 0.5), 2, 12))
    k = min(k, len(files))
    labels, centers = _kmeans(x, k)

    defs, _ = _symbols_by_file(store)
    file_toks = {}
    df: Counter = Counter()
    for f in files:
        toks = set()
        for part in Path(f).with_suffix("").parts:
            toks.update(t for t in tokens(part) if len(t) > 2)
        for s in defs.get(f, ()):
            toks.update(t for t in tokens(s) if len(t) > 2)
        toks -= GENERIC
        file_toks[f] = toks
        df.update(toks)

    groups = []
    for j in range(k):
        members = [files[i] for i in np.where(labels == j)[0]]
        if not members:
            continue
        tf = Counter()
        for f in members:
            tf.update(file_toks[f])
        terms = sorted(tf, key=lambda t: tf[t] / len(members) * np.log(len(files) / df[t]), reverse=True)[:6]
        dirs = Counter("/".join(Path(f).parts[:2]) if len(Path(f).parts) > 2 else (Path(f).parts[0] if len(Path(f).parts) > 1 else ".")
                       for f in members)
        sims = sorted(((float(cents[f] @ centers[j]), f) for f in members), reverse=True)
        syms = Counter()
        for f in members:
            for s in defs.get(f, ()):
                syms[s] += 1
        key_syms = [s for _, f in sims[:6] for s in sorted(defs.get(f, ()), key=len, reverse=True)[:2]][:8]
        groups.append((len(members), terms, dirs.most_common(3), [f for _, f in sims[:6]], key_syms))

    groups.sort(key=lambda g: g[0], reverse=True)
    out = [f"# {store.root.name}: {len(files)} files in {len(groups)} areas"
           + (f" (filter: {path_glob})" if path_glob else "")]
    for n, (size, terms, dirs, reps, key_syms) in enumerate(groups, 1):
        out.append(f"\n## {n}. {' · '.join(terms[:4]) or 'misc'}  ({size} files)")
        out.append("Dirs: " + ", ".join(f"{d} ({c})" for d, c in dirs))
        out.append("Core files: " + ", ".join(reps))
        if key_syms:
            out.append("Key symbols: " + ", ".join(f"`{s}`" for s in key_syms))
    return "\n".join(out)


# ---------------------------------------------------------------- history


def search_history(store: Store, query: str, k: int = 8, path: str | None = None) -> str:
    info = history.update(store.root, store)
    rows = history.commit_rows(store, with_vec=True)
    if not rows:
        return info.get("note") or "No commit history indexed."
    if path:
        rows = [r for r in rows if any(f == path or f.startswith(path.rstrip("/") + "/") for f in r["files"])]
        if not rows:
            return f"No commits touching {path}."
    qv = embed.get().embed_query(query, prefix=HISTORY_PREFIX)
    mat = np.stack([r["vec"] for r in rows])
    sims = mat @ qv
    fused = defaultdict(float)
    for rank, i in enumerate(np.argsort(-sims)[:200]):
        fused[rows[i]["sha"]] += 1.0 / (RRF_K + rank)
    fq = fts_query(query)
    if fq and store.fts:
        try:
            with store.connect() as con:
                kw = [r[0] for r in con.execute(
                    "SELECT c.sha FROM commits_fts f JOIN commits c ON c.rowid=f.rowid "
                    "WHERE commits_fts MATCH ? ORDER BY bm25(commits_fts, 3.0, 1.0, 1.0, 2.0) LIMIT 100", (fq,))]
            allowed = {r["sha"] for r in rows}
            for rank, sha in enumerate(s for s in kw if s in allowed):
                fused[sha] += 0.6 / (RRF_K + rank)
        except sqlite3.OperationalError:
            pass
    by_sha = {r["sha"]: (r, float(s)) for r, s in zip(rows, sims)}
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    out = []
    for n, sha in enumerate(top, 1):
        r, s = by_sha[sha]
        date = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
        out.append(f"### {n}. {sha[:10]} · {date} · {r['author']}  (score {s:.3f})\n{r['subject']}")
        if r["body"]:
            out.append("> " + r["body"].strip().splitlines()[0][:200])
        if r["symbols"]:
            out.append("Changed: " + ", ".join(f"`{x}`" for x in r["symbols"][:10]))
        if r["files"]:
            more = f" (+{len(r['files']) - 8} more)" if len(r["files"]) > 8 else ""
            out.append("Files: " + ", ".join(r["files"][:8]) + more)
    out.append("\nUse `git show <sha>` for the full diff.")
    return "\n".join(out)
