"""Retrieval benchmark on real GitHub issues (SWE-bench Lite + SWE-bench Multilingual).

Each task is an issue from a public repo plus the patch that fixed it. The repo is checked out at the
commit before the fix, indexed, and searched with the issue text; a result is good when it ranks the
files the patch changed near the top. That is the question an agent asks first ("where do I fix
this?"), over 10 languages and dozens of repos, so it says whether a change helps in general rather
than on one repo.

Repos are fetched shallowly (one commit each) into the cache directory and indexed into a separate
bench index, so your own repos and their indexes are untouched.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from . import config

DATASETS = ["SWE-bench/SWE-bench_Lite", "SWE-bench/SWE-bench_Multilingual"]
ROWS_URL = "https://datasets-server.huggingface.co/rows?"
RERANKER = "BAAI/bge-reranker-v2-m3"  # "rerank"; "rerank:<model>" tries any other
LANG_BY_EXT = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
               ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby", ".php": "php",
               ".c": "c", ".h": "c", ".cc": "c++", ".cpp": "c++", ".hpp": "c++", ".cs": "c#", ".swift": "swift"}


def log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def _fetch_rows(dataset: str, cache: Path) -> list[dict]:
    path = cache / (dataset.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": "test",
                                    "offset": offset, "length": 100})
        with urllib.request.urlopen(ROWS_URL + q, timeout=60) as r:
            data = json.loads(r.read())
        batch = [x["row"] for x in data["rows"]]
        rows += [{k: x[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "patch")}
                 for x in batch]
        offset += len(batch)
        if not batch or offset >= data.get("num_rows_total", 0):
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def gold_files(patch: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/", patch, re.M)))


def pick_tasks(rows: list[dict], per_repo: int, n_repos: int, seed: int) -> list[dict]:
    """A deterministic, language-diverse sample: `per_repo` tasks from each of `n_repos` repos."""
    from .indexer import wanted

    rng = random.Random(seed)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        gold = [f for f in gold_files(r["patch"]) if wanted(f)]
        if gold:
            by_repo[r["repo"]].append({**r, "gold": gold})
    by_lang: dict[str, list[str]] = defaultdict(list)
    for repo, tasks in sorted(by_repo.items()):
        # the repo's language: the most common code file across its fixes (not a .json or .md they touched)
        langs = [LANG_BY_EXT[e] for t in tasks for g in t["gold"] if (e := Path(g).suffix.lower()) in LANG_BY_EXT]
        by_lang[max(set(langs), key=langs.count) if langs else "other"].append(repo)
    for repos in by_lang.values():
        rng.shuffle(repos)
    chosen, langs = [], sorted(by_lang)
    while len(chosen) < n_repos and any(by_lang.values()):
        for lang in langs:  # round-robin so every language is represented
            if by_lang[lang] and len(chosen) < n_repos:
                chosen.append((lang, by_lang[lang].pop()))
    out = []
    for lang, repo in chosen:
        tasks = sorted(by_repo[repo], key=lambda t: t["instance_id"])
        rng.shuffle(tasks)
        out += [{**t, "lang": lang} for t in tasks[:per_repo]]
    return out


def _git(cwd: Path, *args: str, timeout: float = 900) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=timeout)


def checkout(repo: str, sha: str, base: Path) -> Path:
    """The repo at `sha`, fetched shallowly; one working copy per repo, reused across its tasks."""
    root = base / repo.replace("/", "__")
    if not (root / ".git").exists():
        root.mkdir(parents=True, exist_ok=True)
        _git(root, "init", "-q")
        _git(root, "remote", "add", "origin", f"https://github.com/{repo}.git")
    _git(root, "-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "origin", sha)
    _git(root, "checkout", "-q", "-f", "--detach", "FETCH_HEAD")
    _git(root, "clean", "-q", "-fdx")
    return root


def _file_ranking(hits: list[dict]) -> list[str]:
    return list(dict.fromkeys(h["row"]["path"] for h in hits))


def _score(ranked: list[str], gold: list[str]) -> dict:
    ranks = [ranked.index(g) + 1 for g in gold if g in ranked]
    first = min(ranks) if ranks else None
    return {"hit@1": float(bool(first and first <= 1)), "hit@5": float(bool(first and first <= 5)),
            "hit@10": float(bool(first and first <= 10)),
            "recall@10": sum(1 for r in ranks if r <= 10) / len(gold),
            "mrr": 1.0 / first if first else 0.0, "first_rank": first}


def run(per_repo: int = 3, n_repos: int = 12, variants: list[str] | None = None, out: Path | None = None,
        seed: int = 0, max_files: int = 6000) -> dict:
    variants = [v.strip() for v in (variants or ["hybrid"]) if v.strip()]
    base = config.CFG.cache_dir / "bench"
    # a separate index so benchmark repos never show up next to the user's own
    os.environ["CODE_EMBED_CACHE"] = str(base / "index")
    os.environ["CODE_EMBED_DAEMON"] = "0"
    if any("jina-reranker-v3" in v for v in variants):
        os.environ["CODE_EMBED_TRUST_REMOTE_CODE"] = "1"  # asked for by name on the command line
    config.reload()
    from . import context, embed, indexer, search
    from .store import Store

    rows = [r for ds in DATASETS for r in _fetch_rows(ds, base / "datasets")]
    tasks = pick_tasks(rows, per_repo, n_repos, seed)
    log(f"{len(tasks)} tasks from {len({t['repo'] for t in tasks})} repos; variants: {', '.join(variants)}")
    embed.get()
    embed.RERANKERS_KEPT = max(1, sum(v.startswith("rerank") for v in variants))

    results, skipped, errors = [], [], {}
    for i, t in enumerate(tasks, 1):
        try:
            root = checkout(t["repo"], t["base_commit"], base / "repos")
        except (subprocess.SubprocessError, OSError) as e:
            skipped.append({"id": t["instance_id"], "why": f"checkout failed: {e}"})
            continue
        n_files = len(indexer.list_files(root))
        if n_files > max_files:
            skipped.append({"id": t["instance_id"], "why": f"{n_files} files > {max_files}"})
            log(f"skip {t['instance_id']}: {n_files} files")
            continue
        t0 = time.time()
        stats = indexer.index(root, quiet=True)
        log(f"[{i}/{len(tasks)}] {t['instance_id']} ({t['lang']}, {n_files} files, "
            f"indexed in {time.time() - t0:.0f}s, {stats['chunks_embedded']} new chunks)")
        store = Store(root)
        gold = [g for g in t["gold"] if (root / g).exists()] or t["gold"]
        row = {"id": t["instance_id"], "repo": t["repo"], "lang": t["lang"], "gold": gold, "variants": {}}
        q = t["problem_statement"]
        for v in variants:
            t1 = time.time()
            try:
                if v == "context":
                    res = context.build([store], q, budget=6000)
                    shown = [f for f in res["files"] if f in res["markdown"]]
                    sc = _score(shown, gold)
                    sc["tokens"] = res["tokens"]
                else:
                    mode = v if v in ("vector", "bm25") else "hybrid"
                    rr = RERANKER if v == "rerank" else v.split(":", 1)[1] if v.startswith("rerank:") else ""
                    hits = search.search([store], q, k=60, mode=mode, rerank_model=rr)
                    sc = _score(_file_ranking(hits), gold)
            except Exception as e:  # one broken variant shouldn't throw away the others; it scores zero
                log(f"{v} failed on {t['instance_id']}: {type(e).__name__}: {str(e)[:200]}")
                errors[v] = errors.get(v, 0) + 1
                sc = _score([], gold)
                if v == "context":
                    sc["tokens"] = 0
            sc["seconds"] = round(time.time() - t1, 3)
            row["variants"][v] = sc
        results.append(row)

    summary = summarize(results, variants)
    summary["skipped"] = skipped
    summary["errors"] = errors
    if out:
        out.write_text(json.dumps({"summary": summary, "tasks": results}, indent=2), encoding="utf-8")
    print(table(summary, variants), file=sys.stderr)
    if errors:
        log(f"failures (scored as misses): {errors}")
    return summary


def summarize(results: list[dict], variants: list[str]) -> dict:
    def agg(rows, v):
        keys = ["hit@1", "hit@5", "hit@10", "recall@10", "mrr", "seconds"] + (["tokens"] if v == "context" else [])
        return {k: round(statistics.mean(r["variants"][v][k] for r in rows), 3) for k in keys} if rows else {}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_lang[r["lang"]].append(r)
    return {"tasks": len(results),
            "overall": {v: agg(results, v) for v in variants},
            "by_language": {lang: {"tasks": len(rs), **{v: agg(rs, v) for v in variants}}
                            for lang, rs in sorted(by_lang.items())}}


def table(summary: dict, variants: list[str]) -> str:
    w = max(10, *(len(v) for v in variants))
    lines = [f"\n{summary['tasks']} tasks", f"{'variant':<{w}} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} "
                                           f"{'rec@10':>7} {'MRR':>6} {'sec':>6}"]
    for v in variants:
        s = summary["overall"].get(v) or {}
        if s:
            extra = f"  ~{s['tokens']:.0f} tokens" if "tokens" in s else ""
            lines.append(f"{v:<{w}} {s['hit@1']:>6.2f} {s['hit@5']:>6.2f} {s['hit@10']:>7.2f} "
                         f"{s['recall@10']:>7.2f} {s['mrr']:>6.2f} {s['seconds']:>6.2f}{extra}")
    lines.append("\nhit@10 by language: " + ", ".join(
        f"{lang} {d[variants[0]]['hit@10']:.2f} ({d['tasks']})" for lang, d in summary["by_language"].items()
        if d.get(variants[0])))
    return "\n".join(lines)
