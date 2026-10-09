"""MCP server exposing code search tools over stdio."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from mcp.types import ToolAnnotations

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import config, embed, history, indexer, search
from .store import Store, all_stores
from .watcher import WATCHERS

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
FIRST_INDEX_WAIT = 40.0  # seconds a tool waits for a first index before answering from a partial one

INSTRUCTIONS = """\
Local semantic code search (EmbeddingGemma 2) over the user's repos; nothing leaves the machine.
Prefer these tools over grep/find when you don't know where something lives:
- search_code: find code by meaning or by symbol name. Omit `path` to use the current project.
- related_files: before editing a file, see what usually changes with it (callers, tests, configs).
- find_tests: tests covering a file or symbol. find_similar: duplicates / patterns to follow.
- search_history: why/when something changed. explain_area: map of an unfamiliar repo.
Indexes update automatically in the background; results may say indexing is in progress.
"""

mcp = _Server("code-search", instructions=INSTRUCTIONS)

_jobs: dict[str, tuple[threading.Thread, dict]] = {}
_jobs_lock = threading.Lock()


def _start_index(root: Path) -> tuple[threading.Thread, dict]:
    with _jobs_lock:
        job = _jobs.get(str(root))
        if job and job[0].is_alive():
            return job
        progress: dict = {}

        def run():
            try:
                indexer.index(root, progress=progress)
                WATCHERS.ensure(root)
            except Exception as e:
                progress["error"] = f"{type(e).__name__}: {e}"
                indexer.log(f"index of {root} failed: {progress['error']}")

        t = threading.Thread(target=run, daemon=True, name=f"index-{root.name}")
        t.start()
        _jobs[str(root)] = (t, progress)
        return t, progress


def _prepare(path: str | None) -> tuple[Store, str]:
    """Resolve the repo, make sure the index is reasonably fresh. Returns (store, note)."""
    root = indexer.resolve_root(path)
    store = Store(root)
    with store.connect() as con:
        has_files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0] > 0
    if has_files and WATCHERS.active(root):
        job = _jobs.get(str(root))
        if job and job[0].is_alive():
            p = job[1]
            return store, f"_(indexing {p.get('done', 0)}/{p.get('total', '?')} files in background)_"
        return store, ""
    t, progress = _start_index(root)
    t.join(timeout=FIRST_INDEX_WAIT if not has_files else 15)
    if progress.get("error"):
        raise RuntimeError(f"Indexing failed: {progress['error']}")
    if t.is_alive():
        return store, (f"_Indexing {root.name} in the background ({progress.get('done', 0)}/"
                       f"{progress.get('total', '?')} files so far); results are partial — try again shortly._")
    return store, ""


def _stores(path: str | None, repos: list[str] | None) -> tuple[list[Store], str]:
    notes = []
    if repos:
        if any(r.strip().lower() == "all" for r in repos):
            paths = {str(s.root) for s in all_stores()} | {str(indexer.resolve_root(p)) for p in config.CFG.repos}
        else:
            paths = {str(indexer.resolve_root(p)) for p in repos}
        stores = []
        for p in sorted(paths):
            s, note = _prepare(p)
            stores.append(s)
            if note:
                notes.append(note)
        return stores, "\n".join(notes)
    s, note = _prepare(path)
    return [s], note


def _with_note(body: str, note: str) -> str:
    return f"{note}\n\n{body}" if note else body


# ---------------------------------------------------------------- tools


@mcp.tool(annotations=READ_ONLY)
def search_code(query: str, path: str | None = None, repos: list[str] | None = None, k: int = 8,
                path_glob: str | None = None, kinds: list[str] | None = None,
                include_tests: bool = True, snippet_lines: int = 25) -> str:
    """Find code by meaning ("where do we retry failed HTTP calls") or by name ("UserStore.save").
    Hybrid of EmbeddingGemma 2 vectors and BM25 keyword search; returns file:line ranges with the
    enclosing symbol and a snippet.

    Args:
      query: natural language, a symbol name, or a code fragment
      path: any directory in the repo (default: current project)
      repos: search several repos at once: list of paths, or ["all"] for every indexed repo
      k: number of results
      path_glob: filter, e.g. "src/**/*.ts", "*.py", or "!tests/**" to exclude; comma-separated
      kinds: only these chunk kinds, e.g. ["function", "method", "class"]
      include_tests: set false to hide test files
      snippet_lines: max lines per snippet
    """
    stores, note = _stores(path, repos)
    hits = search.search(stores, query, k=k, path_glob=path_glob, kinds=kinds, include_tests=include_tests)
    return _with_note(search.format_hits(hits, snippet_lines, multi=len(stores) > 1), note)


@mcp.tool(annotations=READ_ONLY)
def search_by_image(image_path: str, path: str | None = None, repos: list[str] | None = None, k: int = 6) -> str:
    """Find code related to a screenshot, mockup or diagram (EmbeddingGemma 2 puts images and code in the
    same space). `image_path` is a local file. Works best with CODE_EMBED_IMAGES=1 so repo images are indexed too."""
    p = Path(image_path).expanduser()
    if not p.is_file():
        raise ValueError(f"No such image: {p}")
    vec = embed.get().embed_images([str(p)])
    if vec is None:
        raise RuntimeError("The current embedding backend can't embed images (needs the sentence-transformers "
                           "backend with an EmbeddingGemma 2 version that supports image input).")
    stores, note = _stores(path, repos)
    hits = search.search(stores, None, k=k, qvec=vec[0])
    return _with_note(search.format_hits(hits, 20, multi=len(stores) > 1), note)


@mcp.tool(annotations=READ_ONLY)
def find_similar(file: str, line: int, path: str | None = None, k: int = 6, other_files_only: bool = False) -> str:
    """Code similar to the chunk at file:line — spot duplicated logic or find an existing pattern to follow
    before writing a new helper. `file` is relative to the repo root."""
    store, note = _prepare(path)
    hits = search.find_similar(store, file, line, k=k, cross_file_only=other_files_only)
    if not hits:
        return _with_note(f"No indexed chunk at {file}:{line}.", note)
    return _with_note(search.format_hits(hits, 20), note)


@mcp.tool(annotations=READ_ONLY)
def related_files(file: str, path: str | None = None, k: int = 10) -> str:
    """Files likely to need changes together with `file`: combines git co-change history, references in both
    directions (who calls it / what it uses) and semantic similarity. Use before editing to find callers,
    tests and configs you might otherwise miss. `file` is relative to the repo root."""
    store, note = _prepare(path)
    rows = search.related_files(store, file, k=k)
    if not rows:
        return _with_note(f"No related files found for {file}.", note)
    lines = [f"Related to `{file}`:"]
    for r in rows:
        tag = " [test]" if r["test"] else ""
        lines.append(f"- {r['path']}{tag}  ({r['score']}) — {'; '.join(r['why']) or 'related'}")
    return _with_note("\n".join(lines), note)


@mcp.tool(annotations=READ_ONLY)
def find_tests(target: str, path: str | None = None, k: int = 8) -> str:
    """Tests that cover a file ("src/auth.py"), a location ("src/auth.py:42") or a symbol ("validate_token").
    Uses naming conventions, references to the target's symbols, and semantic similarity."""
    store, note = _prepare(path)
    rows = search.find_tests(store, target, k=k)
    if not rows:
        return _with_note(f"No tests found for {target}.", note)
    lines = [f"Tests for `{target}`:"]
    for r in rows:
        best = f" — best match: {r['best_test']}" if r["best_test"] else ""
        lines.append(f"- {r['path']}  ({r['score']}; {'; '.join(r['why']) or 'similar'}){best}")
    return _with_note("\n".join(lines), note)


@mcp.tool(annotations=READ_ONLY)
def explain_area(path: str | None = None, path_glob: str | None = None, clusters: int | None = None) -> str:
    """Map of a codebase: groups files into functional areas by embedding similarity and labels each with
    its main terms, directories, core files and key symbols. Good first call in an unfamiliar repo.
    Narrow with path_glob (e.g. "src/api/**")."""
    store, note = _prepare(path)
    return _with_note(search.explain_area(store, path_glob, clusters), note)


@mcp.tool(annotations=READ_ONLY)
def search_history(query: str, path: str | None = None, file: str | None = None, k: int = 8) -> str:
    """Search git history by meaning: "when did we change retry logic and why", "commit that added rate limiting".
    Indexes commit messages, changed files and changed function names. `file` limits to commits touching that
    file or directory."""
    store, note = _prepare(path)
    return _with_note(search.search_history(store, query, k=k, path=file), note)


@mcp.tool(annotations=READ_ONLY)
def index_repo(path: str | None = None, include_history: bool = True) -> dict:
    """Build or update the index now (normally automatic). Only changed files are re-embedded, and chunks seen
    before (e.g. on another branch) come from the embedding cache."""
    root = indexer.resolve_root(path)
    stats = indexer.index(root)
    WATCHERS.ensure(root)
    if include_history:
        stats["history"] = history.update(root)
    return stats


@mcp.tool(annotations=READ_ONLY)
def index_status(path: str | None = None, all_repos: bool = False) -> dict | list:
    """What is indexed: files, chunks, commits, model, watcher state. all_repos=true lists every indexed repo."""
    stores = all_stores() if all_repos else [Store(indexer.resolve_root(path))]
    out = []
    for s in stores:
        with s.connect() as con:
            files = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            commits = con.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
            cache = con.execute("SELECT COUNT(*) FROM embcache").fetchone()[0]
            langs = dict(con.execute("SELECT lang, COUNT(*) FROM files GROUP BY lang ORDER BY 2 DESC LIMIT 8").fetchall())
            at = s.get_meta(con, "indexed_at")
            head = s.get_meta(con, "head")
        job = _jobs.get(str(s.root))
        out.append({
            "root": str(s.root), "files": files, "chunks": chunks, "commits": commits,
            "embedding_cache": cache, "languages": langs,
            "last_indexed": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(at))) if at else None,
            "head": head[:10] if head else None, "watching": WATCHERS.active(s.root),
            "indexing_now": bool(job and job[0].is_alive()),
            "model": config.CFG.model_tag, "backend": config.CFG.backend,
            "reranker": config.CFG.rerank_model or None, "db": str(s.path),
        })
    return out if all_repos else out[0]


def _warm_configured_repos():
    for p in config.CFG.repos:
        try:
            _start_index(indexer.resolve_root(p))
        except Exception as e:
            indexer.log(f"could not index {p}: {e}")


def run():
    threading.Thread(target=_warm_configured_repos, daemon=True).start()
    try:
        mcp.run()
    finally:
        WATCHERS.stop_all()
