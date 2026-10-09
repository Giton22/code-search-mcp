"""Incremental, git-aware indexing.

Change detection is stat-then-hash per file. Embeddings are looked up by chunk-content hash first,
so a branch switch, rebase or revert only re-embeds chunks whose text was never seen before.
A cross-process lock keeps two MCP clients from indexing the same repo at once.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import chunking, config, embed
from .filelock import FileLock
from .store import Store, content_hash

CODE_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".rs", ".java",
    ".kt", ".kts", ".scala", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs", ".fs", ".rb", ".php",
    ".swift", ".m", ".mm", ".lua", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".r", ".jl", ".dart",
    ".ex", ".exs", ".erl", ".hs", ".ml", ".mli", ".clj", ".cljs", ".elm", ".zig", ".nim", ".v", ".sol",
    ".vue", ".svelte", ".astro", ".html", ".css", ".scss", ".less", ".md", ".mdx", ".rst", ".txt",
    ".toml", ".yaml", ".yml", ".json", ".proto", ".graphql", ".gql", ".tf", ".hcl", ".gradle", ".cmake",
    ".dockerfile", ".ini", ".cfg", ".xml", ".prisma",
}
CODE_NAMES = {"Dockerfile", "Makefile", "Justfile", "Rakefile", "Gemfile", "Procfile", "Vagrantfile",
              "CMakeLists.txt", "BUILD", "WORKSPACE"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "out", "target", "vendor", ".venv", "venv", "env",
    "__pycache__", ".next", ".nuxt", ".svelte-kit", ".cache", "coverage", ".idea", ".vscode", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "Pods", ".gradle", "bin", "obj", ".terraform",
    "site-packages", "bower_components", ".turbo", ".parcel-cache",
}
SKIP_FILES = re.compile(
    r"(\.min\.(js|css)$|\.lock$|-lock\.(json|yaml)$|\.map$|\.snap$|\.pb\.go$|_pb2\.py$|\.generated\.|"
    r"^package-lock\.json$|^yarn\.lock$|^pnpm-lock\.yaml$|^go\.sum$|^Cargo\.lock$|^poetry\.lock$)"
)
MAX_IMAGE_BYTES = 8_000_000
EMBED_GROUP = 256


def log(msg: str) -> None:
    print(f"[code-search] {msg}", file=sys.stderr, flush=True)


def git(root: Path, *args: str, timeout: float = 60, input: str | None = None) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                           timeout=timeout, input=input, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_root(path: str | None) -> Path:
    """Directory -> repo root. Defaults to the client's project (Claude Code sets CLAUDE_PROJECT_DIR)."""
    default = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    p = Path(path or default).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    top = git(p, "rev-parse", "--show-toplevel", timeout=10)
    return Path(top.strip()).resolve() if top and top.strip() else p


def is_git(root: Path) -> bool:
    return (root / ".git").exists()


def wanted(rel: str) -> str | None:
    """'code', 'image' or None."""
    name = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")[:-1]
    if any(p in SKIP_DIRS for p in parts):
        return None
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image" if config.CFG.images else None
    if SKIP_FILES.search(name):
        return None
    return "code" if (name in CODE_NAMES or ext in CODE_EXT) else None


def list_files(root: Path) -> list[str]:
    out = git(root, "ls-files", "-co", "--exclude-standard", "-z", timeout=120)
    if out is not None:
        return sorted(f for f in out.split("\0") if f and wanted(f))
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            if wanted(rel):
                files.append(rel)
    return sorted(files)


def _title(rel: str, c: chunking.Chunk) -> str:
    # the doc "title" carries path + qualified symbol, which helps the model a lot for code
    return f"{rel} {c.symbol}".strip() if c.symbol else rel


def index(root: Path, quiet: bool = False, progress: dict | None = None) -> dict:
    progress = progress if progress is not None else {}
    store = Store(root)
    emb = embed.get()
    t0 = time.time()
    with FileLock(store.path.with_suffix(".lock")):
        with store.connect() as con:
            known = {p: (sha, m) for p, sha, m in con.execute("SELECT path, sha, mtime FROM files")}
        current = list_files(root)
        cur = set(current)
        removed = [p for p in known if p not in cur]

        changed: list[tuple[str, str, float, bytes]] = []
        touched: list[tuple[str, float]] = []
        for rel in current:
            fp = root / rel
            try:
                st = fp.stat()
            except OSError:  # deleted in the working tree but still in git's index
                if rel in known:
                    removed.append(rel)
                continue
            kind = wanted(rel)
            limit = MAX_IMAGE_BYTES if kind == "image" else config.CFG.max_file_bytes
            if st.st_size > limit or st.st_size == 0:
                if rel in known:
                    removed.append(rel)
                continue
            if rel in known and known[rel][1] == st.st_mtime:
                continue
            try:
                raw = fp.read_bytes()
            except OSError:
                continue
            sha = hashlib.sha1(raw).hexdigest()
            if rel in known and known[rel][0] == sha:
                touched.append((rel, st.st_mtime))
                continue
            changed.append((rel, sha, st.st_mtime, raw))

        stats = dict(files_seen=len(current), files_updated=0, files_removed=len(removed),
                     chunks_embedded=0, chunks_from_cache=0, images=0)

        with store.connect() as con:
            con.executemany("UPDATE files SET mtime=? WHERE path=?", [(m, p) for p, m in touched])
            for rel in removed:
                store.delete_file(con, rel)
                con.execute("DELETE FROM files WHERE path=?", (rel,))

        # group files so a huge first index commits progressively and can resume if interrupted
        group: list[tuple[str, str, float, list[tuple[chunking.Chunk, str, str]]]] = []
        n_group = 0
        done = 0

        def flush():
            nonlocal group, n_group
            if not group:
                return
            with store.connect() as con:
                hashes = [h for *_, items in group for _, h, _ in items]
                have = store.cached_vectors(con, hashes)
            need = [(h, title, c.text) for *_, items in group for c, h, title in items if h not in have]
            uniq = {h: (title, text) for h, title, text in need}
            if uniq:
                hs = list(uniq)
                vecs = emb.embed_docs([uniq[h] for h in hs])
                new = dict(zip(hs, vecs))
            else:
                new = {}
            stats["chunks_embedded"] += len(new)
            stats["chunks_from_cache"] += sum(len(items) for *_, items in group) - len(need)
            with store.connect() as con:
                store.cache_vectors(con, list(new.items()))
                for rel, sha, mtime, items in group:
                    store.delete_file(con, rel)
                    lang = items[0][0].lang if items else ""
                    con.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", (rel, sha, mtime, lang))
                    for c, h, _ in items:
                        v = have.get(h)
                        if v is None:
                            v = new[h]
                        store.insert_chunk(con, rel, c, h, v)
                store.bump(con)
            stats["files_updated"] += len(group)
            group, n_group = [], 0

        images: list[tuple[str, str, float]] = []
        progress.update(total=len(changed), done=0)
        for rel, sha, mtime, raw in changed:
            progress["done"] = done
            if wanted(rel) == "image":
                images.append((rel, sha, mtime))
                continue
            if b"\0" in raw[:8192]:
                continue
            text = raw.decode("utf-8", "replace")
            chunks = [] if chunking.looks_generated(text) else chunking.chunk_file(rel, text)
            items = [(c, content_hash(_title(rel, c), c.text), _title(rel, c)) for c in chunks]
            group.append((rel, sha, mtime, items))
            n_group += len(items)
            done += 1
            if n_group >= EMBED_GROUP:
                flush()
                if not quiet:
                    log(f"{root.name}: {done}/{len(changed)} files")
        flush()

        if images:
            stats["images"] = _index_images(store, emb, root, images)

        with store.connect() as con:
            if removed or changed:
                store.bump(con)
            head = git(root, "rev-parse", "HEAD", timeout=10)
            if head:
                store.set_meta(con, "head", head.strip())
            store.set_meta(con, "indexed_at", repr(time.time()))
            store.prune_cache(con)
            stats["total_chunks"] = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    stats["root"] = str(root)
    stats["seconds"] = round(time.time() - t0, 2)
    if not quiet and (stats["files_updated"] or stats["files_removed"]):
        log(f"{root.name}: indexed {stats['files_updated']} files "
            f"({stats['chunks_embedded']} embedded, {stats['chunks_from_cache']} from cache) in {stats['seconds']}s")
    return stats


def _index_images(store: Store, emb, root: Path, images) -> int:
    if not getattr(emb, "supports_images", False):
        return 0
    n = 0
    for i in range(0, len(images), 16):
        part = images[i:i + 16]
        vecs = emb.embed_images([str(root / rel) for rel, _, _ in part])
        if vecs is None:
            return n
        with store.connect() as con:
            for (rel, sha, mtime), v in zip(part, vecs):
                store.delete_file(con, rel)
                con.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", (rel, sha, mtime, "image"))
                c = chunking.Chunk(1, 1, f"[image] {rel}", Path(rel).stem, "image", "", "image")
                store.insert_chunk(con, rel, c, f"img:{sha}", v)
            store.bump(con)
        n += len(part)
    return n


def ensure_index(root: Path) -> Store:
    """Build the index if it doesn't exist yet; otherwise return immediately."""
    store = Store(root)
    with store.connect() as con:
        empty = con.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    if empty:
        index(root)
    return store


def mean_vector(store: Store, paths: list[str]) -> dict[str, np.ndarray]:
    """Per-file centroid of chunk vectors (used for related files and clustering)."""
    ids, mat = store.matrix()
    if not len(ids):
        return {}
    pos = {int(i): n for n, i in enumerate(ids)}
    with store.connect() as con:
        if paths:
            q = ",".join("?" * len(paths))
            rows = con.execute(f"SELECT id, path FROM chunks WHERE path IN ({q})", paths).fetchall()
        else:
            rows = con.execute("SELECT id, path FROM chunks").fetchall()
    acc: dict[str, list[int]] = {}
    for cid, p in rows:
        if cid in pos:
            acc.setdefault(p, []).append(pos[cid])
    out = {}
    for p, idx in acc.items():
        v = mat[idx].mean(axis=0)
        out[p] = v / max(np.linalg.norm(v), 1e-12)
    return out
