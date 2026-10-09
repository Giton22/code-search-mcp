"""SQLite storage: one database per (repo, model, dimension) under the cache directory.

Tables
  files     path -> content sha / mtime (change detection)
  chunks    code chunks with symbol metadata and their vector
  embcache  chunk-content hash -> vector, so switching branches or reverting files re-uses
            embeddings instead of recomputing them
  commits   git history entries with their vector (search_history, co-change)
  *_fts     FTS5 keyword indexes (BM25) over chunks and commits
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import config

SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha TEXT, mtime REAL, lang TEXT);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, path TEXT, start INTEGER, end INTEGER,
    symbol TEXT, kind TEXT, signature TEXT, lang TEXT, text TEXT, hash TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_symbol ON chunks(symbol);
CREATE TABLE IF NOT EXISTS embcache(hash TEXT PRIMARY KEY, vec BLOB, used REAL);
CREATE TABLE IF NOT EXISTS commits(
    sha TEXT PRIMARY KEY, ts INTEGER, author TEXT, subject TEXT, body TEXT,
    files TEXT, symbols TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS commits_ts ON commits(ts);
"""

_FTS = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
    "text, symbol, path, content='chunks', content_rowid='id', tokenize=\"unicode61 tokenchars '_'\")",
    "CREATE VIRTUAL TABLE IF NOT EXISTS commits_fts USING fts5("
    "subject, body, files, symbols, tokenize=\"unicode61 tokenchars '_'\")",
]


def db_path(root: Path) -> Path:
    cfg = config.CFG
    key = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"{root.name}-{key}-{cfg.model_tag}.db"


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()  # storage only; maths happens in float32


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def content_hash(title: str, text: str) -> str:
    return hashlib.sha1(f"{config.CFG.model_tag}\0{title}\0{text}".encode("utf-8", "replace")).hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.path = db_path(root)
        self._init()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def _init(self):
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            ver = con.execute("SELECT v FROM meta WHERE k='schema'").fetchone()
            if ver is None or ver[0] != SCHEMA_VERSION:
                # older layout (or fresh file): rebuild; embeddings are recomputed lazily
                for (name,) in con.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table') AND name NOT LIKE 'sqlite_%'"
                        " AND name NOT LIKE '%fts_%'").fetchall():
                    con.execute(f"DROP TABLE IF EXISTS {name}")
                con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            con.executescript(_SCHEMA)
            fts = True
            for stmt in _FTS:
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError:
                    fts = False
            self.fts = fts
            self.set_meta(con, "schema", SCHEMA_VERSION)
            self.set_meta(con, "root", str(self.root))
            self.set_meta(con, "model", config.CFG.model_tag)

    # ---------------------------------------------------------------- meta
    @staticmethod
    def get_meta(con, k, default=None):
        row = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def set_meta(con, k, v):
        con.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))

    def bump(self, con):
        self.set_meta(con, "updated", repr(time.time()))

    # ---------------------------------------------------------------- chunks
    def delete_file(self, con, path: str):
        if self.fts:
            con.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text, symbol, path) "
                        "SELECT 'delete', id, text, symbol, path FROM chunks WHERE path=?", (path,))
        con.execute("DELETE FROM chunks WHERE path=?", (path,))

    def insert_chunk(self, con, path, c, h, vec) -> int:
        cur = con.execute(
            "INSERT INTO chunks(path,start,end,symbol,kind,signature,lang,text,hash,vec) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (path, c.start, c.end, c.symbol, c.kind, c.signature, c.lang, c.text, h, vec_to_blob(vec)))
        if self.fts:
            con.execute("INSERT INTO chunks_fts(rowid, text, symbol, path) VALUES(?,?,?,?)",
                        (cur.lastrowid, c.text, c.symbol, path))
        return cur.lastrowid

    def cached_vectors(self, con, hashes: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        now = time.time()
        for i in range(0, len(hashes), 500):
            part = hashes[i:i + 500]
            q = ",".join("?" * len(part))
            for h, b in con.execute(f"SELECT hash, vec FROM embcache WHERE hash IN ({q})", part):
                out[h] = blob_to_vec(b)
            con.execute(f"UPDATE embcache SET used=? WHERE hash IN ({q})", [now, *part])
        return out

    def cache_vectors(self, con, items: list[tuple[str, np.ndarray]]):
        now = time.time()
        con.executemany("INSERT OR REPLACE INTO embcache VALUES(?,?,?)",
                        [(h, vec_to_blob(v), now) for h, v in items])

    def prune_cache(self, con, keep_factor: float = 3.0):
        """Keep the embedding cache at most a few times the live index size (oldest first out)."""
        live = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM embcache").fetchone()[0]
        limit = int(max(live * keep_factor, 5000))
        if total > limit:
            con.execute("DELETE FROM embcache WHERE hash IN (SELECT hash FROM embcache "
                        "WHERE hash NOT IN (SELECT hash FROM chunks) ORDER BY used LIMIT ?)", (total - limit,))

    # ---------------------------------------------------------------- vector matrix
    _mat_lock = threading.Lock()
    _mat_cache: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """(ids, float32 matrix) for all chunks, cached in memory and in a .npz snapshot for fast cold starts."""
        with self.connect() as con:
            stamp = self.get_meta(con, "updated", "0")
        key = str(self.path)
        with self._mat_lock:
            hit = self._mat_cache.get(key)
            if hit and hit[0] == stamp:
                return hit[1], hit[2]
            snap = self.path.with_suffix(".vecs.npz")
            ids = mat = None
            if snap.exists():
                try:
                    with np.load(snap) as z:
                        if str(z["stamp"]) == stamp:
                            ids, mat = z["ids"], z["mat"].astype(np.float32)
                except Exception:
                    ids = mat = None
            if ids is None:
                with self.connect() as con:
                    rows = con.execute("SELECT id, vec FROM chunks ORDER BY id").fetchall()
                if rows:
                    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                    mat16 = np.vstack([np.frombuffer(r[1], dtype=np.float16) for r in rows])
                    mat = mat16.astype(np.float32)
                    try:
                        tmp = snap.with_suffix(f".{os.getpid()}.tmp.npz")
                        np.savez(tmp, ids=ids, mat=mat16, stamp=np.array(stamp))
                        tmp.replace(snap)
                    except OSError:
                        pass
                else:
                    ids = np.zeros(0, dtype=np.int64)
                    mat = np.zeros((0, config.CFG.dim), dtype=np.float32)
            self._mat_cache[key] = (stamp, ids, mat)
            return ids, mat

    # ---------------------------------------------------------------- helpers
    def chunk_rows(self, con, ids) -> dict[int, dict]:
        ids = [int(i) for i in ids]
        out = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            q = ",".join("?" * len(part))
            for r in con.execute(
                    f"SELECT id,path,start,end,symbol,kind,signature,lang,text FROM chunks WHERE id IN ({q})", part):
                out[r[0]] = dict(id=r[0], path=r[1], start=r[2], end=r[3], symbol=r[4], kind=r[5],
                                 signature=r[6], lang=r[7], text=r[8])
        return out


def all_stores() -> list[Store]:
    """Every repo indexed with the current model, found from the cache directory."""
    cfg = config.CFG
    out = []
    if not cfg.cache_dir.exists():
        return out
    for p in sorted(cfg.cache_dir.glob(f"*-{cfg.model_tag}.db")):
        try:
            con = sqlite3.connect(p, timeout=10)
            row = con.execute("SELECT v FROM meta WHERE k='root'").fetchone()
            con.close()
        except sqlite3.Error:
            continue
        if row and Path(row[0]).is_dir():
            out.append(Store(Path(row[0])))
    return out


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)
