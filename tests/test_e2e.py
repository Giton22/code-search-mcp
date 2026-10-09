"""End-to-end tests with the offline 'hash' backend (no model download needed).

Run:  CODE_EMBED_BACKEND=hash uv run --extra dev pytest -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

FILES = {
    "src/auth/jwt.py": '''import jwt

from .keys import load_key


def validate_token(token: str) -> dict:
    """Validate a JWT access token and return its claims."""
    key = load_key()
    return jwt.decode(token, key, algorithms=["HS256"])


def refresh_token(token: str) -> str:
    """Issue a new token from a refresh token."""
    claims = validate_token(token)
    return jwt.encode(claims, load_key())
''',
    "src/auth/keys.py": '''import os


def load_key() -> str:
    """Read the signing key from the environment."""
    return os.environ["JWT_KEY"]
''',
    "src/http/client.ts": '''import { sleep } from "./util";

/** Fetch a URL and retry failed requests with exponential backoff. */
export async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fetch(url);
    } catch (err) {
      await sleep(2 ** attempt * 100);
    }
  }
  throw new Error("request failed");
}

export class ApiClient {
  constructor(private base: string) {}

  async getUser(id: string) {
    return fetchWithRetry(`${this.base}/users/${id}`);
  }
}
''',
    "src/http/util.ts": '''export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
''',
    "src/store/users.go": '''package store

// UserStore keeps users in a database table.
type UserStore struct {
\tdb *DB
}

// GetUser loads one user by id from the users table.
func (s *UserStore) GetUser(id string) (*User, error) {
\treturn s.db.QueryUser(id)
}
''',
    "tests/test_jwt.py": '''from src.auth.jwt import validate_token


def test_validate_token_rejects_garbage():
    try:
        validate_token("garbage")
    except Exception:
        return
    raise AssertionError("should fail")
''',
    "src/http/client.test.ts": '''import { fetchWithRetry } from "./client";

test("fetchWithRetry retries", async () => {
  expect(await fetchWithRetry("http://x")).toBeDefined();
});
''',
    "README.md": "# Demo\n\nA demo service.\n\n## Auth\n\nTokens are JWTs signed with JWT_KEY.\n",
    "node_modules/junk/index.js": "function shouldNotBeIndexed() {}\n",
}


def sh(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_EMBED_BACKEND", "hash")
    monkeypatch.setenv("CODE_EMBED_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_EMBED_WATCH_DEBOUNCE", "0.5")
    monkeypatch.setenv("CODE_EMBED_IMAGES", "1")
    from code_search import config, embed

    config.reload()
    embed._instance = None
    return tmp_path


@pytest.fixture()
def repo(env):
    root = env / "demo"
    root.mkdir()
    sh(root, "git", "init", "-q", "-b", "main")
    (root / ".gitignore").write_text("node_modules/\n")
    # history: a few focused commits so co-change and history search have signal
    write(root, "src/auth/keys.py", FILES["src/auth/keys.py"])
    write(root, "src/auth/jwt.py", FILES["src/auth/jwt.py"])
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-qm", "Add JWT validation with signing key from env")
    write(root, "src/http/util.ts", FILES["src/http/util.ts"])
    write(root, "src/http/client.ts", FILES["src/http/client.ts"])
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-qm", "Retry failed HTTP requests with exponential backoff")
    for rel in ("src/auth/jwt.py", "src/auth/keys.py"):
        write(root, rel, FILES[rel] + "\n# touched\n")
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-qm", "Rotate signing key handling for token refresh")
    for rel, text in FILES.items():
        write(root, rel, text)
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-qm", "Add user store and tests for auth and http client")
    return root


def test_index_and_search(repo):
    from code_search import indexer, search
    from code_search.store import Store

    stats = indexer.index(repo)
    assert stats["files_updated"] >= 7
    store = Store(repo)
    with store.connect() as con:
        paths = {r[0] for r in con.execute("SELECT path FROM files")}
        syms = {r[0] for r in con.execute("SELECT symbol FROM chunks")}
    assert "node_modules/junk/index.js" not in paths  # gitignored
    assert {"validate_token", "fetchWithRetry", "ApiClient", "UserStore", "GetUser"} <= syms

    hits = search.search([store], "validate_token", k=3)
    assert hits[0]["row"]["path"] == "src/auth/jwt.py"
    assert hits[0]["row"]["symbol"] == "validate_token"

    hits = search.search([store], "retry failed requests backoff", k=3, include_tests=False)
    assert hits[0]["row"]["path"] == "src/http/client.ts"
    assert not any(search.is_test(h["row"]["path"]) for h in hits)

    hits = search.search([store], "user", k=5, path_glob="*.go")
    assert hits and all(h["row"]["path"].endswith(".go") for h in hits)
    hits = search.search([store], "user", k=5, kinds=["class"])
    assert all(h["row"]["kind"] == "class" for h in hits)
    out = search.format_hits(search.search([store], "load signing key", k=2))
    assert "load_key" in out and "src/auth/keys.py" in out


def test_incremental_delete_and_branch_cache(repo):
    from code_search import indexer

    indexer.index(repo)
    s = indexer.index(repo)
    assert s["files_updated"] == 0 and s["chunks_embedded"] == 0

    # branch with a modified file, then back: second visit must come from the embedding cache
    sh(repo, "git", "checkout", "-qb", "feature")
    write(repo, "src/auth/keys.py", FILES["src/auth/keys.py"] + "\n\ndef rotate_key():\n    return 'new'\n")
    sh(repo, "git", "commit", "-qam", "Add key rotation")
    s1 = indexer.index(repo)
    assert s1["files_updated"] == 1 and s1["chunks_embedded"] >= 1
    sh(repo, "git", "checkout", "-q", "main")
    s2 = indexer.index(repo)
    assert s2["files_updated"] == 1 and s2["chunks_embedded"] == 0 and s2["chunks_from_cache"] >= 1
    sh(repo, "git", "checkout", "-q", "feature")
    s3 = indexer.index(repo)
    assert s3["chunks_embedded"] == 0

    (repo / "src/http/util.ts").unlink()
    s4 = indexer.index(repo)
    assert s4["files_removed"] == 1


def test_related_tests_explain_history(repo):
    from code_search import history, indexer, search
    from code_search.store import Store

    indexer.index(repo)
    store = Store(repo)
    h = history.update(repo, store)
    assert h["commits_added"] == 4

    rel = search.related_files(store, "src/auth/jwt.py")
    paths = [r["path"] for r in rel]
    assert "src/auth/keys.py" in paths[:3]  # co-changed and referenced (load_key)
    assert any("load_key" in w for r in rel for w in r["why"])

    tests = search.find_tests(store, "src/auth/jwt.py")
    assert tests[0]["path"] == "tests/test_jwt.py"
    tests = search.find_tests(store, "fetchWithRetry")
    assert tests[0]["path"] == "src/http/client.test.ts"

    text = search.explain_area(store, clusters=2)
    assert "areas" in text and "Core files" in text

    out = search.search_history(store, "exponential backoff retry", k=2)
    assert "Retry failed HTTP requests" in out.split("###")[1]
    out = search.search_history(store, "signing key", path="src/auth/keys.py")
    file_lines = [ln for ln in out.splitlines() if ln.startswith("Files:")]
    assert file_lines and all("src/auth/keys.py" in ln for ln in file_lines)
    assert "Retry failed HTTP" not in out


def test_multi_repo(env, repo):
    from code_search import indexer, search
    from code_search.store import all_stores

    other = env / "other"
    other.mkdir()
    sh(other, "git", "init", "-q")
    write(other, "billing.py", "def charge_credit_card(amount):\n    '''Charge a card.'''\n    return amount\n")
    sh(other, "git", "add", "-A")
    sh(other, "git", "commit", "-qm", "billing")
    indexer.index(repo)
    indexer.index(other)
    stores = all_stores()
    assert {s.root.name for s in stores} == {"demo", "other"}
    hits = search.search(stores, "charge_credit_card", k=3)
    assert hits[0]["repo"] == "other"
    assert "[other]" in search.format_hits(hits, multi=True)


def test_images(repo):
    from code_search import embed, indexer, search
    from code_search.store import Store

    png = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")
    (repo / "docs").mkdir()
    (repo / "docs/login_screen.png").write_bytes(png)
    s = indexer.index(repo)
    assert s["images"] == 1
    vec = embed.get().embed_images([str(repo / "docs/login_screen.png")])
    hits = search.search([Store(repo)], None, k=1, qvec=vec[0])
    assert hits[0]["row"]["kind"] == "image"


def test_watcher(repo):
    from code_search import indexer
    from code_search.store import Store
    from code_search.watcher import WatchManager

    indexer.index(repo)
    wm = WatchManager()
    assert wm.ensure(repo)
    try:
        write(repo, "src/auth/session.py", "def create_session_cookie(user):\n    return 'cookie'\n")
        deadline = time.time() + 15
        found = False
        while time.time() < deadline and not found:
            time.sleep(0.5)
            with Store(repo).connect() as con:
                found = con.execute("SELECT 1 FROM chunks WHERE symbol='create_session_cookie'").fetchone() is not None
        assert found, "watcher did not re-index the new file"
    finally:
        wm.stop_all()


def test_eval_and_pairs(repo, env):
    from code_search import evaluate, finetune, indexer

    indexer.index(repo)
    out = env / "eval.jsonl"
    n = evaluate.generate(repo, out)
    assert n >= 1
    res = evaluate.run(repo, out, k=5)
    assert res["queries"] == n and 0 <= res["mrr@5"] <= 1
    manual = env / "manual.jsonl"
    manual.write_text(json.dumps({"query": "validate a jwt", "expected": ["src/auth/jwt.py::validate_token"]}) + "\n")
    assert evaluate.run(repo, manual, k=5)["recall@5"] == 1.0
    pairs = finetune.build_pairs(repo)
    assert any(p["src"] == "doc" and "Validate a JWT access token" in p["query"] for p in pairs)
    doc_pair = next(p for p in pairs if p["query"].startswith("Validate a JWT"))
    assert "Validate a JWT" not in doc_pair["text"]  # docstring removed from the positive


def test_concurrent_processes_no_duplicates(repo, env):
    code = ("import sys; sys.path.insert(0, %r); from code_search import indexer; indexer.index(indexer.resolve_root(%r))"
            % (str(Path(__file__).parents[1] / "src"), str(repo)))
    procs = [subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy(),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(3)]
    assert all(p.wait(60) == 0 for p in procs)
    from code_search.store import Store

    with Store(repo).connect() as con:
        dup = con.execute("SELECT path, start, COUNT(*) c FROM chunks GROUP BY path, start, end HAVING c > 1").fetchall()
    assert not dup


def test_mcp_stdio(repo, env):
    src = str(Path(__file__).parents[1] / "src")
    p = subprocess.Popen([sys.executable, "-m", "code_search", "serve"], cwd=repo,
                         env={**os.environ, "PYTHONPATH": src, "CLAUDE_PROJECT_DIR": str(repo)},
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def rpc(i, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method, **({"id": i} if i is not None else {}),
               **({"params": params} if params is not None else {})}
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline()) if i is not None else None

    try:
        init = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "t", "version": "1"}})
        assert "code-search" in json.dumps(init) and "search_code" in init["result"].get("instructions", "")
        rpc(None, "notifications/initialized")
        tools = rpc(2, "tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"search_code", "search_by_image", "find_similar", "related_files", "find_tests",
                         "explain_area", "search_history", "index_repo", "index_status"}
        assert all(t["annotations"]["readOnlyHint"] for t in tools)
        res = rpc(3, "tools/call", {"name": "search_code", "arguments": {"query": "validate_token", "k": 1}})
        text = res["result"]["content"][0]["text"]
        assert "src/auth/jwt.py" in text and "validate_token" in text
        res = rpc(4, "tools/call", {"name": "related_files", "arguments": {"file": "src/auth/jwt.py"}})
        assert "src/auth/keys.py" in res["result"]["content"][0]["text"]
    finally:
        p.kill()
