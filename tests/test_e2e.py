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
    monkeypatch.setenv("CODE_EMBED_DAEMON", "0")  # in-process; test_daemon covers the shared service
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


def test_eval(repo, env):
    from code_search import evaluate, indexer

    indexer.index(repo)
    out = env / "eval.jsonl"
    n = evaluate.generate(repo, out)
    assert n >= 1
    res = evaluate.run(repo, out, k=5)
    assert res["queries"] == n and 0 <= res["mrr@5"] <= 1
    manual = env / "manual.jsonl"
    manual.write_text(json.dumps({"query": "validate a jwt", "expected": ["src/auth/jwt.py::validate_token"]}) + "\n")
    assert evaluate.run(repo, manual, k=5)["recall@5"] == 1.0


def test_concurrent_processes_no_duplicates(repo, env):
    code = ("import sys; sys.path.insert(0, %r); from code_search import indexer; indexer.index(indexer.resolve_root(%r))"
            % (str(Path(__file__).parents[1] / "src"), str(repo)))
    procs = [subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy(), text=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for _ in range(3)]
    errors = [err for p in procs for err in [p.communicate(timeout=60)[1]] if p.returncode]
    assert not errors, errors[0][-2000:]
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
                         "explain_area", "search_history", "index_repo", "index_status", "prepare_context",
                         "expand", "find_definition", "find_references"}
        schema = next(t for t in tools if t["name"] == "search_code")["inputSchema"]
        assert {"query", "path", "k"} <= set(schema["properties"]) and schema["required"] == ["query"]
        assert all(t["annotations"]["readOnlyHint"] for t in tools)
        res = rpc(3, "tools/call", {"name": "search_code", "arguments": {"query": "validate_token", "k": 1}})
        text = res["result"]["content"][0]["text"]
        assert "src/auth/jwt.py" in text and "validate_token" in text
        res = rpc(4, "tools/call", {"name": "related_files", "arguments": {"file": "src/auth/jwt.py"}})
        assert "src/auth/keys.py" in res["result"]["content"][0]["text"]
    finally:
        p.kill()


def test_prompts_keep_whitespace(monkeypatch):
    monkeypatch.setenv("CODE_EMBED_BACKEND", "st")
    from code_search import config, embed

    monkeypatch.setattr(config, "CFG", config.CFG)  # restore the global config after the test
    cfg = config.reload()
    e = embed.Embedder(cfg)
    assert e.query_text("find f") == "task: code retrieval | query: find f"
    assert e.doc_text("a.py f", "x") == "title: a.py f | text: x"
    assert e.doc_text("", "x") == "title: none | text: x"



def test_hardware_detection_and_plan():
    from code_search import hardware as hw

    assert hw.amd_gfx("AMD Radeon RX 9070 XT") == "gfx1201"
    assert hw.amd_gfx("AMD Radeon RX 9060 XT") == "gfx1200"
    assert hw.amd_gfx("AMD Radeon RX 7900 XTX") == "gfx1100"
    assert hw.amd_gfx("AMD Radeon(TM) Graphics") is None
    assert (hw.kfd_gfx(120001), hw.kfd_gfx(110000), hw.kfd_gfx(90010), hw.kfd_gfx(0)) == \
        ("gfx1201", "gfx1100", "gfx90a", None)

    igpu = {"vendor": "amd", "name": "AMD Radeon(TM) Graphics", "gfx": None}
    rdna4 = {"vendor": "amd", "name": "AMD Radeon RX 9070 XT", "gfx": "gfx1201"}
    nvidia = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4090", "gfx": None}
    assert hw.best_gpu([igpu, rdna4]) == rdna4 and hw.best_gpu([igpu]) is None
    assert hw.best_gpu([rdna4, nvidia]) == nvidia

    _, cmd = hw.install_plan(rdna4, uv="uv", python="py")
    assert cmd[cmd.index("--index") + 1] == hw.AMD_INDEX  # --index outranks PyPI; --index-url would not
    assert "torch[device-gfx1201]" in cmd and "--reinstall-package" in cmd
    _, cmd = hw.install_plan(nvidia, uv="uv", python="py")
    assert cmd[cmd.index("--torch-backend") + 1] == "auto"
    assert all("--no-config" in hw.install_plan(g)[1] for g in (rdna4, nvidia, None))  # project excludes torch
    _, cmd = hw.install_plan({"vendor": "apple", "name": "Apple Silicon", "gfx": None})
    assert "--torch-backend" not in cmd and "torch" in cmd  # PyPI's macOS build has MPS
    _, cmd = hw.install_plan(None)
    assert cmd[cmd.index("--torch-backend") + 1] == "cpu"


def test_ui_api(repo, env):
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request
    from http.server import ThreadingHTTPServer

    from code_search import ui

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    ui.Handler.token, ui.Handler.port = "t0k", httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{ui.Handler.port}"

    def get(path, headers=None):
        req = urllib.request.Request(base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        status, page = get("/")
        assert status == 200 and b"t0k" in page  # the page carries the token for its own API calls
        assert get("/api/system")[0] == 403  # no token
        assert get("/api/system", {"X-Token": "t0k", "Host": "evil.example"})[0] == 403  # DNS rebinding
        q = urllib.parse.urlencode({"q": "validate a jwt", "repo": str(repo)})
        status, body = get(f"/api/search?{q}", {"X-Token": "t0k"})
        hits = json.loads(body)["hits"]
        assert status == 200 and any(h["path"] == "src/auth/jwt.py" for h in hits)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_graph_context_expand(repo, env):
    from code_search import context, graph, indexer
    from code_search.store import Store

    indexer.index(repo)
    store = Store(repo)
    defs = graph.definitions(store, "validate_token")
    assert [d["path"] for d in defs] == ["src/auth/jwt.py"]
    refs = graph.references(store, "validate_token")
    assert "src/auth/jwt.py" in refs["files"] and "tests/test_jwt.py" in refs["files"]  # refresh_token + the test
    assert all(u["symbol"] != "validate_token" for u in refs["files"]["src/auth/jwt.py"])  # not its own definition
    calls = graph.callees(store, defs[0]["id"], defs[0]["path"])
    assert any(c["name"] == "load_key" and c["path"] == "src/auth/keys.py" for c in calls)
    assert graph.ref_names('x = "load_key"  # validate_token\nfoo(bar)') == {"foo", "bar"}  # not strings/comments

    res = context.build([store], "validate a JWT access token", budget=1500)
    md = res["markdown"]
    assert "src/auth/jwt.py" in md and res["files"][0] == "src/auth/jwt.py"
    assert "Calls:" in md and "src/auth/keys.py" in md
    assert res["tokens"] <= 1500 * 1.1

    assert "def validate_token" in context.expand(store, "src/auth/jwt.py::validate_token")
    assert "def load_key" in context.expand(store, "src/auth/keys.py:4")
    assert "def load_key" in context.expand(store, "load_key")
    assert "export class ApiClient" in context.expand(store, "src/http/client.ts")


def test_hook(repo, env, tmp_path, monkeypatch):
    import io

    from code_search import daemon, hook, indexer, server
    from code_search.watcher import WATCHERS

    assert not hook.worth_searching("ok thanks")
    assert not hook.worth_searching("/review")
    assert hook.worth_searching("where do we validate JWT tokens before requests?")

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "x"}]}]}}))
    monkeypatch.setattr(hook, "_settings_path", lambda scope: settings)
    hook.install()
    hook.install()  # idempotent
    entries = json.loads(settings.read_text())["hooks"]["UserPromptSubmit"]
    assert len(entries) == 2 and "code-search hook" in entries[1]["hooks"][0]["command"]
    hook.uninstall()
    assert json.loads(settings.read_text())["hooks"]["UserPromptSubmit"] == entries[:1]  # other hooks kept

    indexer.index(repo)
    try:
        text = server.auto_context("how is the JWT access token validated", str(repo), min_score=0)
        assert "src/auth/jwt.py" in text
        assert server.auto_context("how is the JWT access token validated", str(repo), min_score=1.1) == ""
    finally:
        WATCHERS.stop_all()

    monkeypatch.setattr(daemon, "info", lambda timeout=1: {"port": 1, "token": "t"})
    monkeypatch.setattr(daemon, "_request", lambda st, path, body=None, timeout=0: {"result": "- src/a.py:1-2"})
    out = io.StringIO()
    hook.run(io.StringIO(json.dumps({"prompt": "where are JWT tokens validated?", "cwd": str(repo)})), out)
    assert json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] == "- src/a.py:1-2"
    out = io.StringIO()
    hook.run(io.StringIO(json.dumps({"prompt": "thanks", "cwd": str(repo)})), out)
    assert out.getvalue() == ""


def test_daemon(repo, env, monkeypatch):
    import signal

    from code_search import config, daemon

    monkeypatch.setenv("CODE_EMBED_DAEMON", "1")
    monkeypatch.setenv("CODE_EMBED_DAEMON_IDLE_MINUTES", "2")  # a backstop if the kill below never runs
    config.reload()
    st = daemon.ensure(wait=90)
    try:
        assert st, (config.CFG.cache_dir / "daemon.log").read_text(errors="replace")[-2000:]
        out = daemon.call("search_code", {"query": "validate_token", "path": str(repo), "k": 1})
        assert "src/auth/jwt.py" in out
        out = daemon.call("find_references", {"symbol": "load_key", "path": str(repo)})
        assert "src/auth/jwt.py" in out
        assert daemon.ensure()["pid"] == st["pid"]  # reused, not a second daemon
        with pytest.raises(RuntimeError, match="unknown tool"):
            daemon.call("nope", {})
    finally:
        if daemon.state_path().exists():
            os.kill(json.loads(daemon.state_path().read_text())["pid"], signal.SIGTERM)


def test_query_signals():
    from code_search.search import query_signals

    idents, files = query_signals("QuerySet.filter() fails in django/db/models/query.py when `parse_header` "
                                  "gets None; see getUserToken, HTTPServer and the main config")
    assert idents == {"queryset", "filter", "parse_header", "getusertoken", "httpserver"}
    assert files == {"django/db/models/query.py"}
    assert query_signals("where are tokens validated") == (set(), set())
