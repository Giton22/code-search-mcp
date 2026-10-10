"""Command line: `code-search serve` (the MCP server) plus the same tools from the terminal.

Terminal commands run inside the shared daemon (started on demand), so after the first one the
model is already loaded. Set CODE_EMBED_DAEMON=0 to run everything in this process instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="code-search", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="run the MCP server on stdio (default)")
    sub.add_parser("daemon", help="run the shared background service in the foreground")

    p = sub.add_parser("setup", help="install everything, with the PyTorch build for this machine's GPU")
    p.add_argument("--dry-run", action="store_true", help="only show what would be installed")

    p = sub.add_parser("ui", help="open the web UI in your browser")
    p.add_argument("--no-browser", action="store_true", help="just print the URL")

    p = sub.add_parser("hook", help="Claude Code prompt hook (reads the event on stdin); or install / uninstall it")
    p.add_argument("action", nargs="?", choices=["run", "install", "uninstall"], default="run")
    p.add_argument("--scope", choices=["user", "project"], default="user",
                   help="user: ~/.claude/settings.json, project: ./.claude/settings.json")

    p = sub.add_parser("index", help="build / update the index for a repo")
    p.add_argument("path", nargs="?")
    p.add_argument("--no-history", action="store_true")

    p = sub.add_parser("search", help="search code")
    p.add_argument("query")
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--glob")
    p.add_argument("--all", action="store_true", help="search every indexed repo")

    p = sub.add_parser("context", help="everything relevant to a task, within a token budget")
    p.add_argument("task")
    p.add_argument("--path")
    p.add_argument("--budget", type=int, default=6000)

    p = sub.add_parser("expand", help="full code for a handle: path:12-40, path::Symbol, a symbol or a file")
    p.add_argument("handle")
    p.add_argument("--path")

    p = sub.add_parser("def", help="where a symbol is defined")
    p.add_argument("symbol")
    p.add_argument("--path")

    p = sub.add_parser("refs", help="every place that uses a symbol")
    p.add_argument("symbol")
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=30)

    p = sub.add_parser("history", help="search git history")
    p.add_argument("query")
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=8)

    p = sub.add_parser("related", help="files related to a file")
    p.add_argument("file")
    p.add_argument("--path")

    p = sub.add_parser("tests", help="tests for a file, file:line or symbol")
    p.add_argument("target")
    p.add_argument("--path")

    p = sub.add_parser("explain", help="cluster the repo into areas")
    p.add_argument("--path")
    p.add_argument("--glob")
    p.add_argument("--clusters", type=int)

    p = sub.add_parser("status", help="index status")
    p.add_argument("path", nargs="?")
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("bench", help="benchmark retrieval on real GitHub issues (SWE-bench style)")
    p.add_argument("--per-repo", type=int, default=3, help="tasks per repository")
    p.add_argument("--repos", type=int, default=12, help="number of repositories")
    p.add_argument("--variants", default="hybrid",
                   help="comma-separated: hybrid,vector,bm25,context,rerank,rerank:<model>")
    p.add_argument("--max-files", type=int, default=6000, help="skip repos with more files than this")
    p.add_argument("--out", type=Path, help="write per-task results as JSON")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval-generate", help="build an eval set from git history")
    p.add_argument("out", type=Path)
    p.add_argument("--path")
    p.add_argument("-n", type=int, default=100)

    p = sub.add_parser("eval", help="measure recall@k / MRR on an eval set")
    p.add_argument("file", type=Path)
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")

    a = ap.parse_args(argv)

    def show(x):
        print(x if isinstance(x, str) else json.dumps(x, indent=2, ensure_ascii=False))

    # light commands first: the hook runs on every prompt and must not import numpy or torch
    if a.cmd == "hook":
        from . import hook

        if a.action == "run":
            hook.run()
        else:
            path = (hook.install if a.action == "install" else hook.uninstall)(a.scope)
            print(f"{a.action}ed the code-search prompt hook in {path}", file=sys.stderr)
        return
    if a.cmd == "ui":
        from .ui import open_ui

        open_ui(open_browser=not a.no_browser)
        return
    if a.cmd == "daemon":
        from . import daemon

        daemon.run()
        return
    if a.cmd == "setup":
        from . import hardware

        show(hardware.setup(dry_run=a.dry_run))
        return
    if a.cmd in (None, "serve"):
        from .server import run

        run()
        return

    from .server import dispatch

    tools = {
        "index": lambda: ("index_repo", {"path": a.path, "include_history": not a.no_history}),
        "search": lambda: ("search_code", {"query": a.query, "path": a.path, "k": a.k, "path_glob": a.glob,
                                           "repos": ["all"] if a.all else None}),
        "context": lambda: ("prepare_context", {"task": a.task, "path": a.path, "budget_tokens": a.budget}),
        "expand": lambda: ("expand", {"handle": a.handle, "path": a.path}),
        "def": lambda: ("find_definition", {"symbol": a.symbol, "path": a.path}),
        "refs": lambda: ("find_references", {"symbol": a.symbol, "path": a.path, "k": a.k}),
        "history": lambda: ("search_history", {"query": a.query, "path": a.path, "k": a.k}),
        "related": lambda: ("related_files", {"file": a.file, "path": a.path}),
        "tests": lambda: ("find_tests", {"target": a.target, "path": a.path}),
        "explain": lambda: ("explain_area", {"path": a.path, "path_glob": a.glob, "clusters": a.clusters}),
        "status": lambda: ("index_status", {"path": a.path, "all_repos": a.all}),
    }
    if a.cmd in tools:
        name, args = tools[a.cmd]()
        show(dispatch(name, args))
        return

    from . import indexer

    if a.cmd == "bench":
        from . import bench

        show(bench.run(per_repo=a.per_repo, n_repos=a.repos, variants=a.variants.split(","), out=a.out,
                       seed=a.seed, max_files=a.max_files))
    elif a.cmd == "eval-generate":
        from . import evaluate

        n = evaluate.generate(indexer.resolve_root(a.path), a.out, n=a.n)
        print(f"wrote {n} queries to {a.out}", file=sys.stderr)
    elif a.cmd == "eval":
        from . import evaluate

        show(evaluate.run(indexer.resolve_root(a.path), a.file, k=a.k, verbose=a.verbose))


if __name__ == "__main__":
    main()
