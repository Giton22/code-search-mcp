"""Command line: `code-search serve` (the MCP server) plus helpers for indexing, searching and evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="code-search", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="run the MCP server on stdio (default)")

    p = sub.add_parser("index", help="build / update the index for a repo")
    p.add_argument("path", nargs="?")
    p.add_argument("--no-history", action="store_true")

    p = sub.add_parser("search", help="search code from the terminal")
    p.add_argument("query")
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--glob")
    p.add_argument("--all", action="store_true", help="search every indexed repo")

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

    p = sub.add_parser("eval-generate", help="build an eval set from git history")
    p.add_argument("out", type=Path)
    p.add_argument("--path")
    p.add_argument("-n", type=int, default=100)

    p = sub.add_parser("eval", help="measure recall@k / MRR on an eval set")
    p.add_argument("file", type=Path)
    p.add_argument("--path")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("finetune", help="fine-tune the embedding model on this repo")
    p.add_argument("output", type=Path)
    p.add_argument("--path")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--min-pairs", type=int, default=50)
    p.add_argument("--pairs-only", action="store_true", help="just write the training pairs as JSONL")

    a = ap.parse_args(argv)
    if a.cmd in (None, "serve"):
        from .server import run

        run()
        return

    from . import history, indexer, search
    from .store import Store, all_stores

    def root(p):
        return indexer.resolve_root(p)

    def show(x):
        print(x if isinstance(x, str) else json.dumps(x, indent=2, ensure_ascii=False))

    if a.cmd == "index":
        r = root(a.path)
        stats = indexer.index(r)
        if not a.no_history:
            stats["history"] = history.update(r)
        show(stats)
    elif a.cmd == "search":
        stores = all_stores() if a.all else [indexer.ensure_index(root(a.path))]
        show(search.format_hits(search.search(stores, a.query, k=a.k, path_glob=a.glob), multi=a.all))
    elif a.cmd == "history":
        show(search.search_history(indexer.ensure_index(root(a.path)), a.query, k=a.k))
    elif a.cmd == "related":
        show(search.related_files(indexer.ensure_index(root(a.path)), a.file))
    elif a.cmd == "tests":
        show(search.find_tests(indexer.ensure_index(root(a.path)), a.target))
    elif a.cmd == "explain":
        show(search.explain_area(indexer.ensure_index(root(a.path)), a.glob, a.clusters))
    elif a.cmd == "status":
        from .server import index_status

        show(index_status(a.path, all_repos=a.all))
    elif a.cmd == "eval-generate":
        from . import evaluate

        n = evaluate.generate(root(a.path), a.out, n=a.n)
        print(f"wrote {n} queries to {a.out}", file=sys.stderr)
    elif a.cmd == "eval":
        from . import evaluate

        show(evaluate.run(root(a.path), a.file, k=a.k, verbose=a.verbose))
    elif a.cmd == "finetune":
        from . import finetune

        if a.pairs_only:
            pairs = finetune.build_pairs(root(a.path))
            a.output.write_text("".join(json.dumps(p) + "\n" for p in pairs))
            print(f"wrote {len(pairs)} pairs to {a.output}", file=sys.stderr)
        else:
            show(finetune.train(root(a.path), a.output, epochs=a.epochs, batch_size=a.batch_size,
                                lr=a.lr, lora=a.lora, min_pairs=a.min_pairs))


if __name__ == "__main__":
    main()
