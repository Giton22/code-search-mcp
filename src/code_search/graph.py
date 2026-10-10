"""Symbol graph: where names are defined and which chunks reference them.

Built from the indexed chunks (no compiler or language server): a chunk *defines* its symbol and
*references* every identifier in its code outside comments and strings. Lookups are by the last
name segment ("save" for "UserStore.save"), so overloads and same-named methods in different
classes show up together; results say where each one lives so the caller can tell them apart.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

GRAPH_VERSION = "1"
DEF_KINDS = {"function", "method", "class", "interface", "trait", "impl", "struct", "enum", "module",
             "type", "record", "constructor", "macro", "namespace", "protocol", "test", "block"}

_STRINGS = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|`[^`]*`')
_COMMENTS = re.compile(r"/\*[\s\S]*?\*/|(?://|#(?![\[!]))[^\n]*|^\s*--[^\n]*", re.M)  # keeps #[attr] / #!
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# keywords and builtins common across languages; never useful as graph edges
KEYWORDS = set("""
and as assert async await break case catch class const continue def default defer del delete do elif else
enum export extends false final finally for from func function get goto if impl implements import in
instanceof interface is lambda let loop match mod module mut namespace new nil none not null object or
override package pass private protected pub public raise readonly return self static struct super switch
this throw throws trait true try type typeof union unsafe use val var void where while with yield
int str float bool dict list tuple set len range print bytes string boolean number any char byte
long short double unsigned signed sizeof println printf fmt err error string fun suspend internal
data sealed open lateinit companion init constructor require include define ifdef endif undef
""".split())


def ref_names(text: str) -> set[str]:
    """Identifiers a chunk refers to, ignoring comments, strings and keywords."""
    code = _COMMENTS.sub(" ", _STRINGS.sub('""', text))
    return {n for n in _IDENT.findall(code) if n.lower() not in KEYWORDS}


def def_name(symbol: str | None) -> str:
    return (symbol or "").rsplit(".", 1)[-1]


def index_chunk(con, chunk_id: int, symbol: str | None, kind: str | None, text: str):
    """Record what a newly inserted chunk defines and references."""
    own = def_name(symbol)
    if own and (kind or "") in DEF_KINDS:
        con.execute("INSERT INTO defs(name, chunk) VALUES(?,?)", (own, chunk_id))
    if kind == "image":
        return
    con.executemany("INSERT INTO refs(chunk, name) VALUES(?,?)",
                    [(chunk_id, n) for n in ref_names(text) if n != own])


def rebuild(con) -> int:
    """(Re)build both tables from the stored chunks; used once for indexes made before the graph existed."""
    con.execute("DELETE FROM refs")
    con.execute("DELETE FROM defs")
    n = 0
    for cid, sym, kind, text in con.execute("SELECT id, symbol, kind, text FROM chunks").fetchall():
        index_chunk(con, cid, sym, kind, text or "")
        n += 1
    return n


def _lines_with(text: str, name: str, start: int, limit: int = 3) -> list[tuple[int, str]]:
    pat = re.compile(rf"\b{re.escape(name)}\b")
    out = []
    for i, line in enumerate(text.splitlines()):
        if pat.search(line):
            out.append((start + i, line.strip()[:160]))
            if len(out) >= limit:
                break
    return out


def definitions(store, symbol: str) -> list[dict]:
    """Chunks defining `symbol` ("save" or "UserStore.save")."""
    name = def_name(symbol)
    with store.connect() as con:
        ids = [r[0] for r in con.execute("SELECT chunk FROM defs WHERE name=?", (name,))]
        rows = list(store.chunk_rows(con, ids).values())
    if "." in symbol:  # qualified: keep exact matches when there are any
        exact = [r for r in rows if (r["symbol"] or "").endswith(symbol)]
        rows = exact or rows
    return sorted(rows, key=lambda r: (r["path"], r["start"]))


def references(store, symbol: str, k: int = 50) -> dict:
    """Chunks that use `symbol`, grouped by file, with the matching lines. Excludes the definitions."""
    name = def_name(symbol)
    with store.connect() as con:
        ids = [r[0] for r in con.execute("SELECT chunk FROM refs WHERE name=?", (name,))]
        defs = {r[0] for r in con.execute("SELECT chunk FROM defs WHERE name=?", (name,))}
        rows = store.chunk_rows(con, [i for i in ids if i not in defs])
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows.values(), key=lambda r: (r["path"], r["start"])):
        lines = _lines_with(r["text"], name, r["start"])
        if lines:  # the identifier may only have matched inside a stripped string or comment
            by_file[r["path"]].append({"symbol": r["symbol"], "start": r["start"], "end": r["end"], "lines": lines})
    files = sorted(by_file, key=lambda f: -len(by_file[f]))
    total = sum(len(v) for v in by_file.values())
    return {"name": name, "total": total, "files": {f: by_file[f] for f in files[:k]},
            "more_files": max(0, len(files) - k)}


def callees(store, chunk_id: int, path: str, max_defs: int = 4) -> list[dict]:
    """Symbols defined elsewhere in the repo that this chunk uses, nearest definition first."""
    here = PurePosixPath(path).parent
    with store.connect() as con:
        pairs = con.execute(
            "SELECT r.name, d.chunk FROM refs r JOIN defs d ON d.name = r.name WHERE r.chunk=? AND d.chunk != ?",
            (chunk_id, chunk_id)).fetchall()
        by_name: dict[str, list[int]] = defaultdict(list)
        for name, cid in pairs:
            by_name[name].append(cid)
        by_name = {n: c for n, c in by_name.items() if len(c) <= max_defs}  # too many: a generic name
        rows = store.chunk_rows(con, [c for cs in by_name.values() for c in cs])

    def distance(r):
        p = PurePosixPath(r["path"])
        if r["path"] == path:
            return 0
        return 1 if p.parent == here else 2

    out = []
    for name, cids in sorted(by_name.items()):
        best = min((rows[c] for c in cids if c in rows), key=distance, default=None)
        if best:
            out.append({"name": name, "symbol": best["symbol"], "path": best["path"], "start": best["start"],
                        "end": best["end"], "signature": best["signature"], "ambiguous": len(cids) > 1})
    return out


def format_definitions(rows: list[dict], symbol: str) -> str:
    if not rows:
        return f"No definition of `{symbol}` is indexed."
    out = [f"{len(rows)} definition{'s' if len(rows) > 1 else ''} of `{symbol}`:"]
    for r in rows:
        sig = f"\n    {r['signature']}" if r["signature"] else ""
        out.append(f"- {r['path']}:{r['start']}-{r['end']} · {r['kind'] or 'symbol'} `{r['symbol']}`{sig}")
    return "\n".join(out)


def format_references(res: dict) -> str:
    if not res["total"]:
        return f"No references to `{res['name']}` found."
    out = [f"`{res['name']}` is used in {res['total']} places across "
           f"{len(res['files']) + res['more_files']} files:"]
    for f, uses in res["files"].items():
        out.append(f"\n{f}")
        for u in uses[:6]:
            where = f" (in `{u['symbol']}`)" if u["symbol"] else ""
            for ln, code in u["lines"][:2]:
                out.append(f"  {ln}: {code}{where}")
                where = ""
        if len(uses) > 6:
            out.append(f"  … {len(uses) - 6} more in this file")
    if res["more_files"]:
        out.append(f"\n… and {res['more_files']} more files")
    return "\n".join(out)
