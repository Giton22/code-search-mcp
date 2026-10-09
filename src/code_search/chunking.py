"""Split source files into searchable chunks.

Tree-sitter is used for real code so chunks follow function / class / method boundaries and carry
their symbol name and signature. Large classes are split into a header chunk plus one chunk per
method. Unsupported languages and prose fall back to a line-based heuristic.
"""
from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    start: int  # 1-based, inclusive
    end: int
    text: str
    symbol: str = ""  # qualified name, e.g. "UserStore.get_user"
    kind: str = ""  # function / method / class / module / section / ...
    signature: str = ""
    lang: str = ""


OVERLAP = 6

# ---------------------------------------------------------------- tree-sitter

# languages where tree-sitter adds little or the grammar is markup
NO_TS = {"markdown", "markdown_inline", "json", "yaml", "toml", "xml", "html", "css", "scss", "csv",
         "dockerfile", "make", "ini", "properties", "text", "rst"}

_DEF = re.compile(
    r"(function|method|class|struct|impl|interface|trait|enum|module|namespace|protocol|extension|"
    r"object_declaration|record|constructor|macro|type_alias|type_declaration|type_spec|union|"
    r"contract|decorated_definition|singleton|test)",
)
_NOT_DEF = re.compile(r"(call|invocation|_type$|type_identifier|parameters|argument|body|list|reference)")
_CONTAINER = re.compile(r"(class|impl|interface|trait|module|namespace|protocol|extension|object|struct|"
                        r"record|enum|contract|singleton)")

_parsers: dict[str, object] = {}
_bad_langs: set[str] = set()
_ts_lock = threading.Lock()


def detect_language(path: str) -> str:
    try:
        from tree_sitter_language_pack import detect_language_from_path

        return detect_language_from_path(path) or ""
    except Exception:
        return ""


def _parser(lang: str):
    with _ts_lock:
        if lang in _parsers:
            return _parsers[lang]
        if lang in _bad_langs:
            return None
        try:
            from tree_sitter_language_pack import get_parser

            p = get_parser(lang)  # may download the grammar on first use
            _parsers[lang] = p
            return p
        except Exception as e:
            print(f"[code-search] no tree-sitter grammar for {lang}: {e}", file=sys.stderr, flush=True)
            _bad_langs.add(lang)
            return None


def _row(point) -> int:
    return point[0] if isinstance(point, tuple) else getattr(point, "row", point[0])


def _is_def(node) -> bool:
    t = node.type
    if t in ("export_statement", "export_declaration"):
        inner = node.child_by_field_name("declaration")
        return inner is not None and _is_def(inner)
    if t in ("lexical_declaration", "variable_declaration"):
        for c in node.named_children:
            v = c.child_by_field_name("value")
            if v is not None and ("function" in v.type or v.type in ("class", "class_expression")):
                return True
        return False
    return bool(_DEF.search(t)) and not _NOT_DEF.search(t)


def _unwrap(node):
    for field in ("declaration", "definition"):
        inner = node.child_by_field_name(field)
        if inner is not None:
            return _unwrap(inner)
    return node


def _name(node) -> str:
    node = _unwrap(node)
    for field in ("name", "type", "declarator"):
        n = node.child_by_field_name(field)
        if n is not None:
            if field == "declarator":  # C/C++: function_declarator -> identifier
                inner = n.child_by_field_name("declarator")
                n = inner if inner is not None else n
            text = n.text.decode("utf-8", "replace") if n.text else ""
            if text:
                return re.split(r"[\s(<{]", text.strip(), maxsplit=1)[0][:80]
    for c in node.named_children:
        if c.type in ("variable_declarator", "type_spec", "type_alias", "const_spec"):
            return _name(c)
        if "identifier" in c.type or c.type in ("constant", "name"):
            return (c.text or b"").decode("utf-8", "replace")[:80]
    return ""


def _kind(node, parent: str) -> str:
    t = _unwrap(node).type
    for k in ("method", "constructor", "class", "interface", "trait", "impl", "struct", "enum", "module",
              "namespace", "protocol", "record", "macro", "type", "test"):
        if k in t:
            return k
    if "function" in t or t in ("lexical_declaration", "variable_declaration"):
        return "method" if parent else "function"
    return t


def _signature(lines: list[str], start: int, end: int) -> str:
    for i in range(start, min(end + 1, start + 6)):
        s = lines[i].strip()
        if s and not s.startswith(("@", "#[", "//", "/*", "*", "#")):
            sig = s
            if sig.endswith(("(", ",")) and i + 1 <= end:  # multi-line parameter lists
                sig += " " + lines[i + 1].strip()
            return sig.rstrip("{:").strip()[:200]
    return lines[start].strip()[:200]


def _body_children(node):
    node = _unwrap(node)
    body = node.child_by_field_name("body")
    if body is None:
        for c in node.named_children:
            if c.type.endswith(("body", "block", "declaration_list", "field_declaration_list")):
                body = c
                break
    return list(body.named_children) if body is not None else []


def _split_lines(a: int, b: int, max_lines: int) -> list[tuple[int, int]]:
    """0-based inclusive row ranges, each at most max_lines, with overlap."""
    out, s = [], a
    while s <= b:
        e = min(b, s + max_lines - 1)
        out.append((s, e))
        if e >= b:
            break
        s = e - OVERLAP + 1
    return out


def _ts_chunks(text: str, lang: str, max_lines: int) -> list[Chunk] | None:
    parser = _parser(lang)
    if parser is None:
        return None
    src = text.encode("utf-8", "replace")
    try:
        tree = parser.parse(src)
    except Exception:
        return None
    lines = text.splitlines()
    if not lines:
        return []
    last = len(lines) - 1
    out: list[Chunk] = []

    def emit(a: int, b: int, symbol="", kind="", sig="", header=""):
        a, b = max(0, a), min(last, b)
        if b < a:
            return
        for s, e in _split_lines(a, b, max_lines):
            body = "\n".join(lines[s:e + 1]).strip("\n")
            if not body.strip():
                continue
            if header and s > a:  # keep the enclosing signature visible in later parts
                body = f"{header}\n    ...\n{body}"
            out.append(Chunk(s + 1, e + 1, body, symbol, kind, sig, lang))

    def walk(children, parent: str, parent_sig: str, region: tuple[int, int]):
        pending: list[int] = []  # start rows of a run of non-definition nodes
        run_end = -1
        comment_start: int | None = None

        def flush():
            nonlocal pending, run_end
            if pending:
                emit(pending[0], run_end, parent, "module" if not parent else "members", parent_sig, parent_sig)
            pending, run_end = [], -1

        for node in children:
            a, b = _row(node.start_point), _row(node.end_point)
            if "comment" in node.type:
                if comment_start is None:
                    comment_start = a
                continue
            if _is_def(node) and (b - a) >= 1:
                flush()
                start = comment_start if comment_start is not None else a
                comment_start = None
                name = _name(node)
                qual = f"{parent}.{name}" if parent and name else (name or parent)
                kind = _kind(node, parent)
                sig = _signature(lines, a, b)
                if (b - start + 1) > max_lines and _CONTAINER.search(_unwrap(node).type):
                    kids = _body_children(node)
                    first_def = next((_row(k.start_point) for k in kids if _is_def(k)), None)
                    head_end = (first_def - 1) if first_def is not None else min(b, start + max_lines - 1)
                    emit(start, head_end, qual, kind, sig)
                    if first_def is not None:
                        walk([k for k in kids if _row(k.start_point) >= first_def], qual, sig, (first_def, b))
                else:
                    emit(start, b, qual, kind, sig, sig)
            else:
                if comment_start is not None:
                    a = comment_start
                    comment_start = None
                if not pending:
                    pending = [a]
                run_end = max(run_end, b)
                if run_end - pending[0] + 1 >= max_lines:
                    flush()
        if comment_start is not None:
            pending = pending or [comment_start]
            run_end = max(run_end, region[1])
        flush()

    walk(tree.root_node.named_children, "", "", (0, last))
    out.sort(key=lambda c: (c.start, c.end))
    return _merge_small(out, max_lines)


def _merge_small(chunks: list[Chunk], max_lines: int) -> list[Chunk]:
    """Glue tiny module-level fragments (imports, constants) to their neighbour."""
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        tiny = (c.end - c.start) < 3 and c.kind in ("module", "members")
        if prev and tiny and prev.kind == c.kind and (c.end - prev.start) < max_lines and c.start <= prev.end + 2:
            prev.text = prev.text + "\n" + c.text
            prev.end = c.end
            continue
        merged.append(c)
    return merged


# ---------------------------------------------------------------- fallback heuristic

_BOUNDARY = re.compile(
    r"^(?:@\w|def |async def |class |fn |pub |impl |trait |struct |enum |func |"
    r"function |async function |export |const \w+ = (?:async )?\(|interface |type |"
    r"module |namespace |public |private |protected |internal |static |package |"
    r"CREATE |\w[\w<>,\s*&]*\s+\w+\s*\([^;]*$)"
)
_NAME = re.compile(
    r"(?:def|class|fn|func|function|struct|enum|trait|interface|type|module|namespace|impl)\s+"
    r"(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)|const\s+([A-Za-z_]\w*)\s*=|([A-Za-z_]\w*)\s*\("
)


def _heuristic_chunks(text: str, max_lines: int, lang: str) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    markdown = lang in ("markdown", "rst", "mdx")
    boundary = re.compile(r"^#{1,4} ") if markdown else _BOUNDARY
    starts = [0]
    for i, ln in enumerate(lines[1:], 1):
        if ln and not ln[0].isspace() and boundary.match(ln):
            j = i
            while j > starts[-1] + 1 and lines[j - 1].lstrip().startswith(("@", "#", "//", "/*", "*")) and not markdown:
                j -= 1
            if j - starts[-1] >= 3:
                starts.append(j)
    starts.append(len(lines))
    segs: list[tuple[int, int]] = []
    for a, b in zip(starts, starts[1:]):
        if segs and (b - segs[-1][0]) <= max_lines // 2:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    out = []
    for a, b in segs:
        first = lines[a].strip()
        if markdown:
            sym, kind = first.lstrip("#").strip()[:80], "section"
        else:
            m = _NAME.search(first)
            sym = next((g for g in m.groups() if g), "") if m else ""
            kind = "block"
        for s, e in _split_lines(a, b - 1, max_lines):
            body = "\n".join(lines[s:e + 1]).strip("\n")
            if body.strip():
                out.append(Chunk(s + 1, e + 1, body, sym, kind, first[:200] if sym else "", lang))
    return out


# ---------------------------------------------------------------- entry point


def looks_generated(text: str) -> bool:
    lines = text.splitlines() or [""]
    longest = max(len(x) for x in lines[:200])
    return longest > 2000 or (len(text) / max(len(lines), 1)) > 400


def chunk_file(path: str, text: str, max_lines: int | None = None) -> list[Chunk]:
    max_lines = max_lines or config.CFG.max_lines
    lang = detect_language(path)
    if not lang:
        ext = Path(path).suffix.lower()
        lang = {".md": "markdown", ".mdx": "markdown", ".rst": "rst"}.get(ext, "")
    if config.CFG.use_tree_sitter and lang and lang not in NO_TS:
        chunks = _ts_chunks(text, lang, max_lines)
        if chunks:
            return chunks
    return _heuristic_chunks(text, max_lines, lang)
