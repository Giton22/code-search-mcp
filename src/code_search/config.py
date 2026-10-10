"""All settings come from environment variables so every MCP client can configure them the same way."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "", strip: bool = True) -> str:
    v = os.environ.get(name, default)
    return v.strip() if strip else v


def _flag(name: str, default: bool) -> bool:
    v = _env(name)
    if not v:
        return default
    return v.lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Config:
    # embedding backend: "st" (sentence-transformers, default), "http" (OpenAI-compatible
    # endpoint such as llama-server / Ollama / LM Studio) or "hash" (tests only)
    backend: str = field(default_factory=lambda: _env("CODE_EMBED_BACKEND", "st"))
    model: str = field(default_factory=lambda: _env("CODE_EMBED_MODEL", "google/embeddinggemma-2"))
    dim: int = field(default_factory=lambda: int(_env("CODE_EMBED_DIM", "768")))
    device: str | None = field(default_factory=lambda: _env("CODE_EMBED_DEVICE") or None)
    http_url: str = field(default_factory=lambda: _env("CODE_EMBED_URL", "http://127.0.0.1:8080/v1/embeddings"))
    http_model: str = field(default_factory=lambda: _env("CODE_EMBED_HTTP_MODEL", "embeddinggemma2"))
    batch: int = field(default_factory=lambda: int(_env("CODE_EMBED_BATCH", "32")))

    # prompts from the EmbeddingGemma 2 model card (code retrieval task)
    # (not stripped: the trailing space in the prefix matters)
    query_prefix: str = field(default_factory=lambda: _env("CODE_EMBED_QUERY_PREFIX", "task: code retrieval | query: ", strip=False))
    doc_template: str = field(default_factory=lambda: _env("CODE_EMBED_DOC_TEMPLATE", "title: {title} | text: {text}", strip=False))

    # optional cross-encoder reranker, e.g. "BAAI/bge-reranker-v2-m3"
    rerank_model: str = field(default_factory=lambda: _env("CODE_EMBED_RERANK"))
    rerank_top: int = field(default_factory=lambda: int(_env("CODE_EMBED_RERANK_TOP", "30")))

    cache_dir: Path = field(default_factory=lambda: Path(_env("CODE_EMBED_CACHE") or Path.home() / ".cache" / "code-embed"))
    max_lines: int = field(default_factory=lambda: int(_env("CODE_EMBED_MAX_LINES", "60")))
    max_file_bytes: int = field(default_factory=lambda: int(_env("CODE_EMBED_MAX_FILE_BYTES", "512000")))
    use_tree_sitter: bool = field(default_factory=lambda: _flag("CODE_EMBED_TREE_SITTER", True))

    watch: bool = field(default_factory=lambda: _flag("CODE_EMBED_WATCH", True))
    watch_debounce: float = field(default_factory=lambda: float(_env("CODE_EMBED_WATCH_DEBOUNCE", "2.0")))
    images: bool = field(default_factory=lambda: _flag("CODE_EMBED_IMAGES", False))
    history_limit: int = field(default_factory=lambda: int(_env("CODE_EMBED_HISTORY", "3000")))

    # extra repos to search with repos=["all"] and to watch from startup (os.pathsep-separated)
    repos: tuple[str, ...] = field(
        default_factory=lambda: tuple(p for p in _env("CODE_EMBED_REPOS").split(os.pathsep) if p)
    )

    @property
    def model_tag(self) -> str:
        name = self.http_model if self.backend == "http" else self.model
        tag = Path(name).name or name.replace("/", "_")
        return f"{tag}-{self.dim}"


CFG = Config()


def reload() -> Config:
    """Re-read the environment (used by tests and the CLI)."""
    global CFG
    CFG = Config()
    return CFG
