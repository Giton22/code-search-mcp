"""Embedding backends.

- "st":   sentence-transformers, loads EmbeddingGemma 2 (or any ST model / fine-tuned folder)
- "http": any OpenAI-compatible /v1/embeddings endpoint: llama-server (GGUF), Ollama, LM Studio
- "hash": deterministic bag-of-tokens vectors, for tests and offline demos only

All backends return L2-normalised float32 vectors and apply the model-card prompts themselves,
so the same index format works whichever backend produced it (as long as the model matches).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import urllib.request
from typing import Sequence

import numpy as np

from . import config


def _log(msg: str) -> None:
    print(f"[code-search] {msg}", file=sys.stderr, flush=True)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if not np.all(np.isfinite(v)):
        raise RuntimeError(
            "Embedding model returned NaN/inf values. EmbeddingGemma 2 must run in bfloat16 or "
            "float32, never float16; check CODE_EMBED_DEVICE / your GGUF quantisation."
        )
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _truncate(v: np.ndarray, dim: int) -> np.ndarray:
    return v[..., :dim] if v.shape[-1] > dim else v


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SUBWORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokens(text: str) -> list[str]:
    """Identifier-aware tokens: getUserToken -> getusertoken, get, user, token."""
    out = []
    for t in _IDENT.findall(text):
        low = t.lower()
        out.append(low)
        parts = [p.lower() for p in _SUBWORD.findall(t)]
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) > 1)
        elif "_" in t:
            out.extend(p for p in low.split("_") if len(p) > 1)
    return out


class Embedder:
    name = "base"
    supports_images = False

    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    # --- public API -------------------------------------------------------------
    def doc_text(self, title: str, text: str) -> str:
        return self.cfg.doc_template.format(title=title or "none", text=text)

    def query_text(self, query: str, prefix: str | None = None) -> str:
        return f"{self.cfg.query_prefix if prefix is None else prefix}{query}"

    def embed_docs(self, items: Sequence[tuple[str, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)
        texts = [self.doc_text(t, x) for t, x in items]
        return normalize(_truncate(self._encode(texts), self.cfg.dim))

    def embed_query(self, query: str, prefix: str | None = None) -> np.ndarray:
        """prefix=None uses the code-retrieval prompt; history search uses a generic retrieval one."""
        return normalize(_truncate(self._encode([self.query_text(query, prefix)]), self.cfg.dim))[0]

    def embed_images(self, paths: Sequence[str]) -> np.ndarray | None:
        return None

    # --- backend hook -------------------------------------------------------------
    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class STEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, cfg):
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer
        import torch

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        # model card: bf16 or fp32 only; fp16 silently produces NaN / degraded vectors
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
        _log(f"loading {cfg.model} on {device} ({str(dtype).replace('torch.', '')})")
        kw = {"truncate_dim": cfg.dim} if cfg.dim < 768 else {}
        self.model = SentenceTransformer(
            cfg.model, device=device, model_kwargs={"torch_dtype": dtype},
            trust_remote_code=os.environ.get("CODE_EMBED_TRUST_REMOTE_CODE") == "1", **kw,
        )
        self._images_ok: bool | None = None

    def _encode(self, texts):
        # prompt="" stops sentence-transformers adding a default prompt on top of ours
        return self.model.encode(texts, prompt="", batch_size=self.cfg.batch,
                                 show_progress_bar=False, convert_to_numpy=True)

    @property
    def supports_images(self):  # type: ignore[override]
        return self._images_ok is not False

    def embed_images(self, paths):
        if self._images_ok is False:
            return None
        try:
            # interleaved multimodal input, as in the EmbeddingGemma 2 model card
            vecs = [self.model.encode({"text": "<|image|>", "image": [p]}, show_progress_bar=False,
                                      convert_to_numpy=True) for p in paths]
            self._images_ok = True
            return normalize(_truncate(np.vstack([np.atleast_2d(v) for v in vecs]), self.cfg.dim))
        except Exception as e:  # model or ST version without image support
            if self._images_ok is None:
                _log(f"image embedding unavailable ({type(e).__name__}: {e}); skipping images")
            self._images_ok = False
            return None


class HTTPEmbedder(Embedder):
    """OpenAI-compatible embeddings endpoint (llama-server --embeddings, Ollama, LM Studio)."""
    name = "http"

    def _encode(self, texts):
        out = []
        for i in range(0, len(texts), self.cfg.batch):
            body = json.dumps({"model": self.cfg.http_model, "input": texts[i:i + self.cfg.batch],
                               "encoding_format": "float"}).encode()
            req = urllib.request.Request(self.cfg.http_url, data=body,
                                         headers={"Content-Type": "application/json"})
            key = os.environ.get("CODE_EMBED_API_KEY")
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.loads(r.read())
            except OSError as e:
                raise RuntimeError(f"Embedding server at {self.cfg.http_url} unreachable: {e}") from e
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            out.extend(r["embedding"] for r in rows)
        return np.asarray(out, dtype=np.float32)


class HashEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests. Not semantically meaningful."""
    name = "hash"
    supports_images = True

    def query_text(self, query, prefix=None):  # prefixes would only add noise to bag-of-words
        return query

    def doc_text(self, title, text):
        return f"{title} {text}"

    def _encode(self, texts):
        m = np.zeros((len(texts), self.cfg.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokens(t):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                m[i, h % self.cfg.dim] += 1.0 if (h >> 63) == 0 else -1.0
            if not m[i].any():
                m[i, 0] = 1.0
        return m

    def embed_images(self, paths):
        return self.embed_docs([(p, os.path.basename(p)) for p in paths])


_lock = threading.Lock()
_instance: Embedder | None = None


def get() -> Embedder:
    """Lazy singleton so the MCP server starts instantly and loads the model on first use."""
    global _instance
    with _lock:
        cfg = config.CFG
        if _instance is None or _instance.cfg is not cfg:
            cls = {"st": STEmbedder, "http": HTTPEmbedder, "hash": HashEmbedder}.get(cfg.backend)
            if cls is None:
                raise ValueError(f"Unknown CODE_EMBED_BACKEND={cfg.backend!r} (use st, http or hash)")
            _instance = cls(cfg)
        return _instance


# ---------------------------------------------------------------- reranker

_rr_lock = threading.Lock()
_reranker = None


def rerank(query: str, docs: Sequence[str]) -> np.ndarray | None:
    """Score (query, doc) pairs with a cross-encoder if CODE_EMBED_RERANK is set."""
    global _reranker
    name = config.CFG.rerank_model
    if not name or not docs:
        return None
    with _rr_lock:
        if _reranker is None or getattr(_reranker, "_cs_name", None) != name:
            from sentence_transformers import CrossEncoder

            _log(f"loading reranker {name}")
            _reranker = CrossEncoder(name, device=config.CFG.device,
                                     trust_remote_code=os.environ.get("CODE_EMBED_TRUST_REMOTE_CODE") == "1")
            _reranker._cs_name = name
        scores = _reranker.predict([(query, d[:4000]) for d in docs], show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)
