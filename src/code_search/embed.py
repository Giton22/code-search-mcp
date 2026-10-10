"""Embedding backends.

- "st":   sentence-transformers, loads EmbeddingGemma 2 (or any ST model)
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

from . import config, hardware


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
        device, self.device_desc = hardware.device(cfg.device)  # first: explains a missing PyTorch
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "cpu" and not cfg.device and (hint := hardware.cpu_fallback_hint()):
            _log(hint)
        hardware.be_gentle(device)
        # model card: bf16 or fp32 only; fp16 silently produces NaN / degraded vectors
        dtype = hardware.dtype(device)
        _log(f"loading {cfg.model} on {device} [{self.device_desc}] ({str(dtype).replace('torch.', '')})")
        kw = {"truncate_dim": cfg.dim} if cfg.dim < 768 else {}
        self.model = SentenceTransformer(
            cfg.model, device=device, model_kwargs={"dtype": dtype},
            trust_remote_code=os.environ.get("CODE_EMBED_TRUST_REMOTE_CODE") == "1", **kw,
        )
        self._images_ok: bool | None = None
        # characters per batch: a batch of long chunks needs far more memory than one of short
        # ones, so batches are sized by length and the budget halves whenever the GPU runs out
        self._chars = self.cfg.batch * 1500

    def _encode(self, texts):
        import torch

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out: list = [None] * len(texts)
        i = 0
        while i < len(order):
            longest = max(len(texts[order[i]]), 1)
            idx = order[i:i + max(1, min(self.cfg.batch, self._chars // longest))]
            try:
                # prompt="" stops sentence-transformers adding a default prompt on top of ours
                vecs = self.model.encode([texts[j] for j in idx], prompt="", batch_size=len(idx),
                                         show_progress_bar=False, convert_to_numpy=True)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if len(idx) == 1 or "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._chars = max(longest, self._chars // 2)
                _log(f"out of GPU memory; smaller batches from now on ({self._chars} chars)")
                continue
            for j, v in zip(idx, vecs):
                out[j] = v
            i += len(idx)
        return np.stack(out) if out else np.zeros((0, self.cfg.dim), dtype=np.float32)

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
_rerankers: dict = {}  # name -> loaded model; normally one, the benchmark compares several
RERANKERS_KEPT = 1


RERANK_INSTRUCTION = "Given a coding question or issue, judge whether this code is where it is answered or fixed"


class _CrossEncoder:
    """Classic cross-encoders (bge-reranker, gte-reranker, ms-marco ...) via sentence-transformers."""

    def __init__(self, name: str, device: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, device=device, model_kwargs={"dtype": hardware.dtype(device)},
                                  trust_remote_code=os.environ.get("CODE_EMBED_TRUST_REMOTE_CODE") == "1")

    def score(self, query: str, docs: list[str]) -> list[float]:
        return self.model.predict([(query, d) for d in docs], batch_size=8, show_progress_bar=False).tolist()


class _Qwen3:
    """Qwen3-Reranker: a small LLM asked "does this document answer the query? yes/no"; the score is
    P(yes). Prompt format from the model card."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, name: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=hardware.dtype(device)).to(device).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")

    def score(self, query: str, docs: list[str]) -> list[float]:
        import torch

        out = []
        for i in range(0, len(docs), 4):
            texts = [f"{self.PREFIX}<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {d}{self.SUFFIX}"
                     for d in docs[i:i + 4]]
            enc = self.tok(texts, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # only the last position's logits: the full (tokens x 151k vocab) tensor is gigabytes
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, [self.no, self.yes]].float()
            out += torch.softmax(logits, -1)[:, 1].tolist()
        return out


class _JinaV3:
    """jina-reranker-v3: listwise, reads the query and all candidates in one pass. Runs Jina's own model
    code (trust_remote_code) and its weights are CC BY-NC 4.0: non-commercial use only."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModel

        if os.environ.get("CODE_EMBED_TRUST_REMOTE_CODE") != "1":
            raise RuntimeError(f"{name} runs its own model code; set CODE_EMBED_TRUST_REMOTE_CODE=1 to allow it")
        self.model = AutoModel.from_pretrained(name, dtype=hardware.dtype(device), trust_remote_code=True)
        self.model = self.model.to(device).eval()

    def score(self, query: str, docs: list[str]) -> list[float]:
        scores = [0.0] * len(docs)
        for r in self.model.rerank(query, docs):
            scores[r["index"]] = float(r["relevance_score"])
        return scores


def _load_reranker(name: str):
    device = hardware.device(config.CFG.device)[0]
    hardware.be_gentle(device)
    low = name.lower()
    cls = _Qwen3 if "qwen3-reranker" in low else _JinaV3 if "jina-reranker-v3" in low else _CrossEncoder
    _log(f"loading reranker {name} on {device}")
    return cls(name, device)


DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-0.6B"  # best permissively licensed one on `code-search bench`


def reranker_name() -> str:
    """CODE_EMBED_RERANK when set ("off" disables it); otherwise Qwen3-Reranker-0.6B when there's a GPU.
    On CPU it would add seconds to every search, so it stays off there unless asked for."""
    v = config.CFG.rerank_model.strip()
    if v.lower() in {"0", "off", "none", "no", "false"}:
        return ""
    if v or config.CFG.backend != "st":
        return v
    try:
        return DEFAULT_RERANKER if hardware.device(config.CFG.device)[0] != "cpu" else ""
    except RuntimeError:  # no PyTorch
        return ""


def rerank(query: str, docs: Sequence[str], model: str | None = None) -> np.ndarray | None:
    """Relevance of each doc to the query: `model`, or the configured reranker when not given."""
    name = reranker_name() if model is None else model
    if not name or not docs:
        return None
    with _rr_lock:
        if name not in _rerankers:
            while len(_rerankers) >= RERANKERS_KEPT:
                _rerankers.pop(next(iter(_rerankers)))  # free the oldest before loading another
            _rerankers[name] = _load_reranker(name)
        # long issue text and long chunks: the start carries the signal, and this bounds memory
        scores = _rerankers[name].score(query[:2000], [d[:3000] for d in docs])
    return np.asarray(scores, dtype=np.float32)
