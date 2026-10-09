"""Fine-tune the embedding model on your own repo.

Training pairs come from the index you already have:
  1. docstring / leading comment  ->  the code it documents (with the comment removed)
  2. commit subject               ->  current version of the functions that commit changed

Uses sentence-transformers with CachedMultipleNegativesRankingLoss (large effective batch, low memory),
optional LoRA, bf16 on capable GPUs and fp32 otherwise (never fp16 for EmbeddingGemma 2).
Held-out pairs are scored before and after so you can see whether it helped.

Afterwards point CODE_EMBED_MODEL at the output folder; that creates a separate index, so you can
compare with `code-search eval` and switch back any time.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import config, history, indexer
from .store import Store

_PY_DOC = re.compile(r'^[^\n]*\n\s*[rRuU]?("""|\'\'\')(.+?)\1', re.S)
_LEAD_COMMENT = re.compile(r"^((?:[ \t]*(?:#|//|///|/\*\*|/\*|\*|--)[^\n]*\n)+)")


def _clean(doc: str) -> str:
    doc = re.sub(r"^[ \t]*(?:#+|//+|/\*\*?|\*/?|--)\s?", "", doc, flags=re.M)
    doc = re.sub(r"\s+", " ", doc).strip()
    doc = re.split(r"(?<=[.!?])\s|\s@\w+|\s:param|\sArgs:|\sReturns:", doc)[0]
    return doc.strip(" .*/")


def _doc_pair(text: str) -> tuple[str, str] | None:
    m = _LEAD_COMMENT.match(text)
    if m and len(m.group(1).strip()) > 10:
        doc, code = m.group(1), text[m.end():]
    else:
        m = _PY_DOC.match(text)
        if not m:
            return None
        doc = m.group(2)
        before, after = text[: m.start(1)].rstrip(" \t"), text[m.end():]
        code = before + (after[1:] if after.startswith("\n") and before.endswith("\n") else after)
    doc = _clean(doc)
    if len(doc.split()) < 4 or len(code.strip().splitlines()) < 3:
        return None
    return doc, code.strip("\n")


def build_pairs(root: Path) -> list[dict]:
    store = Store(root)
    indexer.index(root, quiet=True)
    history.update(root, store)
    pairs = []
    with store.connect() as con:
        chunks = con.execute("SELECT path, symbol, kind, text FROM chunks WHERE kind != 'image'").fetchall()
    by_sym: dict[tuple[str, str], str] = {}
    for path, sym, kind, text in chunks:
        title = f"{path} {sym}".strip()
        if sym:
            by_sym[(path, sym.rsplit(".", 1)[-1])] = (title, text)
        dp = _doc_pair(text)
        if dp:
            pairs.append({"query": dp[0], "title": title, "text": dp[1], "src": "doc"})
    for c in history.commit_rows(store):
        subj = c["subject"].strip()
        if len(subj.split()) < 4 or len(c["files"]) > 5:
            continue
        for f in c["files"]:
            for s in c["symbols"][:3]:
                hit = by_sym.get((f, s))
                if hit:
                    pairs.append({"query": subj, "title": hit[0], "text": hit[1], "src": "commit"})
    seen, out = set(), []
    for p in pairs:
        key = (p["query"].lower(), p["title"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def train(root: Path, output: Path, epochs: int = 1, batch_size: int = 64, lr: float | None = None,
          lora: bool = False, max_pairs: int = 20000, seed: int = 0, min_pairs: int = 50) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    cfg = config.CFG
    pairs = build_pairs(root)
    if len(pairs) < min_pairs:
        raise SystemExit(f"Only {len(pairs)} training pairs found; need at least {min_pairs} "
                         "(more docstrings / descriptive commits), or lower --min-pairs.")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    n_eval = max(min(20, len(pairs) // 3), len(pairs) // 10)
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
    print(f"{len(train_pairs)} training pairs, {len(eval_pairs)} held out "
          f"({sum(p['src'] == 'doc' for p in pairs)} from docstrings, {sum(p['src'] == 'commit' for p in pairs)} from commits)")

    def anchor(q):
        return f"{cfg.query_prefix}{q}"

    def positive(p):
        return cfg.doc_template.format(title=p["title"], text=p["text"])

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = SentenceTransformer(cfg.model, model_kwargs={"torch_dtype": dtype})
    if lora:
        from peft import LoraConfig, TaskType

        gemma = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
        targets = [t for t in gemma if t in names] or "all-linear"  # Unsloth's recipe for Gemma; fallback otherwise
        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.0,
                                     target_modules=targets))

    # held-out retrieval eval: queries vs every positive in the dataset as corpus
    corpus = {str(i): positive(p) for i, p in enumerate(pairs)}
    queries, relevant = {}, {}
    for i, p in enumerate(eval_pairs):
        queries[f"q{i}"] = anchor(p["query"])
        relevant[f"q{i}"] = {str(j) for j, q in enumerate(pairs) if q["query"] == p["query"] and q["title"] == p["title"]}
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, name="heldout", show_progress_bar=False,
                                              mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
                                              precision_recall_at_k=[10], map_at_k=[10])
    before = evaluator(model)

    ds = Dataset.from_dict({"anchor": [anchor(p["query"]) for p in train_pairs],
                            "positive": [positive(p) for p in train_pairs]})
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr or (1e-4 if lora else 2e-5), warmup_ratio=0.1, bf16=bf16, fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=20, save_strategy="no", report_to=[], seed=seed,
        prompts=None,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    after = evaluator(model)
    output.mkdir(parents=True, exist_ok=True)
    # with --lora this saves a small adapter that references the base model; sentence-transformers
    # applies it automatically when CODE_EMBED_MODEL points at this folder
    model.save_pretrained(str(output))
    key = next((k for k in after if k.endswith("mrr@10")), None)
    summary = {"output": str(output), "pairs": len(pairs),
               "heldout_mrr@10_before": round(before.get(key, 0), 4) if key else None,
               "heldout_mrr@10_after": round(after.get(key, 0), 4) if key else None}
    (output / "code_search_finetune.json").write_text(json.dumps(summary, indent=2))
    return summary
