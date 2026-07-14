#!/usr/bin/env python3
"""Lightweight HuggingFace sentence embeddings for AES 2.0 (CPU, no training).

This extracts FROZEN embeddings only — there is NO fine-tuning and NO gradient
step on any Transformer parameter (requirements 1 & 2). The model is loaded,
put in eval mode, and run under torch.no_grad(); we only read its hidden states
and mean-pool them.

Model (req 3): sentence-transformers/all-MiniLM-L6-v2, loaded directly through
``transformers`` (AutoTokenizer + AutoModel) so we control truncation and
pooling exactly as specified. It is a small 6-layer MiniLM (384-dim), fine on
CPU for a one-off feature-extraction pass.

Graceful skip (req 4): if ``torch``/``transformers`` are missing OR the model
cannot be loaded (e.g. no network for the first download), the script prints a
clear ``[skip]`` and exits 0 without writing anything — the rest of the
pipeline (Ridge/LogReg/SVR/GBDT/blend) is unaffected.

Caching (req 12): if both output files already exist, nothing is recomputed.

Artifacts:
  outputs/train_hf_emb.npy   (n_train, 384) float32
  outputs/test_hf_emb.npy    (n_test,  384) float32
"""
from __future__ import annotations

import os
import sys
import time

# Prefer the local HF cache and avoid network stalls: this box can't reach
# huggingface.co, but the model is already cached. Offline mode makes
# from_pretrained read the cache directly instead of retrying HEAD requests.
# (Set before importing transformers so it takes effect.)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16               # req 7 (8 or 16)
MAX_LENGTH = 256              # req 8 (<= 384)
HEAD_TOKENS = 192             # req 9: keep first 192 ...
TAIL_TOKENS = 64              # ... + last 64 content tokens for long essays
TEXT_COL_PRIORITY = ["full_text", "essay"]

TRAIN_OUT = os.path.join(OUT_DIR, "train_hf_emb.npy")
TEST_OUT = os.path.join(OUT_DIR, "test_hf_emb.npy")

# placeholder-for-append

# Hard ceiling on the final sequence length incl. special tokens (req 8: <=384).
HARD_MAX_LEN = 384


def _try_import():
    """Import torch + transformers, or return (None, None, None) to skip."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        return torch, AutoTokenizer, AutoModel
    except Exception as e:  # ImportError or any load-time error
        print(f"[skip] transformers/torch unavailable: {type(e).__name__}: {e}")
        return None, None, None


def _wrap_special(tokenizer, ids):
    """Prepend/append the model's special tokens around content ids.

    Version-robust: newer transformers dropped
    ``build_inputs_with_special_tokens`` from some tokenizers, so build the
    [CLS] ... [SEP] sequence directly from the tokenizer's special-token ids
    (falling back gracefully if a model has none).
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    prefix = [cls_id] if cls_id is not None else []
    suffix = [sep_id] if sep_id is not None else []
    return prefix + ids + suffix


def _build_input_ids(tokenizer, text: str):
    """Tokenize one text, applying head+tail truncation for long essays (req 9).

    Returns a list of input ids INCLUDING special tokens. Content is capped at
    HEAD_TOKENS + TAIL_TOKENS; with the 2 special tokens the final length never
    exceeds HEAD_TOKENS + TAIL_TOKENS + 2 (<= HARD_MAX_LEN).
    """
    # content tokens only (no CLS/SEP), no truncation yet
    ids = tokenizer.encode(text, add_special_tokens=False)
    budget = MAX_LENGTH - 2  # room for [CLS] and [SEP] in the normal path
    if len(ids) > budget:
        # long essay: keep the opening (thesis/intro) + the closing (conclusion)
        ids = ids[:HEAD_TOKENS] + ids[-TAIL_TOKENS:]
    return _wrap_special(tokenizer, ids)


def _embed(texts, torch, tokenizer, model):
    """Mean-pooled embeddings for a list of texts. Frozen model, no grad."""
    all_vecs = []
    model.eval()  # req 5
    n = len(texts)
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        seqs = [_build_input_ids(tokenizer, t) for t in batch]
        maxlen = min(max(len(s) for s in seqs), HARD_MAX_LEN)  # req 8 ceiling
        pad_id = tokenizer.pad_token_id or 0

        input_ids, attn = [], []
        for s in seqs:
            s = s[:maxlen]
            pad = maxlen - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)

        with torch.no_grad():  # req 6 — no gradients, no parameter updates
            out = model(input_ids=input_ids, attention_mask=attn)
            hidden = out.last_hidden_state              # (B, L, H)
            mask = attn.unsqueeze(-1).type_as(hidden)   # (B, L, 1)
            # mean pooling over non-pad tokens (req 10)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            vecs = (summed / counts).cpu().numpy().astype(np.float32)
        all_vecs.append(vecs)
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"    embedded {min(start + BATCH_SIZE, n)}/{n}")
    return np.vstack(all_vecs)


def _pick_text_col(df):
    for c in TEXT_COL_PRIORITY:
        if c in df.columns:
            return c
    raise KeyError(f"no text column among {TEXT_COL_PRIORITY}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # req 12 — do not recompute if both artifacts already exist
    if os.path.exists(TRAIN_OUT) and os.path.exists(TEST_OUT):
        print(f"[cache] {os.path.basename(TRAIN_OUT)} and "
              f"{os.path.basename(TEST_OUT)} already exist; skipping recompute.")
        return

    torch, AutoTokenizer, AutoModel = _try_import()
    if torch is None:
        print("[skip] no artifacts written; main flow unaffected.")
        return

    # load model (req 3/4) — any failure (e.g. offline first download) -> skip
    try:
        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print(f"[model] loaded {MODEL_NAME} in {time.time()-t_load:.1f}s")
    except Exception as e:
        print(f"[skip] could not load {MODEL_NAME}: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return

    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tcol = _pick_text_col(train)
        print(f"[data] train={len(train)} test={len(test)} text='{tcol}' "
              f"(bs={BATCH_SIZE}, max_len={MAX_LENGTH}, head+tail={HEAD_TOKENS}+{TAIL_TOKENS})")

        t0 = time.time()
        print("[emb] train ...")
        train_emb = _embed(train[tcol].astype(str).tolist(), torch, tokenizer, model)
        print("[emb] test ...")
        test_emb = _embed(test[tcol].astype(str).tolist(), torch, tokenizer, model)
    except Exception as e:
        print(f"[skip] embedding failed: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return

    assert train_emb.shape[0] == len(train) and test_emb.shape[0] == len(test)
    assert not np.isnan(train_emb).any() and not np.isnan(test_emb).any()
    # write atomically-ish: only after both computed successfully
    np.save(TRAIN_OUT, train_emb)
    np.save(TEST_OUT, test_emb)
    print(f"[saved] {os.path.basename(TRAIN_OUT)} {train_emb.shape}  "
          f"{os.path.basename(TEST_OUT)} {test_emb.shape}")
    print(f"[done] total embedding time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover - defensive top-level guard
        print(f"[skip] hf_embeddings aborted: {type(e).__name__}: {e}")
        sys.exit(0)
