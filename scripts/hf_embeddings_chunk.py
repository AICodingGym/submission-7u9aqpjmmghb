#!/usr/bin/env python3
"""Chunk-mean-pooled frozen MiniLM embeddings for AES 2.0 (CPU, no training).

Why a second embedding file when train_hf_emb.npy already exists:
  The existing extraction keeps only head(192)+tail(64) tokens — for the median
  essay (~448 wordpiece tokens, p90 ~733) that covers only ~57% of the text, and
  mean-pooling over a fixed short window discards most of the body AND the
  essay-length signal. This variant instead splits each essay into consecutive
  CHUNK_LEN-token windows (capped at MAX_CHUNKS), embeds every chunk with the
  frozen model, and AVERAGES the chunk vectors — so a long essay's whole body
  contributes. The resulting embedding is a genuinely DIFFERENT view of the same
  text, which is the point: the ensemble is TF-IDF-signal-saturated and needs a
  decorrelated feature source.

Still FROZEN: eval() + torch.no_grad(), no gradient step on any parameter.
Graceful skip if torch/transformers missing or the model can't load offline.
Caching: if both artifacts exist, nothing is recomputed.

Artifacts:
  outputs/train_hf_emb_chunk.npy   (n_train, 384) float32
  outputs/test_hf_emb_chunk.npy    (n_test,  384) float32
"""
from __future__ import annotations

import os
import sys
import time

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
BATCH_SIZE = 32            # chunks per forward pass
CHUNK_LEN = 256           # content tokens per chunk (before special tokens)
MAX_CHUNKS = 6            # cap chunks/essay -> bounds CPU on the longest essays
TEXT_COL_PRIORITY = ["full_text", "essay"]

TRAIN_OUT = os.path.join(OUT_DIR, "train_hf_emb_chunk.npy")
TEST_OUT = os.path.join(OUT_DIR, "test_hf_emb_chunk.npy")

# placeholder-for-append


def _try_import():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        return torch, AutoTokenizer, AutoModel
    except Exception as e:
        print(f"[skip] transformers/torch unavailable: {type(e).__name__}: {e}")
        return None, None, None


def _chunk_ids(tokenizer, text: str):
    """Split one essay into up to MAX_CHUNKS windows of CHUNK_LEN content tokens.

    Returns a list of id-lists, each already wrapped with [CLS]/[SEP]. Empty
    text yields a single [CLS][SEP] chunk so every essay produces >=1 vector.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    prefix = [cls_id] if cls_id is not None else []
    suffix = [sep_id] if sep_id is not None else []
    if not ids:
        return [prefix + suffix]
    chunks = []
    for start in range(0, len(ids), CHUNK_LEN):
        chunks.append(prefix + ids[start:start + CHUNK_LEN] + suffix)
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks


def _embed(texts, torch, tokenizer, model):
    """Chunk-mean-pooled embedding per text. Frozen model, no grad.

    All chunks across all essays in a mini-batch are embedded together for
    throughput, then averaged back per essay. mean-pool over non-pad tokens
    within a chunk, then mean over an essay's chunks.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id or 0
    n = len(texts)
    out = np.zeros((n, model.config.hidden_size), dtype=np.float32)

    # flatten (essay_idx, chunk_ids) so batches are full even for short essays
    flat, owner = [], []
    for i, t in enumerate(texts):
        for ch in _chunk_ids(tokenizer, t):
            flat.append(ch); owner.append(i)
    owner = np.asarray(owner)
    counts = np.bincount(owner, minlength=n).astype(np.float32)
    counts[counts == 0] = 1.0

    done = 0
    for start in range(0, len(flat), BATCH_SIZE):
        seqs = flat[start:start + BATCH_SIZE]
        own = owner[start:start + BATCH_SIZE]
        maxlen = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)
        with torch.no_grad():
            o = model(input_ids=input_ids, attention_mask=attn)
            hidden = o.last_hidden_state
            mask = attn.unsqueeze(-1).type_as(hidden)
            summed = (hidden * mask).sum(dim=1)
            cnt = mask.sum(dim=1).clamp(min=1e-9)
            vecs = (summed / cnt).cpu().numpy().astype(np.float32)  # (B, H) per chunk
        # accumulate each chunk vector onto its owner essay
        np.add.at(out, own, vecs)
        done += len(seqs)
        if (start // BATCH_SIZE) % 25 == 0:
            print(f"    embedded {done}/{len(flat)} chunks "
                  f"({len(flat)} total for {n} essays)")
    out /= counts[:, None]     # essay = mean of its chunk vectors
    return out


def _pick_text_col(df):
    for c in TEXT_COL_PRIORITY:
        if c in df.columns:
            return c
    raise KeyError(f"no text column among {TEXT_COL_PRIORITY}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(TRAIN_OUT) and os.path.exists(TEST_OUT):
        print(f"[cache] chunk embeddings already exist; skipping recompute.")
        return

    torch, AutoTokenizer, AutoModel = _try_import()
    if torch is None:
        print("[skip] no artifacts written; main flow unaffected."); return
    try:
        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print(f"[model] loaded {MODEL_NAME} in {time.time()-t_load:.1f}s "
              f"(chunk_len={CHUNK_LEN}, max_chunks={MAX_CHUNKS})")
    except Exception as e:
        print(f"[skip] could not load {MODEL_NAME}: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected."); return

    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tcol = _pick_text_col(train)
        t0 = time.time()
        print(f"[emb] train ({len(train)}) ...")
        tr_emb = _embed(train[tcol].astype(str).tolist(), torch, tokenizer, model)
        print(f"[emb] test ({len(test)}) ...")
        te_emb = _embed(test[tcol].astype(str).tolist(), torch, tokenizer, model)
    except Exception as e:
        print(f"[skip] embedding failed: {type(e).__name__}: {e}"); return

    assert tr_emb.shape[0] == len(train) and te_emb.shape[0] == len(test)
    assert not np.isnan(tr_emb).any() and not np.isnan(te_emb).any()
    np.save(TRAIN_OUT, tr_emb); np.save(TEST_OUT, te_emb)
    print(f"[saved] train_hf_emb_chunk.npy {tr_emb.shape}  "
          f"test_hf_emb_chunk.npy {te_emb.shape}")
    print(f"[done] total embedding time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"[skip] hf_embeddings_chunk aborted: {type(e).__name__}: {e}")
        sys.exit(0)
