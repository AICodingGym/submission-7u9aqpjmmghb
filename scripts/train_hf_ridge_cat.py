#!/usr/bin/env python3
"""Ridge over CONCATENATED frozen MiniLM views (head/tail ⊕ chunk) for AES 2.0.

Motivation (the "keep length / whole-essay signal" play): each single MiniLM
view is a weak, length-washed summary — head/tail Ridge OOF 0.558, chunk Ridge
0.524. But they see DIFFERENT spans (head/tail = intro+conclusion; chunk = mean
over the whole body), so concatenating the two 384-d vectors and letting Ridge
weight all 768 dims lifts standalone OOF to ~0.605. That is still weaker than
the TF-IDF models, but its errors are only ~0.66-correlated with them, so as an
ensemble member it can add decorrelated signal (hf_ridge already earns a small
non-zero weight in the NNLS stack).

Both embedding files are precomputed & cached (no re-embedding here). Per fold:
StandardScaler + Ridge fit on TRAIN rows only; test predictions averaged across
folds. Thresholds optimized on OOF only. Same leakage rules as the rest.

Artifacts:
  outputs/oof_hf_ridge_cat.npy   (n_train)
  outputs/test_hf_ridge_cat.npy  (n_test)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.thresholds import (  # noqa: E402
    apply_thresholds, optimize_thresholds, quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
HT_TR = os.path.join(OUT_DIR, "train_hf_emb.npy")
HT_TE = os.path.join(OUT_DIR, "test_hf_emb.npy")
CH_TR = os.path.join(OUT_DIR, "train_hf_emb_chunk.npy")
CH_TE = os.path.join(OUT_DIR, "test_hf_emb_chunk.npy")
SEED = 42
N_FOLDS = 5
ALPHA = 10.0

# placeholder-for-append


def run():
    t0 = time.time()
    for p in (HT_TR, HT_TE, CH_TR, CH_TE):
        if not os.path.exists(p):
            print(f"[skip] missing {os.path.basename(p)}; run hf_embeddings*.py first.")
            return
    y = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))["score"].to_numpy(int)
    Xtr = np.hstack([np.load(HT_TR), np.load(CH_TR)])
    Xte = np.hstack([np.load(HT_TE), np.load(CH_TE)])
    print(f"[feat] concat emb dims = {Xtr.shape[1]} (head/tail 384 ⊕ chunk 384)")

    oof = np.zeros(len(y)); test_pred = np.zeros(Xte.shape[0])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(Xtr, y), 1):
        sc = StandardScaler().fit(Xtr[tr])
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr[tr]), y[tr])
        oof[va] = m.predict(sc.transform(Xtr[va]))
        test_pred += m.predict(sc.transform(Xte)) / N_FOLDS

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT_DIR, "oof_hf_ridge_cat.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_hf_ridge_cat.npy"), test_pred)
    q = qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds))
    print(f"[oof] hf_ridge_cat OOF QWK = {q:.4f}  (head/tail 0.558, chunk 0.524)")
    print(f"[saved] oof_hf_ridge_cat.npy test_hf_ridge_cat.npy  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] hf_ridge_cat aborted: {type(e).__name__}: {e}")
        sys.exit(0)
