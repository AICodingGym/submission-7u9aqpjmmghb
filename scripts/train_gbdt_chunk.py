#!/usr/bin/env python3
"""LightGBM over [SVD(TF-IDF) + hand + head/tail emb + CHUNK emb] for AES 2.0.

This is train_gbdt_hf.py's recipe (already the ensemble's best embedding user,
gbdt_hf_ms in the winning Caruana bag) but ALSO feeding the chunk-mean-pooled
MiniLM embedding from hf_embeddings_chunk.py. The chunk embedding sees the whole
essay body (head+tail sees ~57% of the median essay), so it carries semantic
signal the existing features miss; a non-linear GBDT can exploit it where a
linear head (hf_ridge, OOF 0.556) could not.

Feature block per fold:
    TruncatedSVD(word+char TF-IDF)[128] + hand[28] + head/tail emb[384] + chunk emb[384]

Leakage: SVD/vectorizers/scaler fit on the TRAIN fold only; embeddings are
precomputed frozen features indexed per fold; thresholds optimized on OOF only.
Skips gracefully if a GBDT backend or either embedding file is missing.

Artifacts (only on full success):
  outputs/oof_gbdt_chunk.npy   (n_train,)
  outputs/test_gbdt_chunk.npy  (n_test,)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features, FEATURE_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_gbdt import build_svd_features, detect_backend, SVD_DIM  # noqa: E402
from train_cpu_baseline import pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
HT_TRAIN = os.path.join(OUT_DIR, "train_hf_emb.npy")          # head/tail emb
HT_TEST = os.path.join(OUT_DIR, "test_hf_emb.npy")
CH_TRAIN = os.path.join(OUT_DIR, "train_hf_emb_chunk.npy")    # chunk emb
CH_TEST = os.path.join(OUT_DIR, "test_hf_emb_chunk.npy")

SEED = 42
N_FOLDS = 5

# placeholder-for-append


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] no lightgbm/xgboost; nothing written."); return
    for p in (HT_TRAIN, HT_TEST, CH_TRAIN, CH_TEST):
        if not os.path.exists(p):
            print(f"[skip] missing {os.path.basename(p)}; run hf_embeddings*.py first.")
            return
    print(f"[backend] {backend}")

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)

    ht_tr, ht_te = np.load(HT_TRAIN), np.load(HT_TEST)
    ch_tr, ch_te = np.load(CH_TRAIN), np.load(CH_TEST)
    assert ht_tr.shape[0] == len(train) and ch_tr.shape[0] == len(train)
    print(f"[feat] hand[{len(FEATURE_COLUMNS)}] + SVD({SVD_DIM}) + ht_emb[{ht_tr.shape[1]}] "
          f"+ chunk_emb[{ch_tr.shape[1]}]")
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train)); test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fqs = []
    try:
        for fold, (tr, va) in enumerate(skf.split(train, y), 1):
            ft = time.time()
            X_tr, X_va, X_te, _, evr = build_svd_features(
                tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                hand_tr[tr], hand_tr[va], hand_te)
            X_tr = np.hstack([X_tr, ht_tr[tr], ch_tr[tr]])
            X_va = np.hstack([X_va, ht_tr[va], ch_tr[va]])
            X_te = np.hstack([X_te, ht_te, ch_te])
            m = make_model()
            m.fit(X_tr, y[tr])
            oof[va] = m.predict(X_va); test_pred += m.predict(X_te) / N_FOLDS
            fq = qwk(y[va], np.clip(np.round(oof[va]), SCORE_MIN, SCORE_MAX).astype(int))
            fqs.append(fq)
            print(f"[fold {fold}/{N_FOLDS}] feats={X_tr.shape[1]} evr={evr:.3f} "
                  f"QWK(round)={fq:.4f} ({time.time()-ft:.0f}s)")
            del X_tr, X_va, X_te, m
    except Exception as e:
        print(f"[skip] {backend} failed: {type(e).__name__}: {e}"); return

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT_DIR, "oof_gbdt_chunk.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt_chunk.npy"), test_pred)
    print(f"[cv] mean fold QWK(round)={np.mean(fqs):.4f} +/- {np.std(fqs):.4f}")
    q = qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds))
    print(f"[oof] QWK optimized = {q:.4f}  (gbdt_hf=0.8134, gbdt=0.8133)")
    print(f"[saved] oof_gbdt_chunk.npy test_gbdt_chunk.npy  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] gbdt_chunk aborted: {type(e).__name__}: {e}")
        sys.exit(0)
