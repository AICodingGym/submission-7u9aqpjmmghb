#!/usr/bin/env python3
"""LightGBM over [SVD(TF-IDF) + hand features + HF embeddings] for AES 2.0.

Extends the GBDT baseline (``train_gbdt.py``) by concatenating the frozen
MiniLM sentence embeddings (from ``hf_embeddings.py``) onto the existing dense
feature block:

    TruncatedSVD(word+char TF-IDF) [128]  +  hand features [28]  +  HF emb [384]

Rationale: HF-Ridge alone was weak (OOF ~0.556) because mean-pooled short-text
embeddings drop essay-length signal, but its errors are only ~0.65 correlated
with the TF-IDF/GBDT models — so letting a NON-linear model (LightGBM) decide
how to combine the embedding dims WITH the TF-IDF/length features may extract
value a separate linear head could not.

Reuses ``build_svd_features``/``detect_backend`` from train_gbdt so the TF-IDF
+ SVD + scaler logic is identical and fit on TRAIN folds only (no leakage). HF
embeddings are precomputed frozen features, indexed per fold — no leakage.

Skips gracefully (req parity with train_gbdt): no GBDT backend -> skip; missing
HF embeddings -> hint to run hf_embeddings.py; any fold error -> skip, no
partial artifacts. Never recomputes embeddings.

Artifacts (only on full success):
  outputs/oof_gbdt_hf.npy   (n_train,)
  outputs/test_gbdt_hf.npy  (n_test,)
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
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_gbdt import (  # noqa: E402
    build_svd_features,
    detect_backend,
    SVD_DIM,
)
from train_cpu_baseline import (  # noqa: E402
    pick_column,
    TARGET_COL_PRIORITY,
    TEXT_COL_PRIORITY,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
TRAIN_EMB = os.path.join(OUT_DIR, "train_hf_emb.npy")
TEST_EMB = os.path.join(OUT_DIR, "test_hf_emb.npy")

SEED = 42
N_FOLDS = 5

# placeholder-for-append


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] neither lightgbm nor xgboost is installed; nothing written.")
        return
    if not (os.path.exists(TRAIN_EMB) and os.path.exists(TEST_EMB)):
        print("[skip] HF embeddings missing; run `python scripts/hf_embeddings.py` first.")
        print("[skip] no artifacts written; main flow unaffected.")
        return
    print(f"[backend] using {backend}")

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)

    # precomputed frozen embeddings (req: do not recompute)
    hf_train = np.load(TRAIN_EMB)
    hf_test = np.load(TEST_EMB)
    assert hf_train.shape[0] == len(train) and hf_test.shape[0] == len(test), \
        "HF embedding rows do not match train/test"

    print(f"[feat] {len(FEATURE_COLUMNS)} hand + SVD({SVD_DIM}) TF-IDF + HF({hf_train.shape[1]})")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    try:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
            ft = time.time()
            # identical SVD-of-TFIDF + scaled hand features as train_gbdt
            X_tr, X_va, X_te, _, evr = build_svd_features(
                text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
                hand_train[tr_idx], hand_train[va_idx], hand_test,
            )
            # concatenate the frozen HF embeddings (dense) — GBDT handles scale
            X_tr = np.hstack([X_tr, hf_train[tr_idx]])
            X_va = np.hstack([X_va, hf_train[va_idx]])
            X_te = np.hstack([X_te, hf_test])

            model = make_model()
            model.fit(X_tr, y[tr_idx])
            oof[va_idx] = model.predict(X_va)
            test_pred += model.predict(X_te) / N_FOLDS

            va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
            fq = qwk(y[va_idx], va_round)
            fold_qwks.append(fq)
            print(f"[fold {fold}/{N_FOLDS}] feats={X_tr.shape[1]}  svd_evr={evr:.3f}  "
                  f"QWK(round)={fq:.4f}  ({time.time()-ft:.1f}s)")
            del X_tr, X_va, X_te, model
    except Exception as e:
        print(f"[skip] {backend} run failed: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN in predictions; not writing artifacts.")
        return
    np.save(os.path.join(OUT_DIR, "oof_gbdt_hf.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt_hf.npy"), test_pred)

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {qwk(y, apply_thresholds(oof, result.thresholds)):.4f}")
    print(f"[saved] outputs/oof_gbdt_hf.npy ({oof.shape[0]},)  "
          f"outputs/test_gbdt_hf.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover - defensive top-level guard
        print(f"[skip] GBDT+HF run aborted: {type(e).__name__}: {e}")
        sys.exit(0)
