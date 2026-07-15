#!/usr/bin/env python3
"""Ridge over frozen HuggingFace sentence embeddings for AES 2.0 (CPU).

Consumes the precomputed MiniLM embeddings written by ``hf_embeddings.py``
(outputs/train_hf_emb.npy, outputs/test_hf_emb.npy) — it never recomputes them
(requirement 8). If they are missing it prints how to produce them and exits.

Pipeline: StandardScaler -> Ridge, 5-fold StratifiedKFold on the score, OOF +
test predictions saved as continuous scores for later blending. Thresholds are
optimized on OOF only (via src.thresholds) purely to report an OOF QWK.

NOTE: this is an exploratory model. Mean-pooled MiniLM embeddings drop the
essay-length signal that dominates this task, so it scored only ~0.556 OOF QWK
and is intentionally NOT part of the default blend (scripts/blend.py). Kept for
reference; do not wire its outputs into the blend without re-checking they help.

Artifacts:
  outputs/oof_hf_ridge.npy   (n_train,)
  outputs/test_hf_ridge.npy  (n_test,)
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
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

TRAIN_EMB = os.path.join(OUT_DIR, "train_hf_emb.npy")
TEST_EMB = os.path.join(OUT_DIR, "test_hf_emb.npy")

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 1.0
TARGET_COL_PRIORITY = ["score"]

# placeholder-for-append


def pick_target(df: pd.DataFrame) -> str:
    for name in TARGET_COL_PRIORITY:
        if name in df.columns:
            return name
    raise KeyError(f"no target column among {TARGET_COL_PRIORITY}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # req 1: embeddings must already exist; do NOT recompute (req 8)
    if not (os.path.exists(TRAIN_EMB) and os.path.exists(TEST_EMB)):
        print("[error] HF embeddings not found:")
        if not os.path.exists(TRAIN_EMB):
            print(f"          missing {TRAIN_EMB}")
        if not os.path.exists(TEST_EMB):
            print(f"          missing {TEST_EMB}")
        print("[hint] run:  python scripts/hf_embeddings.py   first, then re-run this script.")
        sys.exit(1)

    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    target = pick_target(train)
    y = train[target].to_numpy(dtype=int)

    # req 2: load precomputed embeddings
    X_train = np.load(TRAIN_EMB)
    X_test = np.load(TEST_EMB)
    print(f"[emb] train {X_train.shape}  test {X_test.shape}")
    assert X_train.shape[0] == len(train), \
        f"train emb rows {X_train.shape[0]} != train {len(train)}"
    assert X_test.shape[0] == len(test), \
        f"test emb rows {X_test.shape[0]} != test {len(test)}"
    assert X_train.shape[1] == X_test.shape[1], "train/test emb dim mismatch"

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        ft = time.time()
        # req 3: scaler fit on the TRAIN fold only (no leakage)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[tr_idx])
        Xva = scaler.transform(X_train[va_idx])
        Xte = scaler.transform(X_test)

        model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)  # req 4
        model.fit(Xtr, y[tr_idx])
        oof[va_idx] = model.predict(Xva)
        test_pred += model.predict(Xte) / N_FOLDS

        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fq = qwk(y[va_idx], va_round)
        fold_qwks.append(fq)
        print(f"[fold {fold}/{N_FOLDS}] QWK(round)={fq:.4f}  ({time.time()-ft:.1f}s)")
        del scaler, model, Xtr, Xva, Xte

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")

    # req 6: persist continuous predictions for downstream blending
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any(), "NaN in predictions"
    np.save(os.path.join(OUT_DIR, "oof_hf_ridge.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_hf_ridge.npy"), test_pred)

    # req 7: OOF QWK (round+clip and OOF-optimized thresholds)
    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {qwk(y, apply_thresholds(oof, result.thresholds)):.4f}")
    print(f"[saved] outputs/oof_hf_ridge.npy ({oof.shape[0]},)  "
          f"outputs/test_hf_ridge.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
