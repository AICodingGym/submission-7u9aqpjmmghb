#!/usr/bin/env python3
"""GBDT over [SVD(TF-IDF) + 28 hand + 12 linguistic] features (experiment).

Adds the readability / lexical-sophistication features from src.ling_features
(the only new-feature family whose per-feature correlation with score cleared
the noise level: difficult_words +0.57, long_syllable_ratio / coleman_liau /
syllables_per_word ~0.22) on top of the existing gbdt feature block.

Same leakage rules as train_gbdt (vectorizers/SVD/scaler fit on train fold only;
linguistic features are deterministic, computed once). Writes suffixed
artifacts; does not overwrite production gbdt.
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

from src.features import extract_text_features  # noqa: E402
from src.ling_features import extract_linguistic_features, LING_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_gbdt import build_svd_features, detect_backend  # noqa: E402
from train_cpu_baseline import (  # noqa: E402
    pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5

# placeholder-for-append


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] no lightgbm/xgboost; nothing written."); return
    print(f"[backend] {backend}")

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)

    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)
    print(f"[feat] computing {len(LING_COLUMNS)} linguistic features ...")
    ling_tr = extract_linguistic_features(tr_txt).to_numpy(dtype=np.float64)
    ling_te = extract_linguistic_features(te_txt).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train)); test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fqs = []
    try:
        for fold, (tr, va) in enumerate(skf.split(train, y), 1):
            ft = time.time()
            X_tr, X_va, X_te, _, _ = build_svd_features(
                tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                hand_tr[tr], hand_tr[va], hand_te)
            X_tr = np.hstack([X_tr, ling_tr[tr]])
            X_va = np.hstack([X_va, ling_tr[va]])
            X_te = np.hstack([X_te, ling_te])
            m = make_model(); m.fit(X_tr, y[tr])
            oof[va] = m.predict(X_va); test_pred += m.predict(X_te) / N_FOLDS
            fq = qwk(y[va], np.clip(np.round(oof[va]), SCORE_MIN, SCORE_MAX).astype(int))
            fqs.append(fq)
            print(f"[fold {fold}/{N_FOLDS}] feats={X_tr.shape[1]} (+{len(LING_COLUMNS)} ling) "
                  f"QWK(round)={fq:.4f} ({time.time()-ft:.0f}s)")
            del X_tr, X_va, X_te, m
    except Exception as e:
        print(f"[skip] {backend} failed: {type(e).__name__}: {e}"); return

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT_DIR, "oof_gbdt_ling.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt_ling.npy"), test_pred)

    print(f"[cv] mean fold QWK(round)={np.mean(fqs):.4f} +/- {np.std(fqs):.4f}")
    q_opt = qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds))
    print(f"[oof] QWK optimized-thresholds = {q_opt:.4f}  "
          f"(baseline gbdt=0.8133; gain={q_opt-0.8133:+.4f})")
    print(f"[saved] oof_gbdt_ling.npy test_gbdt_ling.npy  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] gbdt_ling aborted: {type(e).__name__}: {e}")
        sys.exit(0)
