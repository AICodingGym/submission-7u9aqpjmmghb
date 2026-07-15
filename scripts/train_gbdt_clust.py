#!/usr/bin/env python3
"""Experiment #1: add cluster/topic features to the GBDT model.

The shift analysis found distinct essay style-groups but never fed that signal
back into a model. Here each fold fits KMeans on the TRAIN-fold dense block
(SVD(TF-IDF)+hand) and appends the per-essay distance-to-each-centroid vector
(``KMeans.transform``) as K extra features, then trains LightGBM on the
augmented block. Distances encode "which style-group is this essay near and how
strongly" — a coordinate TF-IDF/SVD doesn't hand the tree directly.

Leakage control: KMeans (and all vectorizers/SVD/scaler inside build_svd_features)
fit on the TRAIN fold only; thresholds optimized on OOF only.

Decision rule: adopt only if OOF QWK beats the current gbdt (0.8133) AND the
blend improves by more than the ~0.003 noise floor. Does NOT overwrite the
production oof_gbdt/test_gbdt — writes its own suffixed artifacts.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_gbdt import build_svd_features, detect_backend  # noqa: E402
from train_cpu_baseline import (  # noqa: E402
    pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5
N_CLUSTERS = 4          # distance-to-centroid features to append

# placeholder-for-append


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] no lightgbm/xgboost; nothing written.")
        return
    print(f"[backend] {backend}")

    train = pd.read_csv(os.path.join(REPO_ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(REPO_ROOT, "data", "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        X_tr, X_va, X_te, _, evr = build_svd_features(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        # KMeans fit on TRAIN fold only; append distance-to-centroid features
        km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
        km.fit(StandardScaler().fit_transform(X_tr))
        # transform on the SAME scale each split was built on (X_* already scaled
        # hand + raw SVD; use a fresh scaler consistently for the distance space)
        sc = StandardScaler().fit(X_tr)
        D_tr = km.transform(sc.transform(X_tr))
        D_va = km.transform(sc.transform(X_va))
        D_te = km.transform(sc.transform(X_te))

        X_tr = np.hstack([X_tr, D_tr])
        X_va = np.hstack([X_va, D_va])
        X_te = np.hstack([X_te, D_te])

        model = make_model()
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_te) / N_FOLDS
        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fq = qwk(y[va_idx], va_round)
        fold_qwks.append(fq)
        print(f"[fold {fold}/{N_FOLDS}] feats={X_tr.shape[1]} (+{N_CLUSTERS} clust)  "
              f"QWK(round)={fq:.4f}  ({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model, km

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    np.save(os.path.join(OUT_DIR, "oof_gbdt_clust.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt_clust.npy"), test_pred)

    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    q_opt = qwk(y, apply_thresholds(oof, result.thresholds))
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {q_opt:.4f}")
    print(f"[baseline] gbdt=0.8133  gain={q_opt-0.8133:+.4f} (adopt only if > +0.0030)")
    print(f"[saved] outputs/oof_gbdt_clust.npy  outputs/test_gbdt_clust.npy")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] gbdt_clust aborted: {type(e).__name__}: {e}")
        sys.exit(0)
