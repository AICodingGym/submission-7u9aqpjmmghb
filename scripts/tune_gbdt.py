#!/usr/bin/env python3
"""Hyperparameter sweep for the GBDT model, scored on the SAME 5-fold OOF.

Analysis/experiment tool: it does NOT overwrite the production oof_gbdt/test_gbdt
artifacts. It prints an OOF-QWK ranking of {SVD_DIM x LightGBM config} so we can
decide whether any config beats the current gbdt (OOF 0.8133) by more than the
~0.003 noise floor before adopting it into train_gbdt.py.

Efficiency: TF-IDF + SVD features are built ONCE per SVD_DIM per fold and cached
in memory, then every LightGBM config is trained on those cached matrices — so
we pay the expensive vectorization once, not once per config.

Leakage control mirrors train_gbdt: vectorizers/SVD/scaler fit on the train fold
only; thresholds optimized on OOF only.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
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
from train_cpu_baseline import (  # noqa: E402
    build_char_vectorizer, build_word_vectorizer, pick_column,
    TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5

SVD_DIMS = [128, 256]
# LightGBM configs to sweep (name -> kwargs). Kept small to bound CPU time.
CONFIGS = {
    "base_500":    dict(n_estimators=500,  learning_rate=0.05, num_leaves=31),
    "trees_1000":  dict(n_estimators=1000, learning_rate=0.03, num_leaves=31),
    "leaves_63":   dict(n_estimators=800,  learning_rate=0.03, num_leaves=63),
    "deep_slow":   dict(n_estimators=1500, learning_rate=0.02, num_leaves=63),
}

# placeholder-for-append


def build_svd_block(text_tr, text_va, text_te, hand_tr, hand_va, hand_te, dim):
    """word+char TF-IDF -> TruncatedSVD(dim) + scaled hand feats. Fit on train."""
    wv = build_word_vectorizer()
    Xw_tr, Xw_va, Xw_te = wv.fit_transform(text_tr), wv.transform(text_va), wv.transform(text_te)
    cv = build_char_vectorizer()
    Xc_tr, Xc_va, Xc_te = cv.fit_transform(text_tr), cv.transform(text_va), cv.transform(text_te)
    Xs_tr = sp.hstack([Xw_tr, Xc_tr]).tocsr()
    Xs_va = sp.hstack([Xw_va, Xc_va]).tocsr()
    Xs_te = sp.hstack([Xw_te, Xc_te]).tocsr()
    svd = TruncatedSVD(n_components=dim, random_state=SEED)
    Zt_tr, Zt_va, Zt_te = svd.fit_transform(Xs_tr), svd.transform(Xs_va), svd.transform(Xs_te)
    sc = StandardScaler()
    Ht_tr, Ht_va, Ht_te = sc.fit_transform(hand_tr), sc.transform(hand_va), sc.transform(hand_te)
    return (np.hstack([Zt_tr, Ht_tr]), np.hstack([Zt_va, Ht_va]), np.hstack([Zt_te, Ht_te]))


def main():
    import lightgbm as lgb

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(train, y))

    def score_oof(oof):
        r = optimize_thresholds(oof, y)
        return (qwk(y, np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)),
                qwk(y, apply_thresholds(oof, r.thresholds)))

    results = []
    for dim in SVD_DIMS:
        # build features once per dim, cache per fold
        t_feat = time.time()
        cache = []
        for tr_idx, va_idx in folds:
            X_tr, X_va, X_te = build_svd_block(
                text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
                hand_train[tr_idx], hand_train[va_idx], hand_test, dim)
            cache.append((tr_idx, va_idx, X_tr, X_va, X_te))
        print(f"[svd={dim}] features built for {N_FOLDS} folds "
              f"({time.time()-t_feat:.1f}s)")

        for cname, cfg in CONFIGS.items():
            t0 = time.time()
            oof = np.zeros(len(train), dtype=np.float64)
            for tr_idx, va_idx, X_tr, X_va, X_te in cache:
                model = lgb.LGBMRegressor(
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                    n_jobs=-1, verbose=-1, **cfg)
                model.fit(X_tr, y[tr_idx])
                oof[va_idx] = model.predict(X_va)
            q_round, q_opt = score_oof(oof)
            dt = time.time() - t0
            results.append((dim, cname, q_round, q_opt, dt))
            print(f"[svd={dim} {cname:11s}] QWK round={q_round:.4f} opt={q_opt:.4f} "
                  f"({dt:.1f}s)")
        del cache

    print("\n=== ranking by OOF QWK (optimized thresholds) ===")
    results.sort(key=lambda r: -r[3])
    print(f"{'svd':>4} {'config':12s} {'QWK_opt':>8} {'QWK_round':>10} {'time_s':>8}")
    for dim, cname, qr, qo, dt in results:
        print(f"{dim:>4} {cname:12s} {qo:>8.4f} {qr:>10.4f} {dt:>8.1f}")
    best = results[0]
    print(f"\n[best] svd={best[0]} {best[1]}  OOF_QWK_opt={best[3]:.4f}")
    print(f"[baseline] current gbdt OOF_QWK_opt=0.8133  "
          f"gain={best[3]-0.8133:+.4f} (adopt only if > +0.0030)")


if __name__ == "__main__":
    main()
