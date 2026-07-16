#!/usr/bin/env python3
"""Tune GBDT regularization on cached SVD features (survives job ceiling).

Our gbdt was only swept over trees/leaves before — never its regularization
(reg_alpha, reg_lambda, min_child_samples). This caches the SVD(128)+hand
feature block per fold once (the slow part), then trains many LightGBM configs
fast from cache.

  MODE=build  python scripts/tune_gbdt_reg.py     # cache 5 folds' dense features
  MODE=train  python scripts/tune_gbdt_reg.py      # sweep configs from cache
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

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
from train_gbdt import build_svd_features  # noqa: E402
from train_cpu_baseline import pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
CACHE = os.path.join(OUT_DIR, "_svd_cache")
SEED = 42
N_FOLDS = 5

CONFIGS = {
    "prod":     dict(n_estimators=500,  learning_rate=0.05, num_leaves=31),
    "reg_mid":  dict(n_estimators=800,  learning_rate=0.03, num_leaves=31,
                     reg_alpha=0.1, reg_lambda=1.0, min_child_samples=40),
    "reg_strong": dict(n_estimators=1200, learning_rate=0.02, num_leaves=31,
                       reg_alpha=0.5, reg_lambda=3.0, min_child_samples=60,
                       subsample=0.7, colsample_bytree=0.6),
    "reg_leaves": dict(n_estimators=1000, learning_rate=0.03, num_leaves=15,
                       reg_alpha=0.2, reg_lambda=2.0, min_child_samples=50),
}

# placeholder-for-append


def _common():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    tc = pick_column(train, TEXT_COL_PRIORITY, "text")
    tg = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[tg].to_numpy(dtype=int)
    tr_txt = train[tc].astype(str); te_txt = test[tc].astype(str)
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
    return train, y, tr_txt, te_txt, hand_tr, hand_te, folds


def build(fold_ids):
    os.makedirs(CACHE, exist_ok=True)
    train, y, tr_txt, te_txt, hand_tr, hand_te, folds = _common()
    for k in fold_ids:
        mk = os.path.join(CACHE, f"f{k}_tr.npy")
        if os.path.exists(mk):
            print(f"[build f{k}] cached, skip"); continue
        t0 = time.time(); tr, va = folds[k]
        X_tr, X_va, _, _, _ = build_svd_features(
            tr_txt.iloc[tr], tr_txt.iloc[va], te_txt.iloc[:2],
            hand_tr[tr], hand_tr[va], hand_te[:2])
        np.save(os.path.join(CACHE, f"f{k}_tr.npy"), X_tr.astype(np.float32))
        np.save(os.path.join(CACHE, f"f{k}_va.npy"), X_va.astype(np.float32))
        print(f"[build f{k}] {X_tr.shape} ({time.time()-t0:.0f}s)")


def train():
    import lightgbm as lgb
    train, y, *_ , folds = _common()
    for k in range(N_FOLDS):
        if not os.path.exists(os.path.join(CACHE, f"f{k}_tr.npy")):
            print(f"[train] missing fold {k}; build first."); return
    for cname, cfg in CONFIGS.items():
        t0 = time.time(); oof = np.zeros(len(train))
        for k in range(N_FOLDS):
            tr, va = folds[k]
            Xtr = np.load(os.path.join(CACHE, f"f{k}_tr.npy"))
            Xva = np.load(os.path.join(CACHE, f"f{k}_va.npy"))
            base = dict(random_state=SEED, n_jobs=-1, verbose=-1,
                        subsample=0.8, colsample_bytree=0.8)
            base.update(cfg)
            m = lgb.LGBMRegressor(**base).fit(Xtr, y[tr]); oof[va] = m.predict(Xva)
        thr = optimize_thresholds(oof, y).thresholds
        q = qwk(y, apply_thresholds(oof, thr))
        print(f"[{cname:11s}] OOF QWK={q:.4f} ({time.time()-t0:.0f}s)")
    print("[baseline] prod gbdt OOF=0.8133")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "train")
    if mode == "build":
        build([int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(",")])
    else:
        train()
