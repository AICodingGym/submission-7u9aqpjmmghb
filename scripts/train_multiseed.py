#!/usr/bin/env python3
"""Experiment #2: multi-seed bagging of the two dominant base models.

Reruns the full 5-fold CV with several DIFFERENT fold seeds (each seed also
seeds the model's internal randomness) and averages the per-sample OOF and the
test predictions across seeds. Averaging N complete OOF vectors — each sample
predicted once per seed by its held-out fold — is a proper bagged OOF and
reduces the variance from a single arbitrary fold split.

Scope: ridge (blend weight ~0.2) and gbdt (blend weight ~0.7) = ~90% of the
winning ensemble. LogReg (weight 0.1, lbfgs ~deterministic) is left single-seed.

Leakage control: build_fold_features / build_svd_features fit vectorizers/SVD/
scaler on the TRAIN fold only; thresholds optimized on OOF only.

Artifacts:
  outputs/oof_ridge_ms.npy / test_ridge_ms.npy
  outputs/oof_gbdt_ms.npy  / test_gbdt_ms.npy
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

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
    build_fold_features, pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)
from train_gbdt import build_svd_features, detect_backend  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
N_FOLDS = 5
SEEDS = [42, 7, 123]        # fold + model seeds to average over

# placeholder-for-append


def oof_qwk(v, y):
    return qwk(y, apply_thresholds(v, optimize_thresholds(v, y).thresholds))


def run_ridge(train, y, text_train, text_test, hand_train, hand_test, test, seed):
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, va in skf.split(train, y):
        X_tr, X_va, X_te, _ = build_fold_features(
            text_train.iloc[tr], text_train.iloc[va], text_test,
            hand_train[tr], hand_train[va], hand_test)
        m = Ridge(alpha=1.0, random_state=seed).fit(X_tr, y[tr])
        oof[va] = m.predict(X_va); tp += m.predict(X_te) / N_FOLDS
        del X_tr, X_va, X_te, m
    return oof, tp


def run_gbdt(train, y, text_train, text_test, hand_train, hand_test, test, seed, make):
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, va in skf.split(train, y):
        X_tr, X_va, X_te, _, _ = build_svd_features(
            text_train.iloc[tr], text_train.iloc[va], text_test,
            hand_train[tr], hand_train[va], hand_test)
        m = make(); m.set_params(random_state=seed); m.fit(X_tr, y[tr])
        oof[va] = m.predict(X_va); tp += m.predict(X_te) / N_FOLDS
        del X_tr, X_va, X_te, m
    return oof, tp


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make = detect_backend()
    if backend is None:
        print("[skip] no gbdt backend; nothing written."); return

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str); text_test = test[text_col].astype(str)
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    for name, runner, extra in [("ridge", run_ridge, None), ("gbdt", run_gbdt, make)]:
        oof_sum = np.zeros(len(train)); tp_sum = np.zeros(len(test))
        single = None
        for s in SEEDS:
            st = time.time()
            if extra is None:
                oof, tp = runner(train, y, text_train, text_test, hand_train, hand_test, test, s)
            else:
                oof, tp = runner(train, y, text_train, text_test, hand_train, hand_test, test, s, extra)
            if single is None:
                single = oof_qwk(oof, y)
            print(f"[{name} seed={s}] OOF QWK={oof_qwk(oof,y):.4f} ({time.time()-st:.1f}s)")
            oof_sum += oof / len(SEEDS); tp_sum += tp / len(SEEDS)
        q_ms = oof_qwk(oof_sum, y)
        np.save(os.path.join(OUT_DIR, f"oof_{name}_ms.npy"), oof_sum)
        np.save(os.path.join(OUT_DIR, f"test_{name}_ms.npy"), tp_sum)
        print(f"[{name}] single-seed OOF={single:.4f}  {len(SEEDS)}-seed-avg OOF={q_ms:.4f}  "
              f"gain={q_ms-single:+.4f}")
        print(f"[saved] oof_{name}_ms.npy test_{name}_ms.npy")

    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"[skip] multiseed aborted: {type(e).__name__}: {e}")
        sys.exit(0)
