#!/usr/bin/env python3
"""Tune LogisticRegression C on the cached big TF-IDF (expected-score head).

logreg's C was never tuned (always C=1.0). Reuses the cached big TF-IDF folds
(outputs/_tfidf_cache/big_*), sweeps C, and reports OOF QWK via the expected-
score mapping used in train_logreg.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.thresholds import (  # noqa: E402
    apply_thresholds, optimize_thresholds, quadratic_weighted_kappa as qwk,
)
from train_logreg import expected_score  # noqa: E402
from train_cpu_baseline import pick_column, TARGET_COL_PRIORITY  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
CACHE = os.path.join(OUT_DIR, "_tfidf_cache")
SEED = 42
N_FOLDS = 5
CS = [float(x) for x in os.environ.get("CS", "0.3,1.0,3.0,10.0").split(",")]

# placeholder-for-append


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
    for k in range(N_FOLDS):
        if not os.path.exists(os.path.join(CACHE, f"big_f{k}_tr.npz")):
            print(f"missing cache fold {k}"); return
    for C in CS:
        t0 = time.time(); oof = np.zeros(len(train))
        for k in range(N_FOLDS):
            tr, va = folds[k]
            Xtr = sp.load_npz(os.path.join(CACHE, f"big_f{k}_tr.npz"))
            Xva = sp.load_npz(os.path.join(CACHE, f"big_f{k}_va.npz"))
            m = LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(Xtr, y[tr])
            oof[va] = expected_score(m.predict_proba(Xva), m.classes_)
        thr = optimize_thresholds(oof, y).thresholds
        q = qwk(y, apply_thresholds(oof, thr))
        print(f"[logreg C={C}] OOF QWK={q:.4f} ({time.time()-t0:.0f}s)")
    print("[baseline] prod logreg (C=1, prod tfidf) OOF=0.7921")


if __name__ == "__main__":
    main()
