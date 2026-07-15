#!/usr/bin/env python3
"""Which model pool generalizes best? Honest held-out comparison.

We've seen in-sample OOF QWK go UP while online goes DOWN (search overfits the
OOF). So this does NOT rank pools by in-sample OOF. Instead, for each candidate
pool it repeatedly: splits OOF rows into fit/eval halves (stratified), random-
searches weights on the FIT half, and scores QWK on the EVAL half (weights +
thresholds never saw those rows). The averaged held-out QWK is what actually
predicts leaderboard behavior.

Pools compared (all using the multi-seed bagged _ms predictions):
  all9      : every model incl. weak hf_ridge/sgd/svr
  no_weak   : drop hf_ridge, sgd, svr  (the user's hypothesis)
  strong4   : ridge + logreg + gbdt + one gbdt variant
  core3     : ridge + logreg + gbdt only
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.thresholds import (  # noqa: E402
    optimize_thresholds, apply_thresholds, quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
DRAWS = 3000
N_SPLITS = 4          # repeat held-out splits to average out split noise

POOLS = {
    "all9":    ["ridge", "logreg", "svr", "sgd", "gbdt", "gbdt_hf", "gbdt_clust", "gbdt_spell", "hf_ridge"],
    "no_weak": ["ridge", "logreg", "gbdt", "gbdt_hf", "gbdt_clust", "gbdt_spell"],
    "strong4": ["ridge", "logreg", "gbdt", "gbdt_hf"],
    "core3":   ["ridge", "logreg", "gbdt"],
}

# placeholder-for-append


def fast_q(v, y):
    return qwk(y, np.clip(np.round(v), 1, 6).astype(int))


def search_weights(OOF, y, draws, rng):
    """Random-search weights maximizing round+clip QWK on the given rows."""
    k = OOF.shape[0]
    best_q, best_w = -1.0, None
    for _ in range(draws):
        w = rng.dirichlet(np.ones(k))
        q = fast_q(w @ OOF, y)
        if q > best_q:
            best_q, best_w = q, w
    return best_w


def held_out_qwk(names, y, n_train, splits, draws):
    """Mean held-out QWK: fit weights+thresholds on fit-half, score eval-half."""
    OOF = np.vstack([np.load(os.path.join(OUT_DIR, f"oof_{m}_ms.npy")) for m in names])
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    scores = []
    for si, (fit_idx, eval_idx) in enumerate(skf.split(np.zeros(n_train), y)):
        rng = np.random.default_rng(100 + si)
        w = search_weights(OOF[:, fit_idx], y[fit_idx], draws, rng)
        # thresholds fit on the SAME fit-half only
        thr = optimize_thresholds(w @ OOF[:, fit_idx], y[fit_idx]).thresholds
        q = qwk(y[eval_idx], apply_thresholds(w @ OOF[:, eval_idx], thr))
        scores.append(q)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    y = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))["score"].to_numpy(dtype=int)
    n = len(y)
    print(f"Held-out weight-search comparison ({N_SPLITS} splits x {DRAWS} draws)")
    print("(weights + thresholds fit on fit-half, QWK scored on unseen eval-half)\n")
    rows = []
    for pool, names in POOLS.items():
        m, s = held_out_qwk(names, y, n, N_SPLITS, DRAWS)
        rows.append((pool, m, s, len(names)))
        print(f"  {pool:9s} (k={len(names)}): held-out QWK = {m:.4f} +/- {s:.4f}")
    rows.sort(key=lambda r: -r[1])
    print(f"\n[best pool] {rows[0][0]}  held-out QWK={rows[0][1]:.4f}")
    print("Higher held-out QWK => better expected leaderboard generalization.")


if __name__ == "__main__":
    main()
