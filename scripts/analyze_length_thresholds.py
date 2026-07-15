#!/usr/bin/env python3
"""Diagnostic: would LENGTH-CONDITIONAL QWK thresholds help? (analysis only)

Currently the blend maps continuous predictions -> integers with ONE global set
of 5 thresholds (fit on OOF). Idea: use different thresholds per essay-length
bucket. This script decides whether that is worth doing, using existing OOF
predictions only — it trains nothing and writes no submission.

Two things are measured:
  (1) residual length bias: mean(oof - y) per length tercile. If ~0 everywhere,
      the models already use length well and there's no room.
  (2) HONEST held-out gain: fit thresholds on half the OOF, evaluate on the
      other half (thresholds never see the eval rows), global vs per-bucket.
      This is what would actually transfer to the leaderboard. An in-sample
      number is also shown to expose the overfitting gap.
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

from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    apply_thresholds, optimize_thresholds, quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_BUCKETS = 3

# placeholder-for-append


def load_blend_oof(y):
    """Reconstruct the production blend OOF (0.2 ridge + 0.1 logreg + 0.7 gbdt)."""
    r = np.load(os.path.join(OUT_DIR, "oof_ridge.npy"))
    l = np.load(os.path.join(OUT_DIR, "oof_logreg.npy"))
    g = np.load(os.path.join(OUT_DIR, "oof_gbdt.npy"))
    return 0.2 * r + 0.1 * l + 0.7 * g


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = train["score"].to_numpy(dtype=int)
    words = extract_text_features(train["full_text"].astype(str))["word_count"].to_numpy()
    oof = load_blend_oof(y)

    # length buckets by word-count terciles
    edges = np.quantile(words, np.linspace(0, 1, N_BUCKETS + 1))
    bucket = np.clip(np.digitize(words, edges[1:-1]), 0, N_BUCKETS - 1)

    print("# Length-conditional threshold analysis (blend OOF)\n")
    print(f"global OOF QWK (optimized) = "
          f"{qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds)):.4f}\n")

    # (1) residual length bias
    print("## (1) residual bias by length bucket  (mean(pred - true))")
    for b in range(N_BUCKETS):
        m = bucket == b
        print(f"  bucket {b} (n={m.sum():5d}, words {words[m].min():.0f}-{words[m].max():.0f}): "
              f"mean_resid={np.mean(oof[m]-y[m]):+.3f}  mean_score={y[m].mean():.2f}")
    print()

    # (2) honest held-out comparison: fit on half, eval on other half
    print("## (2) held-out QWK: global vs per-bucket thresholds")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    fit_idx, eval_idx = next(iter(skf.split(oof, y)))

    # global thresholds fit on fit-half
    gthr = optimize_thresholds(oof[fit_idx], y[fit_idx]).thresholds
    q_global = qwk(y[eval_idx], apply_thresholds(oof[eval_idx], gthr))

    # per-bucket thresholds fit on fit-half, applied to eval-half by bucket
    pred_eval = np.zeros(len(eval_idx), dtype=int)
    for b in range(N_BUCKETS):
        fmask = fit_idx[bucket[fit_idx] == b]
        emask_local = bucket[eval_idx] == b
        if fmask.size < 50 or emask_local.sum() == 0:
            thr = gthr  # fall back if a bucket is too small to fit safely
        else:
            thr = optimize_thresholds(oof[fmask], y[fmask]).thresholds
        pred_eval[emask_local] = apply_thresholds(oof[eval_idx][emask_local], thr)
    q_bucket = qwk(y[eval_idx], pred_eval)

    # in-sample (optimistic) per-bucket, for the overfitting gap
    pred_in = np.zeros(len(y), dtype=int)
    for b in range(N_BUCKETS):
        m = bucket == b
        thr = optimize_thresholds(oof[m], y[m]).thresholds
        pred_in[m] = apply_thresholds(oof[m], thr)
    q_bucket_insample = qwk(y, pred_in)

    print(f"  held-out  global   thresholds: {q_global:.4f}")
    print(f"  held-out  per-bucket thresholds: {q_bucket:.4f}   "
          f"(gain {q_bucket - q_global:+.4f})")
    print(f"  in-sample per-bucket (optimistic): {q_bucket_insample:.4f}  "
          f"<- overfitting ceiling, NOT achievable on test")
    print()
    gain = q_bucket - q_global
    if gain > 0.003:
        print(f"[verdict] worth trying: honest held-out gain {gain:+.4f} > 0.003 noise floor.")
    else:
        print(f"[verdict] NOT worth it: honest gain {gain:+.4f} <= 0.003 noise floor; "
              "the extra thresholds mostly overfit.")


if __name__ == "__main__":
    main()
