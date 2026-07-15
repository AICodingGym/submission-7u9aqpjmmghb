#!/usr/bin/env python3
"""Reproducible OOF blend of the per-model predictions for AES 2.0.

Reads the saved out-of-fold / test predictions written by the individual model
scripts (train_cpu_baseline, train_logreg, train_svr, train_gbdt), searches a
convex weight combination that maximizes OOF QWK, freezes rounding thresholds
on OOF, applies them to the blended test predictions, and writes a submission.

This turns the previously ad-hoc shell analysis (best blend ~0.818 OOF QWK)
into a one-command, reproducible artifact.

Leakage control:
  * Weights AND thresholds are chosen on OOF only; test predictions are only
    ever *combined* and *thresholded*, never used to fit anything.
  * Missing models are skipped gracefully so the blend still runs with whatever
    OOF/test pairs are present.

Artifacts:
  outputs/submission_blend.csv   (exact sample_submission schema, int 1..6)
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.thresholds import (  # noqa: E402
    LABELS,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

# model name -> (oof file, test file). Order = search/reporting order.
# The 3-model blend (ridge + logreg + gbdt) scored best online (QWK 0.83301).
# gbdt_hf was tried but only moved OOF by +0.0017 (within noise) and scored
# 0.83275 online — no real gain — so it is left out of the default blend.
MODELS = ["ridge", "logreg", "svr", "gbdt"]
WEIGHT_STEP = 0.1          # simplex grid resolution for the weight search

# placeholder-for-append


def load_predictions():
    """Load available (oof, test) pairs. Returns dict name -> (oof, test)."""
    preds = {}
    for name in MODELS:
        oof_p = os.path.join(OUT_DIR, f"oof_{name}.npy")
        test_p = os.path.join(OUT_DIR, f"test_{name}.npy")
        if os.path.exists(oof_p) and os.path.exists(test_p):
            preds[name] = (np.load(oof_p), np.load(test_p))
    return preds


def oof_qwk(vec: np.ndarray, y: np.ndarray) -> float:
    result = optimize_thresholds(vec, y)
    return qwk(y, apply_thresholds(vec, result.thresholds))


def simplex_weights(k: int, step: float):
    """Yield all weight tuples on the k-simplex at the given grid resolution."""
    n = int(round(1.0 / step))
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            last = n - sum(combo)
            yield tuple(c / n for c in combo) + (last / n,)


def search_weights(names, oofs, y):
    """Grid-search convex weights over models to maximize OOF QWK.

    Returns (best_weights_dict, best_qwk). Falls back to the single best model
    if the search somehow underperforms it.
    """
    best_w, best_q = None, -1.0
    for w in simplex_weights(len(names), WEIGHT_STEP):
        blend = sum(wi * oofs[i] for wi, i in zip(w, range(len(names))))
        q = oof_qwk(blend, y)
        if q > best_q:
            best_q, best_w = q, w

    # safety net: never do worse than the best single model
    for i, name in enumerate(names):
        q = oof_qwk(oofs[i], y)
        if q > best_q:
            onehot = tuple(1.0 if j == i else 0.0 for j in range(len(names)))
            best_q, best_w = q, onehot

    return {name: round(w, 3) for name, w in zip(names, best_w)}, best_q


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    y = train["score"].to_numpy(dtype=int)
    id_col, target = "essay_id", "score"

    preds = load_predictions()
    if not preds:
        print("[skip] no oof_*/test_* prediction pairs found in outputs/; "
              "run the model scripts first. Nothing written.")
        return
    names = [n for n in MODELS if n in preds]
    oofs = [preds[n][0] for n in names]
    tests = [preds[n][1] for n in names]

    # shape sanity: OOF aligns with train, test aligns with test file
    for n, o, t in zip(names, oofs, tests):
        assert o.shape[0] == len(train), f"{n} oof len {o.shape[0]} != train {len(train)}"
        assert t.shape[0] == len(test), f"{n} test len {t.shape[0]} != test {len(test)}"
        assert not np.isnan(o).any() and not np.isnan(t).any(), f"{n} has NaN preds"

    print(f"[models] {names}")
    print("[single] OOF QWK (optimized thresholds):")
    for n, o in zip(names, oofs):
        print(f"          {n:7s}: {oof_qwk(o, y):.4f}")

    weights, blend_qwk = search_weights(names, oofs, y)
    print(f"[blend]  weights = {weights}")
    print(f"[blend]  OOF QWK = {blend_qwk:.4f}")

    # build blended predictions and freeze thresholds ON OOF ONLY
    w_vec = np.array([weights[n] for n in names], dtype=np.float64)
    oof_blend = sum(wi * o for wi, o in zip(w_vec, oofs))
    test_blend = sum(wi * t for wi, t in zip(w_vec, tests))
    result = optimize_thresholds(oof_blend, y)
    print(f"[thr]    OOF thresholds = {np.round(result.thresholds, 4).tolist()} "
          f"(strictly increasing: {bool((np.diff(result.thresholds) > 0).all())})")

    test_labels = apply_thresholds(test_blend, result.thresholds)  # int, clipped 1..6

    # --- submission with EXACT sample_submission schema --------------------- #
    sub = sample.copy()
    id_to_label = dict(zip(test[id_col].to_numpy(), test_labels))
    mapped = sub[id_col].map(id_to_label)
    assert not mapped.isna().any(), \
        f"{int(mapped.isna().sum())} test id(s) had no prediction mapped"
    sub[target] = mapped.astype(int)

    assert list(sub.columns) == list(sample.columns), "column mismatch vs sample"
    assert len(sub) == len(test), "row count != test"
    assert sub[id_col].tolist() == sample[id_col].tolist(), "id order changed"
    assert sub[target].dtype.kind == "i", "score not integer"
    vals = set(sub[target].unique().tolist())
    assert vals.issubset(set(LABELS)), f"scores outside 1..6: {vals}"

    out_path = os.path.join(OUT_DIR, "submission_blend.csv")
    sub.to_csv(out_path, index=False)
    print(f"[sub]    wrote {out_path}")
    print(f"[sub]    columns={list(sub.columns)}  rows={len(sub)}  "
          f"values={sorted(vals)}")
    print(f"[sub]    pred distribution: {sub[target].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
