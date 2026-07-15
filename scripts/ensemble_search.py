#!/usr/bin/env python3
"""Random-search ensemble weight optimizer for AES 2.0.

Discovers every model with BOTH an oof_<name>.npy and test_<name>.npy in
outputs/, validates their lengths, then random-searches non-negative weights
(summing to 1) to maximize OOF QWK — re-optimizing rounding thresholds for each
candidate blend. Writes the best blend as outputs/final_submission.csv.

Discovery is driven by oof_*.npy (then the matching test_*.npy), so 2D feature
dumps like test_hf_emb.npy (no oof_ counterpart) are never picked up.

Leakage control: weights AND thresholds are chosen on OOF only. test predictions
are only combined and thresholded, never used to fit anything.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.thresholds import (  # noqa: E402
    LABELS, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

SEED = 42
N_SEARCH = 1000
ID_COL, TARGET = "essay_id", "score"

# placeholder-for-append


def discover_models(n_train: int, n_test: int):
    """Find models with valid oof/test pairs. Returns (names, oof_mat, test_mat).

    A model is kept only if oof_<name>.npy has n_train rows, test_<name>.npy
    exists with n_test rows, both are 1-D and NaN-free. Skipped models are
    reported with the reason.
    """
    names, oof_list, test_list = [], [], []
    for oof_path in sorted(glob.glob(os.path.join(OUT_DIR, "oof_*.npy"))):
        name = os.path.basename(oof_path)[len("oof_"):-len(".npy")]
        test_path = os.path.join(OUT_DIR, f"test_{name}.npy")
        if not os.path.exists(test_path):
            print(f"  [skip] {name}: no test_{name}.npy")
            continue
        oof, test = np.load(oof_path), np.load(test_path)
        if oof.ndim != 1 or test.ndim != 1:
            print(f"  [skip] {name}: not 1-D (oof {oof.shape}, test {test.shape})")
            continue
        if oof.shape[0] != n_train:
            print(f"  [skip] {name}: oof len {oof.shape[0]} != n_train {n_train}")
            continue
        if test.shape[0] != n_test:
            print(f"  [skip] {name}: test len {test.shape[0]} != n_test {n_test}")
            continue
        if np.isnan(oof).any() or np.isnan(test).any():
            print(f"  [skip] {name}: contains NaN")
            continue
        names.append(name)
        oof_list.append(oof)
        test_list.append(test)
        print(f"  [ok]   {name}: oof({oof.shape[0]}) test({test.shape[0]})")
    if not names:
        return [], None, None
    return names, np.vstack(oof_list), np.vstack(test_list)


def oof_qwk(vec, y):
    thr = optimize_thresholds(vec, y).thresholds
    return qwk(y, apply_thresholds(vec, thr)), thr


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    y = train[TARGET].to_numpy(dtype=int)
    n_train, n_test = len(train), len(test)

    print(f"[discover] scanning outputs/ (n_train={n_train}, n_test={n_test})")
    names, OOF, TEST = discover_models(n_train, n_test)
    if not names:
        print("[error] no valid model pairs found; run the train_* scripts first.")
        sys.exit(1)
    k = len(names)
    print(f"[models] {k} usable: {names}")

    # single-model reference
    print("[single] OOF QWK per model:")
    for i, nm in enumerate(names):
        q, _ = oof_qwk(OOF[i], y)
        print(f"          {nm:10s}: {q:.4f}")

    # --- random search over the weight simplex (Dirichlet gives uniform simplex) ---
    rng = np.random.default_rng(SEED)
    best_q, best_w = -1.0, None
    for _ in range(N_SEARCH):
        w = rng.dirichlet(np.ones(k))
        blend = w @ OOF                      # (n_train,)
        q = qwk(y, apply_thresholds(blend, optimize_thresholds(blend, y).thresholds))
        if q > best_q:
            best_q, best_w = q, w

    # also try the trivial all-in-one-model corners as a safety net
    for i in range(k):
        q, _ = oof_qwk(OOF[i], y)
        if q > best_q:
            best_q = q
            best_w = np.eye(k)[i]

    # finalize thresholds on the best blend's OOF
    best_blend_oof = best_w @ OOF
    final_q, best_thr = oof_qwk(best_blend_oof, y)

    print("\n=== best ensemble ===")
    wdict = {nm: round(float(wi), 4) for nm, wi in zip(names, best_w) if wi > 1e-4}
    print(f"[weights] {wdict}")
    print(f"[thresholds] {np.round(best_thr, 4).tolist()} "
          f"(strictly increasing: {bool((np.diff(best_thr) > 0).all())})")
    print(f"[oof] best OOF QWK = {final_q:.4f}  (over {N_SEARCH} random draws)")

    # --- apply frozen weights + thresholds to TEST ---
    test_blend = best_w @ TEST
    test_labels = apply_thresholds(test_blend, best_thr)   # int, clipped to 1..6

    sub = sample.copy()
    id_to_label = dict(zip(test[ID_COL].to_numpy(), test_labels))
    mapped = sub[ID_COL].map(id_to_label)
    assert not mapped.isna().any(), f"{int(mapped.isna().sum())} ids unmapped"
    sub[TARGET] = mapped.astype(int)

    assert list(sub.columns) == list(sample.columns), "column mismatch"
    assert len(sub) == n_test, "row count != test"
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist(), "id order changed"
    assert sub[TARGET].dtype.kind == "i", "score not integer"
    vals = set(sub[TARGET].unique().tolist())
    assert vals.issubset(set(LABELS)), f"scores outside 1..6: {vals}"

    out_path = os.path.join(OUT_DIR, "final_submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"[sub] wrote {out_path}  cols={list(sub.columns)}  rows={len(sub)}  "
          f"values={sorted(vals)}")
    print(f"[sub] pred distribution: {sub[TARGET].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
