#!/usr/bin/env python3
"""Enhanced ensemble search: more draws + multi-seed + local polish.

Extends ensemble_search.py. Same discovery/validation and OOF-only rule, but:
  * random search over MANY Dirichlet draws across SEVERAL seeds, and
  * a local coordinate-ascent polish around the best random weight vector,
so the reported optimum is less dependent on a single lucky draw.

Writes outputs/final_submission.csv (overwrites) with the best blend.
Leakage control: weights + thresholds chosen on OOF only.
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
ID_COL, TARGET = "essay_id", "score"

DRAWS_PER_SEED = 4000
SEEDS = [0, 1, 2, 3, 42]          # multi-seed to de-lucky the search
POLISH_ROUNDS = 60                # coordinate-ascent steps after random search

# placeholder-for-append


def discover(n_train, n_test):
    names, oof_l, test_l = [], [], []
    for op in sorted(glob.glob(os.path.join(OUT_DIR, "oof_*.npy"))):
        name = os.path.basename(op)[4:-4]
        tp = os.path.join(OUT_DIR, f"test_{name}.npy")
        if not os.path.exists(tp):
            continue
        oof, test = np.load(op), np.load(tp)
        if (oof.ndim == 1 and test.ndim == 1 and oof.shape[0] == n_train
                and test.shape[0] == n_test
                and not np.isnan(oof).any() and not np.isnan(test).any()):
            names.append(name); oof_l.append(oof); test_l.append(test)
            print(f"  [ok] {name}")
        else:
            print(f"  [skip] {name}")
    return names, np.vstack(oof_l), np.vstack(test_l)


def qwk_of(vec, y):
    return qwk(y, apply_thresholds(vec, optimize_thresholds(vec, y).thresholds))


def _fixed_round(vec):
    """Fast integer mapping without threshold optimization (for ranking only)."""
    return np.clip(np.round(vec), LABELS[0], LABELS[-1]).astype(int)


def qwk_fast(vec, y):
    """Cheap QWK used to RANK candidates during search (no Nelder-Mead).

    Optimizing thresholds per candidate is ~100x slower and, empirically, does
    not change which weight vector ranks best — so search on round+clip, then
    optimize thresholds once on the winner.
    """
    return qwk(y, _fixed_round(vec))


def polish(w, OOF, y, rounds):
    """Coordinate ascent: nudge pairs of weights, keep changes that help OOF.

    Ranks with qwk_fast (round+clip) for speed; thresholds are optimized once
    on the final winner in main().
    """
    k = len(w)
    best_w = w.copy()
    best_q = qwk_fast(best_w @ OOF, y)
    steps = [0.05, 0.02, 0.01]
    for _ in range(rounds):
        improved = False
        for step in steps:
            for i in range(k):
                for j in range(k):
                    if i == j or best_w[i] < step:
                        continue
                    cand = best_w.copy()
                    cand[i] -= step
                    cand[j] += step
                    q = qwk_fast(cand @ OOF, y)
                    if q > best_q:
                        best_q, best_w = q, cand
                        improved = True
        if not improved:
            break
    return best_w, best_q


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    y = train[TARGET].to_numpy(dtype=int)
    n_train, n_test = len(train), len(test)

    print(f"[discover] n_train={n_train} n_test={n_test}")
    names, OOF, TEST = discover(n_train, n_test)
    k = len(names)
    print(f"[models] {k}: {names}")

    # random search across seeds (ranked with fast round+clip QWK)
    best_q, best_w = -1.0, None
    total = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ in range(DRAWS_PER_SEED):
            w = rng.dirichlet(np.ones(k))
            q = qwk_fast(w @ OOF, y)
            total += 1
            if q > best_q:
                best_q, best_w = q, w
    print(f"[random] best fast-QWK = {best_q:.4f} over {total} draws ({len(SEEDS)} seeds)")

    # local polish (also fast-ranked)
    pol_w, pol_q = polish(best_w, OOF, y, POLISH_ROUNDS)
    print(f"[polish] fast-QWK = {pol_q:.4f} ({pol_q-best_q:+.4f} vs random best)")
    best_w = pol_w

    # optimize thresholds ONCE on the winning blend -> the reportable OOF QWK
    best_thr = optimize_thresholds(best_w @ OOF, y).thresholds
    final_q = qwk(y, apply_thresholds(best_w @ OOF, best_thr))

    print("\n=== best ensemble ===")
    print(f"[weights] {{{', '.join(f'{n}: {round(float(x),4)}' for n,x in zip(names,best_w) if x>1e-4)}}}")
    print(f"[thresholds] {np.round(best_thr,4).tolist()} "
          f"(incr: {bool((np.diff(best_thr)>0).all())})")
    print(f"[oof] final OOF QWK = {final_q:.4f}")
    print(f"[ref] 3-model blend OOF=0.8183 (online 0.83301); "
          f"9-model rand OOF=0.8204 (online 0.83344)")

    test_labels = apply_thresholds(best_w @ TEST, best_thr)
    sub = sample.copy()
    id_to_label = dict(zip(test[ID_COL].to_numpy(), test_labels))
    mapped = sub[ID_COL].map(id_to_label)
    assert not mapped.isna().any(), "unmapped ids"
    sub[TARGET] = mapped.astype(int)
    assert list(sub.columns) == list(sample.columns)
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist()
    assert sub[TARGET].dtype.kind == "i"
    vals = set(sub[TARGET].unique().tolist())
    assert vals.issubset(set(LABELS)), vals

    out_path = os.path.join(OUT_DIR, "final_submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"[sub] wrote {out_path}  rows={len(sub)}  values={sorted(vals)}")
    print(f"[sub] dist: {sub[TARGET].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
