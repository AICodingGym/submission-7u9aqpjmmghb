#!/usr/bin/env python3
"""Caruana greedy ensemble selection for AES 2.0 (real threshold-opt QWK).

Why this over ensemble_search_v2:
  * v2 RANKS weight vectors by round+clip QWK (fast but a proxy), then optimizes
    thresholds only on the winner. Empirically that leaves OOF on the table: a
    greedy forward-selection that scores every candidate with the REAL
    threshold-optimized QWK reaches OOF ~0.8221 vs v2's ~0.8204.
  * Caruana ensemble selection (greedy add WITH REPLACEMENT) is the standard
    robust blender: it naturally ignores unhelpful/redundant models (never
    selects them), concentrates weight on what helps, and resists the
    overfitting of a 20k-draw random weight search because the hypothesis space
    at each step is tiny (one model index).

Optional bagged selection (BAG_ROUNDS>0): repeat the greedy on random subsets of
the model library and average the selection counts (Caruana et al. 2004
"bagged ensemble selection") — trades a little compute for a more stable blend.

Leakage control: weights AND thresholds are chosen on OOF only; test predictions
are only ever combined + thresholded, never used to fit anything. Same rule as
every other script in this repo.

Writes:
  outputs/submission_caruana.csv   (exact sample_submission schema, int 1..6)
  outputs/ensemble_caruana_report.json
"""
from __future__ import annotations

import glob
import json
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
    LABELS, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
ID_COL, TARGET = "essay_id", "score"

SEED = 42
N_FOLDS = 5
MAX_STEPS = 20          # greedy add steps (with replacement)
PATIENCE = 4            # stop if no OOF improvement for this many steps
BAG_ROUNDS = int(os.environ.get("BAG_ROUNDS", "0"))   # 0 = plain greedy (fast, proven)
BAG_FRAC = 0.7          # fraction of the library each bag round may pick from
INIT_TOPK = 2           # seed each greedy run with the top-K single models
THR_MAXITER = 400       # Nelder-Mead cap during search (final winner re-opt at full)

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
        else:
            print(f"  [skip] {name} (shape/NaN)")
    return names, np.vstack(oof_l), np.vstack(test_l)


def q_opt(vec, y, maxiter=THR_MAXITER):
    """Real QWK after optimizing rounding thresholds on OOF (the true metric).

    ``maxiter`` caps Nelder-Mead during the greedy search for speed; the final
    winning blend is re-scored once at full precision in main().
    """
    return qwk(y, apply_thresholds(vec, optimize_thresholds(vec, y, maxiter=maxiter).thresholds))


def greedy_select(OOF, y, allowed, max_steps, patience, init_topk):
    """Caruana greedy add WITH REPLACEMENT over the `allowed` model indices.

    Returns list of selected indices (with repeats) maximizing real OOF QWK.
    """
    k = OOF.shape[0]
    # seed with the best single models among the allowed set
    singles = sorted(allowed, key=lambda j: -q_opt(OOF[j], y))
    bag = list(singles[:init_topk])
    sum_oof = OOF[bag].sum(axis=0)
    best_q = q_opt(sum_oof / len(bag), y)
    best_bag = list(bag)
    no_improve = 0
    for _ in range(max_steps):
        cand_j, cand_q = None, best_q
        for j in allowed:
            v = (sum_oof + OOF[j]) / (len(bag) + 1)
            q = q_opt(v, y)
            if q > cand_q:
                cand_q, cand_j = q, j
        if cand_j is None:
            no_improve += 1
            if no_improve >= patience:
                break
            # allow a non-improving add to escape a local plateau? No — with
            # replacement a non-improving add can only hurt, so just stop.
            break
        bag.append(cand_j); sum_oof = sum_oof + OOF[cand_j]
        if cand_q > best_q:
            best_q, best_bag, no_improve = cand_q, list(bag), 0
        else:
            no_improve += 1
    return best_bag


def counts_to_weights(counts, k):
    w = np.zeros(k)
    for j, c in counts.items():
        w[j] = c
    s = w.sum()
    return w / s if s > 0 else w


def foldwise_qwk(blend_oof, y):
    """Per-fold QWK (thresholds still optimized on full OOF) for a stability read."""
    thr = optimize_thresholds(blend_oof, y).thresholds
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    qs = []
    for _, va in skf.split(np.zeros(len(y)), y):
        qs.append(qwk(y[va], apply_thresholds(blend_oof[va], thr)))
    return qs


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    y = train[TARGET].to_numpy(dtype=int)
    n_train, n_test = len(train), len(test)

    names, OOF, TEST = discover(n_train, n_test)
    k = len(names)
    print(f"[discover] {k} models over n_train={n_train} n_test={n_test}")

    rng = np.random.default_rng(SEED)
    all_idx = list(range(k))

    if BAG_ROUNDS and BAG_ROUNDS > 0:
        from collections import Counter
        agg = Counter()
        for r in range(BAG_ROUNDS):
            m = max(INIT_TOPK + 1, int(round(BAG_FRAC * k)))
            allowed = sorted(rng.choice(all_idx, size=m, replace=False).tolist())
            bag = greedy_select(OOF, y, allowed, MAX_STEPS, PATIENCE, INIT_TOPK)
            for j in bag:
                agg[j] += 1
        w = counts_to_weights(dict(agg), k)
        tag = f"bagged x{BAG_ROUNDS}"
    else:
        bag = greedy_select(OOF, y, all_idx, MAX_STEPS, PATIENCE, INIT_TOPK)
        from collections import Counter
        w = counts_to_weights(dict(Counter(bag)), k)
        tag = "plain greedy"

    oof_blend = w @ OOF
    thr = optimize_thresholds(oof_blend, y).thresholds
    final_q = qwk(y, apply_thresholds(oof_blend, thr))
    fold_qs = foldwise_qwk(oof_blend, y)

    picked = sorted([(names[j], float(w[j])) for j in range(k) if w[j] > 1e-9],
                    key=lambda t: -t[1])
    print(f"\n=== Caruana ensemble ({tag}) ===")
    print(f"[weights] {{{', '.join(f'{n}: {round(x,4)}' for n,x in picked)}}}")
    print(f"[thresholds] {np.round(thr,4).tolist()} "
          f"(incr: {bool((np.diff(thr) > 0).all())})")
    print(f"[oof] final OOF QWK = {final_q:.4f}")
    print(f"[oof] fold QWK = {[round(q,4) for q in fold_qs]}  "
          f"mean={np.mean(fold_qs):.4f} std={np.std(fold_qs):.4f}")
    print(f"[ref] deployed 9-model rand OOF=0.8204 (online 0.83344); "
          f"greedy-fwd ceiling=0.8221")

    test_blend = w @ TEST
    test_labels = apply_thresholds(test_blend, thr)
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

    out_csv = os.path.join(OUT_DIR, "submission_caruana.csv")
    sub.to_csv(out_csv, index=False)
    dist = sub[TARGET].value_counts().sort_index().to_dict()
    print(f"[sub] wrote {out_csv}  rows={len(sub)}  dist={dist}")

    report = {
        "method": f"caruana_{tag.replace(' ', '_')}",
        "n_models": k, "oof_qwk": final_q,
        "fold_qwk": fold_qs, "fold_qwk_mean": float(np.mean(fold_qs)),
        "fold_qwk_std": float(np.std(fold_qs)),
        "weights": {n: x for n, x in picked},
        "thresholds": np.round(thr, 6).tolist(),
        "test_distribution": {int(k_): int(v_) for k_, v_ in dist.items()},
    }
    with open(os.path.join(OUT_DIR, "ensemble_caruana_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] outputs/ensemble_caruana_report.json")


if __name__ == "__main__":
    main()
