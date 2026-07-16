#!/usr/bin/env python3
"""Phase-1: materialize the shared CV folds + near-duplicate leakage check.

Writes the fixed folds every subsequent model reuses (via src.cv), plus a
near-duplicate report so we can confirm the standard split doesn't straddle
near-verbatim essays. All fold definitions are UNSUPERVISED (text/score-strat
only), never fit on anything test-derived.

Outputs:
  outputs/folds_standard.csv   (essay_id, fold_seed42, fold_seed7)
  outputs/folds_shift.csv      (essay_id, shift_fold, topic_cluster)
  outputs/dedup_report.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.cv import (  # noqa: E402
    SEEDS, fold_id_array, get_shift_folds, near_duplicate_report,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(REPO_ROOT, "outputs")


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = train["score"].to_numpy(int)
    txt = train["full_text"].astype(str)
    ids = train["essay_id"].to_numpy()

    # standard folds for each bagging seed
    std = pd.DataFrame({"essay_id": ids})
    for s in SEEDS:
        std[f"fold_seed{s}"] = fold_id_array(y, seed=s)
    std.to_csv(os.path.join(OUT, "folds_standard.csv"), index=False)

    # shift-aware folds + topic clusters
    shift_folds, groups = get_shift_folds(y, txt, k=8)
    sfid = np.full(len(y), -1, int)
    for f, (_, va) in enumerate(shift_folds):
        sfid[va] = f
    pd.DataFrame({"essay_id": ids, "shift_fold": sfid,
                  "topic_cluster": groups}).to_csv(
        os.path.join(OUT, "folds_shift.csv"), index=False)

    # near-duplicate check on the STANDARD seed-42 split
    nn_sim, pairs = near_duplicate_report(txt, sim_threshold=0.95)
    f42 = fold_id_array(y, seed=42)
    cross = sum(1 for i, j, _ in pairs if f42[i] != f42[j])
    report = {
        "n_train": len(y),
        "near_dup_pairs_ge_0.95": len(pairs),
        "cross_fold_dup_pairs_seed42": int(cross),
        "max_nn_sim": round(float(nn_sim.max()), 4),
        "mean_nn_sim": round(float(nn_sim.mean()), 4),
        "frac_ge_0.90": round(float((nn_sim >= 0.90).mean()), 4),
        "frac_ge_0.95": round(float((nn_sim >= 0.95).mean()), 4),
        "example_pairs": [[i, j, round(s, 4)] for i, j, s in pairs[:10]],
        "shift_fold_score_balance": {
            int(f): pd.Series(y[sfid == f]).value_counts().sort_index().to_dict()
            for f in range(5)},
    }
    with open(os.path.join(OUT, "dedup_report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))
    print("[phase1] wrote folds_standard.csv folds_shift.csv dedup_report.json")


if __name__ == "__main__":
    main()
