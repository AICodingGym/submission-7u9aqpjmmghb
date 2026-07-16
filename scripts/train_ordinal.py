#!/usr/bin/env python3
"""Ordinal regression (Frank & Hall) for AES 2.0 — an ensemble-DIVERSITY play.

Our library is saturated with squared-error regressors (ridge/svr/gbdt) and one
multinomial LogReg. None models the ORDER of the 1..6 scale directly. Frank &
Hall (2001) decomposes a K-class ordinal target into K-1 binary problems:

    for each cut k in 1..5:  P(y > k)  via a binary LogisticRegression
    P(y = j) recovered from consecutive differences; and, using the identity for
    a 1-based ordinal, the expected score is simply
        E[y] = 1 + sum_{k=1..5} P(y > k)

This is a genuinely different loss (five cumulative log-losses vs one squared /
multinomial loss) so its errors should decorrelate from the existing members —
which is the whole point (the bag is TF-IDF-signal-saturated; new members only
help if they are decorrelated, per the 2026-07-16 lesson).

Features: the SAME word+char TF-IDF + 28 hand features as the ridge/logreg
baselines (build_fold_features), fit on the TRAIN fold only. Thresholds for QWK
optimized on OOF only. Same leakage rules as every other script.

Artifacts:
  outputs/oof_ordinal.npy   (n_train, continuous E[y] in [1,6])
  outputs/test_ordinal.npy  (n_test)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features, FEATURE_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import (  # noqa: E402
    build_fold_features, load_data, pick_column,
    TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5
C = 1.0
MAX_ITER = 1000
CUTS = [1, 2, 3, 4, 5]      # P(y > k) for k in CUTS  (labels are 1..6)

# placeholder-for-append


def ordinal_expected(X_tr, y_tr, X_eval):
    """Frank & Hall: train K-1 binary P(y>k) models, return E[y] on X_eval.

    Each binary model may (rarely) see only one class in a fold if a cut is
    degenerate; guard by falling back to a constant probability in that case.
    """
    n = X_eval.shape[0]
    p_gt = np.zeros((n, len(CUTS)), dtype=np.float64)
    for i, k in enumerate(CUTS):
        yb = (y_tr > k).astype(int)
        if yb.min() == yb.max():
            # degenerate cut (all one side) -> constant prob = the observed rate
            p_gt[:, i] = float(yb.mean())
            continue
        clf = LogisticRegression(C=C, max_iter=MAX_ITER, solver="liblinear")
        clf.fit(X_tr, yb)
        # column for class "1" (y>k). classes_ is [0,1] after the guard above.
        pos = list(clf.classes_).index(1)
        p_gt[:, i] = clf.predict_proba(X_eval)[:, pos]
    # enforce monotone non-increasing P(y>k) across k (isotonic-lite): a higher
    # cut cannot have higher exceedance prob. Clip then cumulative-min.
    p_gt = np.clip(p_gt, 0.0, 1.0)
    p_gt = np.minimum.accumulate(p_gt, axis=1)
    # E[y] = 1 + sum_k P(y>k)  for a 1-based ordinal scale
    return 1.0 + p_gt.sum(axis=1)


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test, sample = load_data()
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)
    print(f"[feat] {len(FEATURE_COLUMNS)} hand + word/char TF-IDF (same as baselines)")
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train)); test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fqs = []
    for fold, (tr, va) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        X_tr, X_va, X_te, n_feats = build_fold_features(
            tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
            hand_tr[tr], hand_tr[va], hand_te)
        oof[va] = ordinal_expected(X_tr, y[tr], X_va)
        test_pred += ordinal_expected(X_tr, y[tr], X_te) / N_FOLDS
        fq = qwk(y[va], np.clip(np.round(oof[va]), SCORE_MIN, SCORE_MAX).astype(int))
        fqs.append(fq)
        print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  QWK(round)={fq:.4f} "
              f"({time.time()-ft:.0f}s)")
        del X_tr, X_va, X_te

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT_DIR, "oof_ordinal.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_ordinal.npy"), test_pred)
    print(f"[cv] mean fold QWK(round)={np.mean(fqs):.4f} +/- {np.std(fqs):.4f}")
    q = qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds))
    print(f"[oof] QWK optimized = {q:.4f}  (logreg=0.7942, ridge_tuned=0.8050)")
    print(f"[saved] oof_ordinal.npy test_ordinal.npy  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] ordinal aborted: {type(e).__name__}: {e}")
        sys.exit(0)
