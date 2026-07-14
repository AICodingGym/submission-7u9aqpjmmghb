#!/usr/bin/env python3
"""CPU-only TF-IDF + LinearSVR baseline for AES 2.0 (graceful, skippable).

Reuses the EXACT feature pipeline from ``train_cpu_baseline`` (word TF-IDF (1,2)
+ char_wb TF-IDF (3,5) + 28 ``src.features`` hand features, fit per train fold),
so it never re-implements or drifts from the Ridge/LogReg baselines.

Model: ``LinearSVR`` (liblinear-style, fast on sparse) as a regressor; its
continuous output is thresholded on OOF exactly like the Ridge output.

Requirement 4 — self-skipping: if any fold errors OR a wall-clock budget is
exceeded, the script prints a clear ``[skip]`` reason and exits 0 WITHOUT
writing partial/broken artifacts, so an orchestrator's main flow is unaffected.
Only a fully-completed run writes:
  outputs/oof_svr.npy   (len == n_train)
  outputs/test_svr.npy  (len == n_test)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features, FEATURE_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import (  # noqa: E402
    build_fold_features,
    detect_id_column,
    load_data,
    pick_column,
    TARGET_COL_PRIORITY,
    TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")

SEED = 42
N_FOLDS = 5
C = 1.0
MAX_ITER = 2000
# self-skip budget: if the whole run can't finish within this many seconds,
# skip gracefully (requirement 4). Generous default; lower to fail faster.
TIME_BUDGET_SEC = 1800

# placeholder-for-append


class SkipModel(Exception):
    """Raised to abort the SVR run gracefully (never propagates as a failure)."""


def _train(train, y, text_train, text_test, hand_train, hand_test, test):
    """Run the 5-fold LinearSVR CV. Raises SkipModel on any error/timeout."""
    from sklearn.svm import LinearSVR  # imported here so an import error -> skip

    t0 = time.time()
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if time.time() - t0 > TIME_BUDGET_SEC:
            raise SkipModel(
                f"time budget {TIME_BUDGET_SEC}s exceeded before fold {fold}")
        ft = time.time()
        X_tr, X_va, X_te, n_feats = build_fold_features(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        model = LinearSVR(C=C, max_iter=MAX_ITER, random_state=SEED)
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_te) / N_FOLDS

        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fold_qwk = qwk(y[va_idx], va_round)
        fold_qwks.append(fold_qwk)
        print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  "
              f"QWK(round)={fold_qwk:.4f}  ({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    return oof, test_pred


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test, sample = load_data()

    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    _ = detect_id_column(train, test, target)
    print(f"[cols] text='{text_col}'  target='{target}'")
    print(f"[data] train={len(train)}  test={len(test)}  "
          f"(test length read from file, not hardcoded)")

    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    print(f"[feat] computing {len(FEATURE_COLUMNS)} hand features via src.features")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    try:
        oof, test_pred = _train(
            train, y, text_train, text_test, hand_train, hand_test, test)
    except SkipModel as e:
        print(f"[skip] LinearSVR skipped: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return
    except Exception as e:  # any unexpected training error -> skip, don't crash
        print(f"[skip] LinearSVR skipped due to error: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return

    # only a fully-successful run reaches here -> validate then persist
    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN in predictions; not writing artifacts.")
        return
    np.save(os.path.join(OUT_DIR, "oof_svr.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_svr.npy"), test_pred)

    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {qwk(y, apply_thresholds(oof, result.thresholds)):.4f}")
    print(f"[saved] outputs/oof_svr.npy ({oof.shape[0]},)  "
          f"outputs/test_svr.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    # top-level guard: nothing here should ever crash a calling pipeline
    try:
        run()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[skip] LinearSVR run aborted: {type(e).__name__}: {e}")
        sys.exit(0)
