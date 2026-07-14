#!/usr/bin/env python3
"""CPU-only TF-IDF + LogisticRegression (expected-score) baseline for AES 2.0.

Reuses the EXACT feature pipeline from ``train_cpu_baseline`` — word TF-IDF
(1,2) + char_wb TF-IDF (3,5) + 28 hand features from ``src.features`` — fit on
each training fold only, so nothing here re-implements feature logic or risks
drifting from the Ridge baseline.

Model: multinomial LogisticRegression (lbfgs solver, which uses softmax /
multinomial loss for multiclass by default). Its 6-class ``predict_proba`` is
collapsed to a single continuous "expected score"
    E[score] = sum_k P(class = k) * k,   k in 1..6
which is thresholded on OOF exactly like the Ridge continuous output.

Leakage control mirrors the Ridge script: vectorizers/scaler fit on train folds
only; thresholds optimized on OOF only (``src.thresholds`` has no test param);
test length read from the file, never hardcoded.

Artifacts:
  outputs/oof_logreg.npy   (len == n_train, continuous expected scores)
  outputs/test_logreg.npy  (len == n_test,  continuous expected scores)
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
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
# reuse the identical fold-feature builder + column detection from the Ridge script
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
MAX_ITER = 1000          # lbfgs cap; lower this first if a fold is too slow (req 7)

# placeholder-for-append


def expected_score(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Collapse per-class probabilities to a continuous expected score.

    E[score] = sum_k P(k) * k, where ``classes`` gives the score value of each
    probability column (LogisticRegression sorts them ascending, e.g. 1..6).
    Works even if a rare class is absent from a given train fold — only the
    classes the model actually saw contribute, which is the correct expectation.
    """
    return proba @ classes.astype(np.float64)


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test, sample = load_data()

    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    id_col = detect_id_column(train, test, target)
    print(f"[cols] text='{text_col}'  target='{target}'  id='{id_col}'")
    print(f"[data] train={len(train)}  test={len(test)}  "
          f"(test length read from file, not hardcoded)")

    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)

    print(f"[feat] computing {len(FEATURE_COLUMNS)} hand features via src.features")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        # identical features to the Ridge baseline (fit on train fold only)
        X_tr, X_va, X_te, n_feats = build_fold_features(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        model = LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver="lbfgs",         # lbfgs is multinomial for multiclass by default
        )
        model.fit(X_tr, y[tr_idx])
        classes = model.classes_          # ascending scores actually seen in this fold

        oof[va_idx] = expected_score(model.predict_proba(X_va), classes)
        test_pred += expected_score(model.predict_proba(X_te), classes) / N_FOLDS

        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fold_qwk = qwk(y[va_idx], va_round)
        fold_qwks.append(fold_qwk)
        print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  classes={classes.tolist()}  "
              f"QWK(round)={fold_qwk:.4f}  ({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    return train, y, oof, test_pred, t0


def finalize(train, y, oof, test_pred, t0):
    np.save(os.path.join(OUT_DIR, "oof_logreg.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_logreg.npy"), test_pred)
    assert oof.shape[0] == len(train), "oof length != n_train"
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any(), "NaN in predictions"

    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    qwk_round = qwk(y, oof_round)

    # thresholds optimized ON OOF ONLY, mirroring the Ridge pipeline
    result = optimize_thresholds(oof, y)
    qwk_opt = qwk(y, apply_thresholds(oof, result.thresholds))
    print(f"[oof] QWK round+clip = {qwk_round:.4f}   "
          f"QWK optimized-thresholds = {qwk_opt:.4f}")
    print(f"[oof] thresholds = {np.round(result.thresholds, 4).tolist()}")
    print(f"[saved] outputs/oof_logreg.npy ({oof.shape[0]},)  "
          f"outputs/test_logreg.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    finalize(*run())
