#!/usr/bin/env python3
"""Differentiated weak model: SGDRegressor (huber) on a DISTINCT TF-IDF space.

Purpose is ensemble *diversity*, not raw strength. The existing linear models
(Ridge, LinearSVR) all sit on the same word(1,2)+char(3,5) TF-IDF and end up
highly correlated (SVR ~0.99 with Ridge), so another model on identical features
would add nothing. To decorrelate, this model uses:

  * a DIFFERENT vectorization: word unigrams (1,1) + char_wb (2,4), smaller
    vocab — a coarser, more surface-level view than the main pipeline; plus
  * a DIFFERENT objective: SGDRegressor with the robust ``huber`` loss, which
    down-weights tail outliers instead of Ridge's squared loss.

Everything is fit on the TRAIN fold only (vectorizers + scaler); thresholds are
optimized on OOF only (src.thresholds). CPU-only, no transformer, no GPU.

Artifacts:
  outputs/oof_sgd.npy   (n_train,)
  outputs/test_sgd.npy  (n_test,)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    detect_id_column, load_data, pick_column,
    TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5

# placeholder-for-append


def build_word_vectorizer_alt() -> TfidfVectorizer:
    # unigram-only word view (distinct from the main (1,2) space)
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), max_features=50_000, min_df=3,
        sublinear_tf=True, strip_accents="unicode", token_pattern=r"(?u)\b\w+\b",
    )


def build_char_vectorizer_alt() -> TfidfVectorizer:
    # shorter char n-grams than the main (3,5) space
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=60_000, min_df=3,
        sublinear_tf=True, strip_accents="unicode",
    )


def build_fold_features_alt(text_tr, text_va, text_te, hand_tr, hand_va, hand_te):
    """Distinct TF-IDF space + hand features; fit on TRAIN fold only."""
    wv = build_word_vectorizer_alt()
    Xw_tr, Xw_va, Xw_te = wv.fit_transform(text_tr), wv.transform(text_va), wv.transform(text_te)
    cv = build_char_vectorizer_alt()
    Xc_tr, Xc_va, Xc_te = cv.fit_transform(text_tr), cv.transform(text_va), cv.transform(text_te)
    sc = StandardScaler()
    Hs_tr = sp.csr_matrix(sc.fit_transform(hand_tr))
    Hs_va = sp.csr_matrix(sc.transform(hand_va))
    Hs_te = sp.csr_matrix(sc.transform(hand_te))
    X_tr = sp.hstack([Xw_tr, Xc_tr, Hs_tr]).tocsr()
    X_va = sp.hstack([Xw_va, Xc_va, Hs_va]).tocsr()
    X_te = sp.hstack([Xw_te, Xc_te, Hs_te]).tocsr()
    return X_tr, X_va, X_te, X_tr.shape[1]


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test, sample = load_data()
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    _ = detect_id_column(train, test, target)
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    print(f"[data] train={len(train)} test={len(test)}")
    print(f"[feat] {len(FEATURE_COLUMNS)} hand + word(1,1) + char_wb(2,4) TF-IDF")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        X_tr, X_va, X_te, n_feats = build_fold_features_alt(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        model = SGDRegressor(
            loss="huber", penalty="l2", alpha=1e-4, epsilon=0.1,
            max_iter=50, tol=1e-4, random_state=SEED,
        )
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_te) / N_FOLDS
        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fq = qwk(y[va_idx], va_round)
        fold_qwks.append(fq)
        print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  QWK(round)={fq:.4f}  "
              f"({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")

    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    np.save(os.path.join(OUT_DIR, "oof_sgd.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_sgd.npy"), test_pred)

    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {qwk(y, apply_thresholds(oof, result.thresholds)):.4f}")
    print(f"[saved] outputs/oof_sgd.npy ({oof.shape[0]},)  "
          f"outputs/test_sgd.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
