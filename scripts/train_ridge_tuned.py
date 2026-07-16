#!/usr/bin/env python3
"""Regenerate a TUNED Ridge OOF/test using the cached big TF-IDF (alpha=5).

The alpha sweep showed big TF-IDF + Ridge(alpha=5) -> OOF 0.8050, vs the
original Ridge(alpha=1) 0.7966 (+0.0084). This rebuilds the full ridge OOF and
TEST predictions at that tuned config and saves them as ridge_tuned so the
ensemble can use the stronger base model.

Train-fold OOF comes from the cached fold matrices (outputs/_tfidf_cache/big_*).
The TEST matrix is built once here (fit word/char vectorizers + scaler on ALL
train, transform test) — standard for producing test predictions from a CV'd
model via full-train-refit-per-fold averaging.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    apply_thresholds, optimize_thresholds, quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
CACHE = os.path.join(OUT_DIR, "_tfidf_cache")
SEED = 42
N_FOLDS = 5
ALPHA = 5.0
WCFG = ((1, 3), 150_000)   # word ngram, max
CCFG = ((3, 6), 250_000)   # char ngram, max
MINDF, SUB = 2, True

# placeholder-for-append


def build_test_matrix(txt_train, txt_test, hand_train, hand_test):
    """Fit vectorizers+scaler on ALL train, transform train and test (for the
    per-fold test prediction: each fold's model is refit on its train split via
    cached matrices, but test features must live in the SAME vocab space).

    We build a full-train vocab test matrix AND per-fold we cannot reuse cached
    (fold-specific vocab). So for TEST we refit per fold below using the same
    config. Returns nothing; kept for clarity — see main()."""
    raise NotImplementedError


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
    oof = np.zeros(len(train)); test_pred = np.zeros(len(test))
    wn, wmax = WCFG; cn, cmax = CCFG

    for k, (tr, va) in enumerate(folds):
        ft = time.time()
        # OOF part: reuse cached train/val matrices (built by tfidf_cached big)
        Xtr = sp.load_npz(os.path.join(CACHE, f"big_f{k}_tr.npz"))
        Xva = sp.load_npz(os.path.join(CACHE, f"big_f{k}_va.npz"))
        m = Ridge(alpha=ALPHA).fit(Xtr, y[tr])
        oof[va] = m.predict(Xva)

        # TEST part: refit vectorizers on this fold's TRAIN rows, transform test,
        # so the test features match this fold model's vocab; average over folds.
        wv = TfidfVectorizer(analyzer="word", ngram_range=wn, max_features=wmax,
                             min_df=MINDF, sublinear_tf=SUB, strip_accents="unicode",
                             token_pattern=r"(?u)\b\w+\b")
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cn, max_features=cmax,
                             min_df=MINDF, sublinear_tf=SUB, strip_accents="unicode")
        wv.fit(tr_txt.iloc[tr]); cv.fit(tr_txt.iloc[tr])
        sc = StandardScaler().fit(hand_tr[tr])
        Xte = sp.hstack([wv.transform(te_txt), cv.transform(te_txt),
                         sp.csr_matrix(sc.transform(hand_te))]).tocsr()
        test_pred += m.predict(Xte) / N_FOLDS
        print(f"[fold {k}] refit+test done ({time.time()-ft:.0f}s)")
        del Xtr, Xva, Xte, m, wv, cv

    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    np.save(os.path.join(OUT_DIR, "oof_ridge_tuned.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_ridge_tuned.npy"), test_pred)
    q = qwk(y, apply_thresholds(oof, optimize_thresholds(oof, y).thresholds))
    print(f"[oof] ridge_tuned OOF QWK = {q:.4f}  (orig ridge 0.7966)")
    print(f"[saved] oof_ridge_tuned.npy test_ridge_tuned.npy  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
