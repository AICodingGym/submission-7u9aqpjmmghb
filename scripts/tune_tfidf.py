#!/usr/bin/env python3
"""Sweep TF-IDF capacity x linear model on full sparse features (fast, CPU).

Our production Ridge (word (1,2) 80k + char (3,5) 120k, alpha=1.0) gives OOF
0.7966. The friend's 0.836 CPU result suggests our TF-IDF/linear config is
under-tuned. Linear models on sparse TF-IDF are seconds per fold, so we can
sweep many configs on the SAME 5-fold split and rank by OOF QWK.

Varies: word/char max_features, ngram ranges, min_df, sublinear_tf, and the
model (Ridge alpha, or LinearSVR C). Vectorizers/scaler fit on train fold only;
thresholds optimized on OOF only. Nothing overwritten — this is a search.
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
from sklearn.svm import LinearSVR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
SEED = 42
N_FOLDS = 5

# each config: (word_ngram, word_max, char_ngram, char_max, min_df, sublinear)
TFIDF_CONFIGS = {
    "prod":      ((1, 2), 80_000,  (3, 5), 120_000, 2, True),
    "big":       ((1, 3), 150_000, (3, 6), 250_000, 2, True),
    "huge":      ((1, 3), 300_000, (2, 6), 400_000, 2, True),
    "mindf1":    ((1, 2), 150_000, (3, 5), 200_000, 1, True),
}

# placeholder-for-append


def make_feats(cfg, tr_txt, va_txt, te_txt, hand_tr, hand_va, hand_te):
    wn, wmax, cn, cmax, mindf, sub = cfg
    wv = TfidfVectorizer(analyzer="word", ngram_range=wn, max_features=wmax,
                         min_df=mindf, sublinear_tf=sub, strip_accents="unicode",
                         token_pattern=r"(?u)\b\w+\b")
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cn, max_features=cmax,
                         min_df=mindf, sublinear_tf=sub, strip_accents="unicode")
    Xw_tr = wv.fit_transform(tr_txt); Xw_va = wv.transform(va_txt); Xw_te = wv.transform(te_txt)
    Xc_tr = cv.fit_transform(tr_txt); Xc_va = cv.transform(va_txt); Xc_te = cv.transform(te_txt)
    sc = MaxAbsScaler()  # keeps sparsity; hand feats scaled separately below
    hsc = StandardScaler()
    H_tr = sp.csr_matrix(hsc.fit_transform(hand_tr))
    H_va = sp.csr_matrix(hsc.transform(hand_va))
    H_te = sp.csr_matrix(hsc.transform(hand_te))
    Xtr = sp.hstack([Xw_tr, Xc_tr, H_tr]).tocsr()
    Xva = sp.hstack([Xw_va, Xc_va, H_va]).tocsr()
    Xte = sp.hstack([Xw_te, Xc_te, H_te]).tocsr()
    return Xtr, Xva, Xte, Xtr.shape[1]


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))

    def score(oof):
        thr = optimize_thresholds(oof, y).thresholds
        return qwk(y, apply_thresholds(oof, thr))

    models = {"ridge_a3": ("ridge", 3.0), "ridge_a5": ("ridge", 5.0)}
    if os.environ.get("WITH_SVR"):
        models["svr_c1"] = ("svr", 1.0)

    results = []
    only = set(sys.argv[1:])
    configs = {k: v for k, v in TFIDF_CONFIGS.items() if not only or k in only}
    for cname, cfg in configs.items():
        tf0 = time.time()
        # build features once per config per fold, cache
        cache = []
        for tr, va in folds:
            Xtr, Xva, Xte, nf = make_feats(cfg, tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                           hand_tr[tr], hand_tr[va], hand_te)
            cache.append((tr, va, Xtr, Xva))
        print(f"[tfidf {cname}] n_feats={nf} built ({time.time()-tf0:.0f}s)")
        for mname, (kind, hp) in models.items():
            t0 = time.time()
            oof = np.zeros(len(train))
            for tr, va, Xtr, Xva in cache:
                mdl = Ridge(alpha=hp) if kind == "ridge" else LinearSVR(C=hp, max_iter=2000, random_state=SEED)
                mdl.fit(Xtr, y[tr]); oof[va] = mdl.predict(Xva)
            q = score(oof)
            results.append((cname, mname, nf, q, time.time()-t0))
            print(f"    {mname:9s} OOF QWK={q:.4f} ({time.time()-t0:.0f}s)")
        del cache

    results.sort(key=lambda r: -r[3])
    print("\n=== ranking (OOF QWK, optimized thresholds) ===")
    print(f"{'tfidf':8s} {'model':9s} {'nfeat':>8} {'QWK':>8} {'s':>6}")
    for cn, mn, nf, q, dt in results:
        print(f"{cn:8s} {mn:9s} {nf:>8} {q:>8.4f} {dt:>6.0f}")
    print(f"\n[baseline] prod ridge OOF=0.7966 | SVD-gbdt=0.8133")
    print(f"[best] {results[0][0]}/{results[0][1]} OOF={results[0][3]:.4f}")


if __name__ == "__main__":
    main()
