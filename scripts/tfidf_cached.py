#!/usr/bin/env python3
"""Cache-then-train TF-IDF sweep: survives short job ceilings.

Two phases (idempotent, resumable):
  phase 1 (build): for a config, build per-fold sparse features once and save to
    outputs/_tfidf_cache/<config>_f<k>.npz. Skips folds already cached.
  phase 2 (train): load cached folds, sweep Ridge alphas, print OOF QWK.

Run build phase in chunks (a couple folds per invocation) so no single job runs
long enough to be killed; then run train phase (fast).

Usage:
  MODE=build CFG=big FOLDS=0,1  python scripts/tfidf_cached.py
  MODE=build CFG=big FOLDS=2,3,4 python scripts/tfidf_cached.py
  MODE=train CFG=big            python scripts/tfidf_cached.py
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

CONFIGS = {
    "prod": ((1, 2), 80_000,  (3, 5), 120_000, 2, True),
    "big":  ((1, 3), 150_000, (3, 6), 250_000, 2, True),
    "huge": ((1, 3), 300_000, (2, 6), 400_000, 2, True),
}

# placeholder-for-append


def _load_common():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    txt = train[text_col].astype(str)
    hand = extract_text_features(txt).to_numpy(dtype=np.float64)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(train, y))
    return train, y, txt, hand, folds


def build(cfg_name, fold_ids):
    os.makedirs(CACHE, exist_ok=True)
    wn, wmax, cn, cmax, mindf, sub = CONFIGS[cfg_name]
    train, y, txt, hand, folds = _load_common()
    for k in fold_ids:
        out = os.path.join(CACHE, f"{cfg_name}_f{k}.npz")
        if os.path.exists(out):
            print(f"[build {cfg_name} f{k}] cached, skip"); continue
        t0 = time.time()
        tr, va = folds[k]
        wv = TfidfVectorizer(analyzer="word", ngram_range=wn, max_features=wmax,
                             min_df=mindf, sublinear_tf=sub, strip_accents="unicode",
                             token_pattern=r"(?u)\b\w+\b")
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=cn, max_features=cmax,
                             min_df=mindf, sublinear_tf=sub, strip_accents="unicode")
        Xw_tr = wv.fit_transform(txt.iloc[tr]); Xw_va = wv.transform(txt.iloc[va])
        Xc_tr = cv.fit_transform(txt.iloc[tr]); Xc_va = cv.transform(txt.iloc[va])
        sc = StandardScaler()
        H_tr = sp.csr_matrix(sc.fit_transform(hand[tr]))
        H_va = sp.csr_matrix(sc.transform(hand[va]))
        Xtr = sp.hstack([Xw_tr, Xc_tr, H_tr]).tocsr()
        Xva = sp.hstack([Xw_va, Xc_va, H_va]).tocsr()
        sp.save_npz(os.path.join(CACHE, f"{cfg_name}_f{k}_tr.npz"), Xtr)
        sp.save_npz(os.path.join(CACHE, f"{cfg_name}_f{k}_va.npz"), Xva)
        np.savez(out, tr=tr, va=va)  # marker + indices
        print(f"[build {cfg_name} f{k}] nfeat={Xtr.shape[1]} ({time.time()-t0:.0f}s)")


def train(cfg_name, alphas=(1.0, 3.0, 5.0, 8.0)):
    train_df, y, _, _, folds = _load_common()
    # verify all folds cached
    for k in range(N_FOLDS):
        if not os.path.exists(os.path.join(CACHE, f"{cfg_name}_f{k}_tr.npz")):
            print(f"[train] missing cache for fold {k}; run build first."); return
    for a in alphas:
        oof = np.zeros(len(train_df))
        for k in range(N_FOLDS):
            tr, va = folds[k]
            Xtr = sp.load_npz(os.path.join(CACHE, f"{cfg_name}_f{k}_tr.npz"))
            Xva = sp.load_npz(os.path.join(CACHE, f"{cfg_name}_f{k}_va.npz"))
            m = Ridge(alpha=a).fit(Xtr, y[tr]); oof[va] = m.predict(Xva)
        thr = optimize_thresholds(oof, y).thresholds
        q = qwk(y, apply_thresholds(oof, thr))
        print(f"[{cfg_name} ridge a={a}] OOF QWK = {q:.4f}")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "train")
    cfg = os.environ.get("CFG", "big")
    if mode == "build":
        fids = [int(x) for x in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
        build(cfg, fids)
    else:
        train(cfg)
