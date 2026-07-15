#!/usr/bin/env python3
"""Multi-seed bagging for EVERY model strategy in the pipeline.

For each base model, reruns the full 5-fold CV under several fold+model seeds
and averages the OOF and test predictions across seeds (a proper bagged OOF:
each sample is predicted once per seed by its held-out fold). This reduces the
single-split variance for every strategy, not just ridge/gbdt.

Covered strategies (same feature/leakage rules as their original scripts —
vectorizers/SVD/scaler fit on the TRAIN fold only; thresholds are optimized on
OOF only, here just to report QWK):
  ridge, logreg, svr, sgd  (linear/margin on TF-IDF+hand)
  gbdt, gbdt_hf, gbdt_clust, gbdt_spell  (LightGBM on SVD(+extras))
  hf_ridge  (Ridge on frozen HF embeddings)

Writes outputs/oof_<name>_ms.npy / test_<name>_ms.npy for each (skips any whose
prerequisite artifacts, e.g. HF embeddings, are missing).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge, LogisticRegression, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.cluster import KMeans
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
from train_cpu_baseline import (  # noqa: E402
    build_fold_features, pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)
from train_gbdt import build_svd_features, detect_backend  # noqa: E402
from train_sgd import build_fold_features_alt  # noqa: E402
from train_logreg import expected_score  # noqa: E402
from train_gbdt_spell import spelling_features, load_dictionary  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
N_FOLDS = 5
SEEDS = [42, 7, 123]
# allow overriding seed count from the env for slow models (e.g. SEEDS=42,7)
if os.environ.get("MS_SEEDS"):
    SEEDS = [int(s) for s in os.environ["MS_SEEDS"].split(",")]

# placeholder-for-append


def oof_qwk(v, y):
    return qwk(y, apply_thresholds(v, optimize_thresholds(v, y).thresholds))


def _cv(train, y, test, seed, fit_predict):
    """Generic 5-fold CV driver; fit_predict(tr,va) -> (oof_va, test_pred)."""
    oof = np.zeros(len(train)); tp = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, va in skf.split(train, y):
        p_va, p_te = fit_predict(tr, va, seed)
        oof[va] = p_va; tp += p_te / N_FOLDS
    return oof, tp


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make_gbdt = detect_backend()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    tr_txt = train[text_col].astype(str); te_txt = test[text_col].astype(str)
    hand_tr = extract_text_features(tr_txt).to_numpy(dtype=np.float64)
    hand_te = extract_text_features(te_txt).to_numpy(dtype=np.float64)

    # optional prerequisites
    hf_tr = hf_te = None
    if os.path.exists(os.path.join(OUT_DIR, "train_hf_emb.npy")):
        hf_tr = np.load(os.path.join(OUT_DIR, "train_hf_emb.npy"))
        hf_te = np.load(os.path.join(OUT_DIR, "test_hf_emb.npy"))
    vocab = load_dictionary()
    sp_tr = spelling_features(tr_txt, vocab) if vocab else None
    sp_te = spelling_features(te_txt, vocab) if vocab else None

    # ---- per-model fold callables ---------------------------------------- #
    def fp_ridge(tr, va, seed):
        Xtr, Xva, Xte, _ = build_fold_features(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                               hand_tr[tr], hand_tr[va], hand_te)
        m = Ridge(alpha=1.0, random_state=seed).fit(Xtr, y[tr])
        return m.predict(Xva), m.predict(Xte)

    def fp_logreg(tr, va, seed):
        Xtr, Xva, Xte, _ = build_fold_features(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                               hand_tr[tr], hand_tr[va], hand_te)
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xtr, y[tr])
        c = m.classes_
        return expected_score(m.predict_proba(Xva), c), expected_score(m.predict_proba(Xte), c)

    def fp_svr(tr, va, seed):
        Xtr, Xva, Xte, _ = build_fold_features(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                               hand_tr[tr], hand_tr[va], hand_te)
        m = LinearSVR(C=1.0, max_iter=2000, random_state=seed).fit(Xtr, y[tr])
        return m.predict(Xva), m.predict(Xte)

    def fp_sgd(tr, va, seed):
        Xtr, Xva, Xte, _ = build_fold_features_alt(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                                   hand_tr[tr], hand_tr[va], hand_te)
        m = SGDRegressor(loss="huber", penalty="l2", alpha=1e-4, epsilon=0.1,
                         max_iter=50, tol=1e-4, random_state=seed).fit(Xtr, y[tr])
        return m.predict(Xva), m.predict(Xte)

    def _gbdt_fp(extra_tr, extra_te):
        def fp(tr, va, seed):
            Xtr, Xva, Xte, _, _ = build_svd_features(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                                     hand_tr[tr], hand_tr[va], hand_te)
            if extra_tr is not None:
                Xtr = np.hstack([Xtr, extra_tr[tr]]); Xva = np.hstack([Xva, extra_tr[va]])
                Xte = np.hstack([Xte, extra_te])
            m = make_gbdt(); m.set_params(random_state=seed); m.fit(Xtr, y[tr])
            return m.predict(Xva), m.predict(Xte)
        return fp

    def fp_gbdt_clust(tr, va, seed):
        Xtr, Xva, Xte, _, _ = build_svd_features(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt,
                                                 hand_tr[tr], hand_tr[va], hand_te)
        km = KMeans(n_clusters=4, random_state=seed, n_init=10)
        sc = StandardScaler().fit(Xtr)
        km.fit(sc.transform(Xtr))
        Xtr = np.hstack([Xtr, km.transform(sc.transform(Xtr))])
        Xva = np.hstack([Xva, km.transform(sc.transform(Xva))])
        Xte = np.hstack([Xte, km.transform(sc.transform(Xte))])
        m = make_gbdt(); m.set_params(random_state=seed); m.fit(Xtr, y[tr])
        return m.predict(Xva), m.predict(Xte)

    def fp_hf_ridge(tr, va, seed):
        sc = StandardScaler(); Xtr = sc.fit_transform(hf_tr[tr])
        m = Ridge(alpha=1.0, random_state=seed).fit(Xtr, y[tr])
        return m.predict(sc.transform(hf_tr[va])), m.predict(sc.transform(hf_te))

    jobs = [("ridge", fp_ridge), ("logreg", fp_logreg), ("svr", fp_svr), ("sgd", fp_sgd)]
    if backend is not None:
        jobs += [("gbdt", _gbdt_fp(None, None)),
                 ("gbdt_clust", fp_gbdt_clust)]
        if hf_tr is not None:
            jobs.append(("gbdt_hf", _gbdt_fp(hf_tr, hf_te)))
        if sp_tr is not None:
            jobs.append(("gbdt_spell", _gbdt_fp(sp_tr, sp_te)))
    if hf_tr is not None:
        jobs.append(("hf_ridge", fp_hf_ridge))

    # optional CLI filter: run only the named models (batching to survive long runs)
    only = set(sys.argv[1:])
    if only:
        jobs = [(n, fp) for n, fp in jobs if n in only]
        print(f"[filter] running only: {[n for n, _ in jobs]}")

    # per-seed persist mode: MS_ONE_SEED=<s> runs a SINGLE seed and writes
    # oof_<name>_s<s>.npy / test_<name>_s<s>.npy (for slow models that can't fit
    # multiple seeds in one job). Combine later with MS_COMBINE=1.
    one_seed = os.environ.get("MS_ONE_SEED")
    if one_seed is not None:
        s = int(one_seed)
        for name, fp in jobs:
            st = time.time()
            oof, tp = _cv(train, y, test, s, fp)
            np.save(os.path.join(OUT_DIR, f"oof_{name}_s{s}.npy"), oof)
            np.save(os.path.join(OUT_DIR, f"test_{name}_s{s}.npy"), tp)
            print(f"[{name}_s{s}] OOF={oof_qwk(oof, y):.4f} [{time.time()-st:.0f}s] "
                  f"-> oof_{name}_s{s}.npy")
        return

    if os.environ.get("MS_COMBINE"):
        import glob
        for name, _ in jobs:
            paths = sorted(glob.glob(os.path.join(OUT_DIR, f"oof_{name}_s*.npy")))
            if not paths:
                continue
            seeds = [os.path.basename(p).split("_s")[-1][:-4] for p in paths]
            oof = np.mean([np.load(p) for p in paths], axis=0)
            tp = np.mean([np.load(p.replace("oof_", "test_")) for p in paths], axis=0)
            np.save(os.path.join(OUT_DIR, f"oof_{name}_ms.npy"), oof)
            np.save(os.path.join(OUT_DIR, f"test_{name}_ms.npy"), tp)
            print(f"[{name}] combined seeds {seeds} -> avg_OOF={oof_qwk(oof, y):.4f}  "
                  f"oof_{name}_ms.npy")
        return

    for name, fp in jobs:
        st = time.time()
        oof_sum = np.zeros(len(train)); tp_sum = np.zeros(len(test)); singles = []
        for s in SEEDS:
            oof, tp = _cv(train, y, test, s, fp)
            singles.append(oof_qwk(oof, y))
            oof_sum += oof / len(SEEDS); tp_sum += tp / len(SEEDS)
        q_ms = oof_qwk(oof_sum, y)
        np.save(os.path.join(OUT_DIR, f"oof_{name}_ms.npy"), oof_sum)
        np.save(os.path.join(OUT_DIR, f"test_{name}_ms.npy"), tp_sum)
        print(f"[{name:11s}] seeds={[f'{q:.4f}' for q in singles]} "
              f"avg_OOF={q_ms:.4f} (single={singles[0]:.4f}, gain={q_ms-singles[0]:+.4f}) "
              f"[{time.time()-st:.0f}s]")

    print(f"[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"[skip] multiseed_all aborted: {type(e).__name__}: {e}")
        sys.exit(0)
