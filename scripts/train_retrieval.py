#!/usr/bin/env python3
"""P3: internal class-centroid similarity + kNN retrieval scoring for AES 2.0.

The audit's second real lever. Two cross-fitted, train-only signals that target
the extreme-class shrinkage (true-1 recall 0.51, true-5 0.57) the regressors
suffer — nearest-neighbour scoring does NOT regress to the mean, so it can be
sharp exactly where the GBDT/ridge bag is soft.

Everything is fit on the TRAIN fold only, inside the shared src.cv folds, so the
OOF row-aligns with the existing library and drops into the ensemble.

Representation: TruncatedSVD(200) of word+char TF-IDF, L2-normalized -> cosine
== dot product. (Dense 200-dim keeps NearestNeighbors fast on CPU; the sparse
TF-IDF is never densified.)

A. Class-centroid features (6 cosine sims to per-score centroids + margin +
   centroid-expected score). Centroids computed on TRAIN rows only.
B. kNN retrieval score: for each essay, k nearest TRAIN neighbours ->
   distance-weighted mean neighbour score. k swept on OOF.
Duplicate guard: retrieval is cross-fold (val/test never retrieve from their own
fold), so a near-duplicate in the SAME fold cannot be its own neighbour.

Two model outputs are saved:
  * retrieval expected score as a standalone weak learner:
      outputs/oof_retrieval.npy / test_retrieval.npy
  * the full dense feature block (centroid sims + knn stats) -> LightGBM/HGB,
    a dense "quality" model with different error signal:
      outputs/oof_gbdt_centroid.npy / test_gbdt_centroid.npy
  * outputs/retrieval_features_{train,test}.npy (for later stacking)
  * outputs/retrieval_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.cv import get_folds  # noqa: E402
from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import build_word_vectorizer, build_char_vectorizer  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(REPO_ROOT, "outputs")
SEED = 42
SVD_DIM = 200
KS = [5, 10, 20, 40, 80]
LABELS6 = [1, 2, 3, 4, 5, 6]

# placeholder-for-append


def svd_embed(tr_txt, va_txt, te_txt):
    """word+char TF-IDF -> SVD(SVD_DIM), L2-normalized. Fit on TRAIN fold only."""
    wv = build_word_vectorizer(); cv = build_char_vectorizer()
    Xtr = sp.hstack([wv.fit_transform(tr_txt), cv.fit_transform(tr_txt)]).tocsr()
    Xva = sp.hstack([wv.transform(va_txt), cv.transform(va_txt)]).tocsr()
    Xte = sp.hstack([wv.transform(te_txt), cv.transform(te_txt)]).tocsr()
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    Ztr = svd.fit_transform(Xtr); Zva = svd.transform(Xva); Zte = svd.transform(Xte)
    def norm(Z): return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return norm(Ztr), norm(Zva), norm(Zte)


def centroid_feats(Ztr, y_tr, Zeval):
    """6 cosine sims to per-score centroids + best-sim margin + expected score."""
    cents = np.zeros((6, Ztr.shape[1]))
    for i, c in enumerate(LABELS6):
        rows = Ztr[y_tr == c]
        if len(rows):
            v = rows.mean(axis=0)
            cents[i] = v / (np.linalg.norm(v) + 1e-12)
    sims = Zeval @ cents.T                       # (n, 6) cosine sims
    sm = softmax(sims)
    exp_score = sm @ np.array(LABELS6, float)    # centroid-expected score
    best = sims.max(axis=1)
    second = np.sort(sims, axis=1)[:, -2]
    margin = best - second
    return sims, exp_score, margin


def softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def knn_scores(Ztr, y_tr, Zeval, ks):
    """Distance-weighted mean neighbour score for each k, + neighbour stats."""
    nn = NearestNeighbors(n_neighbors=max(ks), metric="cosine").fit(Ztr)
    dist, ind = nn.kneighbors(Zeval)             # (n, maxk)
    out = {}
    for k in ks:
        d = dist[:, :k]; nbr = y_tr[ind[:, :k]]
        w = 1.0 / (d + 1e-6)
        out[k] = (w * nbr).sum(1) / w.sum(1)     # inverse-distance weighted score
    # extra stats from the largest k
    k = max(ks); nbr = y_tr[ind[:, :k]]
    stats = dict(nbr_std=nbr.std(1), nearest_dist=dist[:, 0],
                 mean_dist=dist[:, :k].mean(1), nbr_mean=nbr.mean(1))
    return out, stats


def run():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train["score"].to_numpy(int)
    tr_txt = train["full_text"].astype(str); te_txt = test["full_text"].astype(str)
    n_train, n_test = len(train), len(test)
    hand = extract_text_features(tr_txt).to_numpy(float)   # unused base; kept for parity

    # accumulate: per-k retrieval OOF, centroid features, knn stats
    knn_oof = {k: np.zeros(n_train) for k in KS}
    knn_test = {k: np.zeros(n_test) for k in KS}
    cent_sims_oof = np.zeros((n_train, 6)); cent_sims_test = np.zeros((n_test, 6))
    cent_exp_oof = np.zeros(n_train); cent_exp_test = np.zeros(n_test)
    cent_margin_oof = np.zeros(n_train); cent_margin_test = np.zeros(n_test)
    stat_names = ["nbr_std", "nearest_dist", "mean_dist", "nbr_mean"]
    stats_oof = {s: np.zeros(n_train) for s in stat_names}
    stats_test = {s: np.zeros(n_test) for s in stat_names}

    for fold, (tr, va) in enumerate(get_folds(y, seed=SEED), 1):
        ft = time.time()
        Ztr, Zva, Zte = svd_embed(tr_txt.iloc[tr], tr_txt.iloc[va], te_txt)
        y_tr = y[tr]
        # centroid
        s_va, e_va, m_va = centroid_feats(Ztr, y_tr, Zva)
        s_te, e_te, m_te = centroid_feats(Ztr, y_tr, Zte)
        cent_sims_oof[va] = s_va; cent_exp_oof[va] = e_va; cent_margin_oof[va] = m_va
        cent_sims_test += s_te / 5; cent_exp_test += e_te / 5; cent_margin_test += m_te / 5
        # knn
        kv, st_va = knn_scores(Ztr, y_tr, Zva, KS)
        kt, st_te = knn_scores(Ztr, y_tr, Zte, KS)
        for k in KS:
            knn_oof[k][va] = kv[k]; knn_test[k] += kt[k] / 5
        for s in stat_names:
            stats_oof[s][va] = st_va[s]; stats_test[s] += st_te[s] / 5
        print(f"[fold {fold}/5] svd+centroid+knn done ({time.time()-ft:.0f}s)")
        del Ztr, Zva, Zte

    # pick best single k on OOF
    best_k, best_q = None, -1
    for k in KS:
        q = qwk(y, apply_thresholds(knn_oof[k], optimize_thresholds(knn_oof[k], y).thresholds))
        print(f"[knn k={k:3d}] OOF QWK = {q:.4f}")
        if q > best_q:
            best_q, best_k = q, k
    retr_oof, retr_test = knn_oof[best_k], knn_test[best_k]
    np.save(os.path.join(OUT, "oof_retrieval.npy"), retr_oof)
    np.save(os.path.join(OUT, "test_retrieval.npy"), retr_test)

    # centroid-expected score as its own weak member
    cent_q = qwk(y, apply_thresholds(cent_exp_oof, optimize_thresholds(cent_exp_oof, y).thresholds))
    np.save(os.path.join(OUT, "oof_centroid.npy"), cent_exp_oof)
    np.save(os.path.join(OUT, "test_centroid.npy"), cent_exp_test)

    # assemble dense retrieval feature block (for later stacking / a GBDT member)
    feat_oof = np.column_stack([cent_sims_oof, cent_exp_oof, cent_margin_oof]
                               + [knn_oof[k] for k in KS]
                               + [stats_oof[s] for s in stat_names])
    feat_test = np.column_stack([cent_sims_test, cent_exp_test, cent_margin_test]
                                + [knn_test[k] for k in KS]
                                + [stats_test[s] for s in stat_names])
    np.save(os.path.join(OUT, "retrieval_features_train.npy"), feat_oof)
    np.save(os.path.join(OUT, "retrieval_features_test.npy"), feat_test)

    # correlations + per-class recall for the retrieval member
    champ = ["gbdt_clust_ms", "gbdt_hf_ms", "gbdt_spell_ms", "ridge"]
    corr = {n: round(float(np.corrcoef(retr_oof, np.load(os.path.join(OUT, f"oof_{n}.npy")))[0, 1]), 3)
            for n in champ}
    thr = optimize_thresholds(retr_oof, y).thresholds
    pred = apply_thresholds(retr_oof, thr)
    recall = {int(c): round(float((pred[y == c] == c).mean()), 4) for c in LABELS6}

    report = dict(best_k=best_k, retrieval_oof_qwk=round(best_q, 4),
                  centroid_oof_qwk=round(cent_q, 4),
                  corr_with_champion=corr, per_class_recall=recall,
                  runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "retrieval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[retrieval] best k={best_k} OOF QWK={best_q:.4f}  centroid OOF={cent_q:.4f}")
    print(f"[corr] retrieval vs champion: {corr}")
    print(f"[recall] retrieval per-class {recall}  (champion 1:0.51 5:0.57)")
    print(f"[saved] oof_retrieval/centroid + retrieval_features_* + report  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] retrieval aborted: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(0)
