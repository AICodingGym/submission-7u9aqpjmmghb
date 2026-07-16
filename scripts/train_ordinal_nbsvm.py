#!/usr/bin/env python3
"""P2 (top priority): cumulative ORDINAL NB-SVM for AES 2.0.

WHY this model, per the Phase-0 audit: the library is severely homogeneous
(mean pairwise OOF corr 0.90; champion vs best-single 0.995) and extremes are
shrunk to the middle (true-1 recall 0.51, predicted mean 1.55; true-5 recall
0.57). A plain ordinal LogReg on raw TF-IDF was 0.967-corr with the bag — same
representation, so no diversity. The NB-SVM (Wang & Manning 2012) instead feeds
each linear model a Naive-Bayes LOG-COUNT-RATIO-transformed feature space, a
genuinely different representation; combined with a cumulative-ordinal
decomposition (a different loss than squared error) it should decorrelate AND,
with per-cut class balancing, lift the under-recalled extreme classes.

Cumulative ordinal (Frank & Hall): for k in 1..5 train a binary model on
(score > k). Recover a continuous expected score
    E[y] = 1 + sum_k P(y > k)
after enforcing monotonic P(y>1) >= ... >= P(y>5). QWK thresholds are then
optimized on OOF, so any monotone bias from class-balancing is absorbed.

NB transform (per cut, fit on the TRAIN fold only -> no leakage):
    presence f = (X > 0)
    p = alpha + sum_{pos} f ;  q = alpha + sum_{neg} f
    r = log( (p/||p||_1) / (q/||q||_1) )
    X_nb = X.multiply(r)              # element-wise column scaling
then LogisticRegression on X_nb.

Folds come from src.cv (shared standard split, seed 42 + 7 for bagging), so the
OOF is row-aligned with the existing 30 models and drops straight into the
ensemble. Reports per-cut AUC / log-loss.

Artifacts:
  outputs/oof_ordinal_nbsvm.npy    outputs/test_ordinal_nbsvm.npy
  outputs/ordinal_nbsvm_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.cv import SEEDS, get_folds  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import build_word_vectorizer, build_char_vectorizer  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(REPO_ROOT, "outputs")
CUTS = [1, 2, 3, 4, 5]
C = 1.0
NB_ALPHA = 1.0
CLASS_WEIGHT = os.environ.get("CLASS_WEIGHT", "balanced")  # 'balanced' or 'none'
if CLASS_WEIGHT == "none":
    CLASS_WEIGHT = None

# placeholder-for-append


def nb_ratio(X, yb, alpha=NB_ALPHA):
    """NB log-count-ratio r over features, from TRAIN presence only."""
    Xb = (X > 0)
    p = np.asarray(Xb[yb == 1].sum(axis=0)).ravel() + alpha
    q = np.asarray(Xb[yb == 0].sum(axis=0)).ravel() + alpha
    r = np.log((p / p.sum()) / (q / q.sum()))
    return r


def fit_cut(X_tr, yb, X_eval_list):
    """NB-transform + LogReg for one binary cut. Returns P(pos) for each eval X."""
    r = nb_ratio(X_tr, yb)
    R = sp.diags(r)
    Xt = X_tr @ R
    clf = LogisticRegression(C=C, max_iter=1000, solver="liblinear",
                             class_weight=CLASS_WEIGHT)
    clf.fit(Xt, yb)
    pos = list(clf.classes_).index(1)
    return [clf.predict_proba(Xe @ R)[:, pos] for Xe in X_eval_list]


def build_tfidf(tr_txt, va_txt, te_txt):
    """word(1,2)+char_wb(3,5) TF-IDF, fit on TRAIN fold only. Returns 3 CSRs."""
    wv = build_word_vectorizer()
    Xw = (wv.fit_transform(tr_txt), wv.transform(va_txt), wv.transform(te_txt))
    cv = build_char_vectorizer()
    Xc = (cv.fit_transform(tr_txt), cv.transform(va_txt), cv.transform(te_txt))
    return (sp.hstack([Xw[0], Xc[0]]).tocsr(),
            sp.hstack([Xw[1], Xc[1]]).tocsr(),
            sp.hstack([Xw[2], Xc[2]]).tocsr())


def run_one_seed(seed, tr_txt, te_txt, y):
    """Full 5-fold cumulative-ordinal NB-SVM for one seed. Returns oof, test, diag."""
    n_train, n_test = len(tr_txt), len(te_txt)
    oof = np.zeros(n_train)
    test_pred = np.zeros(n_test)
    # per-cut cross-fitted P(y>k) for AUC/logloss diagnostics
    oof_pgt = np.zeros((n_train, len(CUTS)))
    for fold, (tr, va) in enumerate(get_folds(y, seed=seed), 1):
        ft = time.time()
        X_tr, X_va, X_te = build_tfidf(
            tr_txt.iloc[tr], tr_txt.iloc[va], te_txt)
        pgt_va = np.zeros((len(va), len(CUTS)))
        pgt_te = np.zeros((n_test, len(CUTS)))
        for ci, k in enumerate(CUTS):
            yb = (y[tr] > k).astype(int)
            if yb.min() == yb.max():
                rate = float(yb.mean())
                pgt_va[:, ci] = rate; pgt_te[:, ci] = rate
                continue
            pv, pt = fit_cut(X_tr, yb, [X_va, X_te])
            pgt_va[:, ci] = pv; pgt_te[:, ci] = pt
        # monotone non-increasing across cuts (P(y>1) >= ... >= P(y>5))
        pgt_va = np.minimum.accumulate(np.clip(pgt_va, 0, 1), axis=1)
        pgt_te = np.minimum.accumulate(np.clip(pgt_te, 0, 1), axis=1)
        oof[va] = 1.0 + pgt_va.sum(axis=1)
        oof_pgt[va] = pgt_va
        test_pred += (1.0 + pgt_te.sum(axis=1)) / 5.0
        fq = qwk(y[va], np.clip(np.round(oof[va]), SCORE_MIN, SCORE_MAX).astype(int))
        print(f"  [seed{seed} fold{fold}] QWK(round)={fq:.4f} ({time.time()-ft:.0f}s)")
        del X_tr, X_va, X_te
    # per-cut diagnostics on OOF
    diag = {}
    for ci, k in enumerate(CUTS):
        yb = (y > k).astype(int)
        p = np.clip(oof_pgt[:, ci], 1e-6, 1 - 1e-6)
        diag[f"cut>{k}"] = dict(pos_rate=round(float(yb.mean()), 4),
                                auc=round(float(roc_auc_score(yb, p)), 4),
                                logloss=round(float(log_loss(yb, p)), 4))
    return oof, test_pred, diag


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train["score"].to_numpy(int)
    tr_txt = train["full_text"].astype(str)
    te_txt = test["full_text"].astype(str)
    print(f"[nbsvm] class_weight={CLASS_WEIGHT} C={C} seeds={SEEDS}")

    oof_acc = np.zeros(len(train)); test_acc = np.zeros(len(test))
    diags = {}
    for s in SEEDS:
        oof_s, test_s, diag = run_one_seed(s, tr_txt, te_txt, y)
        oof_acc += oof_s / len(SEEDS)
        test_acc += test_s / len(SEEDS)
        diags[f"seed{s}"] = diag

    if np.isnan(oof_acc).any() or np.isnan(test_acc).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT, "oof_ordinal_nbsvm.npy"), oof_acc)
    np.save(os.path.join(OUT, "test_ordinal_nbsvm.npy"), test_acc)

    thr = optimize_thresholds(oof_acc, y).thresholds
    q = qwk(y, apply_thresholds(oof_acc, thr))
    fold_q = [qwk(y[va], apply_thresholds(oof_acc[va], thr))
              for _, va in get_folds(y, seed=SEEDS[0])]
    # correlation with the champion members
    champ = ["gbdt_clust_ms", "gbdt_hf_ms", "gbdt_spell_ms", "ridge"]
    corrs = {n: round(float(np.corrcoef(oof_acc, np.load(os.path.join(OUT, f"oof_{n}.npy")))[0, 1]), 3)
             for n in champ if os.path.exists(os.path.join(OUT, f"oof_{n}.npy"))}
    pred = apply_thresholds(oof_acc, thr)
    per_class = {int(c): round(float((pred[y == c] == c).mean()), 4) for c in range(1, 7)}

    print(f"\n[oof] ordinal_nbsvm OOF QWK = {q:.4f}  (plain ordinal=0.8052, "
          f"logreg=0.7942, best GBDT single=0.8170)")
    print(f"[oof] fold QWK {[round(v,4) for v in fold_q]} std={np.std(fold_q):.4f}")
    print(f"[corr] vs champion members: {corrs}  (want << 0.97)")
    print(f"[recall] per-class {per_class}  (champion 1:0.51 5:0.57)")
    print(f"[dist] test {pd.Series(apply_thresholds(test_acc, thr)).value_counts().sort_index().to_dict()}")

    report = dict(model="ordinal_nbsvm", class_weight=str(CLASS_WEIGHT), C=C,
                  seeds=SEEDS, oof_qwk=round(q, 4),
                  fold_qwk=[round(v, 4) for v in fold_q],
                  fold_std=round(float(np.std(fold_q)), 4),
                  corr_with_champion=corrs, per_class_recall=per_class,
                  per_cut=diags, runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "ordinal_nbsvm_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] oof_ordinal_nbsvm.npy test_ordinal_nbsvm.npy "
          f"ordinal_nbsvm_report.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"[skip] ordinal_nbsvm aborted: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(0)
