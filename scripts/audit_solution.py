#!/usr/bin/env python3
"""Phase-0 audit of the EXISTING AES 2.0 solution. Trains NO new scoring model.

Reads every outputs/oof_*.npy + test_*.npy pair, reconstructs the shared seed-42
5-fold split every training script uses, evaluates the deployed champion bag,
and emits the five diagnostic artifacts the audit asks for:
  outputs/model_correlation.csv
  outputs/per_class_metrics.csv
  outputs/residual_analysis.csv        (+ residual_by_cluster / _by_lenquantile)
  outputs/fold_stability.csv
plus a JSON summary printed to stdout that the Markdown audit is written from.

Leakage note: only READS existing predictions and computes UNSUPERVISED
diagnostics (KMeans clusters, cross-fitted adversarial train-vs-test prob). It
fits nothing on test labels and writes no submission.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import extract_text_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    LABELS, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(REPO_ROOT, "outputs")
SEED, N_FOLDS = 42, 5
CHAMPION = ["gbdt_clust_ms", "gbdt_hf_ms", "gbdt_spell_ms", "ridge"]
REDUNDANT_CORR = 0.995

# placeholder-for-append


def q_opt(v, y):
    return qwk(y, apply_thresholds(v, optimize_thresholds(v, y).thresholds))


def load_all(n_train, n_test):
    """Discover valid oof/test pairs. Returns names, OOF matrix, TEST matrix."""
    names, oof_l, test_l = [], [], []
    for op in sorted(glob.glob(os.path.join(OUT, "oof_*.npy"))):
        name = os.path.basename(op)[4:-4]
        tp = os.path.join(OUT, f"test_{name}.npy")
        if not os.path.exists(tp):
            continue
        oof, test = np.load(op), np.load(tp)
        if (oof.ndim == 1 and test.ndim == 1 and oof.shape[0] == n_train
                and test.shape[0] == n_test
                and not np.isnan(oof).any() and not np.isnan(test).any()):
            names.append(name); oof_l.append(oof); test_l.append(test)
    return names, np.vstack(oof_l), np.vstack(test_l)


def build_clusters(tr_txt, k=8):
    """Unsupervised topic/style clusters: SVD(word+char TF-IDF)+struct, KMeans."""
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=40_000,
                         min_df=2, sublinear_tf=True, strip_accents="unicode")
    Xw = wv.fit_transform(tr_txt)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    Z = svd.fit_transform(Xw)
    km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10)
    return km.fit_predict(Z)


def adversarial_prob(tr_txt, te_txt, n_train):
    """Cross-fitted train-vs-test likeness prob for each TRAIN row (5-fold)."""
    all_txt = pd.concat([tr_txt, te_txt], ignore_index=True)
    lab = np.r_[np.zeros(n_train), np.ones(len(te_txt))].astype(int)
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=40_000,
                         min_df=2, sublinear_tf=True, strip_accents="unicode")
    X = wv.fit_transform(all_txt)
    prob = np.zeros(len(lab))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, va in skf.split(X, lab):
        m = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear").fit(X[tr], lab[tr])
        prob[va] = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(lab[va], prob[va]))
    return prob[:n_train], float(np.mean(aucs))


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train["score"].to_numpy(int)
    n_train, n_test = len(train), len(test)
    tr_txt = train["full_text"].astype(str)
    te_txt = test["full_text"].astype(str)

    names, OOF, TEST = load_all(n_train, n_test)
    idx = {n: i for i, n in enumerate(names)}
    print(f"[audit] {len(names)} models; n_train={n_train} n_test={n_test}")

    # ---- per-model single OOF QWK + fold-wise QWK + test distribution -------
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(n_train), y))
    rows_fold, single = [], {}
    for n in names:
        v = OOF[idx[n]]
        thr = optimize_thresholds(v, y).thresholds
        oq = qwk(y, apply_thresholds(v, thr))
        fq = [qwk(y[va], apply_thresholds(v[va], thr)) for _, va in folds]
        single[n] = oq
        td = pd.Series(apply_thresholds(TEST[idx[n]], thr)).value_counts().sort_index().to_dict()
        rows_fold.append(dict(model=n, oof_qwk=round(oq, 4),
                              fold_mean=round(float(np.mean(fq)), 4),
                              fold_std=round(float(np.std(fq)), 4),
                              fold_min=round(float(np.min(fq)), 4),
                              fold_max=round(float(np.max(fq)), 4),
                              test_dist=str(td)))
    pd.DataFrame(rows_fold).sort_values("oof_qwk", ascending=False).to_csv(
        os.path.join(OUT, "fold_stability.csv"), index=False)

    # ---- correlation matrix --------------------------------------------------
    C = np.corrcoef(OOF)
    pd.DataFrame(C, index=names, columns=names).round(4).to_csv(
        os.path.join(OUT, "model_correlation.csv"))

    # ---- champion bag prediction --------------------------------------------
    w = np.zeros(len(names))
    for n in CHAMPION:
        w[idx[n]] = 1.0 / len(CHAMPION)
    champ_oof = w @ OOF
    champ_test = w @ TEST
    champ_thr = optimize_thresholds(champ_oof, y).thresholds
    champ_q = qwk(y, apply_thresholds(champ_oof, champ_thr))
    champ_fold = [qwk(y[va], apply_thresholds(champ_oof[va], champ_thr)) for _, va in folds]
    champ_pred = apply_thresholds(champ_oof, champ_thr)

    # champion-vs-best-single correlation + homogeneity
    best_single = max(single, key=single.get)
    corr_champ_best = float(np.corrcoef(champ_oof, OOF[idx[best_single]])[0, 1])
    offdiag = C[np.triu_indices(len(names), 1)]

    # ---- per-class metrics (recall + QWK-ish confusion on champion OOF) ------
    per_class = []
    for c in LABELS:
        mask = y == c
        n_c = int(mask.sum())
        rec = float((champ_pred[mask] == c).mean()) if n_c else 0.0
        mae = float(np.abs(champ_pred[mask] - c).mean()) if n_c else 0.0
        # shrink-to-mean check: mean predicted score for this true class
        mean_pred = float(champ_pred[mask].mean()) if n_c else 0.0
        per_class.append(dict(true_score=c, n=n_c, recall=round(rec, 4),
                              mae=round(mae, 4), mean_pred=round(mean_pred, 3)))
    pd.DataFrame(per_class).to_csv(os.path.join(OUT, "per_class_metrics.csv"), index=False)

    # ---- residual analysis: champion continuous residual vs covariates ------
    hand = extract_text_features(tr_txt)
    resid = y - champ_oof   # signed residual on the continuous champion pred
    clusters = build_clusters(tr_txt, k=8)
    tlike, adv_auc = adversarial_prob(tr_txt, te_txt, n_train)

    def corr(a):
        a = np.asarray(a, float)
        return float(np.corrcoef(a, resid)[0, 1]) if a.std() > 0 else 0.0

    res_rows = [
        dict(variable="word_count", corr_with_residual=round(corr(hand["word_count"]), 4)),
        dict(variable="paragraph_count", corr_with_residual=round(corr(hand["paragraph_count"]), 4)),
        dict(variable="sentence_count", corr_with_residual=round(corr(hand["sentence_count"]), 4)),
        dict(variable="avg_sentence_len", corr_with_residual=round(corr(hand["avg_sentence_len"]), 4)),
        dict(variable="unique_word_ratio", corr_with_residual=round(corr(hand["unique_word_ratio"]), 4)),
        dict(variable="target_like_prob", corr_with_residual=round(corr(tlike), 4)),
        dict(variable="abs_residual_mean", corr_with_residual=round(float(np.abs(resid).mean()), 4)),
    ]
    pd.DataFrame(res_rows).to_csv(os.path.join(OUT, "residual_analysis.csv"), index=False)

    # residual by length-quantile and by cluster (mean signed + mean abs)
    lq = pd.qcut(hand["word_count"], 5, labels=False, duplicates="drop")
    by_len = pd.DataFrame({"len_quintile": lq, "resid": resid, "abs_resid": np.abs(resid)}) \
        .groupby("len_quintile").agg(n=("resid", "size"),
                                     mean_resid=("resid", "mean"),
                                     mean_abs_resid=("abs_resid", "mean")).round(4)
    by_len.to_csv(os.path.join(OUT, "residual_by_lenquantile.csv"))
    by_cl = pd.DataFrame({"cluster": clusters, "score": y, "resid": resid,
                          "abs_resid": np.abs(resid), "tlike": tlike}) \
        .groupby("cluster").agg(n=("resid", "size"), mean_score=("score", "mean"),
                                mean_resid=("resid", "mean"),
                                mean_abs_resid=("abs_resid", "mean"),
                                mean_tlike=("tlike", "mean")).round(4)
    by_cl.to_csv(os.path.join(OUT, "residual_by_cluster.csv"))

    summary = dict(
        n_models=len(names), champion=CHAMPION,
        champion_oof_qwk=round(champ_q, 4),
        champion_fold=[round(q, 4) for q in champ_fold],
        champion_fold_std=round(float(np.std(champ_fold)), 4),
        best_single=best_single, best_single_qwk=round(single[best_single], 4),
        champ_gain_over_best_single=round(champ_q - single[best_single], 4),
        corr_champion_vs_best_single=round(corr_champ_best, 4),
        mean_offdiag_corr=round(float(offdiag.mean()), 4),
        min_offdiag_corr=round(float(offdiag.min()), 4),
        max_offdiag_corr=round(float(offdiag.max()), 4),
        n_pairs_above_redundant=int((offdiag > REDUNDANT_CORR).sum()),
        adversarial_auc=round(adv_auc, 4),
        champ_test_dist=str(pd.Series(apply_thresholds(champ_test, champ_thr))
                            .value_counts().sort_index().to_dict()),
        train_score_dist=str(pd.Series(y).value_counts().sort_index().to_dict()),
        champ_thresholds=[round(float(t), 4) for t in champ_thr],
        runtime_s=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[audit] wrote model_correlation.csv per_class_metrics.csv "
          f"residual_analysis.csv fold_stability.csv (+cluster/len) audit_summary.json")


if __name__ == "__main__":
    main()
