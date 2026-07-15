#!/usr/bin/env python3
"""Style-group clustering + train/test shift analysis for AES 2.0.

ANALYSIS ONLY — this script trains no scoring model and writes no predictions.
It answers two questions and dumps them to outputs/shift_report.md:

  1. Do the training essays fall into distinct "style groups"? We cluster on
     cheap structural features (length, paragraph count, punctuation, ...) plus
     an SVD projection of TF-IDF, using KMeans (default) or GaussianMixture into
     2-4 groups, and report each group's size, score distribution and
     word-count distribution.

  2. Is there a train/test distribution shift? A lightweight adversarial
     validation labels train=0 / test=1, fits TF-IDF + LogisticRegression, and
     reports cross-validated ROC-AUC. AUC ~0.5 => train and test look alike;
     AUC close to 1.0 => an easily-separable shift.

Reuses src.features (28 hand features) so clustering sees the same structural
signal the models use. CPU-only, no leakage concerns (nothing is fed back into
the scoring pipeline).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import extract_text_features  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
OUT_FILE = os.path.join(OUT_DIR, "shift_report.md")

SEED = 42
N_CLUSTERS = 3            # in [2, 4]
CLUSTER_METHOD = "kmeans"  # "kmeans" or "gmm"
SVD_DIM = 50              # TF-IDF -> dense for clustering
TEXT_COL_PRIORITY = ["full_text", "essay"]
TARGET_COL_PRIORITY = ["score"]

# placeholder-for-append


def _pick(df, priority, kind):
    for name in priority:
        if name in df.columns:
            return name
    raise KeyError(f"no {kind} column among {priority}")


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def cluster_analysis(train, text_col, target, out):
    """Cluster training essays into style groups and describe each group."""
    texts = train[text_col].astype(str)
    hand = extract_text_features(texts)  # 28 structural features

    # TF-IDF -> SVD (dense) so lexical style also informs the clustering
    tfidf = TfidfVectorizer(max_features=20_000, min_df=3,
                            ngram_range=(1, 2), sublinear_tf=True)
    Xtf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    Ztf = svd.fit_transform(Xtf)

    feats = np.hstack([hand.to_numpy(dtype=np.float64), Ztf])
    Xs = StandardScaler().fit_transform(feats)

    if CLUSTER_METHOD == "gmm":
        model = GaussianMixture(n_components=N_CLUSTERS, random_state=SEED)
        labels = model.fit_predict(Xs)
    else:
        model = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
        labels = model.fit_predict(Xs)

    train = train.copy()
    train["_cluster"] = labels
    train["_words"] = hand["word_count"].to_numpy()
    train["_paras"] = hand["paragraph_count"].to_numpy()

    out.append(f"## 1. Style-group clustering ({CLUSTER_METHOD}, k={N_CLUSTERS})")
    out.append("")
    out.append(f"- Features: 28 hand features + TruncatedSVD({SVD_DIM}) of TF-IDF, "
               f"z-scored.")
    out.append(f"- SVD explained variance (TF-IDF): "
               f"{float(svd.explained_variance_ratio_.sum()):.3f}")
    out.append("")

    # per-cluster summary
    rows = []
    overall_mean = train[target].mean()
    for c in range(N_CLUSTERS):
        sub = train[train["_cluster"] == c]
        rows.append([
            c, len(sub), f"{100*len(sub)/len(train):.1f}%",
            f"{sub[target].mean():.2f}",
            f"{sub['_words'].median():.0f}",
            f"{sub['_paras'].median():.0f}",
        ])
    out.append(_md_table(
        ["cluster", "n", "pct", "mean_score", "median_words", "median_paras"],
        rows))
    out.append("")
    out.append(f"- Overall mean score: {overall_mean:.2f}")
    out.append("")

    # score distribution per cluster (row = cluster, col = score)
    scores = sorted(train[target].unique().tolist())
    dist_rows = []
    for c in range(N_CLUSTERS):
        sub = train[train["_cluster"] == c]
        vc = sub[target].value_counts(normalize=True)
        dist_rows.append([c] + [f"{100*vc.get(s, 0):.1f}%" for s in scores])
    out.append("**Score distribution within each cluster (row-normalized):**")
    out.append("")
    out.append(_md_table(["cluster"] + [f"score={s}" for s in scores], dist_rows))
    out.append("")

    # word-count distribution per cluster
    wc_rows = []
    for c in range(N_CLUSTERS):
        w = train[train["_cluster"] == c]["_words"]
        wc_rows.append([c, f"{w.mean():.0f}", f"{w.std():.0f}",
                        f"{w.min():.0f}", f"{w.median():.0f}", f"{w.max():.0f}"])
    out.append("**Word-count distribution within each cluster:**")
    out.append("")
    out.append(_md_table(["cluster", "mean", "std", "min", "median", "max"], wc_rows))
    out.append("")
    return out


def adversarial_validation(train, test, text_col, out):
    """Label train=0 / test=1, classify, report CV ROC-AUC as a shift measure."""
    txt = pd.concat([train[text_col].astype(str), test[text_col].astype(str)],
                    ignore_index=True)
    is_test = np.concatenate([np.zeros(len(train)), np.ones(len(test))]).astype(int)

    vec = TfidfVectorizer(max_features=20_000, min_df=3,
                          ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(txt)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, is_test, cv=skf,
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(is_test, proba)

    if auc < 0.55:
        verdict = ("**No meaningful shift.** train and test are ~indistinguishable "
                   "(AUC ≈ 0.5); models should transfer well.")
    elif auc < 0.65:
        verdict = ("**Mild shift.** slightly separable; usually harmless but worth "
                   "noting for threshold stability.")
    elif auc < 0.8:
        verdict = ("**Moderate shift.** train and test differ noticeably; "
                   "consider adversarial-weighting or robust validation.")
    else:
        verdict = ("**Strong shift.** train/test are easily separable; local CV "
                   "may over/underestimate leaderboard performance.")

    out.append("## 2. Adversarial validation (train vs test)")
    out.append("")
    out.append("- Setup: label train=0 / test=1, TF-IDF(1,2) + LogisticRegression, "
               "5-fold CV ROC-AUC.")
    out.append(f"- **ROC-AUC = {auc:.4f}**")
    out.append(f"- n_train={len(train)}  n_test={len(test)}")
    out.append("")
    out.append(verdict)
    out.append("")

    # top tokens pushing toward "test" (diagnostic of what differs)
    clf.fit(X, is_test)
    coefs = clf.coef_[0]
    names = np.array(vec.get_feature_names_out())
    top_test = names[np.argsort(coefs)[-12:]][::-1]
    top_train = names[np.argsort(coefs)[:12]]
    out.append(f"- Tokens most predictive of **test**: `{', '.join(top_test)}`")
    out.append(f"- Tokens most predictive of **train**: `{', '.join(top_train)}`")
    out.append("")
    return out, auc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    text_col = _pick(train, TEXT_COL_PRIORITY, "text")
    target = _pick(train, TARGET_COL_PRIORITY, "target")

    out = ["# Train Style-Groups & Train/Test Shift Analysis", "",
           "_Analysis only — no scoring model trained, no predictions written._", ""]
    out = cluster_analysis(train, text_col, target, out)
    out, auc = adversarial_validation(train, test, text_col, out)

    out.append("## 3. Takeaways")
    out.append("")
    out.append("- Clusters above reflect essay *style/length* groups, not the "
               "prompt labels (which aren't provided).")
    out.append(f"- Adversarial AUC={auc:.3f} is the headline shift number; "
               "see the verdict in section 2.")
    out.append("- This report is diagnostic; the scoring pipeline is unchanged.")

    report = "\n".join(out) + "\n"
    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(report)
    print(f"[written] {OUT_FILE}")


if __name__ == "__main__":
    main()
