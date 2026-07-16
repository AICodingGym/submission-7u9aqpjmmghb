#!/usr/bin/env python3
"""P4: dense "essay-quality" GBDT for AES 2.0 — a decorrelated member.

Per the Phase-0 audit, every strong model is a GBDT/ridge over SVD(TF-IDF); they
are 0.99-correlated. This model deliberately uses NO TF-IDF. Its feature block is
purely dense and interpretable, so its error structure should differ:

  28 structural (src.features)
  + readability (src.ling_features, textstat)
  + 26 quality/coherence (src.quality_features)
  + CROSS-FITTED vocabulary-maturity log-odds features (fit per TRAIN fold):
      for each token, mean train score of essays containing it -> aggregate over
      an essay (presence-weighted mean, and high/low-score-word ratios). This is
      supervised, so it is computed INSIDE each fold on train rows only (no leak).

Model: LightGBM regressor (installed) on the dense block. Folds from src.cv
(shared standard split, seeds 42+7 bagged) so OOF row-aligns with the library.

Artifacts:
  outputs/oof_gbdt_quality.npy  outputs/test_gbdt_quality.npy
  outputs/gbdt_quality_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.cv import SEEDS, get_folds  # noqa: E402
from src.features import extract_text_features  # noqa: E402
from src.ling_features import extract_linguistic_features  # noqa: E402
from src.quality_features import extract_quality_features  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX, SCORE_MIN, apply_thresholds, optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(REPO_ROOT, "outputs")
CACHE = os.path.join(OUT, "_quality_cache")

# placeholder-for-append


def vocab_maturity(tr_txt, y_tr, *eval_txts):
    """Cross-fitted supervised-lexicon aggregates. Fit on TRAIN fold only.

    Token score = mean train-fold essay score among essays containing the token.
    Per essay we then compute presence-weighted mean token score + high/low
    "mature word" ratios (tokens whose mean score is >=4.5 / <=2.5). All learned
    from train rows only -> leakage-free.
    """
    cv = CountVectorizer(binary=True, min_df=5, ngram_range=(1, 1),
                         token_pattern=r"(?u)\b\w+\b", strip_accents="unicode")
    Xtr = cv.fit_transform(tr_txt)                    # (n_tr, V) binary
    df = np.asarray(Xtr.sum(axis=0)).ravel() + 1e-9   # doc-freq per token
    tok_score = (Xtr.T @ y_tr) / df                   # mean score per token
    high = (tok_score >= 4.5).astype(np.float64)
    low = (tok_score <= 2.5).astype(np.float64)

    def feats(txt):
        X = cv.transform(txt)                          # (n, V) binary
        pres = np.asarray(X.sum(axis=1)).ravel() + 1e-9
        mean_tok = np.asarray(X @ tok_score).ravel() / pres
        hi_ratio = np.asarray(X @ high).ravel() / pres
        lo_ratio = np.asarray(X @ low).ravel() / pres
        # max/min mature word score present (extreme-word signal)
        Xc = X.tocsc()
        return np.column_stack([mean_tok, hi_ratio, lo_ratio])

    return [feats(t) for t in eval_txts]


def dense_block(txt):
    """Corpus-free dense features (cached across seeds — deterministic)."""
    h = extract_text_features(txt).to_numpy(np.float64)
    l = extract_linguistic_features(txt).to_numpy(np.float64)
    q = extract_quality_features(txt).to_numpy(np.float64)
    return np.hstack([h, l, q])


def cached_dense(name, txt):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"dense_{name}.npy")
    if os.path.exists(p):
        return np.load(p)
    X = dense_block(txt)
    np.save(p, X)
    return X


def make_model():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             min_child_samples=40, random_state=42, n_jobs=-1,
                             verbose=-1)


def run_one_seed(seed, base_tr, base_te, tr_txt, te_txt, y):
    oof = np.zeros(len(y)); test_pred = np.zeros(base_te.shape[0])
    for fold, (tr, va) in enumerate(get_folds(y, seed=seed), 1):
        ft = time.time()
        vm_tr, vm_va, vm_te = vocab_maturity(
            tr_txt.iloc[tr], y[tr].astype(float),
            tr_txt.iloc[tr], tr_txt.iloc[va], te_txt)
        X_tr = np.hstack([base_tr[tr], vm_tr])
        X_va = np.hstack([base_tr[va], vm_va])
        X_te = np.hstack([base_te, vm_te])
        m = make_model(); m.fit(X_tr, y[tr])
        oof[va] = m.predict(X_va); test_pred += m.predict(X_te) / 5
        fq = qwk(y[va], np.clip(np.round(oof[va]), SCORE_MIN, SCORE_MAX).astype(int))
        print(f"  [seed{seed} fold{fold}] feats={X_tr.shape[1]} QWK(round)={fq:.4f} ({time.time()-ft:.0f}s)")
        del X_tr, X_va, X_te, m
    return oof, test_pred


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train["score"].to_numpy(int)
    tr_txt = train["full_text"].astype(str); te_txt = test["full_text"].astype(str)

    print("[feat] building dense block (hand+ling+quality) ...")
    base_tr = cached_dense("train", tr_txt)
    base_te = cached_dense("test", te_txt)
    print(f"[feat] dense base dims = {base_tr.shape[1]} (+3 cross-fitted vocab per fold)")

    oof = np.zeros(len(y)); test_pred = np.zeros(len(test))
    for s in SEEDS:
        o, t = run_one_seed(s, base_tr, base_te, tr_txt, te_txt, y)
        oof += o / len(SEEDS); test_pred += t / len(SEEDS)

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN; not writing."); return
    np.save(os.path.join(OUT, "oof_gbdt_quality.npy"), oof)
    np.save(os.path.join(OUT, "test_gbdt_quality.npy"), test_pred)

    thr = optimize_thresholds(oof, y).thresholds
    q = qwk(y, apply_thresholds(oof, thr))
    fold_q = [qwk(y[va], apply_thresholds(oof[va], thr)) for _, va in get_folds(y, seed=SEEDS[0])]
    champ = ["gbdt_clust_ms", "gbdt_hf_ms", "gbdt_spell_ms", "ridge", "ordinal_nbsvm"]
    corr = {n: round(float(np.corrcoef(oof, np.load(os.path.join(OUT, f"oof_{n}.npy")))[0, 1]), 3)
            for n in champ if os.path.exists(os.path.join(OUT, f"oof_{n}.npy"))}
    pred = apply_thresholds(oof, thr)
    recall = {int(c): round(float((pred[y == c] == c).mean()), 4) for c in range(1, 7)}
    print(f"\n[oof] gbdt_quality OOF QWK = {q:.4f}  (best GBDT single 0.8170)")
    print(f"[oof] fold QWK {[round(v,4) for v in fold_q]} std={np.std(fold_q):.4f}")
    print(f"[corr] vs champion members: {corr}")
    print(f"[recall] per-class {recall}  (champion 1:0.51 5:0.57)")
    print(f"[dist] test {pd.Series(apply_thresholds(test_pred, thr)).value_counts().sort_index().to_dict()}")

    report = dict(model="gbdt_quality", seeds=SEEDS, oof_qwk=round(q, 4),
                  fold_qwk=[round(v, 4) for v in fold_q],
                  fold_std=round(float(np.std(fold_q)), 4),
                  corr_with_champion=corr, per_class_recall=recall,
                  runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "gbdt_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] oof_gbdt_quality.npy test_gbdt_quality.npy report [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print(f"[skip] gbdt_quality aborted: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(0)
