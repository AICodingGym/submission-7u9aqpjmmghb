#!/usr/bin/env python3
"""Experiment #2: spelling / non-dictionary-word features -> extend hand feats.

The essays contain heavy misspelling (electorol, preesident, ...). Error density
plausibly correlates with score yet is not captured explicitly by TF-IDF or the
current hand features. This module computes cheap, dictionary-based spelling
features per essay using the system word list (/usr/share/dict/words) — no
network, no heavy NLP package.

Features per essay (all ratios guard /0):
  oov_ratio            fraction of alphabetic tokens NOT in the dictionary
  oov_unique_ratio     same over the unique token set
  mean_oov_word_len    mean length of out-of-vocabulary tokens
  repeated_letter_oov  OOV tokens containing a 3+ letter run (e.g. 'sooo')

It then trains a GBDT on [SVD(TF-IDF) + 28 hand + these 4 spelling feats] and
reports OOF QWK vs the 0.8133 baseline. Fits nothing global (dictionary is a
fixed resource) -> no leakage. Writes suffixed artifacts, never overwrites gbdt.
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
from train_gbdt import build_svd_features, detect_backend  # noqa: E402
from train_cpu_baseline import (  # noqa: E402
    pick_column, TARGET_COL_PRIORITY, TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
DICT_PATH = "/usr/share/dict/words"
SEED = 42
N_FOLDS = 5
_WORD_RE = re.compile(r"[a-zA-Z]+")
_RUN_RE = re.compile(r"([a-z])\1\1")

# placeholder-for-append


def load_dictionary():
    """Load a lowercase English word set, or None if unavailable (-> skip feats)."""
    if not os.path.exists(DICT_PATH):
        return None
    with open(DICT_PATH, encoding="utf-8", errors="ignore") as f:
        return {w.strip().lower() for w in f if w.strip().isalpha()}


def spelling_features(texts, vocab) -> np.ndarray:
    """4 dictionary-based spelling features per essay (leakage-free: fixed vocab)."""
    out = []
    for t in texts.astype(str):
        toks = [w.lower() for w in _WORD_RE.findall(t)]
        n = len(toks)
        if n == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        oov = [w for w in toks if w not in vocab]
        uniq = set(toks)
        oov_uniq = [w for w in uniq if w not in vocab]
        rep_oov = sum(1 for w in oov if _RUN_RE.search(w))
        out.append([
            len(oov) / n,
            len(oov_uniq) / (len(uniq) + 1),
            float(np.mean([len(w) for w in oov])) if oov else 0.0,
            rep_oov / n,
        ])
    return np.asarray(out, dtype=np.float64)


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] no lightgbm/xgboost; nothing written.")
        return
    vocab = load_dictionary()
    if vocab is None:
        print(f"[skip] no dictionary at {DICT_PATH}; cannot build spelling feats.")
        return
    print(f"[backend] {backend}  |  dictionary words: {len(vocab)}")

    train = pd.read_csv(os.path.join(REPO_ROOT, "data", "train.csv"))
    test = pd.read_csv(os.path.join(REPO_ROOT, "data", "test.csv"))
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    # spelling features computed once (fixed dictionary -> no leakage)
    print("[feat] computing spelling features ...")
    sp_train = spelling_features(text_train, vocab)
    sp_test = spelling_features(text_test, vocab)
    # quick signal check: correlation of oov_ratio with score
    corr = np.corrcoef(sp_train[:, 0], y)[0, 1]
    print(f"[signal] corr(oov_ratio, score) = {corr:.4f}")

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        X_tr, X_va, X_te, _, evr = build_svd_features(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        X_tr = np.hstack([X_tr, sp_train[tr_idx]])
        X_va = np.hstack([X_va, sp_train[va_idx]])
        X_te = np.hstack([X_te, sp_test])

        model = make_model()
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_te) / N_FOLDS
        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fq = qwk(y[va_idx], va_round)
        fold_qwks.append(fq)
        print(f"[fold {fold}/{N_FOLDS}] feats={X_tr.shape[1]} (+4 spell)  "
              f"QWK(round)={fq:.4f}  ({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    assert not np.isnan(oof).any() and not np.isnan(test_pred).any()
    np.save(os.path.join(OUT_DIR, "oof_gbdt_spell.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt_spell.npy"), test_pred)

    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    q_opt = qwk(y, apply_thresholds(oof, result.thresholds))
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {q_opt:.4f}")
    print(f"[baseline] gbdt=0.8133  gain={q_opt-0.8133:+.4f} (adopt only if > +0.0030)")
    print(f"[saved] outputs/oof_gbdt_spell.npy  outputs/test_gbdt_spell.npy")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover
        print(f"[skip] gbdt_spell aborted: {type(e).__name__}: {e}")
        sys.exit(0)
