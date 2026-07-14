#!/usr/bin/env python3
"""CPU-only TF-IDF + Ridge baseline for Automated Essay Scoring 2.0.

No transformers, no GPU, no HuggingFace. Pipeline:
  word TF-IDF (1,2) + char_wb TF-IDF (3,5) + hand-crafted text features
  -> Ridge regression -> OOF-optimized rounding thresholds -> QWK.

Leakage control:
  * All vectorizers / scalers / thresholds are fit on TRAIN folds only.
  * Rounding thresholds are optimized on the full OOF vector (never on test).
  * Test length is taken from the file (never hardcoded) — the rerun test set
    is ~8k rows, so anything hardcoded to the local 1731 would break.

Artifacts written to outputs/:
  oof_ridge.npy, test_ridge.npy, submission_ridge.csv
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

# make `src` importable regardless of the current working directory
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import extract_text_features, FEATURE_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    LABELS,
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")

SEED = 42
N_FOLDS = 5
RIDGE_ALPHA = 1.0

TEXT_COL_PRIORITY = ["full_text", "essay"]
TARGET_COL_PRIORITY = ["score"]


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def pick_column(df: pd.DataFrame, priority: list[str], kind: str) -> str:
    for name in priority:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find a {kind} column; tried {priority}, "
                   f"available: {list(df.columns)}")


def detect_id_column(train: pd.DataFrame, test: pd.DataFrame, target: str) -> str:
    # prefer a shared, unique, *id-named* column
    shared = [c for c in train.columns if c in test.columns and c != target]
    for c in shared:
        if "id" in c.lower() and train[c].is_unique:
            return c
    for c in shared:
        if train[c].is_unique:
            return c
    raise KeyError(f"Could not find an id column among shared columns {shared}")


# --------------------------------------------------------------------------- #
# Feature builders (fit on TRAIN only, transform train+test)
# --------------------------------------------------------------------------- #
def build_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=80_000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=120_000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    return train, test, sample


def build_fold_features(text_tr, text_va, text_te, hand_tr, hand_va, hand_te):
    """Fit vectorizers + scaler on the TRAIN fold only, transform all three.

    Returns (X_tr, X_va, X_te) as horizontally stacked CSR matrices.
    """
    word_vec = build_word_vectorizer()
    Xw_tr = word_vec.fit_transform(text_tr)
    Xw_va = word_vec.transform(text_va)
    Xw_te = word_vec.transform(text_te)

    char_vec = build_char_vectorizer()
    Xc_tr = char_vec.fit_transform(text_tr)
    Xc_va = char_vec.transform(text_va)
    Xc_te = char_vec.transform(text_te)

    scaler = StandardScaler()
    Hs_tr = sp.csr_matrix(scaler.fit_transform(hand_tr))
    Hs_va = sp.csr_matrix(scaler.transform(hand_va))
    Hs_te = sp.csr_matrix(scaler.transform(hand_te))

    X_tr = sp.hstack([Xw_tr, Xc_tr, Hs_tr]).tocsr()
    X_va = sp.hstack([Xw_va, Xc_va, Hs_va]).tocsr()
    X_te = sp.hstack([Xw_te, Xc_te, Hs_te]).tocsr()
    n_feats = X_tr.shape[1]
    del word_vec, char_vec, Xw_tr, Xw_va, Xw_te, Xc_tr, Xc_va, Xc_te
    return X_tr, X_va, X_te, n_feats


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    train, test, sample = load_data()

    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    id_col = detect_id_column(train, test, target)
    print(f"[cols] text='{text_col}'  target='{target}'  id='{id_col}'")
    print(f"[data] train={len(train)}  test={len(test)}  (test length read from file, not hardcoded)")

    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)

    # hand features computed once (no fitting -> no leakage); scaled per-fold.
    # Uses the shared src.features implementation (28 dense features).
    print(f"[feat] computing {len(FEATURE_COLUMNS)} hand-crafted features "
          f"via src.features.extract_text_features")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)
    assert hand_train.shape[1] == len(FEATURE_COLUMNS), \
        f"hand feature count {hand_train.shape[1]} != names {len(FEATURE_COLUMNS)}"

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        ft = time.time()
        X_tr, X_va, X_te, n_feats = build_fold_features(
            text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
            hand_train[tr_idx], hand_train[va_idx], hand_test,
        )
        model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict(X_va)
        test_pred += model.predict(X_te) / N_FOLDS

        # per-fold QWK via simple round+clip (honest, non-optimized readout)
        va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
        fold_qwk = qwk(y[va_idx], va_round)
        fold_qwks.append(fold_qwk)
        print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  "
              f"QWK(round)={fold_qwk:.4f}  ({time.time()-ft:.1f}s)")
        del X_tr, X_va, X_te, model

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    return train, test, sample, id_col, target, y, oof, test_pred, t0


# --------------------------------------------------------------------------- #
# Post-processing: OOF-only thresholds -> integer submission
# --------------------------------------------------------------------------- #
def finalize(train, test, sample, id_col, target, y, oof, test_pred, t0):
    # save raw continuous predictions before any rounding
    np.save(os.path.join(OUT_DIR, "oof_ridge.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_ridge.npy"), test_pred)

    # baseline readout: plain round+clip on OOF
    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    qwk_round = qwk(y, oof_round)

    # thresholds optimized ON OOF ONLY (src.thresholds has no `test` param),
    # then frozen and applied unchanged to the test predictions.
    result = optimize_thresholds(oof, y)
    oof_opt = apply_thresholds(oof, result.thresholds)
    qwk_opt = qwk(y, oof_opt)
    print(f"[oof] QWK round+clip = {qwk_round:.4f}   "
          f"QWK optimized-thresholds = {qwk_opt:.4f}")
    print(f"[oof] thresholds = {np.round(result.thresholds, 4).tolist()}  "
          f"(strictly increasing: {bool((np.diff(result.thresholds) > 0).all())})")

    # apply_thresholds already clips into [1, 6] and returns int
    test_labels = apply_thresholds(test_pred, result.thresholds)

    # --- build submission with EXACT sample_submission columns/order --------- #
    sub = sample.copy()
    id_to_label = dict(zip(test[id_col].to_numpy(), test_labels))
    mapped = sub[id_col].map(id_to_label)
    # check for unmapped ids BEFORE casting: NaN.astype(int) would raise a cryptic
    # IntCastingNaNError first and swallow this friendlier message otherwise.
    assert not mapped.isna().any(), \
        f"{int(mapped.isna().sum())} test id(s) had no prediction mapped"
    sub[target] = mapped.astype(int)

    # hard invariants before writing
    assert list(sub.columns) == list(sample.columns), \
        f"column mismatch: {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"
    assert sub[id_col].tolist() == sample[id_col].tolist(), "id order changed"
    assert sub[target].dtype.kind == "i", f"score not integer: {sub[target].dtype}"
    vals = set(sub[target].unique().tolist())
    assert vals.issubset(set(LABELS)), f"scores outside 1..6: {vals}"

    out_path = os.path.join(OUT_DIR, "submission_ridge.csv")
    sub.to_csv(out_path, index=False)

    print(f"[sub] wrote {out_path}")
    print(f"[sub] columns={list(sub.columns)}  rows={len(sub)}")
    print(f"[sub] score dtype={sub[target].dtype}  values={sorted(vals)}")
    dist = sub[target].value_counts().sort_index()
    print(f"[sub] pred distribution: {dist.to_dict()}")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    finalize(*run())


