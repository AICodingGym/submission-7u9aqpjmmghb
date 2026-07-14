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
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "outputs")

SEED = 42
N_FOLDS = 5
LABELS = [1, 2, 3, 4, 5, 6]          # holistic score scale (fixed)
SCORE_MIN, SCORE_MAX = 1, 6
RIDGE_ALPHA = 1.0

TEXT_COL_PRIORITY = ["full_text", "essay"]
TARGET_COL_PRIORITY = ["score"]

# placeholder markers filled in by later chunks


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
# Hand-crafted text features (dense, cheap, CPU-friendly)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"\b\w+\b")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def hand_features(texts: pd.Series) -> np.ndarray:
    """Structural / stylometric features. No fitting, so leakage-free by design."""
    feats = []
    for t in texts.astype(str):
        n_chars = len(t)
        words = _WORD_RE.findall(t)
        n_words = len(words)
        uniq_words = len(set(w.lower() for w in words))
        sents = [s for s in _SENT_SPLIT_RE.split(t) if s.strip()]
        n_sents = len(sents)
        word_lens = [len(w) for w in words]
        n_para = t.count("\n\n") + 1
        n_commas = t.count(",")
        n_exclaim = t.count("!")
        n_question = t.count("?")
        n_upper = sum(1 for ch in t if ch.isupper())
        n_digits = sum(1 for ch in t if ch.isdigit())
        feats.append([
            n_chars,
            n_words,
            uniq_words,
            uniq_words / (n_words + 1),            # lexical diversity
            n_sents,
            n_words / (n_sents + 1),               # avg words per sentence
            float(np.mean(word_lens)) if word_lens else 0.0,
            float(np.std(word_lens)) if word_lens else 0.0,
            n_chars / (n_words + 1),               # avg chars per word
            n_para,
            n_commas / (n_sents + 1),
            n_exclaim,
            n_question,
            n_upper / (n_chars + 1),
            n_digits / (n_chars + 1),
        ])
    return np.asarray(feats, dtype=np.float64)


HAND_FEATURE_NAMES = [
    "n_chars", "n_words", "uniq_words", "lexical_diversity", "n_sents",
    "avg_words_per_sent", "mean_word_len", "std_word_len", "avg_chars_per_word",
    "n_para", "commas_per_sent", "n_exclaim", "n_question",
    "upper_ratio", "digit_ratio",
]


# --------------------------------------------------------------------------- #
# Metric + rounding
# --------------------------------------------------------------------------- #
def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa on the fixed 1..6 label set.

    labels=LABELS is passed explicitly so the confusion matrix is always 6x6
    even if some score never appears in y_pred — otherwise the (N-1)^2
    normalization would use the wrong N and distort the score.
    """
    return cohen_kappa_score(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        weights="quadratic",
        labels=LABELS,
    )


def apply_thresholds(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Map continuous predictions to integer classes 1..6 via 5 cut points."""
    # thresholds is sorted ascending, length = len(LABELS) - 1
    return np.searchsorted(thresholds, x) + SCORE_MIN


class OptimizedRounder:
    """Optimize 5 rounding thresholds to maximize QWK on OOF predictions only.

    Fit is called with the OOF vector and OOF truth; the learned cut points are
    then applied unchanged to the test predictions. Test labels are never seen.
    """

    def __init__(self):
        # initial guesses at the boundaries between adjacent scores
        self.thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)

    def _neg_qwk(self, thresholds, x, y):
        preds = apply_thresholds(x, np.sort(thresholds))
        return -qwk(y, preds)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OptimizedRounder":
        res = minimize(
            self._neg_qwk,
            self.thresholds,
            args=(x, y),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5},
        )
        self.thresholds = np.sort(res.x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return apply_thresholds(x, self.thresholds)


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

    # hand features computed once (no fitting -> no leakage); scaled per-fold
    print(f"[feat] computing {len(HAND_FEATURE_NAMES)} hand-crafted features: "
          f"{', '.join(HAND_FEATURE_NAMES)}")
    hand_train = hand_features(text_train)
    hand_test = hand_features(text_test)
    assert hand_train.shape[1] == len(HAND_FEATURE_NAMES), \
        f"hand feature count {hand_train.shape[1]} != names {len(HAND_FEATURE_NAMES)}"

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

    # thresholds optimized ON OOF ONLY, then frozen and applied to test
    rounder = OptimizedRounder().fit(oof, y)
    oof_opt = rounder.predict(oof)
    qwk_opt = qwk(y, oof_opt)
    print(f"[oof] QWK round+clip = {qwk_round:.4f}   "
          f"QWK optimized-thresholds = {qwk_opt:.4f}")
    print(f"[oof] thresholds = {np.round(rounder.thresholds, 4).tolist()}")

    test_labels = rounder.predict(test_pred)
    test_labels = np.clip(test_labels, SCORE_MIN, SCORE_MAX).astype(int)

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


