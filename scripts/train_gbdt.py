#!/usr/bin/env python3
"""CPU-only TruncatedSVD(TF-IDF) + GBDT baseline for AES 2.0 (skippable).

Requirement 1/2 — backend detection: prefers ``lightgbm``, falls back to
``xgboost``; if NEITHER is importable the script prints a clear ``[skip]`` and
exits 0 without writing anything, so a calling pipeline is unaffected.

Features (requirement 1/3/4): the SAME sources as the Ridge/LogReg/SVR
baselines — word TF-IDF (1,2) + char_wb TF-IDF (3,5) from
``train_cpu_baseline`` — but compressed with TruncatedSVD to ``SVD_DIM``
components (GBDTs need dense, low-dim input), then the 28 ``src.features`` hand
features are concatenated. Vectorizers, SVD and the hand-feature scaler are all
fit on the TRAIN fold only (no leakage). SVD also sidesteps the sparse->dense
memory blow-up: we never densify the 200k-dim TF-IDF, only its 128-dim
projection.

Artifacts (only on a fully successful run):
  outputs/oof_gbdt.npy   (len == n_train)
  outputs/test_gbdt.npy  (len == n_test)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.features import extract_text_features, FEATURE_COLUMNS  # noqa: E402
from src.thresholds import (  # noqa: E402
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa as qwk,
)
from train_cpu_baseline import (  # noqa: E402
    build_char_vectorizer,
    build_word_vectorizer,
    detect_id_column,
    load_data,
    pick_column,
    TARGET_COL_PRIORITY,
    TEXT_COL_PRIORITY,
)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")

SEED = 42
N_FOLDS = 5
SVD_DIM = 128            # 128 or 256 (req 3); 128 keeps SVD + GBDT fast on CPU

# placeholder-for-append


def detect_backend():
    """Return (name, make_regressor) for lightgbm or xgboost, or (None, None)."""
    try:
        import lightgbm as lgb  # noqa: F401

        def make():
            return lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
        return "lightgbm", make
    except ImportError:
        pass
    try:
        import xgboost as xgb  # noqa: F401

        def make():
            return xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
            )
        return "xgboost", make
    except ImportError:
        pass
    return None, None


def build_svd_features(text_tr, text_va, text_te, hand_tr, hand_va, hand_te):
    """word+char TF-IDF -> TruncatedSVD(SVD_DIM) -> concat scaled hand feats.

    Everything is fit on the TRAIN fold only. Returns three DENSE arrays; the
    full-dim sparse TF-IDF is never densified (only its SVD projection).
    """
    word_vec = build_word_vectorizer()
    Xw_tr, Xw_va, Xw_te = (word_vec.fit_transform(text_tr),
                           word_vec.transform(text_va),
                           word_vec.transform(text_te))
    char_vec = build_char_vectorizer()
    Xc_tr, Xc_va, Xc_te = (char_vec.fit_transform(text_tr),
                           char_vec.transform(text_va),
                           char_vec.transform(text_te))

    # stack sparse, then reduce with a single SVD over the combined space
    Xs_tr = sp.hstack([Xw_tr, Xc_tr]).tocsr()
    Xs_va = sp.hstack([Xw_va, Xc_va]).tocsr()
    Xs_te = sp.hstack([Xw_te, Xc_te]).tocsr()

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    Zt_tr = svd.fit_transform(Xs_tr)      # dense (n, SVD_DIM)
    Zt_va = svd.transform(Xs_va)
    Zt_te = svd.transform(Xs_te)

    scaler = StandardScaler()
    Ht_tr = scaler.fit_transform(hand_tr)
    Ht_va = scaler.transform(hand_va)
    Ht_te = scaler.transform(hand_te)

    X_tr = np.hstack([Zt_tr, Ht_tr])
    X_va = np.hstack([Zt_va, Ht_va])
    X_te = np.hstack([Zt_te, Ht_te])
    evr = float(svd.explained_variance_ratio_.sum())
    del word_vec, char_vec, Xw_tr, Xw_va, Xw_te, Xc_tr, Xc_va, Xc_te, Xs_tr, Xs_va, Xs_te
    return X_tr, X_va, X_te, X_tr.shape[1], evr


def run():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    backend, make_model = detect_backend()
    if backend is None:
        print("[skip] neither lightgbm nor xgboost is installed; nothing written.")
        return
    print(f"[backend] using {backend}")

    train, test, sample = load_data()
    text_col = pick_column(train, TEXT_COL_PRIORITY, "text")
    target = pick_column(train, TARGET_COL_PRIORITY, "target")
    _ = detect_id_column(train, test, target)
    print(f"[cols] text='{text_col}'  target='{target}'")
    print(f"[data] train={len(train)}  test={len(test)}  "
          f"(test length read from file, not hardcoded)")

    y = train[target].to_numpy(dtype=int)
    text_train = train[text_col].astype(str)
    text_test = test[text_col].astype(str)
    print(f"[feat] {len(FEATURE_COLUMNS)} hand features + SVD({SVD_DIM}) of TF-IDF")
    hand_train = extract_text_features(text_train).to_numpy(dtype=np.float64)
    hand_test = extract_text_features(text_test).to_numpy(dtype=np.float64)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_qwks = []
    try:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
            ft = time.time()
            X_tr, X_va, X_te, n_feats, evr = build_svd_features(
                text_train.iloc[tr_idx], text_train.iloc[va_idx], text_test,
                hand_train[tr_idx], hand_train[va_idx], hand_test,
            )
            model = make_model()
            model.fit(X_tr, y[tr_idx])
            oof[va_idx] = model.predict(X_va)
            test_pred += model.predict(X_te) / N_FOLDS

            va_round = np.clip(np.round(oof[va_idx]), SCORE_MIN, SCORE_MAX).astype(int)
            fq = qwk(y[va_idx], va_round)
            fold_qwks.append(fq)
            print(f"[fold {fold}/{N_FOLDS}] feats={n_feats}  svd_evr={evr:.3f}  "
                  f"QWK(round)={fq:.4f}  ({time.time()-ft:.1f}s)")
            del X_tr, X_va, X_te, model
    except Exception as e:  # any training error -> skip cleanly (req 2 spirit)
        print(f"[skip] {backend} run failed: {type(e).__name__}: {e}")
        print("[skip] no artifacts written; main flow unaffected.")
        return

    if np.isnan(oof).any() or np.isnan(test_pred).any():
        print("[skip] NaN in predictions; not writing artifacts.")
        return
    np.save(os.path.join(OUT_DIR, "oof_gbdt.npy"), oof)
    np.save(os.path.join(OUT_DIR, "test_gbdt.npy"), test_pred)

    print(f"[cv] mean fold QWK(round) = {np.mean(fold_qwks):.4f} "
          f"+/- {np.std(fold_qwks):.4f}")
    oof_round = np.clip(np.round(oof), SCORE_MIN, SCORE_MAX).astype(int)
    result = optimize_thresholds(oof, y)
    print(f"[oof] QWK round+clip = {qwk(y, oof_round):.4f}   "
          f"QWK optimized-thresholds = {qwk(y, apply_thresholds(oof, result.thresholds)):.4f}")
    print(f"[saved] outputs/oof_gbdt.npy ({oof.shape[0]},)  "
          f"outputs/test_gbdt.npy ({test_pred.shape[0]},)")
    print(f"[done] total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # pragma: no cover - defensive top-level guard
        print(f"[skip] GBDT run aborted: {type(e).__name__}: {e}")
        sys.exit(0)
