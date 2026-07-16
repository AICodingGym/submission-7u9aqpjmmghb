"""Single source of truth for cross-validation folds in the AES 2.0 pipeline.

Every training script MUST get its folds here so all OOF predictions are
row-aligned and mutually comparable for ensembling. Two schemes:

  * STANDARD  — StratifiedKFold(shuffle, seed) stratified by score. This is the
    split every existing script already uses (seed=42), so new models built on
    it stay ensemble-compatible with the 30 existing OOF files.
  * SHIFT-AWARE — StratifiedGroupKFold with groups = unsupervised topic/style
    clusters, so near-duplicate / same-style essays cannot straddle the
    train/val boundary. Audit found adversarial AUC≈0.50 (no train/test shift),
    so shift-CV ≈ standard-CV here; kept for diagnostics and leakage safety.

Also exposes a near-duplicate detector (char-TF-IDF -> SVD -> nearest neighbour
cosine) so we can verify no exact/near-duplicate leaks across folds.

No leakage: clusters and duplicate flags are UNSUPERVISED (text only, never the
score), so using them to define folds does not leak the target.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

SEED = 42                 # canonical seed used by every existing OOF file
SEEDS = [42, 7]           # multi-seed bagging set (matches existing *_s7 convention)
N_FOLDS = 5

# placeholder-for-append


def get_folds(y, seed: int = SEED, n_splits: int = N_FOLDS):
    """Canonical standard split: StratifiedKFold(shuffle, seed) by score.

    Returns a list of (train_idx, val_idx). Identical to what train_cpu_baseline
    and every other script constructs, so OOF rows line up across models.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def fold_id_array(y, seed: int = SEED, n_splits: int = N_FOLDS) -> np.ndarray:
    """Per-row fold id (0..n_splits-1) for the standard split — for saving/inspection."""
    ids = np.full(len(y), -1, dtype=int)
    for f, (_, va) in enumerate(get_folds(y, seed, n_splits)):
        ids[va] = f
    assert (ids >= 0).all()
    return ids


def text_clusters(texts, k: int = 8, svd_dim: int = 50, seed: int = SEED) -> np.ndarray:
    """Unsupervised topic/style clusters from word TF-IDF -> SVD -> KMeans.

    Text-only (no score), so safe to use as CV groups.
    """
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=40_000,
                         min_df=2, sublinear_tf=True, strip_accents="unicode")
    Z = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(wv.fit_transform(texts))
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)


def get_shift_folds(y, texts, k: int = 8, n_splits: int = N_FOLDS, seed: int = SEED):
    """Shift-aware split: StratifiedGroupKFold with topic-cluster groups.

    Keeps same-style/near-duplicate essays on one side of each split while still
    stratifying by score. Returns (folds, group_ids).
    """
    y = np.asarray(y)
    groups = text_clusters(texts, k=k, seed=seed)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups)), groups


def near_duplicate_report(texts, svd_dim: int = 100, sim_threshold: float = 0.95,
                          seed: int = SEED):
    """Nearest-neighbour cosine on char-TF-IDF -> SVD; flag near-duplicate pairs.

    Returns (nn_sim, pairs): nn_sim[i] = cosine similarity of row i to its most
    similar OTHER row; pairs = list of (i, j, sim) with sim >= sim_threshold.
    Char n-grams catch formatting/near-verbatim copies a word model would miss.
    """
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60_000,
                         min_df=2, sublinear_tf=True, strip_accents="unicode")
    X = cv.fit_transform(texts)
    Z = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(X)
    # normalize so Euclidean NN == cosine ranking
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Zn)
    dist, ind = nn.kneighbors(Zn)          # col 0 is self (dist 0)
    nn_sim = 1.0 - dist[:, 1]
    nbr = ind[:, 1]
    pairs = []
    seen = set()
    for i in np.where(nn_sim >= sim_threshold)[0]:
        j = int(nbr[i])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((int(key[0]), int(key[1]), float(nn_sim[i])))
    return nn_sim, pairs
