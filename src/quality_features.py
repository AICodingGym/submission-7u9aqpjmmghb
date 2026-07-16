"""Essay-quality features for AES 2.0 — deterministic, corpus-free, CPU-only.

Complements src.features (28 structural) and src.ling_features (readability)
with the signal the Phase-0 audit says the champion is MISSING: coherence
(sentence/paragraph adjacency similarity) and finer discourse/vocabulary
structure. Everything here is a deterministic function of a SINGLE essay — no
vocabulary, IDF or statistic is learned from the corpus — so it is leakage-free
across folds and computed once, then cached.

(The cross-fitted "vocabulary maturity" log-odds features — which DO depend on
the training labels — live in the training script and are fit per fold, not
here, precisely so this module stays leakage-free.)

`extract_quality_features(texts)` -> DataFrame (n, len(QUALITY_COLUMNS)).
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_ALNUM_RE = re.compile(r"[a-z0-9]+")

SHORT_SENT_WORDS = 5
LONG_SENT_WORDS = 30

QUALITY_COLUMNS = [
    # A. structural / discourse
    "n_paragraphs", "para_len_mean", "para_len_std", "para_len_max",
    "first_para_len", "last_para_len", "single_sent_para_ratio", "blank_line_count",
    # B. sentence quality
    "sent_len_mean", "sent_len_median", "sent_len_std", "sent_len_max",
    "short_sent_ratio", "long_sent_ratio", "sent_capital_start_ratio",
    # C. vocabulary richness
    "hapax_ratio", "moving_avg_ttr", "repeated_adjacent_word_ratio",
    "repeated_bigram_ratio", "repeated_trigram_ratio", "long_word_ratio",
    # E. coherence (per-essay local TF-IDF, corpus-free)
    "adj_sent_cos_mean", "adj_sent_cos_std", "first_last_sent_cos",
    "adj_para_cos_mean", "low_coherence_sent_ratio",
]

# placeholder-for-append


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]


def _local_cos(tok_lists: list[list[str]]) -> np.ndarray:
    """Pairwise-adjacent cosine over a small set of token multisets.

    Builds a local vocabulary from ONLY these token lists (one essay's
    sentences/paragraphs) — no corpus IDF — and returns cosine similarity of
    each consecutive pair. Corpus-free => leakage-free.
    """
    vocab: dict[str, int] = {}
    for toks in tok_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    if not vocab or len(tok_lists) < 2:
        return np.array([])
    mats = np.zeros((len(tok_lists), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tok_lists):
        for t in toks:
            mats[i, vocab[t]] += 1.0
    norms = np.linalg.norm(mats, axis=1)
    cos = []
    for i in range(len(tok_lists) - 1):
        a, b = norms[i], norms[i + 1]
        cos.append(float(mats[i] @ mats[i + 1] / (a * b)) if a > 0 and b > 0 else 0.0)
    return np.asarray(cos, dtype=np.float64)


def _features_for_text(text: str) -> list[float]:
    t = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    words = _WORD_RE.findall(t)
    lower = [w.lower() for w in words]
    n_words = len(words)

    sents = _sentences(t)
    paras = _paragraphs(t)
    sent_wc = [len(_WORD_RE.findall(s)) for s in sents]
    para_wc = [len(_WORD_RE.findall(p)) for p in paras]
    para_sc = [len(_sentences(p)) for p in paras]

    # A. structural
    n_paras = len(paras)
    first_para = float(para_wc[0]) if para_wc else 0.0
    last_para = float(para_wc[-1]) if para_wc else 0.0
    single_sent_para = (sum(1 for c in para_sc if c <= 1) / n_paras) if n_paras else 0.0
    blank_lines = float(len(_PARA_SPLIT_RE.findall(t)))

    # B. sentence
    n_sents = len(sents)
    short_sent = (sum(1 for c in sent_wc if c <= SHORT_SENT_WORDS) / n_sents) if n_sents else 0.0
    long_sent = (sum(1 for c in sent_wc if c >= LONG_SENT_WORDS) / n_sents) if n_sents else 0.0
    cap_start = (sum(1 for s in sents if s[:1].isupper()) / n_sents) if n_sents else 0.0

    # C. vocab
    from collections import Counter
    cnt = Counter(lower)
    hapax = (sum(1 for w, c in cnt.items() if c == 1) / n_words) if n_words else 0.0
    # moving-average TTR over windows of 50 tokens (length-robust diversity)
    win = 50
    if n_words >= win:
        ttrs = [len(set(lower[i:i + win])) / win for i in range(0, n_words - win + 1, win)]
        ma_ttr = float(np.mean(ttrs)) if ttrs else 0.0
    else:
        ma_ttr = (len(set(lower)) / n_words) if n_words else 0.0
    adj_rep = (sum(1 for a, b in zip(lower, lower[1:]) if a == b) / (n_words - 1)) if n_words > 1 else 0.0
    bigrams = list(zip(lower, lower[1:]))
    trigrams = list(zip(lower, lower[1:], lower[2:]))
    rep_bi = (1 - len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    rep_tri = (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
    long_word = (sum(1 for w in words if len(w) >= 8) / n_words) if n_words else 0.0

    # E. coherence (corpus-free local cosine)
    sent_toks = [_ALNUM_RE.findall(s.lower()) for s in sents]
    para_toks = [_ALNUM_RE.findall(p.lower()) for p in paras]
    adj_sent = _local_cos(sent_toks)
    adj_para = _local_cos(para_toks)
    adj_sent_mean = float(adj_sent.mean()) if adj_sent.size else 0.0
    adj_sent_std = float(adj_sent.std()) if adj_sent.size else 0.0
    low_coh = float((adj_sent < 0.05).mean()) if adj_sent.size else 0.0
    if len(sent_toks) >= 2:
        fl = _local_cos([sent_toks[0], sent_toks[-1]])
        first_last = float(fl[0]) if fl.size else 0.0
    else:
        first_last = 0.0
    adj_para_mean = float(adj_para.mean()) if adj_para.size else 0.0

    return [
        float(n_paras), float(np.mean(para_wc)) if para_wc else 0.0,
        float(np.std(para_wc)) if para_wc else 0.0, float(max(para_wc)) if para_wc else 0.0,
        first_para, last_para, single_sent_para, blank_lines,
        float(np.mean(sent_wc)) if sent_wc else 0.0,
        float(np.median(sent_wc)) if sent_wc else 0.0,
        float(np.std(sent_wc)) if sent_wc else 0.0,
        float(max(sent_wc)) if sent_wc else 0.0,
        short_sent, long_sent, cap_start,
        hapax, ma_ttr, adj_rep, rep_bi, rep_tri, long_word,
        adj_sent_mean, adj_sent_std, first_last, adj_para_mean, low_coh,
    ]


def extract_quality_features(texts: Iterable[str]) -> pd.DataFrame:
    """Deterministic, corpus-free essay-quality features. Numeric, 0-filled."""
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    index = series.index
    if len(series) == 0:
        return pd.DataFrame(columns=QUALITY_COLUMNS, dtype=np.float64)
    rows = [_features_for_text(t) for t in series]
    df = pd.DataFrame(rows, columns=QUALITY_COLUMNS, index=index)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
