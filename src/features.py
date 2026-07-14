"""Lightweight hand-crafted text features for essay scoring.

`extract_text_features(texts)` returns a pandas DataFrame of 28 dense,
interpretable features. Design constraints:

  * No heavy NLP deps — only the standard library (`re`, `string`), numpy,
    and pandas. No spaCy / NLTK / transformers.
  * Every feature is numeric; missing / undefined values are filled with 0,
    so the frame is safe to feed straight into `StandardScaler`.
  * Output is a plain dense DataFrame; convert with
    `scipy.sparse.csr_matrix(df.values)` to `hstack` alongside TF-IDF.

The function fits nothing (no vocabulary, no statistics learned from the
corpus), so it introduces no train/test leakage when reused across folds.
"""
from __future__ import annotations

import re
import string
from typing import Iterable

import numpy as np
import pandas as pd

# thresholds (documented, not learned)
LONG_WORD_LEN = 7          # a "long" word has >= this many chars
SHORT_WORD_LEN = 3         # a "short" word has <= this many chars
LONG_SENTENCE_WORDS = 20   # a "long" sentence has > this many words
REPEAT_RUN_MIN = 3         # a char repeated >= this many times in a row is "repeated"

# Note on the two diversity ratios (both requested in the spec):
#   unique_word_ratio = unique / total          (plain type-token ratio)
#   type_token_ratio  = unique / sqrt(total)     (Guiraud's root TTR)
# The root variant is length-robust, so the two columns carry different signal.

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_REPEAT_RE = re.compile(r"(.)\1{%d,}" % (REPEAT_RUN_MIN - 1))
_QUOTE_CHARS = "\"'“”‘’"  # straight + curly quotes
_PUNCT_SET = set(string.punctuation) | set(_QUOTE_CHARS)

# canonical column order (28 features)
FEATURE_COLUMNS = [
    # length
    "char_count", "word_count", "sentence_count", "paragraph_count",
    "avg_word_len", "max_word_len",
    # vocabulary diversity
    "unique_word_count", "unique_word_ratio", "type_token_ratio",
    "long_word_ratio", "short_word_ratio",
    # punctuation & formatting
    "comma_count", "period_count", "question_count", "exclamation_count",
    "quote_count", "semicolon_count", "colon_count", "punctuation_ratio",
    "uppercase_ratio", "digit_count", "newline_count",
    # sentence complexity
    "avg_sentence_len", "sentence_len_std", "long_sentence_ratio",
    # anomalous text
    "repeated_char_count", "all_caps_word_count", "non_alpha_ratio",
]

# placeholder-for-append


def _features_for_text(text: str) -> dict[str, float]:
    """Compute all 28 features for a single essay. Ratios guard against /0."""
    # Treat None and NaN as empty text. pandas coerces None -> float NaN inside
    # an object Series, so a bare `text is None` check would let NaN through and
    # str(nan) == "nan" would be counted as a real one-word essay.
    if text is None or (isinstance(text, float) and np.isnan(text)):
        t = ""
    else:
        t = str(text)
    n_chars = len(t)

    words = _WORD_RE.findall(t)
    n_words = len(words)
    word_lens = [len(w) for w in words]
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    n_unique = len(unique_words)

    sents = [s for s in _SENT_SPLIT_RE.split(t) if s.strip()]
    n_sents = len(sents)
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sents]

    n_paras = len([p for p in _PARA_SPLIT_RE.split(t) if p.strip()])

    n_alpha = sum(ch.isalpha() for ch in t)
    n_upper = sum(ch.isupper() for ch in t)
    n_digits = sum(ch.isdigit() for ch in t)
    n_punct = sum(ch in _PUNCT_SET for ch in t)

    long_words = sum(1 for L in word_lens if L >= LONG_WORD_LEN)
    short_words = sum(1 for L in word_lens if L <= SHORT_WORD_LEN)
    all_caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    long_sents = sum(1 for c in sent_word_counts if c > LONG_SENTENCE_WORDS)
    repeated = len(_REPEAT_RE.findall(t))

    feats = {
        # length
        "char_count": float(n_chars),
        "word_count": float(n_words),
        "sentence_count": float(n_sents),
        "paragraph_count": float(n_paras),
        "avg_word_len": float(np.mean(word_lens)) if word_lens else 0.0,
        "max_word_len": float(max(word_lens)) if word_lens else 0.0,
        # vocabulary diversity
        "unique_word_count": float(n_unique),
        "unique_word_ratio": n_unique / n_words if n_words else 0.0,
        # Guiraud's root type-token ratio (unique / sqrt(total)): unlike the
        # plain ratio above it is far less sensitive to essay length, so it
        # carries distinct signal rather than duplicating unique_word_ratio.
        "type_token_ratio": n_unique / np.sqrt(n_words) if n_words else 0.0,
        "long_word_ratio": long_words / n_words if n_words else 0.0,
        "short_word_ratio": short_words / n_words if n_words else 0.0,
        # punctuation & formatting
        "comma_count": float(t.count(",")),
        "period_count": float(t.count(".")),
        "question_count": float(t.count("?")),
        "exclamation_count": float(t.count("!")),
        "quote_count": float(sum(t.count(q) for q in _QUOTE_CHARS)),
        "semicolon_count": float(t.count(";")),
        "colon_count": float(t.count(":")),
        "punctuation_ratio": n_punct / n_chars if n_chars else 0.0,
        "uppercase_ratio": n_upper / n_chars if n_chars else 0.0,
        "digit_count": float(n_digits),
        "newline_count": float(t.count("\n")),
        # sentence complexity
        "avg_sentence_len": float(np.mean(sent_word_counts)) if sent_word_counts else 0.0,
        "sentence_len_std": float(np.std(sent_word_counts)) if sent_word_counts else 0.0,
        "long_sentence_ratio": long_sents / n_sents if n_sents else 0.0,
        # anomalous text
        "repeated_char_count": float(repeated),
        "all_caps_word_count": float(all_caps),
        "non_alpha_ratio": (n_chars - n_alpha) / n_chars if n_chars else 0.0,
    }
    return feats


def extract_text_features(texts: Iterable[str]) -> pd.DataFrame:
    """Extract 28 lightweight text features for a collection of essays.

    Parameters
    ----------
    texts : iterable of str
        One string per essay (a list, pandas Series, numpy array, ...).
        None / NaN entries are treated as empty strings.

    Returns
    -------
    pandas.DataFrame
        Shape (len(texts), 28) with columns in ``FEATURE_COLUMNS`` order.
        Purely numeric, missing values filled with 0 -> ready for
        StandardScaler and, once cast to a sparse matrix, scipy hstack.
    """
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    index = series.index

    if len(series) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=np.float64)

    rows = [_features_for_text(t) for t in series]
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=index)

    # numeric coercion + fill: turn any inf/NaN that slipped through into 0
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    return df
