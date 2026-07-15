"""Linguistic / readability features for essay scoring (CPU, lightweight).

Complements the 28 structural features in ``src.features`` with classic
readability indices and lexical-sophistication measures — the kind of signal
traditional Automated Essay Scoring systems rely on and that a TF-IDF bag of
words does not directly expose.

Only depends on ``textstat`` (pure-Python, CPU) + numpy/pandas. No spaCy/NLTK,
no network. Fits nothing (indices are deterministic functions of the text), so
it is leakage-free across folds. Missing/degenerate values are filled with 0.

``extract_linguistic_features(texts)`` -> DataFrame (n, len(LING_COLUMNS)).
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import textstat

_WORD_RE = re.compile(r"[a-zA-Z]+")

# readability indices exposed by textstat (name -> callable attribute)
_READABILITY = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "coleman_liau_index",
    "automated_readability_index",
    "dale_chall_readability_score",
    "linsear_write_formula",
]

LING_COLUMNS = _READABILITY + [
    "difficult_words",         # textstat count of "hard" words
    "difficult_word_ratio",    # normalized by word count
    "syllables_per_word",      # mean syllables / token
    "long_syllable_ratio",     # fraction of words with >=3 syllables (polysyllabic)
]

# placeholder-for-append


def _safe(fn, text: str) -> float:
    try:
        v = float(fn(text))
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _features_for_text(text: str) -> list[float]:
    t = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    if not t.strip():
        return [0.0] * len(LING_COLUMNS)

    vals = [_safe(getattr(textstat, name), t) for name in _READABILITY]

    words = _WORD_RE.findall(t)
    n_words = len(words)
    diff = _safe(textstat.difficult_words, t)
    try:
        syll = float(textstat.syllable_count(t))
    except Exception:
        syll = 0.0
    # polysyllabic words (>=3 syllables) per word
    try:
        poly = sum(1 for w in words if textstat.syllable_count(w) >= 3)
    except Exception:
        poly = 0

    vals += [
        diff,
        diff / n_words if n_words else 0.0,
        syll / n_words if n_words else 0.0,
        poly / n_words if n_words else 0.0,
    ]
    return vals


def extract_linguistic_features(texts: Iterable[str]) -> pd.DataFrame:
    """Readability + lexical-sophistication features. Numeric, 0-filled, CPU."""
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    index = series.index
    if len(series) == 0:
        return pd.DataFrame(columns=LING_COLUMNS, dtype=np.float64)
    rows = [_features_for_text(t) for t in series]
    df = pd.DataFrame(rows, columns=LING_COLUMNS, index=index)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    return df
