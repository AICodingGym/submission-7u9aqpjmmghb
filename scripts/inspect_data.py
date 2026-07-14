#!/usr/bin/env python3
"""Data inspection for the Automated Essay Scoring 2.0 MLE challenge.

Read-only exploration: no modeling. Prints findings to stdout and writes a
Markdown report to outputs/findings.md.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUT_FILE = os.path.join(OUT_DIR, "findings.md")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    return train, test, sample


def _is_stringlike(s: pd.Series) -> bool:
    """True for object/str/string columns (pandas 2 uses object, pandas 3 uses str)."""
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def detect_columns(train: pd.DataFrame, test: pd.DataFrame):
    """Heuristically identify id / text / target columns."""
    # target = column in train but not in test
    target_cands = [c for c in train.columns if c not in test.columns]
    target = target_cands[0] if target_cands else None

    # id = a unique-valued, string-like column with short values (prefer name hint)
    id_col = None
    best_id_len = None
    for c in train.columns:
        if c == target:
            continue
        if train[c].nunique(dropna=False) != len(train):
            continue
        mean_len = train[c].astype(str).str.len().mean()
        # prefer a column literally named *id*, else the shortest unique column
        if "id" in c.lower():
            id_col, best_id_len = c, mean_len
            break
        if best_id_len is None or mean_len < best_id_len:
            id_col, best_id_len = c, mean_len

    # text = string-like column with the largest mean length that isn't the id
    text_col = None
    best_len = -1.0
    for c in train.columns:
        if c in (target, id_col):
            continue
        if _is_stringlike(train[c]):
            mean_len = train[c].astype(str).str.len().mean()
            if mean_len > best_len:
                best_len = mean_len
                text_col = c
    return id_col, text_col, target


def _fmt(v) -> str:
    """Render a cell value cleanly for a markdown table."""
    if isinstance(v, float):
        # drop trailing .0 for whole numbers, else keep given precision
        return f"{v:g}"
    return str(v)


def df_to_md(df: pd.DataFrame, index_label: str) -> str:
    """Render a DataFrame (with a meaningful index) as a GitHub markdown table."""
    headers = [index_label] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for idx, row in df.iterrows():
        cells = [_fmt(idx)] + [_fmt(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def series_to_md(s: pd.Series, key_label: str, val_label: str) -> str:
    """Render a Series as a two-column GitHub markdown table."""
    lines = [f"| {key_label} | {val_label} |", "| --- | --- |"]
    for idx, v in s.items():
        lines.append(f"| {_fmt(idx)} | {_fmt(v)} |")
    return "\n".join(lines)


def describe_df(name: str, df: pd.DataFrame):
    print(f"### {name}")
    print(f"- shape: {df.shape[0]} rows x {df.shape[1]} cols")
    cols = ", ".join(f"`{c}`" for c in df.columns)
    dtypes = ", ".join(f"`{c}`: {t}" for c, t in df.dtypes.astype(str).items())
    print(f"- columns: {cols}")
    print(f"- dtypes: {dtypes}")
    print()
    print("First 3 rows (text truncated to 160 chars):")
    print()
    preview = df.head(3).copy()
    for c in preview.columns:
        if _is_stringlike(preview[c]):
            full = preview[c].astype(str)
            truncated = full.str.slice(0, 160).str.replace("\n", " ", regex=False)
            suffix = full.str.len().apply(lambda n: f" …[{n} chars]" if n > 160 else "")
            preview[c] = truncated + suffix
    for i, (_, row) in enumerate(preview.iterrows(), 1):
        print(f"- **row {i}**")
        for c in preview.columns:
            print(f"    - `{c}`: {row[c]}")
        print()


def length_stats(s: pd.Series, label: str):
    chars = s.astype(str).str.len()
    words = s.astype(str).str.split().apply(len)
    pcts = [0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
    print(f"**{label} — character length**")
    print()
    print(series_to_md(chars.describe(percentiles=pcts).round(1), "stat", "chars"))
    print()
    print(f"**{label} — word count**")
    print()
    print(series_to_md(words.describe(percentiles=pcts).round(1), "stat", "words"))
    print()
    return chars, words


def main():
    train, test, sample = load()
    id_col, text_col, target = detect_columns(train, test)

    print("# Automated Essay Scoring 2.0 — Data Findings")
    print()
    print("_Read-only data inspection. No modeling performed._")
    print()

    print("## 1. Files & shapes")
    print()
    describe_df("train.csv", train)
    describe_df("test.csv", test)
    describe_df("sample_submission.csv", sample)

    print("## 2. Auto-detected column roles")
    print()
    print(f"- **id column**: `{id_col}`")
    print(f"- **text column**: `{text_col}`")
    print(f"- **target column**: `{target}` (present in train, absent in test)")
    print()

    print("## 3. Target (score) distribution")
    print()
    if target is not None:
        vc = train[target].value_counts().sort_index()
        pct = (vc / len(train) * 100).round(2)
        dist = pd.DataFrame({"count": vc, "pct_%": pct})
        print(df_to_md(dist, "score"))
        print()
        print(f"- score dtype: {train[target].dtype}")
        print(f"- min / max: {train[target].min()} / {train[target].max()}")
        print(f"- mean: {train[target].mean():.4f}  median: {train[target].median()}  std: {train[target].std():.4f}")
        print(f"- unique score values: {sorted(train[target].dropna().unique().tolist())}")
        # baseline: predict rounded mean (used by efficiency-prize baseline)
        mean_pred = train[target].mean()
        print(f"- baseline mean prediction (efficiency baseline reference): {mean_pred:.4f} -> rounded {round(mean_pred)}")
        print()

    print("## 4. Text length distribution")
    print()
    if text_col is not None:
        tr_chars, tr_words = length_stats(train[text_col], "train.full_text")
        te_chars, te_words = length_stats(test[text_col], "test.full_text")

        print("### Score vs. text length (train)")
        print()
        if target is not None:
            tmp = pd.DataFrame({
                "score": train[target],
                "n_chars": tr_chars,
                "n_words": tr_words,
            })
            grp = tmp.groupby("score").agg(
                n=("n_chars", "size"),
                mean_chars=("n_chars", "mean"),
                median_chars=("n_chars", "median"),
                mean_words=("n_words", "mean"),
                median_words=("n_words", "median"),
            ).round(1)
            print(df_to_md(grp, "score"))
            corr = tmp["score"].corr(tmp["n_words"])
            print()
            print(f"- Pearson corr(score, word_count) = {corr:.4f}")
            print()

    print("## 5. Missing values")
    print()
    for name, df in [("train", train), ("test", test), ("sample_submission", sample)]:
        miss = df.isna().sum()
        miss = miss[miss > 0]
        if len(miss) == 0:
            print(f"- {name}: no missing values")
        else:
            print(f"- {name}: {miss.to_dict()}")
    # empty / whitespace-only text
    if text_col is not None:
        empty_tr = (train[text_col].astype(str).str.strip() == "").sum()
        empty_te = (test[text_col].astype(str).str.strip() == "").sum()
        print(f"- train empty/whitespace-only {text_col}: {empty_tr}")
        print(f"- test empty/whitespace-only {text_col}: {empty_te}")
    print()

    print("## 6. Duplicates")
    print()
    if id_col is not None:
        print(f"- train duplicate {id_col}: {train[id_col].duplicated().sum()}")
        print(f"- test duplicate {id_col}: {test[id_col].duplicated().sum()}")
        # overlap of ids between train and test
        overlap = set(train[id_col]) & set(test[id_col])
        print(f"- train/test {id_col} overlap: {len(overlap)}")
    if text_col is not None:
        dup_text_tr = train[text_col].duplicated().sum()
        dup_text_te = test[text_col].duplicated().sum()
        print(f"- train duplicate essays (exact {text_col}): {dup_text_tr}")
        print(f"- test duplicate essays (exact {text_col}): {dup_text_te}")
        # cross-set essay leakage
        text_overlap = set(train[text_col]) & set(test[text_col])
        print(f"- train/test identical essays: {len(text_overlap)}")
        if dup_text_tr > 0 and target is not None:
            # do duplicated essays get consistent scores?
            dupmask = train[text_col].duplicated(keep=False)
            grp = train[dupmask].groupby(text_col)[target].nunique()
            inconsistent = (grp > 1).sum()
            print(f"- duplicated train essays with inconsistent scores: {inconsistent} / {len(grp)} groups")
    print()

    print("## 7. Submission format sanity check")
    print()
    print(f"- sample_submission columns: {list(sample.columns)}")
    print(f"- sample_submission rows: {len(sample)}  |  test rows: {len(test)}  |  match: {len(sample) == len(test)}")
    if id_col in sample.columns:
        same_ids = set(sample[id_col]) == set(test[id_col])
        print(f"- sample ids == test ids: {same_ids}")
    if target in sample.columns:
        print(f"- sample score values: {sorted(sample[target].unique().tolist())}")
    print()
    print("## 8. Notes / next steps")
    print()
    print("- Metric: **Quadratic Weighted Kappa (QWK)** on integer scores 1-6.")
    print("- CPU-only environment; efficiency prize is CPU-only. Favor lightweight features "
          "(TF-IDF / linear / gradient-boosting) over large transformers.")
    print("- Predictions must be integers 1-6; QWK is sensitive to rounding/clipping thresholds.")
    print("- Baseline reference = mean-of-target; must beat it to be efficiency-eligible.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    report = buf.getvalue()
    print(report)  # to console
    with open(OUT_FILE, "w") as f:
        f.write(report)
    print(f"\n[written] {os.path.abspath(OUT_FILE)}")
