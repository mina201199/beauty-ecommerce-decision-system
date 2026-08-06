"""資料載入與 schema 檢查。

第一週唯一要跑的東西：
    python -m src.data --describe --dataset hillstrom

它會印出實際欄位。**不要憑印象填 config.py 的欄位名**，用這個確認。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import DATASETS, RAW_DIR, SEED, Dataset


def _find_file(ds: Dataset) -> "pd.DataFrame":
    folder = RAW_DIR / ds.key
    if not folder.exists():
        raise FileNotFoundError(
            f"找不到 {folder}。先跑：python scripts/download_data.py --dataset {ds.key}"
        )
    if ds.filename:
        target = folder / ds.filename
        if target.exists():
            return pd.read_csv(target)
    csvs = sorted(folder.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"{folder} 裡沒有 csv。實際內容：{list(folder.iterdir())}")
    if ds.filename:
        print(f"[warn] config 寫的 {ds.filename} 不存在，改用 {csvs[0].name}", file=sys.stderr)
    return pd.read_csv(csvs[0])


def describe(key: str) -> None:
    """印出真實 schema。設計欄位對應之前一定要先跑這個。"""
    ds = DATASETS[key]
    df = _find_file(ds)
    print(f"\n=== {key} ({ds.kaggle_slug}) ===")
    print(f"shape: {df.shape[0]:,} 列 x {df.shape[1]} 欄\n")
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "null_%": (df.isna().mean() * 100).round(2),
            "nunique": df.nunique(),
        }
    )
    info["sample"] = [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns]
    print(info.to_string())
    for col in (ds.treatment_col, ds.outcome_col):
        if col and col in df.columns:
            print(f"\n{col} 分布:\n{df[col].value_counts(dropna=False).to_string()}")
        elif col:
            print(f"\n[!] config 裡的 '{col}' 不在資料中 — 請改 src/config.py")
    if ds.note:
        print(f"\n注意: {ds.note}")


def load(key: str, binarize: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """回傳 (X 共變數, t 處理指派 0/1, y 成效 0/1)。

    會主動移除 config 裡標記的 post-treatment 欄位 —— 把處理後才發生的
    變數當共變數，是 uplift 專案最常見也最致命的錯誤，而且不會報錯。
    """
    ds = DATASETS[key]
    df = _find_file(ds)

    if ds.treatment_col not in df.columns:
        raise KeyError(
            f"treatment 欄 '{ds.treatment_col}' 不存在。實際欄位：{list(df.columns)}\n"
            f"請修改 src/config.py 的 DATASETS['{key}']。"
        )
    if ds.outcome_col not in df.columns:
        raise KeyError(
            f"outcome 欄 '{ds.outcome_col}' 不存在。實際欄位：{list(df.columns)}"
        )

    raw_t = df[ds.treatment_col]
    if ds.treated_values or ds.control_values:
        keep = raw_t.isin(ds.treated_values + ds.control_values)
        dropped = (~keep).sum()
        if dropped:
            print(f"[info] 依 config 過濾掉 {dropped:,} 列不屬於 treated/control 的資料")
        df = df.loc[keep].reset_index(drop=True)
        t = df[ds.treatment_col].isin(ds.treated_values).astype(int)
    else:
        t = pd.to_numeric(raw_t, errors="coerce").fillna(0).astype(int)
        if binarize:
            t = (t > 0).astype(int)

    y = pd.to_numeric(df[ds.outcome_col], errors="coerce").fillna(0)
    if binarize:
        y = (y > 0).astype(int)

    drop = set(ds.drop_cols) | {ds.treatment_col, ds.outcome_col}
    present = sorted(drop & set(df.columns))
    if ds.drop_cols:
        print(f"[info] 移除 post-treatment / 目標欄位：{present}")
    X = df.drop(columns=[c for c in drop if c in df.columns])

    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat:
        print(f"[info] one-hot 編碼類別欄位：{cat}")
        X = pd.get_dummies(X, columns=cat, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    print(
        f"[ok] {key}: n={len(X):,}  共變數={X.shape[1]}  "
        f"treated={int(t.sum()):,} ({t.mean():.1%})  outcome 率={y.mean():.4f}"
    )
    return X, t, y


def split(X, t, y, test_size: float, seed: int = SEED):
    """分層切分：同時對 treatment 與 outcome 分層，避免小樣本下比例跑掉。"""
    from sklearn.model_selection import train_test_split

    strat = t.astype(str) + "_" + y.astype(str)
    return train_test_split(X, t, y, test_size=test_size, random_state=seed, stratify=strat)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hillstrom", choices=list(DATASETS))
    p.add_argument("--describe", action="store_true")
    a = p.parse_args()
    if a.describe:
        describe(a.dataset)
    else:
        np.random.seed(SEED)
        load(a.dataset)
