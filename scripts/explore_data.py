"""Exploratory data analysis for the FIT5217 A2 cooking dataset.

Loads the three CSV splits under ``data/raw/``, parses the JSON list columns
(``Ingredients`` and ``Recipe``) into Python lists, and reports per-split
summary statistics (row counts, list-length distributions, approximate token
counts, null/empty diagnostics) plus three random sample rows from the
training split. The script is strictly read-only and never writes back to
``data/raw/``.
"""

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# 脚本位于 scripts/ 下, 项目根是 parent.parent, 原始数据在根目录的 data/raw/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = ("train", "dev", "test")


def safe_loads(x):
    """Parse a JSON string into a Python object; return None on failure.

    Used to robustly decode the Ingredients and Recipe columns so we can count
    malformed rows downstream instead of crashing the script.
    """
    if not isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except (json.JSONDecodeError, ValueError):
        return None


def load_split(name: str) -> pd.DataFrame:
    """Load a split CSV and parse its JSON list columns into Python lists.

    Parameters
    ----------
    name : str
        Split name without extension (e.g. ``"train"``).

    Returns
    -------
    pd.DataFrame
        DataFrame whose Ingredients and Recipe columns now contain ``list`` (on
        success) or ``None`` (on parse failure / missing value).
    """
    path = RAW_DIR / f"{name}.csv"
    # 用 pandas 读 CSV, Title 列保持 string, Ingredients/Recipe 是 JSON string 单独解析
    df = pd.read_csv(path)
    df["Ingredients"] = df["Ingredients"].map(safe_loads)
    df["Recipe"] = df["Recipe"].map(safe_loads)
    return df


def percentile_summary(values: Sequence[int]) -> dict:
    """Return min / max / mean / median / p95 of a non-empty numeric sequence."""
    arr = np.asarray(values)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def fmt_stats(label: str, stats: dict) -> str:
    """Format a stats dict into a single aligned console line."""
    return (
        f"  {label:<30} min={stats['min']:<5} max={stats['max']:<6} "
        f"mean={stats['mean']:>8.2f}  median={stats['median']:>6.1f}  "
        f"p95={stats['p95']:>7.1f}"
    )


def summarize_split(name: str, df: pd.DataFrame) -> None:
    """Print row counts, null/empty diagnostics, and three distributions."""
    print(f"\n=== Split: {name}  (rows = {len(df):,}) ===")

    # ---- 缺失值 / 解析失败 / 空列表的统计 ----
    n_title_null = df["Title"].isna().sum()
    n_ing_null = df["Ingredients"].isna().sum()
    n_rec_null = df["Recipe"].isna().sum()
    n_ing_empty = sum(isinstance(x, list) and len(x) == 0 for x in df["Ingredients"])
    n_rec_empty = sum(isinstance(x, list) and len(x) == 0 for x in df["Recipe"])
    print(
        f"  null/parse-failed   Title={n_title_null}  "
        f"Ingredients={n_ing_null}  Recipe={n_rec_null}"
    )
    print(f"  empty lists         Ingredients={n_ing_empty}  Recipe={n_rec_empty}")

    # ---- 分布统计: 只对成功解析为 list 的行做统计, 避免 None 干扰 ----
    valid_ing = [x for x in df["Ingredients"] if isinstance(x, list)]
    valid_rec = [x for x in df["Recipe"] if isinstance(x, list)]

    ing_lens = [len(x) for x in valid_ing]
    step_counts = [len(x) for x in valid_rec]
    # Recipe 拼接为一段文本后按空格 split 估算 token 数 (粗略, 不是真正的 tokenizer)
    token_counts = [len(" ".join(x).split()) for x in valid_rec]

    if ing_lens:
        print(fmt_stats("Ingredients list length", percentile_summary(ing_lens)))
    if step_counts:
        print(fmt_stats("Recipe step count", percentile_summary(step_counts)))
    if token_counts:
        print(fmt_stats("Recipe total tokens (split)", percentile_summary(token_counts)))


def print_samples(df: pd.DataFrame, k: int = 3, seed: int = 42) -> None:
    """Pretty-print ``k`` randomly chosen rows for visual inspection."""
    # 用独立的 Random 实例, 不污染全局 random 状态
    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), k)
    print(f"\n=== {k} random training samples (seed={seed}) ===")
    for i, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        print(f"\n--- Sample {i} (row #{idx}) ---")
        print(f"Title       : {row['Title']}")
        ing = row["Ingredients"]
        rec = row["Recipe"]
        ing_n = len(ing) if isinstance(ing, list) else "N/A"
        rec_n = len(rec) if isinstance(rec, list) else "N/A"
        print(f"Ingredients ({ing_n} items):")
        if isinstance(ing, list):
            for j, item in enumerate(ing, 1):
                print(f"  {j}. {item}")
        print(f"Recipe ({rec_n} steps):")
        if isinstance(rec, list):
            for j, step in enumerate(rec, 1):
                print(f"  {j}. {step}")


def main() -> None:
    """Entry point: load every split, summarise, then show three train samples."""
    print(f"Reading from: {RAW_DIR}")
    # 一次性加载, 后续多个统计函数共享同一份 DataFrame
    dfs = {name: load_split(name) for name in SPLITS}
    for name in SPLITS:
        summarize_split(name, dfs[name])
    print_samples(dfs["train"], k=3, seed=42)


if __name__ == "__main__":
    main()
