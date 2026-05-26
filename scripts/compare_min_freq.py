"""Compare vocabulary characteristics across multiple ``min_freq`` thresholds.

For each candidate threshold this script:

1. Builds a shared vocabulary from the training split (Ingredients + Recipe).
2. Reports the resulting vocabulary size and build time.
3. Measures the OOV (out-of-vocabulary) rate on the development split — both
   as a fraction of *token occurrences* (the standard definition) and as a
   fraction of *unique token types* (useful for spotting head-vs-tail effects).
4. Prints a comparison table to help pick a final ``min_freq``.

The script is read-only with respect to ``data/raw/``.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


# 让 scripts/ 下能 import 项目根 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import Vocabulary, tokenize  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
# 题目要求对比的三个候选阈值
CANDIDATES: Tuple[int, ...] = (3, 5, 10)


def load_joined_texts(name: str) -> Tuple[List[str], List[str]]:
    """Load a split CSV and return (ingredients_texts, recipe_texts).

    Each element is the JSON list joined by a single space, ready for the
    tokenizer. Parsing errors would surface immediately rather than silently
    drop rows — the explore script already verified the data is clean.
    """
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    ing = df["Ingredients"].map(lambda s: " ".join(json.loads(s))).tolist()
    rec = df["Recipe"].map(lambda s: " ".join(json.loads(s))).tolist()
    return ing, rec


def tokenize_all(texts: Sequence[str]) -> List[List[str]]:
    """Tokenize every string in the input list."""
    return [tokenize(t) for t in texts]


def occurrence_oov(
    token_lists: Sequence[Sequence[str]], vocab: Vocabulary
) -> Tuple[int, int]:
    """Return (oov_occurrence_count, total_occurrence_count) against ``vocab``."""
    total = 0
    oov = 0
    for tokens in token_lists:
        for t in tokens:
            total += 1
            if t not in vocab:
                oov += 1
    return oov, total


def type_oov(
    token_lists: Sequence[Sequence[str]], vocab: Vocabulary
) -> Tuple[int, int]:
    """Return (oov_type_count, total_unique_type_count) against ``vocab``."""
    # 先把所有 dev token 去重, 再统计有多少种类型不在 vocab 中
    types = set()
    for tokens in token_lists:
        types.update(tokens)
    oov_types = sum(1 for t in types if t not in vocab)
    return oov_types, len(types)


def main() -> None:
    """Build vocabularies at three thresholds and print the comparison table."""
    # ----- 1) 加载 train + dev, 并对两侧都做拼接 -----
    print("Loading splits ...")
    train_ing, train_rec = load_joined_texts("train")
    dev_ing, dev_rec = load_joined_texts("dev")

    # ----- 2) 一次性分词, 之后所有阈值复用同一份 token list -----
    print("Tokenising ...")
    t0 = time.time()
    train_tokens = tokenize_all(train_ing) + tokenize_all(train_rec)
    dev_tokens = tokenize_all(dev_ing) + tokenize_all(dev_rec)
    print(f"  done in {time.time() - t0:.1f}s "
          f"(train docs={len(train_tokens):,}  dev docs={len(dev_tokens):,})")

    # ----- 3) 每个阈值各构建一次 vocab, 同时统计 occurrence / type 两种 OOV -----
    rows = []
    for mf in CANDIDATES:
        t0 = time.time()
        vocab = Vocabulary.build_from_texts(train_tokens, min_freq=mf)
        build_s = time.time() - t0

        occ_oov, occ_total = occurrence_oov(dev_tokens, vocab)
        typ_oov, typ_total = type_oov(dev_tokens, vocab)

        rows.append({
            "min_freq": mf,
            "vocab_size": len(vocab),
            "build_s": build_s,
            "occ_oov": occ_oov,
            "occ_total": occ_total,
            "occ_pct": 100.0 * occ_oov / occ_total if occ_total else 0.0,
            "typ_oov": typ_oov,
            "typ_total": typ_total,
            "typ_pct": 100.0 * typ_oov / typ_total if typ_total else 0.0,
        })

    # ----- 4) 打印对比表 -----
    print("\n=== min_freq comparison (vocab built from train; OOV measured on dev) ===")
    header = (
        f"{'min_freq':>8}  {'vocab_size':>11}  {'build (s)':>9}  "
        f"{'dev OOV occ %':>14}  {'dev OOV type %':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['min_freq']:>8}  {r['vocab_size']:>11,}  {r['build_s']:>9.1f}  "
            f"{r['occ_pct']:>13.3f}%  {r['typ_pct']:>14.2f}%"
        )

    # 顺便给出 dev 的绝对计数, 方便核对
    print("\nDev split absolute counts:")
    r = rows[0]
    print(f"  total token occurrences = {r['occ_total']:,}")
    print(f"  unique token types      = {r['typ_total']:,}")


if __name__ == "__main__":
    main()
