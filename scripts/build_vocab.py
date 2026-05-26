"""Build the final Task 1.1 artifacts in a single pass.

Outputs (all under ``outputs/``):

* ``vocab.pkl`` — the chosen ``Vocabulary`` (``min_freq=5``).
* ``min_freq_comparison.json`` — the 3-way comparison table (``min_freq`` ∈
  {3, 5, 10}) including dev and test OOV counts. Kept for reporting later.
* ``processed/{train,dev,test}_ids.pkl`` — each split encoded into a list of
  ``{"ingredients_ids": [...], "recipe_ids": [...]}`` records, where
  ``recipe_ids`` is wrapped with ``<SOS>`` / ``<EOS>``.

The script is idempotent: it can be re-run and will overwrite previous
artifacts. Tokenization happens once per split and is reused for both
vocabulary construction and encoding to keep the runtime short.
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import Vocabulary, tokenize  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"

# 最终选用的频率阈值; 同时保留 3 / 10 用于对比表
CHOSEN_MIN_FREQ = 5
COMPARISON_THRESHOLDS = (3, 5, 10)
SPLITS = ("train", "dev", "test")


def load_split(name: str) -> pd.DataFrame:
    """Load a split CSV and parse its two JSON list columns."""
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    # explore 步骤已确认无解析错误, 这里直接 json.loads 即可
    df["Ingredients"] = df["Ingredients"].map(json.loads)
    df["Recipe"] = df["Recipe"].map(json.loads)
    return df


def tokenize_rows(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Tokenize each row of a JSON-list column into a list of tokens."""
    # 单行 list 先用空格拼成段落, 再交给统一的 tokenize 规则
    return [tokenize(" ".join(x)) for x in rows]


def oov_block(token_lists: Sequence[Sequence[str]], vocab: Vocabulary) -> Dict:
    """Return occurrence- and type-level OOV counts of ``token_lists`` vs ``vocab``."""
    total = 0
    oov = 0
    types: set = set()
    for tokens in token_lists:
        types.update(tokens)
        for t in tokens:
            total += 1
            if t not in vocab:
                oov += 1
    oov_types = sum(1 for t in types if t not in vocab)
    return {
        "occ_oov": oov,
        "occ_total": total,
        "occ_pct": round(100 * oov / total, 4) if total else 0.0,
        "type_oov": oov_types,
        "type_total": len(types),
        "type_pct": round(100 * oov_types / len(types), 4) if types else 0.0,
    }


def main() -> None:
    """Entry point: load → tokenize → compare → save vocab → encode → cache."""
    # ---- 1) 加载三个 split ----
    print("Loading splits ...")
    dfs = {name: load_split(name) for name in SPLITS}
    for name, df in dfs.items():
        print(f"  {name}: {len(df):,} rows")

    # ---- 2) 一次性分词 (按行, 双侧分开), 后面 vocab 构建和 encode 都复用这份结果 ----
    print("\nTokenising per row (both sides) ...")
    t0 = time.time()
    ing_tokens: Dict[str, List[List[str]]] = {
        name: tokenize_rows(dfs[name]["Ingredients"]) for name in SPLITS
    }
    rec_tokens: Dict[str, List[List[str]]] = {
        name: tokenize_rows(dfs[name]["Recipe"]) for name in SPLITS
    }
    print(f"  done in {time.time() - t0:.1f}s")

    # train 的两侧合在一起喂给 vocab 构建 (共享词表)
    train_corpus = ing_tokens["train"] + rec_tokens["train"]
    dev_corpus = ing_tokens["dev"] + rec_tokens["dev"]
    test_corpus = ing_tokens["test"] + rec_tokens["test"]

    # ---- 3) 构建 3 个候选词表, 同时记录 dev / test OOV ----
    print(f"\nBuilding {len(COMPARISON_THRESHOLDS)} candidate vocabularies ...")
    comparison = []
    chosen_vocab = None
    for mf in COMPARISON_THRESHOLDS:
        v = Vocabulary.build_from_texts(train_corpus, min_freq=mf)
        row = {
            "min_freq": mf,
            "vocab_size": len(v),
            "dev": oov_block(dev_corpus, v),
            "test": oov_block(test_corpus, v),
            "chosen": mf == CHOSEN_MIN_FREQ,
        }
        comparison.append(row)
        print(
            f"  min_freq={mf:>2}  vocab={len(v):>6,}  "
            f"dev_oov={row['dev']['occ_pct']:>6.3f}%  "
            f"test_oov={row['test']['occ_pct']:>6.3f}%"
        )
        if mf == CHOSEN_MIN_FREQ:
            chosen_vocab = v

    assert chosen_vocab is not None, "CHOSEN_MIN_FREQ must appear in COMPARISON_THRESHOLDS"

    # ---- 4) 持久化对比表 JSON + 选中的词表 ----
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUTS_DIR / "min_freq_comparison.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "chosen_min_freq": CHOSEN_MIN_FREQ,
                "thresholds": list(COMPARISON_THRESHOLDS),
                "rows": comparison,
            },
            f,
            indent=2,
        )

    vocab_path = OUTPUTS_DIR / "vocab.pkl"
    chosen_vocab.save(vocab_path)

    # ---- 5) 用最终词表 encode 三个 split, 写入 processed/ ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("\nEncoding splits ...")
    cached = {}
    for name in SPLITS:
        t0 = time.time()
        records = []
        for ing_toks, rec_toks in zip(ing_tokens[name], rec_tokens[name]):
            # 直接复用已分词结果, 不再调 ingredients_to_ids / recipe_to_ids 以免二次分词
            records.append(
                {
                    "ingredients_ids": chosen_vocab.encode(ing_toks),
                    "recipe_ids": (
                        [Vocabulary.SOS_ID]
                        + chosen_vocab.encode(rec_toks)
                        + [Vocabulary.EOS_ID]
                    ),
                }
            )
        out_path = PROCESSED_DIR / f"{name}_ids.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = out_path.stat().st_size / 1024 / 1024
        cached[name] = {"records": len(records), "size_mb": size_mb}
        print(f"  {name}: {len(records):,} records ({time.time() - t0:.1f}s) "
              f"-> outputs/processed/{name}_ids.pkl ({size_mb:.1f} MB)")

    # ---- 6) Summary ----
    print("\n=== Summary ===")
    chosen_row = next(r for r in comparison if r["chosen"])
    print(f"Vocabulary size (min_freq={CHOSEN_MIN_FREQ}): {len(chosen_vocab):,}")
    print(
        f"Dev  OOV (occurrences): {chosen_row['dev']['occ_pct']}%  "
        f"({chosen_row['dev']['occ_oov']:,} / {chosen_row['dev']['occ_total']:,})"
    )
    print(
        f"Test OOV (occurrences): {chosen_row['test']['occ_pct']}%  "
        f"({chosen_row['test']['occ_oov']:,} / {chosen_row['test']['occ_total']:,})"
    )
    print()
    print("File sizes:")
    print(f"  outputs/vocab.pkl                       "
          f"{vocab_path.stat().st_size / 1024:>7.1f} KB")
    print(f"  outputs/min_freq_comparison.json        "
          f"{json_path.stat().st_size / 1024:>7.1f} KB")
    for name in SPLITS:
        c = cached[name]
        print(f"  outputs/processed/{name}_ids.pkl"
              f"{'':<{len('train_ids.pkl') - len(name + '_ids.pkl')}}        "
              f"{c['size_mb']:>7.1f} MB   ({c['records']:>7,} records)")


if __name__ == "__main__":
    main()
