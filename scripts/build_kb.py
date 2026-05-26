"""Build the Task 3 recipe knowledge base over train + dev + test."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.knowledge_base import RecipeKnowledgeBase  # noqa: E402


def main() -> None:
    """Load all splits, build BM25, save index, and print statistics."""
    raw_dir = ROOT / "data" / "raw"
    output_dir = ROOT / "outputs" / "kb_index"
    frames = []
    for split in ("train", "dev", "test"):
        path = raw_dir / f"{split}.csv"
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
        print(f"loaded {split}: {len(df):,} rows")

    all_df = pd.concat(frames, ignore_index=True)
    kb = RecipeKnowledgeBase()
    kb.build_index(all_df, save_path=output_dir)

    stats = kb.get_stats()
    print("\n=== Knowledge Base Stats ===")
    print(f"num_docs: {stats['num_docs']:,}")
    print(f"indexing_time_sec: {stats['indexing_time_sec']:.2f}")
    print(f"avg_doc_length_tokens: {stats['avg_doc_length_tokens']:.2f}")
    print(f"retrieval_latency_ms: {stats['retrieval_latency_ms']:.2f}")
    print(f"saved_to: {output_dir}")


if __name__ == "__main__":
    main()
