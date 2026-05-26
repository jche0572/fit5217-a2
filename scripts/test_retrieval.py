"""Smoke-test the Task 3 recipe knowledge base retrieval quality."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.knowledge_base import RecipeKnowledgeBase  # noqa: E402


QUERIES = [
    "Give me some chicken recipes.",
    "Give me some dessert recipes.",
    "Give me some Italian recipes.",
]


def main() -> None:
    """Load the saved KB and print top-5 results for three demo queries."""
    kb = RecipeKnowledgeBase()
    kb.load_index(ROOT / "outputs" / "kb_index")

    for query in QUERIES:
        print(f"\n=== Query: {query} ===")
        results = kb.retrieve(query, top_k=5)
        for rank, item in enumerate(results, start=1):
            ingredients = ", ".join(item["ingredients"][:5])
            if len(item["ingredients"]) > 5:
                ingredients += ", ..."
            print(f"{rank}. {item['title']}  score={item['score']:.3f}")
            print(f"   ingredients: {ingredients}")
        print(f"retrieval_latency_ms: {kb.get_stats()['retrieval_latency_ms']:.2f}")

    print("\n=== Final Stats ===")
    for key, value in kb.get_stats().items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
