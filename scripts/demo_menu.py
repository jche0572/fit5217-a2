"""Run Task 3.2 RAG menu-generation demos and save JSON outputs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.knowledge_base import RecipeKnowledgeBase  # noqa: E402
from src.rag.llm_client import GroqClient  # noqa: E402
from src.rag.menu_designer import MenuDesigner  # noqa: E402


DEMO_QUERIES = [
    "Suggest a 3-course dinner menu: a soup starter, a chicken main dish, and a cake for dessert.",
    "Plan a 3-course Italian dinner (starter, pasta main, dessert) for a dinner party.",
    "Create a low-fat, vegetarian 3-course dinner (soup, main, dessert). Avoid using cheese or cream.",
]


def main() -> None:
    """Generate and save the three required demo menus."""
    output_dir = ROOT / "outputs" / "menus"
    output_dir.mkdir(parents=True, exist_ok=True)

    kb = RecipeKnowledgeBase()
    kb.load_index(ROOT / "outputs" / "kb_index")
    llm = GroqClient(model="llama-3.3-70b-versatile", timeout=90, max_retries=3)
    designer = MenuDesigner(kb, llm)

    for i, query in enumerate(DEMO_QUERIES, start=1):
        print(f"\n=== Demo {i}: {query} ===")
        result = designer.generate_menu(query, top_k=12)
        out_path = output_dir / f"demo_{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(json.dumps(result["menu"], indent=2, ensure_ascii=False))
        if result.get("overall_notes"):
            print("overall_notes:", result["overall_notes"])
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
