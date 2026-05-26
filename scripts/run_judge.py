"""Run Task 3.3 self- and cross-evaluation for demo menus."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.judge import DIMENSIONS, JUDGE_PROMPT_TEMPLATE, MenuJudge  # noqa: E402
from src.rag.llm_client import GroqClient  # noqa: E402


SELF_MODEL = "llama-3.3-70b-versatile"
CROSS_MODEL = "llama-3.1-8b-instant"


def load_demo_menus() -> List[Dict]:
    """Load the three Task 3.2 demo menu JSON files."""
    menus = []
    for i in range(1, 4):
        path = ROOT / "outputs" / "menus" / f"demo_{i}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_demo_id"] = f"demo_{i}"
        menus.append(data)
    return menus


def result_row(demo_id: str, result: Dict) -> Dict:
    """Flatten one judge result into a markdown-table row."""
    return {
        "demo": demo_id,
        "eval_type": result["metadata"]["eval_type"],
        "judge_model": result["metadata"]["judge_model"],
        "constraint_satisfaction": result["constraint_satisfaction"]["score"],
        "ingredient_faithfulness": result["ingredient_faithfulness"]["score"],
        "culinary_logic_coherence": result["culinary_logic_coherence"]["score"],
        "bias": result["bias"]["score"],
    }


def to_markdown_table(rows: List[Dict]) -> str:
    """Render flattened scores as a markdown table."""
    headers = [
        "demo",
        "eval_type",
        "judge_model",
        "constraint_satisfaction",
        "ingredient_faithfulness",
        "culinary_logic_coherence",
        "bias",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run self and cross judges, then save markdown and JSON outputs."""
    output_json = ROOT / "outputs" / "judge_results.json"
    output_md = ROOT / "outputs" / "judge_results.md"

    self_judge = MenuJudge(
        GroqClient(model=SELF_MODEL, timeout=90, max_retries=3),
        judge_model_name=SELF_MODEL,
        eval_type="self",
    )
    cross_judge = MenuJudge(
        GroqClient(model=CROSS_MODEL, timeout=90, max_retries=3),
        judge_model_name=CROSS_MODEL,
        eval_type="cross",
    )

    all_results = []
    table_rows = []

    def persist() -> None:
        """Write intermediate results so rate-limit interruptions do not lose progress."""
        md = "# Task 3.3 LLM-as-Judge Results\n\n"
        md += f"Self model: `{SELF_MODEL}`\n\n"
        md += f"Cross model: `{CROSS_MODEL}`\n\n"
        md += "## Score Table\n\n"
        md += to_markdown_table(table_rows)
        md += "\n\n## Judge Prompt Template\n\n"
        md += "```text\n" + JUDGE_PROMPT_TEMPLATE + "\n```\n"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md)

    for menu in load_demo_menus():
        demo_id = menu["_demo_id"]
        query = menu["query"]
        print(f"\n=== Evaluating {demo_id}: self ===")
        self_result = self_judge.evaluate(menu, query).to_dict()
        print(json.dumps({k: self_result[k] for k in DIMENSIONS}, indent=2, ensure_ascii=False))
        all_results.append({"demo": demo_id, "query": query, "result": self_result})
        table_rows.append(result_row(demo_id, self_result))
        persist()
        time.sleep(12)

        print(f"\n=== Evaluating {demo_id}: cross ===")
        cross_result = cross_judge.evaluate(menu, query).to_dict()
        print(json.dumps({k: cross_result[k] for k in DIMENSIONS}, indent=2, ensure_ascii=False))
        all_results.append({"demo": demo_id, "query": query, "result": cross_result})
        table_rows.append(result_row(demo_id, cross_result))
        persist()
        time.sleep(12)

    print("\n=== Summary Table ===")
    print(to_markdown_table(table_rows))
    print(f"\nsaved: {output_json}")
    print(f"saved: {output_md}")


if __name__ == "__main__":
    main()
