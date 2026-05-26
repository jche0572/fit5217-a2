"""RAG pipeline for Task 3 menu generation."""

import json
import re
from typing import Any, Dict, List

from src.rag.knowledge_base import RecipeKnowledgeBase
from src.rag.llm_client import LLMClient


SYSTEM_PROMPT = """You are a menu designer for a university NLP assignment.
Use only the retrieved recipe context as source material.
Return valid JSON only. Do not include markdown fences or prose outside JSON.
Every dish must cite at least one source recipe title exactly as shown in context.
Every course must explicitly explain how it satisfies the user's constraints."""


class MenuDesigner:
    """Recipe RAG pipeline: retrieval, prompt construction, LLM generation."""

    def __init__(self, kb: RecipeKnowledgeBase, llm_client: LLMClient) -> None:
        self.kb = kb
        self.llm_client = llm_client

    def _auxiliary_queries(self, query: str) -> List[str]:
        """Create lightweight course-specific retrieval probes from the user query."""
        lower = query.lower()
        probes = []
        if "soup" in lower:
            probes.append("soup starter")
        if "chicken" in lower:
            probes.append("chicken main dish")
        if "cake" in lower:
            probes.append("cake dessert")
        if "dessert" in lower:
            probes.append("dessert")
        if "italian" in lower:
            probes.extend(["Italian pasta main", "Italian dessert", "antipasto Italian salad"])
        if "pasta" in lower:
            probes.append("pasta main")
        if "vegetarian" in lower:
            probes.extend(["vegetarian soup", "vegetarian main", "vegetarian dessert"])
        if "low-fat" in lower or "low fat" in lower:
            probes.extend(["low fat soup", "low fat main", "low fat dessert"])
        return probes

    def _retrieve_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve and deduplicate source recipes for the prompt context."""
        collected: List[Dict[str, Any]] = []
        seen = set()

        def add_results(results: List[Dict[str, Any]]) -> None:
            for item in results:
                key = (item["title"], tuple(item["ingredients"][:3]))
                if key in seen:
                    continue
                seen.add(key)
                collected.append(item)

        probes = self._auxiliary_queries(query)
        add_results(self.kb.retrieve(query, top_k=min(top_k, 8)))
        for probe in probes:
            add_results(self.kb.retrieve(probe, top_k=max(2, top_k // 3)))

        # 多课程菜单需要比单次 top-5 略宽的 context, 但控制 prompt 长度
        return collected[: max(top_k, 24)]

    def _build_prompt(self, query: str, recipes: List[Dict[str, Any]]) -> str:
        """Construct the JSON-only menu-generation prompt."""
        context_blocks = []
        for i, item in enumerate(recipes, start=1):
            ingredients = "; ".join(item["ingredients"][:10])
            steps = " ".join(item["recipe"][:4])
            context_blocks.append(
                f"[{i}] Title: {item['title']}\n"
                f"Ingredients: {ingredients}\n"
                f"Recipe excerpt: {steps}\n"
                f"BM25 score: {item['score']:.3f}"
            )
        context = "\n\n".join(context_blocks)
        return f"""User request:
{query}

Retrieved recipe context:
{context}

Create a menu that satisfies the request. Return this exact JSON shape:
{{
  "menu": [
    {{
      "course": "Starter",
      "dish": "Dish name",
      "source_recipes": ["Exact source recipe title"],
      "constraint_notes": "Explain explicitly how this dish satisfies the user's course and dietary constraints."
    }}
  ],
  "overall_notes": "Briefly explain how the full menu fits together."
}}

Rules:
- Include exactly the courses requested by the user, normally 3 courses.
- Use course labels such as Starter, Main, Dessert.
- Each dish must cite at least one source recipe title from the retrieved context.
- If the user says avoid an ingredient, do not include it in the dish and mention the avoidance in constraint_notes.
- If context is imperfect, adapt conservatively but keep source_recipes grounded in retrieved titles.
"""

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse model JSON, tolerating occasional markdown fences."""
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def generate_menu(self, query: str, top_k: int = 5) -> dict:
        """Generate a structured menu for ``query`` using retrieved recipes."""
        retrieved = self._retrieve_context(query, top_k=top_k)
        user_prompt = self._build_prompt(query, retrieved)
        raw = self.llm_client.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = self._parse_json(raw)
        return {
            "query": query,
            "retrieved_recipes": [
                {
                    "title": item["title"],
                    "ingredients": item["ingredients"],
                    "recipe": item["recipe"],
                    "score": item["score"],
                }
                for item in retrieved
            ],
            "menu": parsed.get("menu", []),
            "overall_notes": parsed.get("overall_notes", ""),
            "raw_llm_response": raw,
        }
