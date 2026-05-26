"""LLM-as-judge evaluation for Task 3 generated menus."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict

from src.rag.llm_client import LLMClient


DIMENSIONS = [
    "constraint_satisfaction",
    "ingredient_faithfulness",
    "culinary_logic_coherence",
    "bias",
]


JUDGE_SYSTEM_PROMPT = """You are a careful evaluator for a university NLP menu-generation assignment.
Evaluate the generated menu against the original user request and source recipe evidence.
You must reason first, then assign scores. Return valid JSON only."""


JUDGE_PROMPT_TEMPLATE = """Original user request:
{query}

Generated menu JSON:
{menu_json}

Retrieved source recipe evidence:
{sources_json}

Evaluate the menu on four dimensions:

1. Constraint Satisfaction
- Does the menu follow explicit user constraints such as number of courses, course types, cuisine, dietary restrictions, and avoided ingredients?

2. Ingredient Faithfulness
- Are dishes and claims grounded in the retrieved source recipes?
- Are source recipe titles cited accurately?
- Are adaptations reasonable and transparent?

3. Culinary Logic & Coherence
- Would the menu work as a coherent meal?
- Are course order, dish pairing, preparation logic, and flavor balance plausible?

4. Bias
- Does the menu avoid stereotypes, exclusionary assumptions, or culturally careless claims?
- Score high when the menu is neutral, respectful, and avoids unsupported cultural generalizations.

Rubric for each dimension:
1 = Poor: major failures or contradictions.
2 = Weak: several important issues.
3 = Adequate: mostly acceptable but with noticeable gaps.
4 = Good: satisfies the dimension with minor issues.
5 = Excellent: fully satisfies the dimension with clear evidence.

Important instructions:
- For each dimension, write the reasoning first, then choose the score.
- Scores must be integers from 1 to 5.
- Be critical but fair. Do not reward unsupported claims.
- Return this exact JSON shape:
{{
  "constraint_satisfaction": {{"reasoning": "...", "score": 1}},
  "ingredient_faithfulness": {{"reasoning": "...", "score": 1}},
  "culinary_logic_coherence": {{"reasoning": "...", "score": 1}},
  "bias": {{"reasoning": "...", "score": 1}}
}}
"""


@dataclass
class JudgeScore:
    """One dimension score and reasoning."""

    score: int
    reasoning: str


@dataclass
class JudgeResult:
    """Structured judge result with metadata."""

    constraint_satisfaction: JudgeScore
    ingredient_faithfulness: JudgeScore
    culinary_logic_coherence: JudgeScore
    bias: JudgeScore
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert nested dataclasses to a JSON-serializable dict."""
        return asdict(self)


class MenuJudge:
    """Evaluate generated menus using an LLM judge."""

    def __init__(self, llm_client: LLMClient, judge_model_name: str, eval_type: str) -> None:
        self.llm_client = llm_client
        self.judge_model_name = judge_model_name
        self.eval_type = eval_type

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse judge JSON with tolerance for occasional markdown fences."""
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

    def _score_from_obj(self, obj: Dict[str, Any], key: str) -> JudgeScore:
        """Extract and clamp one dimension from parsed LLM JSON."""
        value = obj.get(key, {})
        score = int(value.get("score", 1))
        score = max(1, min(5, score))
        reasoning = str(value.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = "No reasoning provided by judge."
        return JudgeScore(score=score, reasoning=reasoning)

    def evaluate(self, menu: dict, original_query: str) -> JudgeResult:
        """Evaluate one generated menu and return structured scores."""
        compact_sources = [
            {
                "title": item.get("title", ""),
                "ingredients": item.get("ingredients", [])[:8],
                "recipe_excerpt": item.get("recipe", [])[:3],
            }
            for item in menu.get("retrieved_recipes", [])[:12]
        ]
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=original_query,
            menu_json=json.dumps(menu.get("menu", []), ensure_ascii=False, indent=2),
            sources_json=json.dumps(compact_sources, ensure_ascii=False, indent=2),
        )

        raw = self.llm_client.generate(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )

        metadata: Dict[str, Any] = {
            "judge_model": self.judge_model_name,
            "eval_type": self.eval_type,
            "raw_judge_response": raw,
        }
        try:
            parsed = self._parse_json(raw)
            return JudgeResult(
                constraint_satisfaction=self._score_from_obj(parsed, "constraint_satisfaction"),
                ingredient_faithfulness=self._score_from_obj(parsed, "ingredient_faithfulness"),
                culinary_logic_coherence=self._score_from_obj(parsed, "culinary_logic_coherence"),
                bias=self._score_from_obj(parsed, "bias"),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - JSON repair boundary
            metadata["parse_error"] = str(exc)
            fallback = JudgeScore(score=1, reasoning=f"Judge output could not be parsed: {exc}")
            return JudgeResult(
                constraint_satisfaction=fallback,
                ingredient_faithfulness=fallback,
                culinary_logic_coherence=fallback,
                bias=fallback,
                metadata=metadata,
            )
