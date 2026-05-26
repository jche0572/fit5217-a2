"""BM25 recipe knowledge base for Task 3 menu-design RAG."""

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "for",
    "give",
    "i",
    "me",
    "of",
    "please",
    "recipe",
    "recipes",
    "show",
    "some",
    "the",
    "to",
    "want",
    "with",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 retrieval."""
    tokens = []
    for token in _TOKEN_RE.findall(str(text).lower()):
        if token in _STOPWORDS or len(token) <= 1:
            continue
        # 简单复数归一化, 让 desserts / recipes / tomatoes 这类词更容易匹配
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _safe_json_list(value: Any) -> list:
    """Parse a JSON-list string or pass through an existing list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


class RecipeKnowledgeBase:
    """A CPU-friendly BM25 index over recipe title, ingredients, and steps."""

    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.indexing_time_sec: float = 0.0
        self.avg_doc_length_tokens: float = 0.0
        self.retrieval_latency_ms: float = 0.0

    def _row_to_doc(self, row: pd.Series) -> Dict[str, Any]:
        """Convert one dataframe row into a retrievable document."""
        title = "" if pd.isna(row.get("Title", "")) else str(row.get("Title", ""))
        ingredients = _safe_json_list(row.get("Ingredients", []))
        recipe = _safe_json_list(row.get("Recipe", []))
        # Title 捕捉菜名和菜系词; Ingredients 捕捉食材约束; Recipe 正文补足 dessert/Italian 等描述词
        text = " ".join([title, " ".join(ingredients), " ".join(recipe)])
        return {
            "title": title,
            "ingredients": ingredients,
            "recipe": recipe,
            "text": text,
        }

    def build_index(self, df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> None:
        """Build a BM25 index from a dataframe and optionally save it."""
        start = time.perf_counter()
        self.documents = [self._row_to_doc(row) for _, row in df.iterrows()]
        self.tokenized_docs = [_tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.indexing_time_sec = time.perf_counter() - start
        lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length_tokens = sum(lengths) / max(len(lengths), 1)
        self.retrieval_latency_ms = 0.0

        if save_path is not None:
            self.save_index(save_path)

    def save_index(self, save_path: Union[str, Path]) -> None:
        """Persist the BM25 index and metadata to ``save_path``."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "bm25": self.bm25,
            "indexing_time_sec": self.indexing_time_sec,
            "avg_doc_length_tokens": self.avg_doc_length_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
        }
        with open(save_path / "kb.pkl", "wb") as f:
            pickle.dump(state, f)

    def load_index(self, load_path: Union[str, Path]) -> None:
        """Load a previously saved BM25 index from ``load_path``."""
        load_path = Path(load_path)
        with open(load_path / "kb.pkl", "rb") as f:
            state = pickle.load(f)
        self.documents = state["documents"]
        self.tokenized_docs = state["tokenized_docs"]
        self.bm25 = state["bm25"]
        self.indexing_time_sec = state["indexing_time_sec"]
        self.avg_doc_length_tokens = state["avg_doc_length_tokens"]
        self.retrieval_latency_ms = state.get("retrieval_latency_ms", 0.0)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the top-k recipes for ``query``."""
        if self.bm25 is None:
            raise RuntimeError("Knowledge base index has not been built or loaded.")

        start = time.perf_counter()
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        self.retrieval_latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "title": doc["title"],
                    "ingredients": doc["ingredients"],
                    "recipe": doc["recipe"],
                    "score": float(scores[idx]),
                }
            )
        return results

    def get_stats(self) -> dict:
        """Return knowledge-base size, timing, and document-length stats."""
        return {
            "num_docs": len(self.documents),
            "indexing_time_sec": self.indexing_time_sec,
            "avg_doc_length_tokens": self.avg_doc_length_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
        }
