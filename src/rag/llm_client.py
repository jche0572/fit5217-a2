"""LLM client abstractions for Task 3 menu generation."""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - Colab/local env may already export keys
    load_dotenv = None


class LLMClient(ABC):
    """Abstract chat-completion client interface."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text from chat messages."""
        raise NotImplementedError


class GroqClient(LLMClient):
    """Groq OpenAI-compatible chat client using ``GROQ_API_KEY``."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Put it in .env or export it before running.")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Call Groq chat completions with retry and JSON-response support."""
        payload: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.2),
            "max_tokens": kwargs.pop("max_tokens", 1600),
            "response_format": kwargs.pop("response_format", {"type": "json_object"}),
        }
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - keep retry simple at API boundary
                last_error = exc
                if attempt == self.max_retries:
                    break
                # 免费 API 偶尔限流, 指数退避比立刻重试稳定
                time.sleep(2 ** (attempt - 1))

        raise RuntimeError(f"Groq API request failed after {self.max_retries} attempts: {last_error}")
