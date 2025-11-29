"""
LLM integration for docpack workflow planning.

Local-first approach using Ollama with fallback to OpenAI-compatible APIs.
Default model: qwen3:4b (Qwen3-4B-Thinking-2507) for excellent tool use.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


# Default model - bleeding edge 4B with strong reasoning
DEFAULT_MODEL = "qwen3:4b"


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    raw_response: Any = None


class LLMError(Exception):
    """Error from LLM call."""

    pass


def get_llm_client():
    """Get the appropriate LLM client based on environment."""
    # Check for custom API endpoint (OpenAI-compatible)
    base_url = os.environ.get("DOCPACK_LLM_BASE_URL")
    api_key = os.environ.get("DOCPACK_LLM_API_KEY")

    if base_url and api_key:
        return OpenAICompatibleClient(base_url, api_key)

    # Default to Ollama
    return OllamaClient()


class OllamaClient:
    """Client for local Ollama LLM."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        """Send a chat request to Ollama."""
        try:
            from ollama import chat
        except ImportError:
            raise LLMError(
                "Ollama package not installed. Install with: pip install ollama\n"
                "Also ensure Ollama is running: ollama serve"
            )

        model = model or self.model

        try:
            kwargs = {"model": model, "messages": messages}
            if format:
                kwargs["format"] = format

            response = chat(**kwargs)
            return LLMResponse(
                content=response.message.content,
                model=model,
                raw_response=response,
            )
        except Exception as e:
            raise LLMError(f"Ollama error: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            from ollama import list as ollama_list

            ollama_list()
            return True
        except Exception:
            return False


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs."""

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        format: str | None = None,
    ) -> LLMResponse:
        """Send a chat request to OpenAI-compatible API."""
        try:
            import httpx
        except ImportError:
            raise LLMError("httpx package not installed. Install with: pip install httpx")

        model = model or self.model

        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "response_format": {"type": "json_object"} if format == "json" else None,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=model,
                    raw_response=data,
                )
        except Exception as e:
            raise LLMError(f"API error: {e}") from e

    def is_available(self) -> bool:
        """Check if API is available."""
        try:
            import httpx

            with httpx.Client() as client:
                response = client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False


def chat_with_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    format: str | None = None,
) -> LLMResponse:
    """
    Send a chat request to the configured LLM.

    Uses Ollama by default, falls back to OpenAI-compatible API if configured.

    Args:
        messages: Chat messages in OpenAI format
        model: Model to use (default: qwen3:4b for Ollama)
        format: Response format ("json" for structured output)

    Returns:
        LLMResponse with the model's response
    """
    client = get_llm_client()
    return client.chat(messages, model=model, format=format)


def is_llm_available() -> bool:
    """Check if any LLM is available."""
    client = get_llm_client()
    return client.is_available()
