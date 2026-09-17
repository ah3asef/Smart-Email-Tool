"""Small, replaceable client for Ollama's local chat API."""

from __future__ import annotations

import json
import socket
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from helpers.config import get_settings


class OllamaError(RuntimeError):
    """Base error for safe, user-facing Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unusable generation response."""


class OllamaClient:
    """Call ``POST /api/chat`` with configurable RAG generation options."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = settings.ollama_temperature if temperature is None else temperature
        self.top_p = settings.ollama_top_p if top_p is None else top_p
        self.num_predict = settings.ollama_num_predict if num_predict is None else num_predict
        self.timeout = settings.ollama_timeout if timeout is None else timeout

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate one non-streaming chat response from Ollama."""
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8")
        except socket.timeout as exc:
            raise OllamaConnectionError("Ollama generation timed out") from exc
        except HTTPError as exc:
            raise OllamaConnectionError("Ollama rejected the generation request") from exc
        except URLError as exc:
            raise OllamaConnectionError("Could not connect to Ollama") from exc

        try:
            response_data: Any = json.loads(raw_response)
            content = response_data["message"]["content"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaResponseError("Ollama returned an invalid response") from exc

        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("Ollama returned an empty response")
        return content.strip()
