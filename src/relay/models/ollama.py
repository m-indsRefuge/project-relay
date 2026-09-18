"""Minimal direct Ollama adapter for Project Relay M1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelInvocationError(RuntimeError):
    """Raised when a local model invocation cannot produce valid output."""


class StructuredModelClient(Protocol):
    """Small model-client contract required by LangGraph nodes."""

    def invoke(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> OutputT:
        """Invoke one model and validate its structured output."""


@dataclass(frozen=True)
class OllamaClient:
    """Direct client for Ollama's local HTTP API."""

    host: str = "http://localhost:11434"
    timeout_seconds: float = 180.0
    keep_alive: int = 0
    transport: httpx.BaseTransport | None = None

    def invoke(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> OutputT:
        """Generate and validate one structured response."""

        request = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": output_type.model_json_schema(),
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.2,
                "num_predict": 384,
            },
        }

        try:
            with httpx.Client(
                base_url=self.host,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post("/api/generate", json=request)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelInvocationError(
                f"Ollama request failed for model {model!r}: {exc}"
            ) from exc

        raw = payload.get("response")

        if not isinstance(raw, str) or not raw.strip():
            raise ModelInvocationError(
                f"Ollama returned no structured response for model {model!r}."
            )

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelInvocationError(
                f"Model {model!r} returned malformed JSON."
            ) from exc

        try:
            return output_type.model_validate(decoded)
        except ValidationError as exc:
            raise ModelInvocationError(
                f"Model {model!r} returned JSON that violates the required schema."
            ) from exc

    def list_models(self) -> set[str]:
        """Return model names currently reported by Ollama."""

        try:
            with httpx.Client(
                base_url=self.host,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelInvocationError(
                f"Unable to query Ollama model inventory: {exc}"
            ) from exc

        models = payload.get("models", [])

        return {
            item["name"]
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

    def ensure_models_available(self, required_models: tuple[str, ...]) -> None:
        """Fail before execution if any required Relay model is unavailable."""

        available = self.list_models()
        missing = [model for model in required_models if model not in available]

        if missing:
            joined = ", ".join(missing)
            raise ModelInvocationError(
                f"Required Ollama model(s) not installed or not visible: {joined}"
            )