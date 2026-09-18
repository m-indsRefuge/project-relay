"""Minimal Ollama embedding adapter for Relay RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from relay.config import EMBEDDING_MODEL


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or is malformed."""


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass(frozen=True)
class OllamaEmbeddingClient:
    host: str = "http://localhost:11434"
    model: str = EMBEDDING_MODEL
    timeout_seconds: float = 180.0
    transport: httpx.BaseTransport | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with httpx.Client(
                base_url=self.host, timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = client.post(
                    "/api/embed",
                    json={
                        "model": self.model,
                        "input": texts,
                        "truncate": True,
                        "keep_alive": "0s",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed for {self.model!r}: {exc}"
            ) from exc
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError("Ollama embedding response count does not match input count.")
        normalized: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Ollama returned an invalid embedding vector.")
            try:
                normalized.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    "Ollama embedding vector contains a non-numeric value."
                ) from exc
        return normalized
