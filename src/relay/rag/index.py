"""Small deterministic in-memory vector index for Relay M3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from relay.rag.embeddings import EmbeddingClient


class RagError(RuntimeError):
    """Raised when the local knowledge index cannot be used safely."""


class RagSearch(Protocol):
    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Return the top local knowledge passages."""


@dataclass(frozen=True)
class KnowledgeChunk:
    source_id: str
    title: str
    source: str
    content: str


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem.replace("_", " ").title()


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        projected = size + len(paragraph) + (2 if current else 0)
        if current and projected > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            size = len(paragraph)
        else:
            current.append(paragraph)
            size = projected
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RagError("Embedding dimensions do not match.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class KnowledgeIndex:
    knowledge_dir: Path
    embedder: EmbeddingClient
    _chunks: list[KnowledgeChunk] = field(default_factory=list, init=False)
    _vectors: list[list[float]] = field(default_factory=list, init=False)
    _built: bool = field(default=False, init=False)

    def _build(self) -> None:
        paths = sorted(self.knowledge_dir.glob("*.md"))
        if not paths:
            raise RagError(f"No Markdown knowledge documents found in {self.knowledge_dir}.")
        chunks: list[KnowledgeChunk] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            title = _title_for(path, text)
            for index, content in enumerate(_chunk_text(text)):
                chunks.append(
                    KnowledgeChunk(
                        source_id=f"rag:{path.name}#chunk-{index:03d}",
                        title=title,
                        source=str(path),
                        content=content,
                    )
                )
        vectors = self.embedder.embed([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RagError("RAG embedding count does not match chunk count.")
        self._chunks = chunks
        self._vectors = vectors
        self._built = True

    def search(self, query: str, limit: int = 3) -> list[dict]:
        if not query.strip():
            raise RagError("RAG query cannot be empty.")
        if limit < 1:
            raise RagError("RAG result limit must be at least one.")
        if not self._built:
            self._build()
        query_vector = self.embedder.embed([query])[0]
        ranked = sorted(
            (
                (_cosine(query_vector, vector), chunk)
                for chunk, vector in zip(self._chunks, self._vectors, strict=True)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        retrieved_at = datetime.now(UTC).isoformat()
        return [
            {
                "source_id": chunk.source_id,
                "kind": "rag",
                "title": chunk.title,
                "source": chunk.source,
                "content": chunk.content,
                "score": score,
                "retrieved_at": retrieved_at,
            }
            for score, chunk in ranked[:limit]
        ]
