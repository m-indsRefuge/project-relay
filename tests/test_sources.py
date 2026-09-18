"""M3 source planning, embedding, and retrieval tests."""

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from relay.nodes.schemas import SourcePlanOutput
from relay.nodes.sources import _resolve_source_plan
from relay.rag.embeddings import EmbeddingError, OllamaEmbeddingClient
from relay.rag.index import KnowledgeIndex


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(text.lower().count("langgraph")), float(text.lower().count("incident")), 1.0]
            for text in texts
        ]


def source_state() -> dict:
    return {
        "task": "Use local Relay docs and current public LangGraph information.",
        "goal": {
            "objective": "Compare Relay's architecture with current LangGraph documentation.",
            "success_criteria": ["Use both evidence classes."],
        },
    }


def test_source_plan_allows_missing_query_for_enabled_source() -> None:
    output = SourcePlanOutput(
        use_rag=True,
        rag_query=None,
        use_web=True,
        web_query=None,
        rationale="Both source classes are required.",
    )

    plan = _resolve_source_plan(output, source_state())

    expected = "Compare Relay's architecture with current LangGraph documentation."
    assert plan["rag_query"] == expected
    assert plan["web_query"] == expected


def test_source_plan_preserves_model_query_when_present() -> None:
    output = SourcePlanOutput(
        use_rag=True,
        rag_query="Relay orchestration architecture",
        use_web=True,
        web_query="current LangGraph workflow state nodes edges",
        rationale="Use focused queries.",
    )

    plan = _resolve_source_plan(output, source_state())

    assert plan["rag_query"] == "Relay orchestration architecture"
    assert plan["web_query"] == "current LangGraph workflow state nodes edges"


def test_source_plan_rejects_query_for_disabled_source() -> None:
    with pytest.raises(ValidationError):
        SourcePlanOutput(
            use_rag=False,
            rag_query="bad",
            use_web=False,
            web_query=None,
            rationale="Invalid",
        )


def test_ollama_embed_batches_and_validates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    client = OllamaEmbeddingClient(transport=httpx.MockTransport(handler))
    assert client.embed(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_ollama_embed_rejects_wrong_vector_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    client = OllamaEmbeddingClient(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="count"):
        client.embed(["alpha", "beta"])


def test_knowledge_index_returns_semantic_top_hit(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "graph.md").write_text(
        "# Graph Notes\n\nLangGraph coordinates state, nodes, and edges.",
        encoding="utf-8",
    )
    (knowledge / "incident.md").write_text(
        "# Incident Notes\n\nIncident response starts with evidence.",
        encoding="utf-8",
    )
    index = KnowledgeIndex(knowledge_dir=knowledge, embedder=FakeEmbedder())
    results = index.search("How does LangGraph work?", limit=1)
    assert results[0]["source_id"].startswith("rag:graph.md#chunk-")
    assert results[0]["kind"] == "rag"
