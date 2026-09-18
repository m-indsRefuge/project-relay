"""Deterministic provenance tests for the M3 grounding boundary."""

from conftest import FakeModelClient

from relay.nodes.grounding import make_grounding_auditor_node


def make_state(*, citations: list[str]) -> dict:
    return {
        "task": "Compare local Relay knowledge with current web evidence.",
        "rag_results": [
            {
                "source_id": "rag:fixture.md#chunk-000",
                "kind": "rag",
                "title": "Fixture local knowledge",
                "source": "knowledge/fixture.md",
                "content": "Relay uses LangGraph.",
                "retrieved_at": "2026-09-18T00:00:00+00:00",
            }
        ],
        "web_results": [
            {
                "source_id": "web:001",
                "kind": "web",
                "title": "Fixture web evidence",
                "source": "https://example.test/langgraph",
                "snippet": "Current public LangGraph information.",
                "retrieved_at": "2026-09-18T00:00:00+00:00",
            }
        ],
        "synthesizer_output": {
            "answer": "Candidate answer.",
            "supporting_points": ["Candidate point."],
            "citations": citations,
        },
        "model_trace": [],
        "visited_nodes": [],
    }


def test_unknown_citation_forces_revision() -> None:
    node = make_grounding_auditor_node(FakeModelClient())

    result = node(make_state(citations=["web:not-real"]))

    assert result["grounding_audit"]["assessment"] == "needs_revision"
    assert any(
        issue.startswith("Unknown citation IDs:")
        for issue in result["grounding_audit"]["unsupported_claims"]
    )


def test_missing_web_citation_forces_revision() -> None:
    node = make_grounding_auditor_node(FakeModelClient())

    result = node(make_state(citations=["rag:fixture.md#chunk-000"]))

    assert result["grounding_audit"]["assessment"] == "needs_revision"
    assert (
        "Web evidence was supplied but no web source ID was cited."
        in result["grounding_audit"]["unsupported_claims"]
    )


def test_rag_and_web_citations_can_pass_provenance_gate() -> None:
    node = make_grounding_auditor_node(FakeModelClient())

    result = node(
        make_state(
            citations=[
                "rag:fixture.md#chunk-000",
                "web:001",
            ]
        )
    )

    assert result["grounding_audit"]["assessment"] == "grounded"
    assert result["grounding_audit"]["unsupported_claims"] == []
