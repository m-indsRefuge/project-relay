"""Focused Qwen grounding audit for M3."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import GroundingAuditOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Grounding Auditor.
Audit the candidate against the original task, local RAG evidence, and web evidence.
A model hypothesis is not evidence. RAG claims must be supportable by RAG passages.
External/current claims must be supportable by supplied web results. Recommendations are allowed
when clearly framed. Source IDs must refer to actual evidence records.
Return `grounded` only when there are no material unsupported factual claims.
Return only the required structured output."""


def _evidence_source_ids(state: RelayState) -> set[str]:
    return {
        str(record["source_id"])
        for record in [*state.get("rag_results", []), *state.get("web_results", [])]
        if isinstance(record, dict) and record.get("source_id")
    }


def make_grounding_auditor_node(client: StructuredModelClient):
    def grounding_audit(state: RelayState) -> dict:
        evidence_ids = _evidence_source_ids(state)
        rag_ids = {
            record["source_id"]
            for record in state.get("rag_results", [])
            if record.get("source_id")
        }
        web_ids = {
            record["source_id"]
            for record in state.get("web_results", [])
            if record.get("source_id")
        }
        citations = set(state.get("synthesizer_output", {}).get("citations", []))
        deterministic_issues: list[str] = []
        unknown = sorted(citations - evidence_ids)
        if unknown:
            deterministic_issues.append("Unknown citation IDs: " + ", ".join(unknown))
        if rag_ids and not citations.intersection(rag_ids):
            deterministic_issues.append("RAG evidence was supplied but no RAG source ID was cited.")
        if web_ids and not citations.intersection(web_ids):
            deterministic_issues.append("Web evidence was supplied but no web source ID was cited.")
        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(
                {
                    "original_user_task": state["task"],
                    "rag_results": state.get("rag_results", []),
                    "web_results": state.get("web_results", []),
                    "candidate": state["synthesizer_output"],
                },
                ensure_ascii=False,
            ),
            output_type=GroundingAuditOutput,
        )
        if deterministic_issues:
            output = GroundingAuditOutput(
                assessment="needs_revision",
                unsupported_claims=deterministic_issues,
                rationale="Deterministic source-provenance validation failed.",
            )
        return {
            "grounding_audit": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "grounding_audit"],
        }

    return grounding_audit
