"""Shared M3 test fixtures."""

import json
from collections.abc import Generator
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from relay.config import RESEARCHER_MODEL, RETRIEVER_MODEL, SCOUT_MODEL, SYNTHESIZER_MODEL
from relay.nodes.schemas import (
    GoalEvaluationOutput,
    GoalOutput,
    GroundingAuditOutput,
    ResearcherOutput,
    RetrieverOutput,
    ScoutOutput,
    SourcePlanOutput,
    SynthesizerOutput,
)


@dataclass
class FakeRagSearch:
    calls: list[str] = field(default_factory=list)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        self.calls.append(query)
        return [
            {
                "source_id": "rag:fixture.md#chunk-000",
                "kind": "rag",
                "title": "Fixture Local Knowledge",
                "source": "knowledge/fixture.md",
                "content": "Relay uses LangGraph as its orchestration authority.",
                "score": 0.99,
                "retrieved_at": "2026-09-18T00:00:00+00:00",
            }
        ][:limit]


@dataclass
class FakeWebSearch:
    calls: list[str] = field(default_factory=list)

    def search(self, query: str, limit: int = 3) -> list[dict]:
        self.calls.append(query)
        return [
            {
                "source_id": "web:001",
                "kind": "web",
                "title": "Fixture LangGraph Docs",
                "source": "https://example.test/langgraph",
                "snippet": "LangGraph workflows use state, nodes, and edges.",
                "retrieved_at": "2026-09-18T00:00:00+00:00",
            }
        ][:limit]


@dataclass
class FakeModelClient:
    use_rag: bool = False
    use_web: bool = False
    evaluation_statuses: list[str] = field(default_factory=lambda: ["satisfied"])
    gap_routes: list[str] = field(default_factory=lambda: ["researcher"])
    grounding_assessments: list[str] = field(default_factory=lambda: ["grounded"])
    grounding_issues: list[list[str]] = field(default_factory=lambda: [[]])
    calls: list[str] = field(default_factory=list)
    invocations: list[dict] = field(default_factory=list)
    evaluation_count: int = 0
    grounding_count: int = 0

    def invoke(
        self, *, model: str, system: str, prompt: str, output_type: type[BaseModel]
    ) -> BaseModel:
        self.calls.append(model)
        self.invocations.append(
            {"model": model, "system": system, "prompt": prompt, "output_type": output_type}
        )
        if output_type is GoalOutput:
            return GoalOutput(
                objective="Produce a cautious answer to the supplied task.",
                success_criteria=["Address the task directly.", "Preserve uncertainty."],
            )
        if output_type is SourcePlanOutput:
            return SourcePlanOutput(
                use_rag=self.use_rag,
                rag_query="Relay architecture" if self.use_rag else None,
                use_web=self.use_web,
                web_query="current LangGraph graph API" if self.use_web else None,
                rationale="Deterministic source plan for tests.",
            )
        if model == SCOUT_MODEL and output_type is ScoutOutput:
            return ScoutOutput(
                known=["The task is available."],
                questions=["What constraints matter?"],
                handoff="Organize context.",
            )
        if model == RETRIEVER_MODEL and output_type is RetrieverOutput:
            return RetrieverOutput(
                organized_context=["Available evidence is organized."],
                gaps=["Keep unsupported claims explicit."],
                handoff="Analyze bounded evidence.",
            )
        if model == RESEARCHER_MODEL and output_type is ResearcherOutput:
            return ResearcherOutput(
                hypotheses=["A bounded interpretation is appropriate."],
                risks=["Do not confuse reasoning with evidence."],
                handoff="Synthesize supplied context.",
            )
        if model == SYNTHESIZER_MODEL and output_type is SynthesizerOutput:
            context = json.loads(prompt)
            citations = []
            for record in [*context.get("rag_results", []), *context.get("web_results", [])]:
                if record.get("source_id"):
                    citations.append(record["source_id"])
            return SynthesizerOutput(
                answer="Proceed using the available bounded evidence.",
                supporting_points=["The answer preserves the evidence boundary."],
                citations=citations,
            )
        if model == SYNTHESIZER_MODEL and output_type is GroundingAuditOutput:
            index = min(self.grounding_count, len(self.grounding_assessments) - 1)
            issue_index = min(self.grounding_count, len(self.grounding_issues) - 1)
            self.grounding_count += 1
            return GroundingAuditOutput(
                assessment=self.grounding_assessments[index],
                unsupported_claims=self.grounding_issues[issue_index],
                rationale="Deterministic grounding result.",
            )
        if model == SYNTHESIZER_MODEL and output_type is GoalEvaluationOutput:
            index = min(self.evaluation_count, len(self.evaluation_statuses) - 1)
            route_index = min(self.evaluation_count, len(self.gap_routes) - 1)
            self.evaluation_count += 1
            status = self.evaluation_statuses[index]
            if status == "satisfied":
                return GoalEvaluationOutput(
                    status="satisfied", rationale="Candidate meets fake criteria."
                )
            if status == "failed":
                return GoalEvaluationOutput(
                    status="failed", rationale="Fake evaluator cannot complete the goal."
                )
            return GoalEvaluationOutput(
                status="missing_information",
                rationale="One more pass is useful.",
                subgoal="Gather missing evidence.",
                route=self.gap_routes[route_index],
            )
        raise AssertionError(
            f"Unexpected fake model invocation: {model!r}, {output_type.__name__!r}"
        )


@pytest.fixture
def fake_model_client() -> Generator[FakeModelClient, None, None]:
    yield FakeModelClient()


@pytest.fixture
def fake_rag_search() -> Generator[FakeRagSearch, None, None]:
    yield FakeRagSearch()


@pytest.fixture
def fake_web_search() -> Generator[FakeWebSearch, None, None]:
    yield FakeWebSearch()
