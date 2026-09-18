"""Shared M2 test fixtures."""

from collections.abc import Generator
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from relay.config import (
    RESEARCHER_MODEL,
    RETRIEVER_MODEL,
    SCOUT_MODEL,
    SYNTHESIZER_MODEL,
)
from relay.nodes.schemas import (
    GoalEvaluationOutput,
    GoalOutput,
    GroundingAuditOutput,
    ResearcherOutput,
    RetrieverOutput,
    ScoutOutput,
    SynthesizerOutput,
)


@dataclass
class FakeModelClient:
    """Deterministic structured-model client for graph tests."""

    evaluation_statuses: list[str] = field(
        default_factory=lambda: ["satisfied"]
    )
    gap_routes: list[str] = field(default_factory=lambda: ["researcher"])

    grounding_assessments: list[str] = field(
        default_factory=lambda: ["grounded"]
    )
    grounding_issues: list[list[str]] = field(
        default_factory=lambda: [[]]
    )

    calls: list[str] = field(default_factory=list)
    invocations: list[dict] = field(default_factory=list)

    evaluation_count: int = 0
    grounding_count: int = 0

    def invoke(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        self.calls.append(model)
        self.invocations.append(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "output_type": output_type,
            }
        )

        if output_type is GoalOutput:
            return GoalOutput(
                objective="Produce a cautious answer to the supplied task.",
                success_criteria=[
                    "Address the task directly.",
                    "Preserve uncertainty where evidence is missing.",
                ],
            )

        if model == SCOUT_MODEL and output_type is ScoutOutput:
            return ScoutOutput(
                known=["The task is available."],
                questions=["What constraints matter?"],
                handoff="Organize the available context.",
            )

        if model == RETRIEVER_MODEL and output_type is RetrieverOutput:
            return RetrieverOutput(
                organized_context=[
                    "Task and Scout decomposition are available."
                ],
                gaps=["No external evidence is available in M2."],
                handoff="Analyze bounded hypotheses without external claims.",
            )

        if model == RESEARCHER_MODEL and output_type is ResearcherOutput:
            return ResearcherOutput(
                hypotheses=[
                    "A bounded diagnostic approach is appropriate."
                ],
                risks=["Evidence is intentionally limited in M2."],
                handoff="Synthesize only from supplied context.",
            )

        if model == SYNTHESIZER_MODEL and output_type is SynthesizerOutput:
            return SynthesizerOutput(
                answer="Proceed using the available bounded context.",
                supporting_points=[
                    "The answer preserves the M2 evidence boundary."
                ],
            )

        if model == SYNTHESIZER_MODEL and output_type is GroundingAuditOutput:
            index = min(
                self.grounding_count,
                len(self.grounding_assessments) - 1,
            )
            issue_index = min(
                self.grounding_count,
                len(self.grounding_issues) - 1,
            )

            assessment = self.grounding_assessments[index]
            issues = self.grounding_issues[issue_index]
            self.grounding_count += 1

            return GroundingAuditOutput(
                assessment=assessment,
                unsupported_claims=issues,
                rationale=(
                    "Fake grounding audit result for deterministic tests."
                ),
            )

        if model == SYNTHESIZER_MODEL and output_type is GoalEvaluationOutput:
            index = min(
                self.evaluation_count,
                len(self.evaluation_statuses) - 1,
            )
            status = self.evaluation_statuses[index]
            route_index = min(
                self.evaluation_count,
                len(self.gap_routes) - 1,
            )
            route = self.gap_routes[route_index]
            self.evaluation_count += 1

            if status == "satisfied":
                return GoalEvaluationOutput(
                    status="satisfied",
                    rationale="The candidate answer meets the fake criteria.",
                )

            if status == "failed":
                return GoalEvaluationOutput(
                    status="failed",
                    rationale="The fake evaluator cannot complete the goal.",
                )

            return GoalEvaluationOutput(
                status="missing_information",
                rationale="One more bounded reasoning pass is useful.",
                subgoal="Re-examine the main risk before finalizing.",
                route=route,
            )

        raise AssertionError(
            f"Unexpected fake model invocation: model={model!r}, "
            f"output_type={output_type.__name__!r}"
        )


@pytest.fixture
def fake_model_client() -> Generator[FakeModelClient, None, None]:
    yield FakeModelClient()