"""Structured outputs used by Project Relay model-backed nodes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictOutput(BaseModel):
    """Base class that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class GoalOutput(StrictOutput):
    """Qwen goal formation output."""

    objective: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)


class ScoutOutput(StrictOutput):
    """Phi Scout decomposition."""

    known: list[str]
    questions: list[str]
    handoff: str = Field(min_length=1)


class RetrieverOutput(StrictOutput):
    """Ministral context organization."""

    organized_context: list[str]
    gaps: list[str]
    handoff: str = Field(min_length=1)


class ResearcherOutput(StrictOutput):
    """Gemma analysis of the supplied context."""

    hypotheses: list[str]
    risks: list[str]
    handoff: str = Field(min_length=1)


class SynthesizerOutput(StrictOutput):
    """Qwen synthesis."""

    answer: str = Field(min_length=1)
    supporting_points: list[str]


class GroundingAuditOutput(StrictOutput):
    """Focused audit of candidate factual grounding."""

    assessment: Literal["grounded", "needs_revision"]
    unsupported_claims: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grounding_decision(self) -> "GroundingAuditOutput":
        if self.assessment == "grounded" and self.unsupported_claims:
            raise ValueError(
                "grounded assessment cannot contain unsupported claims"
            )

        if self.assessment == "needs_revision" and not self.unsupported_claims:
            raise ValueError(
                "needs_revision requires at least one unsupported claim"
            )

        return self


class GoalEvaluationOutput(StrictOutput):
    """Qwen decision about whether the active goal is satisfied."""

    status: Literal["satisfied", "missing_information", "failed"]
    rationale: str = Field(min_length=1)
    subgoal: str | None = None
    route: Literal[
        "scout",
        "retriever",
        "researcher",
        "synthesizer",
    ] | None = None

    @model_validator(mode="after")
    def validate_goal_decision(self) -> "GoalEvaluationOutput":
        if self.status == "missing_information":
            if not self.subgoal or self.route is None:
                raise ValueError(
                    "missing_information requires both subgoal and route"
                )
        elif self.subgoal is not None or self.route is not None:
            raise ValueError(
                "satisfied/failed decisions must not include subgoal or route"
            )

        return self