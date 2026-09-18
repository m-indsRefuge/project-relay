"""Structured outputs used by Project Relay model-backed nodes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalOutput(StrictOutput):
    objective: str = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)


class SourcePlanOutput(StrictOutput):
    use_rag: bool
    rag_query: str | None = None
    use_web: bool
    web_query: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disabled_source_queries(self) -> "SourcePlanOutput":
        if not self.use_rag and self.rag_query is not None:
            raise ValueError("rag_query must be null when use_rag is false")
        if not self.use_web and self.web_query is not None:
            raise ValueError("web_query must be null when use_web is false")
        return self


class ScoutOutput(StrictOutput):
    known: list[str]
    questions: list[str]
    handoff: str = Field(min_length=1)


class RetrieverOutput(StrictOutput):
    organized_context: list[str]
    gaps: list[str]
    handoff: str = Field(min_length=1)


class ResearcherOutput(StrictOutput):
    hypotheses: list[str]
    risks: list[str]
    handoff: str = Field(min_length=1)


class SynthesizerOutput(StrictOutput):
    answer: str = Field(min_length=1)
    supporting_points: list[str]
    citations: list[str]


class GroundingAuditOutput(StrictOutput):
    assessment: Literal["grounded", "needs_revision"]
    unsupported_claims: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grounding_decision(self) -> "GroundingAuditOutput":
        if self.assessment == "grounded" and self.unsupported_claims:
            raise ValueError("grounded assessment cannot contain unsupported claims")
        if self.assessment == "needs_revision" and not self.unsupported_claims:
            raise ValueError("needs_revision requires at least one unsupported claim")
        return self


class GoalEvaluationOutput(StrictOutput):
    status: Literal["satisfied", "missing_information", "failed"]
    rationale: str = Field(min_length=1)
    subgoal: str | None = None
    route: Literal["scout", "retriever", "researcher", "synthesizer", "rag", "web"] | None = None

    @model_validator(mode="after")
    def validate_goal_decision(self) -> "GoalEvaluationOutput":
        if self.status == "missing_information":
            if not self.subgoal or self.route is None:
                raise ValueError("missing_information requires both subgoal and route")
        elif self.subgoal is not None or self.route is not None:
            raise ValueError("satisfied/failed decisions must not include subgoal or route")
        return self
