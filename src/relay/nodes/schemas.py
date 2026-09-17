"""Structured outputs used by the four M1 model-backed nodes."""

from pydantic import BaseModel, ConfigDict, Field


class StrictOutput(BaseModel):
    """Base class that rejects unexpected fields."""

    model_config = ConfigDict(extra="forbid")


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
    """Gemma analysis of the supplied M1 context."""

    hypotheses: list[str]
    risks: list[str]
    handoff: str = Field(min_length=1)


class SynthesizerOutput(StrictOutput):
    """Qwen final M1 synthesis."""

    answer: str = Field(min_length=1)
    supporting_points: list[str]