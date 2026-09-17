"""Shared M1 test fixtures."""

from collections.abc import Generator

import pytest
from pydantic import BaseModel

from relay.config import (
    RESEARCHER_MODEL,
    RETRIEVER_MODEL,
    SCOUT_MODEL,
    SYNTHESIZER_MODEL,
)
from relay.nodes.schemas import (
    ResearcherOutput,
    RetrieverOutput,
    ScoutOutput,
    SynthesizerOutput,
)


class FakeModelClient:
    """Deterministic structured-model client for graph tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        del system, prompt
        self.calls.append(model)

        if model == SCOUT_MODEL and output_type is ScoutOutput:
            return ScoutOutput(
                known=["The task is available."],
                questions=["What constraints matter?"],
                handoff="Organize the available context.",
            )

        if model == RETRIEVER_MODEL and output_type is RetrieverOutput:
            return RetrieverOutput(
                organized_context=["Task and Scout decomposition are available."],
                gaps=["No external evidence is available in M1."],
                handoff="Analyze bounded hypotheses without external claims.",
            )

        if model == RESEARCHER_MODEL and output_type is ResearcherOutput:
            return ResearcherOutput(
                hypotheses=["A bounded diagnostic approach is appropriate."],
                risks=["Evidence is intentionally limited in M1."],
                handoff="Synthesize only from supplied context.",
            )

        if model == SYNTHESIZER_MODEL and output_type is SynthesizerOutput:
            return SynthesizerOutput(
                answer="Proceed using the available bounded context.",
                supporting_points=["All four M1 specialist nodes executed."],
            )

        raise AssertionError(
            f"Unexpected fake model invocation: model={model!r}, "
            f"output_type={output_type.__name__!r}"
        )


@pytest.fixture
def fake_model_client() -> Generator[FakeModelClient, None, None]:
    yield FakeModelClient()