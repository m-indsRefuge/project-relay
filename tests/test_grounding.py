"""Regression tests for M2 evidence-boundary contracts."""

from relay.nodes.goal import EVALUATION_SYSTEM
from relay.nodes.researcher import SYSTEM as RESEARCHER_SYSTEM
from relay.nodes.retriever import SYSTEM as RETRIEVER_SYSTEM
from relay.nodes.scout import SYSTEM as SCOUT_SYSTEM
from relay.nodes.synthesizer import SYSTEM as SYNTHESIZER_SYSTEM


def normalize(text: str) -> str:
    """Normalize formatting so tests validate semantics, not Markdown punctuation."""

    return " ".join(text.replace("`", "").split())


def test_scout_known_is_limited_to_supplied_facts() -> None:
    contract = normalize(SCOUT_SYSTEM)

    assert "known may contain only facts explicitly supplied" in contract
    assert "does NOT establish that Y was the only recent event" in contract
    assert "possible causes" in contract


def test_retriever_must_not_promote_inference_to_fact() -> None:
    contract = normalize(RETRIEVER_SYSTEM)

    assert "original task is the authoritative source of supplied facts" in contract
    assert "Do not promote Scout questions, implications, or inferred claims" in contract


def test_researcher_keeps_hypotheses_nonfactual() -> None:
    contract = normalize(RESEARCHER_SYSTEM)

    assert "hypotheses are possibilities, not established facts" in contract
    assert "Do not convert temporal correlation into causation" in contract


def test_synthesizer_rejects_unsupported_supporting_points() -> None:
    contract = normalize(SYNTHESIZER_SYSTEM)

    assert "supporting_points must not contain unsupported real-world factual claims" in (
        contract
    )
    assert "original user task is authoritative for supplied facts" in contract


def test_goal_evaluator_respects_grounding_audit_authority() -> None:
    contract = normalize(EVALUATION_SYSTEM)

    assert "focused grounding audit is authoritative" in contract
    assert "goal cannot be satisfied" in contract
    assert "until the Synthesizer revises them" in contract