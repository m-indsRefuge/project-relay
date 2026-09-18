"""Regression tests for preserving the user's requested deliverable."""

import json
from pathlib import Path

from relay.graph import open_graph
from relay.nodes.goal import EVALUATION_SYSTEM, GOAL_SYSTEM
from relay.nodes.schemas import GoalEvaluationOutput, GoalOutput


def make_state(*, episode_id: str, task: str) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "should_interrupt": False,
        "route_reason": None,
        "terminal_status": "running",
        "visited_nodes": [],
        "model_trace": [],
        "goal_iterations": 0,
    }


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_goal_contract_explicitly_preserves_requested_deliverable() -> None:
    assert "requested deliverable is authoritative" in GOAL_SYSTEM
    assert 'Do not transform "describe how to investigate X" into "investigate X"' in (
        GOAL_SYSTEM
    )


def test_evaluator_contract_distinguishes_describing_from_performing() -> None:
    assert "original user task is authoritative" in EVALUATION_SYSTEM
    assert "It does NOT require the described real-world action" in EVALUATION_SYSTEM

    normalized_contract = " ".join(EVALUATION_SYSTEM.split())
    assert "only when graph state contains evidence" in normalized_contract


def test_goal_evaluator_receives_original_user_task(
    tmp_path: Path,
    fake_model_client,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    task = (
        "Describe a cautious diagnostic approach using only the information supplied."
    )

    with open_graph(
        checkpoint,
        model_client=fake_model_client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-FIDELITY-001",
                task=task,
            ),
            make_config("thread-goal-fidelity"),
        )

    goal_call = next(
        call
        for call in fake_model_client.invocations
        if call["output_type"] is GoalOutput
    )
    evaluation_call = next(
        call
        for call in fake_model_client.invocations
        if call["output_type"] is GoalEvaluationOutput
    )

    assert task in goal_call["prompt"]

    evaluation_context = json.loads(evaluation_call["prompt"])

    assert evaluation_context["original_user_task"] == task
    assert evaluation_context["goal"] == result["goal"]
    assert evaluation_context["synthesizer_output"] == result["synthesizer_output"]