"""M2 graph behaviour tests."""

from pathlib import Path

from conftest import FakeModelClient

from relay.config import (
    RESEARCHER_MODEL,
    RETRIEVER_MODEL,
    SCOUT_MODEL,
    SYNTHESIZER_MODEL,
)
from relay.graph import open_graph
from relay.receipt import build_receipt


def make_state(
    *,
    episode_id: str,
    task: str,
    should_interrupt: bool = False,
) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "should_interrupt": should_interrupt,
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


def test_satisfied_goal_runs_grounded_goal_path(
    tmp_path: Path,
    fake_model_client,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(
        checkpoint,
        model_client=fake_model_client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-001",
                task="Exercise M2 goal pursuit.",
            ),
            make_config("thread-valid"),
        )

    assert result["terminal_status"] == "completed"
    assert result["route_reason"] == "goal satisfied"
    assert result["goal_iterations"] == 1
    assert result["goal"]["objective"]
    assert result["grounding_audit"]["assessment"] == "grounded"

    assert result["visited_nodes"] == [
        "initialize_state",
        "form_goal",
        "scout",
        "retriever",
        "researcher",
        "synthesizer",
        "grounding_audit",
        "evaluate_goal",
        "complete",
    ]

    assert result["model_trace"] == [
        SYNTHESIZER_MODEL,
        SCOUT_MODEL,
        RETRIEVER_MODEL,
        RESEARCHER_MODEL,
        SYNTHESIZER_MODEL,
        SYNTHESIZER_MODEL,
        SYNTHESIZER_MODEL,
    ]


def test_empty_task_fails_before_goal_or_model_call(
    tmp_path: Path,
    fake_model_client,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(
        checkpoint,
        model_client=fake_model_client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-002",
                task="   ",
            ),
            make_config("thread-invalid"),
        )

    assert result["terminal_status"] == "failed"
    assert result["route_reason"] == "task is empty"
    assert result["model_trace"] == []
    assert fake_model_client.calls == []
    assert result["visited_nodes"] == [
        "initialize_state",
        "fail",
    ]


def test_goal_gap_can_route_back_to_researcher(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    client = FakeModelClient(
        evaluation_statuses=[
            "missing_information",
            "satisfied",
        ],
        gap_routes=["researcher"],
    )

    with open_graph(
        checkpoint,
        model_client=client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-003",
                task="Require one bounded additional pass.",
            ),
            make_config("thread-loop"),
        )

    assert result["terminal_status"] == "completed"
    assert result["goal_iterations"] == 2
    assert result["active_subgoal"] is None

    assert result["visited_nodes"] == [
        "initialize_state",
        "form_goal",
        "scout",
        "retriever",
        "researcher",
        "synthesizer",
        "grounding_audit",
        "evaluate_goal",
        "researcher",
        "synthesizer",
        "grounding_audit",
        "evaluate_goal",
        "complete",
    ]


def test_goal_gap_can_route_back_to_scout(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    client = FakeModelClient(
        evaluation_statuses=[
            "missing_information",
            "satisfied",
        ],
        gap_routes=["scout"],
    )

    with open_graph(
        checkpoint,
        model_client=client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-004",
                task="Require a complete second reasoning pass.",
            ),
            make_config("thread-loop-scout"),
        )

    assert result["terminal_status"] == "completed"
    assert result["goal_iterations"] == 2
    assert result["visited_nodes"].count("scout") == 2
    assert result["visited_nodes"].count("retriever") == 2
    assert result["visited_nodes"].count("researcher") == 2
    assert result["visited_nodes"].count("grounding_audit") == 2


def test_goal_iteration_budget_fails_closed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    client = FakeModelClient(
        evaluation_statuses=[
            "missing_information",
            "missing_information",
            "missing_information",
        ],
        gap_routes=["researcher"],
    )

    with open_graph(
        checkpoint,
        model_client=client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-005",
                task="Never satisfy the fake goal.",
            ),
            make_config("thread-budget"),
        )

    assert result["terminal_status"] == "failed"
    assert result["goal_iterations"] == 3
    assert result["route_reason"] == "goal iteration budget exhausted"
    assert result["visited_nodes"][-1] == "fail"


def test_explicit_goal_failure_terminates(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    client = FakeModelClient(
        evaluation_statuses=["failed"],
    )

    with open_graph(
        checkpoint,
        model_client=client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-006",
                task="Exercise explicit failure.",
            ),
            make_config("thread-failed-goal"),
        )

    assert result["terminal_status"] == "failed"
    assert result["goal_iterations"] == 1
    assert result["route_reason"] == "goal evaluation failed"


def test_grounding_audit_forces_synthesizer_revision(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    client = FakeModelClient(
        evaluation_statuses=[
            "satisfied",
            "satisfied",
        ],
        grounding_assessments=[
            "needs_revision",
            "grounded",
        ],
        grounding_issues=[
            ["Unsupported causal claim."],
            [],
        ],
    )

    with open_graph(
        checkpoint,
        model_client=client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-007",
                task="Force one grounding revision.",
            ),
            make_config("thread-grounding-revision"),
        )

    assert result["terminal_status"] == "completed"
    assert result["goal_iterations"] == 2
    assert result["visited_nodes"].count("synthesizer") == 2
    assert result["visited_nodes"].count("grounding_audit") == 2
    assert result["grounding_audit"]["assessment"] == "grounded"


def test_execution_receipt_records_goal_and_iterations(
    tmp_path: Path,
    fake_model_client,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(
        checkpoint,
        model_client=fake_model_client,
    ) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M2-008",
                task="Create an M2 receipt.",
            ),
            make_config("thread-receipt"),
        )

    receipt = build_receipt(result)

    assert receipt["terminal_status"] == "completed"
    assert receipt["goal"] is not None
    assert receipt["goal_iterations"] == 1
    assert receipt["final_answer"] == "Proceed using the available bounded context."