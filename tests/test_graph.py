"""M0 graph behaviour tests."""

from pathlib import Path

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
    }


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_valid_task_reaches_completed_terminal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-001",
                task="Prove the valid M0 route.",
            ),
            make_config("thread-valid"),
        )

    assert result["terminal_status"] == "completed"
    assert result["route_reason"] == "task is valid"
    assert result["visited_nodes"] == [
        "initialize_state",
        "complete",
    ]


def test_empty_task_reaches_failed_terminal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-002",
                task="   ",
            ),
            make_config("thread-invalid"),
        )

    assert result["terminal_status"] == "failed"
    assert result["route_reason"] == "task is empty"
    assert result["visited_nodes"] == [
        "initialize_state",
        "fail",
    ]


def test_execution_receipt_is_derived_from_terminal_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-003",
                task="Create an execution receipt.",
            ),
            make_config("thread-receipt"),
        )

    receipt = build_receipt(result)

    assert receipt == {
        "episode_id": "E-M0-003",
        "task": "Create an execution receipt.",
        "terminal_status": "completed",
        "route_reason": "task is valid",
        "visited_nodes": [
            "initialize_state",
            "complete",
        ],
    }