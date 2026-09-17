"""M1 graph behaviour tests."""

from pathlib import Path

from relay.config import RELAY_MODELS
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
    }


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_valid_task_runs_all_four_models(
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
                episode_id="E-M1-001",
                task="Exercise the four-model graph.",
            ),
            make_config("thread-valid"),
        )

    assert result["terminal_status"] == "completed"
    assert result["route_reason"] == "task is valid"
    assert result["model_trace"] == list(RELAY_MODELS)
    assert fake_model_client.calls == list(RELAY_MODELS)
    assert result["visited_nodes"] == [
        "initialize_state",
        "scout",
        "retriever",
        "researcher",
        "synthesizer",
        "complete",
    ]
    assert result["synthesizer_output"]["answer"]


def test_empty_task_fails_before_any_model_call(
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
                episode_id="E-M1-002",
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


def test_execution_receipt_records_model_path(
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
                episode_id="E-M1-003",
                task="Create an M1 receipt.",
            ),
            make_config("thread-receipt"),
        )

    receipt = build_receipt(result)

    assert receipt["terminal_status"] == "completed"
    assert receipt["model_trace"] == list(RELAY_MODELS)
    assert receipt["final_answer"] == "Proceed using the available bounded context."