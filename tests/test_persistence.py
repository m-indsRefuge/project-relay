"""M0 checkpoint persistence and isolation tests."""

from pathlib import Path

from langgraph.types import Command

from relay.graph import open_graph


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


def test_checkpoint_survives_database_reopen(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    config = make_config("persistent-thread")

    with open_graph(checkpoint) as graph:
        graph.invoke(
            make_state(
                episode_id="E-M0-010",
                task="Persist this state.",
            ),
            config,
        )

    with open_graph(checkpoint) as graph:
        snapshot = graph.get_state(config)

    assert snapshot.values["episode_id"] == "E-M0-010"
    assert snapshot.values["task"] == "Persist this state."
    assert snapshot.values["terminal_status"] == "completed"


def test_threads_do_not_contaminate_each_other(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    config_a = make_config("thread-a")
    config_b = make_config("thread-b")

    with open_graph(checkpoint) as graph:
        graph.invoke(
            make_state(
                episode_id="E-M0-A",
                task="State belonging to A.",
            ),
            config_a,
        )

        graph.invoke(
            make_state(
                episode_id="E-M0-B",
                task="State belonging to B.",
            ),
            config_b,
        )

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)

    assert state_a.values["episode_id"] == "E-M0-A"
    assert state_a.values["task"] == "State belonging to A."

    assert state_b.values["episode_id"] == "E-M0-B"
    assert state_b.values["task"] == "State belonging to B."


def test_interrupt_can_resume_after_database_reopen(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    config = make_config("interrupt-thread")

    with open_graph(checkpoint) as graph:
        events = list(
            graph.stream(
                make_state(
                    episode_id="E-M0-020",
                    task="Exercise checkpointed resume.",
                    should_interrupt=True,
                ),
                config,
            )
        )

        assert any("__interrupt__" in event for event in events)

        snapshot = graph.get_state(config)

        assert snapshot.next
        assert "pause_for_resume" in snapshot.next

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            Command(resume="M0 resume accepted"),
            config,
        )

    assert result["terminal_status"] == "completed"
    assert result["resume_value"] == "M0 resume accepted"
    assert result["route_reason"] == "checkpointed interrupt resumed"
    assert result["visited_nodes"] == [
        "initialize_state",
        "pause_for_resume",
        "complete",
    ]