"""Deterministic execution receipt for Project Relay."""

from typing import Any, TypedDict

from relay.state import RelayState


class ExecutionReceipt(TypedDict):
    episode_id: str
    task: str
    terminal_status: str
    route_reason: str | None
    visited_nodes: list[str]
    model_trace: list[str]
    final_answer: str | None


def build_receipt(state: RelayState) -> ExecutionReceipt:
    """Create the operator-facing receipt from terminal graph state."""

    synthesis: dict[str, Any] = state.get("synthesizer_output", {})

    answer = synthesis.get("answer")
    final_answer = answer if isinstance(answer, str) else None

    return {
        "episode_id": state["episode_id"],
        "task": state["task"],
        "terminal_status": state["terminal_status"],
        "route_reason": state.get("route_reason"),
        "visited_nodes": list(state.get("visited_nodes", [])),
        "model_trace": list(state.get("model_trace", [])),
        "final_answer": final_answer,
    }