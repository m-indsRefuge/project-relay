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
    source_trace: list[str]
    goal: dict[str, Any] | None
    goal_iterations: int
    source_plan: dict[str, Any] | None
    citations: list[str]
    final_answer: str | None


def build_receipt(state: RelayState) -> ExecutionReceipt:
    synthesis: dict[str, Any] = state.get("synthesizer_output", {})
    answer = synthesis.get("answer")
    citations = synthesis.get("citations", [])
    return {
        "episode_id": state["episode_id"],
        "task": state["task"],
        "terminal_status": state["terminal_status"],
        "route_reason": state.get("route_reason"),
        "visited_nodes": list(state.get("visited_nodes", [])),
        "model_trace": list(state.get("model_trace", [])),
        "source_trace": list(state.get("source_trace", [])),
        "goal": state.get("goal"),
        "goal_iterations": state.get("goal_iterations", 0),
        "source_plan": state.get("source_plan"),
        "citations": list(citations) if isinstance(citations, list) else [],
        "final_answer": answer if isinstance(answer, str) else None,
    }
