"""Minimal deterministic M0 execution receipt."""

from typing import TypedDict

from relay.state import RelayState


class ExecutionReceipt(TypedDict):
    episode_id: str
    task: str
    terminal_status: str
    route_reason: str | None
    visited_nodes: list[str]


def build_receipt(state: RelayState) -> ExecutionReceipt:
    """Create the small operator-facing receipt required by M0."""

    return {
        "episode_id": state["episode_id"],
        "task": state["task"],
        "terminal_status": state["terminal_status"],
        "route_reason": state.get("route_reason"),
        "visited_nodes": list(state.get("visited_nodes", [])),
    }