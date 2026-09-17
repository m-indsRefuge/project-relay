"""Shared LangGraph state for Project Relay M0."""

from typing import Literal, NotRequired, TypedDict

TerminalStatus = Literal[
    "running",
    "completed",
    "failed",
]


class RelayState(TypedDict):
    """Minimal shared state required by the M0 graph."""

    episode_id: str
    task: str
    should_interrupt: bool
    route_reason: str | None
    terminal_status: TerminalStatus
    visited_nodes: list[str]
    resume_value: NotRequired[str]