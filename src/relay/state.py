"""LangGraph state definitions for Project Relay.

M0 deliberately starts with a very small state surface. Later milestones
extend this schema only when new graph capabilities require additional state.
"""

from typing import Literal, TypedDict


TerminalStatus = Literal[
    "running",
    "completed",
    "failed",
]


class RelayState(TypedDict):
    """Minimal M0 shared graph state."""

    episode_id: str
    task: str
    valid: bool
    route_reason: str | None
    terminal_status: TerminalStatus