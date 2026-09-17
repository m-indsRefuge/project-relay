"""Shared LangGraph state for Project Relay M1."""

from typing import Any, Literal, NotRequired, TypedDict

TerminalStatus = Literal[
    "running",
    "completed",
    "failed",
]


class RelayState(TypedDict):
    """Shared state carried through the deterministic and model-backed graph."""

    episode_id: str
    task: str

    # Retained from M0 to preserve interrupt/resume regression coverage.
    should_interrupt: bool

    route_reason: str | None
    terminal_status: TerminalStatus

    visited_nodes: list[str]
    model_trace: list[str]

    resume_value: NotRequired[str]

    scout_output: NotRequired[dict[str, Any]]
    retriever_output: NotRequired[dict[str, Any]]
    researcher_output: NotRequired[dict[str, Any]]
    synthesizer_output: NotRequired[dict[str, Any]]