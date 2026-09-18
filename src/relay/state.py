"""Shared LangGraph state for Project Relay M3."""

from typing import Any, Literal, NotRequired, TypedDict

TerminalStatus = Literal["running", "completed", "failed"]


class RelayState(TypedDict):
    episode_id: str
    task: str
    should_interrupt: bool
    route_reason: str | None
    terminal_status: TerminalStatus
    visited_nodes: list[str]
    model_trace: list[str]
    source_trace: list[str]
    resume_value: NotRequired[str]
    goal: NotRequired[dict[str, Any]]
    active_subgoal: NotRequired[dict[str, Any] | None]
    goal_evaluation: NotRequired[dict[str, Any]]
    goal_iterations: int
    source_plan: NotRequired[dict[str, Any]]
    rag_results: NotRequired[list[dict[str, Any]]]
    web_results: NotRequired[list[dict[str, Any]]]
    scout_output: NotRequired[dict[str, Any]]
    retriever_output: NotRequired[dict[str, Any]]
    researcher_output: NotRequired[dict[str, Any]]
    synthesizer_output: NotRequired[dict[str, Any]]
    grounding_audit: NotRequired[dict[str, Any]]
