"""Project Relay M0 LangGraph."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from relay.state import RelayState

Route = Literal["complete", "fail", "pause_for_resume"]


def initialize_state(state: RelayState) -> dict:
    """Validate the minimal task and prepare deterministic routing."""

    visited = [*state.get("visited_nodes", []), "initialize_state"]
    task = state.get("task", "")

    if not task.strip():
        return {
            "route_reason": "task is empty",
            "terminal_status": "running",
            "visited_nodes": visited,
        }

    if state.get("should_interrupt", False):
        return {
            "route_reason": "resume probe requested",
            "terminal_status": "running",
            "visited_nodes": visited,
        }

    return {
        "route_reason": "task is valid",
        "terminal_status": "running",
        "visited_nodes": visited,
    }


def route_after_initialize(state: RelayState) -> Route:
    """Choose the next graph node from explicit state."""

    if not state.get("task", "").strip():
        return "fail"

    if state.get("should_interrupt", False):
        return "pause_for_resume"

    return "complete"


def pause_for_resume(state: RelayState) -> dict:
    """Pause graph execution and prove persisted resume behaviour."""

    resume_value = interrupt(
        {
            "kind": "m0_resume_probe",
            "episode_id": state["episode_id"],
            "message": "Resume Project Relay M0.",
        }
    )

    return {
        "resume_value": str(resume_value),
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "pause_for_resume",
        ],
        "route_reason": "checkpointed interrupt resumed",
    }


def complete(state: RelayState) -> dict:
    """Mark the M0 execution successful."""

    return {
        "terminal_status": "completed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "complete",
        ],
    }


def fail(state: RelayState) -> dict:
    """Terminate invalid M0 input visibly."""

    return {
        "terminal_status": "failed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "fail",
        ],
    }


def build_graph(*, checkpointer: SqliteSaver):
    """Build and compile the M0 LangGraph."""

    builder = StateGraph(RelayState)

    builder.add_node("initialize_state", initialize_state)
    builder.add_node("pause_for_resume", pause_for_resume)
    builder.add_node("complete", complete)
    builder.add_node("fail", fail)

    builder.add_edge(START, "initialize_state")

    builder.add_conditional_edges(
        "initialize_state",
        route_after_initialize,
        {
            "complete": "complete",
            "fail": "fail",
            "pause_for_resume": "pause_for_resume",
        },
    )

    builder.add_edge("pause_for_resume", "complete")
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_graph(checkpoint_path: Path) -> Iterator:
    """Open a compiled Relay graph backed by a local SQLite checkpointer."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        yield build_graph(checkpointer=checkpointer)