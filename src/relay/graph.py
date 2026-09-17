"""Project Relay M1 LangGraph."""

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

from relay.config import DEFAULT_CONFIG
from relay.models import OllamaClient, StructuredModelClient
from relay.nodes.researcher import make_researcher_node
from relay.nodes.retriever import make_retriever_node
from relay.nodes.scout import make_scout_node
from relay.nodes.synthesizer import make_synthesizer_node
from relay.state import RelayState

Route = Literal["scout", "fail", "pause_for_resume"]


def initialize_state(state: RelayState) -> dict:
    """Validate M1 input and prepare deterministic routing."""

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
    """Choose the first execution path from explicit state."""

    if not state.get("task", "").strip():
        return "fail"

    if state.get("should_interrupt", False):
        return "pause_for_resume"

    return "scout"


def pause_for_resume(state: RelayState) -> dict:
    """Retain M0 checkpointed interrupt/resume behaviour."""

    resume_value = interrupt(
        {
            "kind": "m1_resume_probe",
            "episode_id": state["episode_id"],
            "message": "Resume Project Relay M1.",
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
    """Mark the M1 execution successful."""

    return {
        "terminal_status": "completed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "complete",
        ],
    }


def fail(state: RelayState) -> dict:
    """Terminate invalid M1 input visibly."""

    return {
        "terminal_status": "failed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "fail",
        ],
    }


def build_graph(
    *,
    checkpointer: SqliteSaver,
    model_client: StructuredModelClient,
):
    """Build and compile the M1 four-model LangGraph."""

    builder = StateGraph(RelayState)

    builder.add_node("initialize_state", initialize_state)
    builder.add_node("pause_for_resume", pause_for_resume)
    builder.add_node("scout", make_scout_node(model_client))
    builder.add_node("retriever", make_retriever_node(model_client))
    builder.add_node("researcher", make_researcher_node(model_client))
    builder.add_node("synthesizer", make_synthesizer_node(model_client))
    builder.add_node("complete", complete)
    builder.add_node("fail", fail)

    builder.add_edge(START, "initialize_state")

    builder.add_conditional_edges(
        "initialize_state",
        route_after_initialize,
        {
            "scout": "scout",
            "fail": "fail",
            "pause_for_resume": "pause_for_resume",
        },
    )

    builder.add_edge("pause_for_resume", "scout")
    builder.add_edge("scout", "retriever")
    builder.add_edge("retriever", "researcher")
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "complete")
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_graph(
    checkpoint_path: Path,
    *,
    model_client: StructuredModelClient | None = None,
) -> Iterator:
    """Open an M1 graph backed by a local SQLite checkpointer."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    client = model_client or OllamaClient(
        host=DEFAULT_CONFIG.ollama_host,
        timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds,
    )

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        yield build_graph(
            checkpointer=checkpointer,
            model_client=client,
        )