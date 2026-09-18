"""Project Relay M3 LangGraph."""

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
from relay.nodes.goal import make_goal_evaluator_node, make_goal_former_node
from relay.nodes.grounding import make_grounding_auditor_node
from relay.nodes.researcher import make_researcher_node
from relay.nodes.retriever import make_retriever_node
from relay.nodes.scout import make_scout_node
from relay.nodes.sources import make_rag_node, make_source_planner_node, make_web_node
from relay.nodes.synthesizer import make_synthesizer_node
from relay.rag import KnowledgeIndex, OllamaEmbeddingClient, RagSearch
from relay.state import RelayState
from relay.tools import DDGSWebSearch, WebSearch

InitialRoute = Literal["form_goal", "fail", "pause_for_resume"]
SourceRoute = Literal["rag", "web", "scout"]
AfterRagRoute = Literal["web", "scout", "synthesizer"]
AfterWebRoute = Literal["scout", "synthesizer"]
GoalRoute = Literal[
    "complete", "fail", "scout", "retriever", "researcher", "synthesizer", "rag", "web"
]


def initialize_state(state: RelayState) -> dict:
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
    return {"route_reason": "task is valid", "terminal_status": "running", "visited_nodes": visited}


def route_after_initialize(state: RelayState) -> InitialRoute:
    if not state.get("task", "").strip():
        return "fail"
    if state.get("should_interrupt", False):
        return "pause_for_resume"
    return "form_goal"


def route_after_source_plan(state: RelayState) -> SourceRoute:
    plan = state["source_plan"]
    if plan["use_rag"]:
        return "rag"
    if plan["use_web"]:
        return "web"
    return "scout"


def route_after_rag(state: RelayState) -> AfterRagRoute:
    active = state.get("active_subgoal")
    if active and active.get("route") == "rag":
        return "synthesizer"
    if state["source_plan"]["use_web"]:
        return "web"
    return "scout"


def route_after_web(state: RelayState) -> AfterWebRoute:
    active = state.get("active_subgoal")
    if active and active.get("route") == "web":
        return "synthesizer"
    return "scout"


def route_after_goal_evaluation(state: RelayState) -> GoalRoute:
    evaluation = state["goal_evaluation"]
    if evaluation["status"] == "satisfied":
        return "complete"
    if evaluation["status"] == "failed":
        return "fail"
    if state["goal_iterations"] >= DEFAULT_CONFIG.max_goal_iterations:
        return "fail"
    route = evaluation.get("route")
    if route not in {"scout", "retriever", "researcher", "synthesizer", "rag", "web"}:
        return "fail"
    return route


def pause_for_resume(state: RelayState) -> dict:
    resume_value = interrupt(
        {
            "kind": "m3_resume_probe",
            "episode_id": state["episode_id"],
            "message": "Resume Project Relay M3.",
        }
    )
    return {
        "resume_value": str(resume_value),
        "visited_nodes": [*state.get("visited_nodes", []), "pause_for_resume"],
        "route_reason": "checkpointed interrupt resumed",
    }


def complete(state: RelayState) -> dict:
    return {
        "terminal_status": "completed",
        "visited_nodes": [*state.get("visited_nodes", []), "complete"],
    }


def fail(state: RelayState) -> dict:
    reason = state.get("route_reason")
    if (
        state.get("goal_evaluation", {}).get("status") == "missing_information"
        and state.get("goal_iterations", 0) >= DEFAULT_CONFIG.max_goal_iterations
    ):
        reason = "goal iteration budget exhausted"
    return {
        "route_reason": reason,
        "terminal_status": "failed",
        "visited_nodes": [*state.get("visited_nodes", []), "fail"],
    }


def build_graph(
    *,
    checkpointer: SqliteSaver,
    model_client: StructuredModelClient,
    rag_search: RagSearch,
    web_search: WebSearch,
):
    builder = StateGraph(RelayState)
    builder.add_node("initialize_state", initialize_state)
    builder.add_node("pause_for_resume", pause_for_resume)
    builder.add_node("form_goal", make_goal_former_node(model_client))
    builder.add_node("plan_sources", make_source_planner_node(model_client))
    builder.add_node("rag_retrieve", make_rag_node(rag_search))
    builder.add_node("web_search", make_web_node(web_search))
    builder.add_node("scout", make_scout_node(model_client))
    builder.add_node("retriever", make_retriever_node(model_client))
    builder.add_node("researcher", make_researcher_node(model_client))
    builder.add_node("synthesizer", make_synthesizer_node(model_client))
    builder.add_node("grounding_audit", make_grounding_auditor_node(model_client))
    builder.add_node("evaluate_goal", make_goal_evaluator_node(model_client))
    builder.add_node("complete", complete)
    builder.add_node("fail", fail)
    builder.add_edge(START, "initialize_state")
    builder.add_conditional_edges(
        "initialize_state",
        route_after_initialize,
        {"form_goal": "form_goal", "fail": "fail", "pause_for_resume": "pause_for_resume"},
    )
    builder.add_edge("pause_for_resume", "form_goal")
    builder.add_edge("form_goal", "plan_sources")
    builder.add_conditional_edges(
        "plan_sources",
        route_after_source_plan,
        {"rag": "rag_retrieve", "web": "web_search", "scout": "scout"},
    )
    builder.add_conditional_edges(
        "rag_retrieve",
        route_after_rag,
        {"web": "web_search", "scout": "scout", "synthesizer": "synthesizer"},
    )
    builder.add_conditional_edges(
        "web_search", route_after_web, {"scout": "scout", "synthesizer": "synthesizer"}
    )
    builder.add_edge("scout", "retriever")
    builder.add_edge("retriever", "researcher")
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "grounding_audit")
    builder.add_edge("grounding_audit", "evaluate_goal")
    builder.add_conditional_edges(
        "evaluate_goal",
        route_after_goal_evaluation,
        {
            "complete": "complete",
            "fail": "fail",
            "scout": "scout",
            "retriever": "retriever",
            "researcher": "researcher",
            "synthesizer": "synthesizer",
            "rag": "rag_retrieve",
            "web": "web_search",
        },
    )
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_graph(
    checkpoint_path: Path,
    *,
    model_client: StructuredModelClient | None = None,
    rag_search: RagSearch | None = None,
    web_search: WebSearch | None = None,
) -> Iterator:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    client = model_client or OllamaClient(
        host=DEFAULT_CONFIG.ollama_host, timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds
    )
    rag = rag_search or KnowledgeIndex(
        knowledge_dir=DEFAULT_CONFIG.knowledge_dir,
        embedder=OllamaEmbeddingClient(
            host=DEFAULT_CONFIG.ollama_host, timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds
        ),
    )
    web = web_search or DDGSWebSearch()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        yield build_graph(
            checkpointer=checkpointer, model_client=client, rag_search=rag, web_search=web
        )
