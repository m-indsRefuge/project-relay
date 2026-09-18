"""Source planning and retrieval nodes for Relay M3."""

import json

from relay.config import RAG_RESULT_LIMIT, SYNTHESIZER_MODEL, WEB_RESULT_LIMIT
from relay.models import StructuredModelClient
from relay.nodes.schemas import SourcePlanOutput
from relay.rag import RagSearch
from relay.state import RelayState
from relay.tools import WebSearch

SOURCE_PLAN_SYSTEM = """You are Relay's Source Planner.
Decide whether the task needs local RAG, live web search, both, or neither.
If the user explicitly requests BOTH local/project documentation and current web evidence,
set both use_rag and use_web to true.
Use RAG for Relay-specific/local documentation. Use web for current or external public information.
Do not use web merely because it exists. Return only the required structured output."""


def make_source_planner_node(client: StructuredModelClient):
    def plan_sources(state: RelayState) -> dict:
        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=SOURCE_PLAN_SYSTEM,
            prompt=json.dumps(
                {"original_user_task": state["task"], "goal": state["goal"]}, ensure_ascii=False
            ),
            output_type=SourcePlanOutput,
        )
        return {
            "source_plan": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "plan_sources"],
        }

    return plan_sources


def make_rag_node(rag_search: RagSearch):
    def retrieve_rag(state: RelayState) -> dict:
        active = state.get("active_subgoal")
        query = (
            str(active["objective"])
            if active and active.get("route") == "rag"
            else str(state["source_plan"]["rag_query"])
        )
        results = rag_search.search(query, limit=RAG_RESULT_LIMIT)
        return {
            "rag_results": results,
            "source_trace": [*state.get("source_trace", []), "rag"],
            "visited_nodes": [*state.get("visited_nodes", []), "rag_retrieve"],
        }

    return retrieve_rag


def make_web_node(web_search: WebSearch):
    def search_web(state: RelayState) -> dict:
        active = state.get("active_subgoal")
        query = (
            str(active["objective"])
            if active and active.get("route") == "web"
            else str(state["source_plan"]["web_query"])
        )
        results = web_search.search(query, limit=WEB_RESULT_LIMIT)
        return {
            "web_results": results,
            "source_trace": [*state.get("source_trace", []), "web"],
            "visited_nodes": [*state.get("visited_nodes", []), "web_search"],
        }

    return search_web
