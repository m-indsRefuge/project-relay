"""Gemma Researcher node."""

import json

from relay.config import RESEARCHER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ResearcherOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Researcher.
Reason over the task, goal, active subgoal, and retrieved evidence.
Evidence boundary:
- `hypotheses` are possibilities, not established facts.
- RAG represents local documents; web represents external search evidence.
- Preserve source IDs when reasoning depends on retrieved evidence.
- Do not convert temporal correlation into causation.
- Do not strengthen an upstream inference merely because another model wrote it.
Return only the required structured output."""


def make_researcher_node(client: StructuredModelClient):
    def researcher(state: RelayState) -> dict:
        output = client.invoke(
            model=RESEARCHER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(
                {
                    "task": state["task"],
                    "goal": state["goal"],
                    "active_subgoal": state.get("active_subgoal"),
                    "rag_results": state.get("rag_results", []),
                    "web_results": state.get("web_results", []),
                    "scout": state.get("scout_output"),
                    "retriever": state.get("retriever_output"),
                },
                ensure_ascii=False,
            ),
            output_type=ResearcherOutput,
        )
        return {
            "researcher_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), RESEARCHER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "researcher"],
        }

    return researcher
