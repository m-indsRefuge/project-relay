"""Ministral Retriever node."""

import json

from relay.config import RETRIEVER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import RetrieverOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Retriever.
Organize the task, goal, Scout output, and M3 evidence for Researcher.
Evidence boundary:
- The original task is the authoritative source of supplied facts.
- Keep RAG and web evidence distinguishable and preserve source IDs.
- Do not promote Scout questions, implications, or inferred claims into facts.
- Clearly separate supplied facts from missing evidence.
Identify gaps instead of inventing evidence. Return only the required structured output."""


def make_retriever_node(client: StructuredModelClient):
    def retriever(state: RelayState) -> dict:
        output = client.invoke(
            model=RETRIEVER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(
                {
                    "task": state["task"],
                    "goal": state["goal"],
                    "active_subgoal": state.get("active_subgoal"),
                    "source_plan": state.get("source_plan"),
                    "rag_results": state.get("rag_results", []),
                    "web_results": state.get("web_results", []),
                    "scout": state.get("scout_output"),
                },
                ensure_ascii=False,
            ),
            output_type=RetrieverOutput,
        )
        return {
            "retriever_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), RETRIEVER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "retriever"],
        }

    return retriever
