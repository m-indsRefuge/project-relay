"""Ministral Retriever node."""

import json

from relay.config import RETRIEVER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import RetrieverOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Retriever.
M2 has no RAG or long-term memory yet.
Organize only the task, goal, active subgoal, and Scout output into useful
context for the next specialist.

Evidence boundary:
- The original task is the authoritative source of supplied facts.
- Do not promote Scout questions, implications, or inferred claims into facts.
- If a Scout statement appears stronger than the original task supports,
  preserve it only as a gap/question rather than as established context.
- Clearly separate supplied facts from missing evidence.

Identify information gaps instead of inventing evidence.
Return only the required structured output."""


def make_retriever_node(client: StructuredModelClient):
    """Create the Retriever LangGraph node."""

    def retriever(state: RelayState) -> dict:
        context = {
            "task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
            "scout": state.get("scout_output"),
        }

        output = client.invoke(
            model=RETRIEVER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=RetrieverOutput,
        )

        return {
            "retriever_output": output.model_dump(mode="json"),
            "model_trace": [
                *state.get("model_trace", []),
                RETRIEVER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "retriever",
            ],
        }

    return retriever