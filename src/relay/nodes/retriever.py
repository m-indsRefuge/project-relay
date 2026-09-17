"""Ministral Retriever node for M1."""

import json

from relay.config import RETRIEVER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import RetrieverOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Retriever.
M1 has no RAG or long-term memory yet.
Organize only the task and Scout output into useful context for the next
specialist. Identify information gaps instead of inventing missing evidence.
Return only the required structured output."""


def make_retriever_node(client: StructuredModelClient):
    """Create the Retriever LangGraph node."""

    def retriever(state: RelayState) -> dict:
        scout = json.dumps(state["scout_output"], ensure_ascii=False)

        output = client.invoke(
            model=RETRIEVER_MODEL,
            system=SYSTEM,
            prompt=(
                f"Task:\n{state['task']}\n\n"
                f"Scout output:\n{scout}"
            ),
            output_type=RetrieverOutput,
        )

        return {
            "retriever_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), RETRIEVER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "retriever"],
        }

    return retriever