"""Phi-4 Mini Scout node for M1."""

from relay.config import SCOUT_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ScoutOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Scout.
Your only job is to decompose the supplied task into what is already known
and what questions another specialist should consider.
Do not pretend to use tools, memory, RAG, or the web.
Return only the required structured output."""


def make_scout_node(client: StructuredModelClient):
    """Create the Scout LangGraph node."""

    def scout(state: RelayState) -> dict:
        output = client.invoke(
            model=SCOUT_MODEL,
            system=SYSTEM,
            prompt=f"Task:\n{state['task']}",
            output_type=ScoutOutput,
        )

        return {
            "scout_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SCOUT_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "scout"],
        }

    return scout