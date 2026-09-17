"""Gemma Researcher node for M1."""

import json

from relay.config import RESEARCHER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ResearcherOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Researcher.
M1 deliberately gives you no web or external tools.
Reason only over the supplied task and prior agent outputs.
Develop bounded hypotheses and risks, and explicitly preserve uncertainty.
Do not claim to have performed external research.
Return only the required structured output."""


def make_researcher_node(client: StructuredModelClient):
    """Create the Researcher LangGraph node."""

    def researcher(state: RelayState) -> dict:
        context = {
            "scout": state["scout_output"],
            "retriever": state["retriever_output"],
        }

        output = client.invoke(
            model=RESEARCHER_MODEL,
            system=SYSTEM,
            prompt=(
                f"Task:\n{state['task']}\n\n"
                f"Prior context:\n{json.dumps(context, ensure_ascii=False)}"
            ),
            output_type=ResearcherOutput,
        )

        return {
            "researcher_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), RESEARCHER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "researcher"],
        }

    return researcher