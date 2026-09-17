"""Qwen Synthesizer node for M1."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import SynthesizerOutput
from relay.state import RelayState

SYSTEM = """You are Relay's M1 Synthesizer.
Produce a concise answer using only the supplied task and prior specialist
outputs. Do not invent web, RAG, tool, or long-term-memory evidence.
If context is uncertain, preserve that uncertainty in the answer.
Return only the required structured output."""


def make_synthesizer_node(client: StructuredModelClient):
    """Create the Synthesizer LangGraph node."""

    def synthesizer(state: RelayState) -> dict:
        context = {
            "scout": state["scout_output"],
            "retriever": state["retriever_output"],
            "researcher": state["researcher_output"],
        }

        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=SYSTEM,
            prompt=(
                f"Task:\n{state['task']}\n\n"
                f"Specialist outputs:\n{json.dumps(context, ensure_ascii=False)}"
            ),
            output_type=SynthesizerOutput,
        )

        return {
            "synthesizer_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "synthesizer"],
        }

    return synthesizer