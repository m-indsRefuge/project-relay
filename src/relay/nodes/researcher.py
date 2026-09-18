"""Gemma Researcher node."""

import json

from relay.config import RESEARCHER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ResearcherOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Researcher.
M2 deliberately gives you no web or external tools.
Reason only over the supplied task, goal, active subgoal, and prior agent
outputs.

Evidence boundary:
- `hypotheses` are possibilities, not established facts.
- Do not convert temporal correlation into causation.
- Do not strengthen an upstream inference merely because another model wrote it.
- Preserve uncertainty whenever the original task lacks confirming evidence.

Develop bounded hypotheses and risks while preserving uncertainty.
Do not claim to have performed external research.
Return only the required structured output."""


def make_researcher_node(client: StructuredModelClient):
    """Create the Researcher LangGraph node."""

    def researcher(state: RelayState) -> dict:
        context = {
            "task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
            "scout": state.get("scout_output"),
            "retriever": state.get("retriever_output"),
        }

        output = client.invoke(
            model=RESEARCHER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=ResearcherOutput,
        )

        return {
            "researcher_output": output.model_dump(mode="json"),
            "model_trace": [
                *state.get("model_trace", []),
                RESEARCHER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "researcher",
            ],
        }

    return researcher