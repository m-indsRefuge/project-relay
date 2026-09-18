"""Qwen Synthesizer node."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import SynthesizerOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Synthesizer.
Produce the best candidate answer for the original goal using only supplied
state. If an active subgoal exists, incorporate the newest reasoning into the
candidate answer.

Evidence boundary:
- The original user task is authoritative for supplied facts.
- Do not repeat an upstream model claim as fact unless the original task
  supports it.
- `supporting_points` must not contain unsupported real-world factual claims.
- Recommendations may describe actions to take; hypotheses must remain framed
  as possibilities.
- Temporal sequence alone does not prove causation or that no other event
  occurred.
- If a prior grounding audit is present, revise the answer to remove or
  explicitly reframe every listed unsupported claim.

Do not invent web, RAG, tool, or long-term-memory evidence.
Preserve uncertainty where evidence is incomplete.
Return only the required structured output."""


def make_synthesizer_node(client: StructuredModelClient):
    """Create the Synthesizer LangGraph node."""

    def synthesizer(state: RelayState) -> dict:
        context = {
            "task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
            "scout": state.get("scout_output"),
            "retriever": state.get("retriever_output"),
            "researcher": state.get("researcher_output"),
            "prior_grounding_audit": state.get("grounding_audit"),
        }

        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=SynthesizerOutput,
        )

        return {
            "synthesizer_output": output.model_dump(mode="json"),
            "model_trace": [
                *state.get("model_trace", []),
                SYNTHESIZER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "synthesizer",
            ],
        }

    return synthesizer