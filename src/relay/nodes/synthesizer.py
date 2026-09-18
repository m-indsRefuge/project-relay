"""Qwen Synthesizer node."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import SynthesizerOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Synthesizer.
Produce the best candidate answer for the original goal using supplied state.
Evidence boundary:
- The original user task is authoritative for supplied facts.
- RAG may support Relay/local-document claims; web may support external/current claims.
- A model hypothesis is not evidence merely because it appears in graph state.
- `supporting_points` must not contain unsupported real-world factual claims.
- If retrieved evidence materially supports the answer, include exact `source_id` values in `citations`.
- Never invent a source ID.
- If a prior grounding audit exists, revise every listed unsupported claim.
Return only the required structured output."""


def make_synthesizer_node(client: StructuredModelClient):
    def synthesizer(state: RelayState) -> dict:
        output = client.invoke(
            model=SYNTHESIZER_MODEL,
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
                    "researcher": state.get("researcher_output"),
                    "prior_grounding_audit": state.get("grounding_audit"),
                },
                ensure_ascii=False,
            ),
            output_type=SynthesizerOutput,
        )
        return {
            "synthesizer_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "synthesizer"],
        }

    return synthesizer
