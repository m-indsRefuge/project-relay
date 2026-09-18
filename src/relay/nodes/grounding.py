"""Focused Qwen grounding audit for M2."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import GroundingAuditOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Grounding Auditor.

Audit the candidate answer and supporting points against the ORIGINAL USER
TASK. The original task is the authority for supplied real-world facts.

Flag a claim as unsupported when it is stated as an established real-world
fact but is not explicitly supplied by, or safely entailed by, the original
task.

Important:
- Temporal sequence does not prove causation.
- "X happened after Y" does not prove Y caused X.
- A hypothesis from another Relay model is not evidence.
- Agreement among Relay models is not independent evidence.
- Recommendations and diagnostic possibilities are allowed when clearly
  framed as actions to consider or hypotheses rather than established facts.
- Do not penalize the answer merely for describing sensible checks.

Return `grounded` only when there are no material unsupported factual claims.
Return only the required structured output."""


def make_grounding_auditor_node(client: StructuredModelClient):
    """Create the focused grounding-audit node."""

    def grounding_audit(state: RelayState) -> dict:
        context = {
            "original_user_task": state["task"],
            "goal": state["goal"],
            "candidate": state["synthesizer_output"],
        }

        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=GroundingAuditOutput,
        )

        return {
            "grounding_audit": output.model_dump(mode="json"),
            "model_trace": [
                *state.get("model_trace", []),
                SYNTHESIZER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "grounding_audit",
            ],
        }

    return grounding_audit