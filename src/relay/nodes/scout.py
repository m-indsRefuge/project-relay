"""Phi-4 Mini Scout node."""

import json

from relay.config import SCOUT_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ScoutOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Scout.
Decompose the active objective into what is known and what questions remain.
Evidence boundary:
- `known` may contain only facts explicitly supplied by the original user task OR facts directly supported by retrieved evidence.
- Do not convert temporal sequence into causation or exclusivity.
- For example, X immediately after Y does NOT establish that Y was the only recent event or that Y caused X.
- Preserve source IDs when a known item depends on RAG or web evidence.
- Put uncertainty and possible causes into `questions`, not `known`.
Return only the required structured output."""


def make_scout_node(client: StructuredModelClient):
    def scout(state: RelayState) -> dict:
        output = client.invoke(
            model=SCOUT_MODEL,
            system=SYSTEM,
            prompt=json.dumps(
                {
                    "task": state["task"],
                    "goal": state["goal"],
                    "active_subgoal": state.get("active_subgoal"),
                    "rag_results": state.get("rag_results", []),
                    "web_results": state.get("web_results", []),
                },
                ensure_ascii=False,
            ),
            output_type=ScoutOutput,
        )
        return {
            "scout_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SCOUT_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "scout"],
        }

    return scout
