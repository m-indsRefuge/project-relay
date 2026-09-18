"""Phi-4 Mini Scout node."""

import json

from relay.config import SCOUT_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import ScoutOutput
from relay.state import RelayState

SYSTEM = """You are Relay's Scout.
Decompose the active objective into what is already known and what questions
the downstream specialists should consider.
If an active subgoal exists, focus on that subgoal.

Evidence boundary:
- `known` may contain only facts explicitly supplied by the original user task
  or conservatively restated from it.
- Do not convert temporal sequence into causation or exclusivity.
- For example, "X happened immediately after Y" does NOT establish that Y was
  the only recent event or that Y caused X.
- Put uncertainty, possible causes, and missing information into `questions`,
  not `known`.

Do not pretend to use tools, memory, RAG, or the web.
Return only the required structured output."""


def make_scout_node(client: StructuredModelClient):
    """Create the Scout LangGraph node."""

    def scout(state: RelayState) -> dict:
        context = {
            "task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
        }

        output = client.invoke(
            model=SCOUT_MODEL,
            system=SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=ScoutOutput,
        )

        return {
            "scout_output": output.model_dump(mode="json"),
            "model_trace": [*state.get("model_trace", []), SCOUT_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "scout"],
        }

    return scout