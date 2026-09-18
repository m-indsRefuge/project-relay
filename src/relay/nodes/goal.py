"""Qwen goal-formation and goal-evaluation nodes for M2."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import GoalEvaluationOutput, GoalOutput
from relay.state import RelayState

GOAL_SYSTEM = """You are Relay's Goal Former.
Convert the user's task into one explicit objective and a short list of
observable success criteria.

The user's requested deliverable is authoritative.
Preserve the speech act and scope of the request.

Examples:
- If the user asks to describe, explain, outline, compare, analyze, or plan,
  the objective must remain a descriptive or analytical deliverable.
- Do not transform "describe how to investigate X" into "investigate X".
- Do not transform "explain X" into "prove X".
- If the user explicitly asks Relay to perform an action, the objective may
  require that action only when the required capability exists.

M2 has no RAG, web, tools, or long-term memory.
Do not create success criteria that require unavailable observations,
external evidence, or actions that Relay cannot actually perform.

Do not solve the task.
Return only the required structured output."""


EVALUATION_SYSTEM = """You are Relay's Goal Evaluator.
The original user task is authoritative.

Judge whether the current candidate answer satisfies BOTH:
1. the original user-requested deliverable; and
2. the explicit goal and success criteria.

Do not evaluate a model-generated goal in isolation from the original task.

Important distinction:
- If the user asked Relay to describe, explain, outline, analyze, compare, or
  plan an action, satisfaction means the candidate successfully provides that
  requested description, explanation, analysis, comparison, or plan.
- It does NOT require the described real-world action to have actually been
  performed.
- If the user explicitly asked Relay to perform, inspect, search, retrieve,
  verify, modify, or otherwise take an action, mark it satisfied only when
  graph state contains evidence that the required action actually occurred.

M2 has no RAG, web, external tools, or long-term memory.
Never claim unavailable actions occurred.

Grounding check:
- The focused grounding audit is authoritative for whether the current
  candidate contains unsupported factual claims.
- If the grounding audit reports unsupported claims, the goal cannot be
  satisfied until the Synthesizer revises them.

Allowed outcomes:
- satisfied: the requested deliverable and success criteria are met.
- missing_information: one focused additional reasoning pass is useful.
- failed: the requested deliverable cannot responsibly be completed from the
  available M2 context.

For missing_information, create exactly one subgoal and route it to one of:
scout, retriever, researcher, synthesizer.

Never route to capabilities that do not exist in M2.
Return only the required structured output."""


def make_goal_former_node(client: StructuredModelClient):
    """Create the Qwen goal-formation node."""

    def form_goal(state: RelayState) -> dict:
        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=GOAL_SYSTEM,
            prompt=f"Original user task:\n{state['task']}",
            output_type=GoalOutput,
        )

        return {
            "goal": output.model_dump(mode="json"),
            "active_subgoal": None,
            "goal_iterations": 0,
            "model_trace": [
                *state.get("model_trace", []),
                SYNTHESIZER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "form_goal",
            ],
        }

    return form_goal


def make_goal_evaluator_node(client: StructuredModelClient):
    """Create the Qwen goal-evaluation node."""

    def evaluate_goal(state: RelayState) -> dict:
        context = {
            "original_user_task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
            "grounding_audit": state.get("grounding_audit"),
            "scout_output": state.get("scout_output"),
            "retriever_output": state.get("retriever_output"),
            "researcher_output": state.get("researcher_output"),
            "synthesizer_output": state.get("synthesizer_output"),
        }

        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=EVALUATION_SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=GoalEvaluationOutput,
        )

        audit = state.get("grounding_audit", {})
        unsupported = audit.get("unsupported_claims", [])

        if unsupported:
            output = GoalEvaluationOutput(
                status="missing_information",
                rationale=(
                    "Grounding audit found unsupported factual claims that "
                    "must be revised before completion."
                ),
                subgoal=(
                    "Revise the candidate answer so every unsupported factual "
                    "claim identified by the grounding audit is removed or "
                    "explicitly reframed as a hypothesis or recommendation."
                ),
                route="synthesizer",
            )

        iteration = state.get("goal_iterations", 0) + 1
        updates: dict = {
            "goal_evaluation": output.model_dump(mode="json"),
            "goal_iterations": iteration,
            "model_trace": [
                *state.get("model_trace", []),
                SYNTHESIZER_MODEL,
            ],
            "visited_nodes": [
                *state.get("visited_nodes", []),
                "evaluate_goal",
            ],
        }

        if output.status == "satisfied":
            updates["active_subgoal"] = None
            updates["route_reason"] = "goal satisfied"
        elif output.status == "failed":
            updates["active_subgoal"] = None
            updates["route_reason"] = "goal evaluation failed"
        else:
            updates["active_subgoal"] = {
                "objective": output.subgoal,
                "route": output.route,
            }
            updates["route_reason"] = f"goal gap routed to {output.route}"

        return updates

    return evaluate_goal