"""Qwen goal-formation and goal-evaluation nodes."""

import json

from relay.config import SYNTHESIZER_MODEL
from relay.models import StructuredModelClient
from relay.nodes.schemas import GoalEvaluationOutput, GoalOutput
from relay.state import RelayState

GOAL_SYSTEM = """You are Relay's Goal Former.
The user's requested deliverable is authoritative.
Preserve the speech act and scope of the request.
Do not transform "describe how to investigate X" into "investigate X".
Do not transform "explain X" into "prove X".
Convert the task into one objective and observable success criteria. Do not solve it.
Return only the required structured output."""

EVALUATION_SYSTEM = """You are Relay's Goal Evaluator.
The original user task is authoritative.
Judge whether the candidate satisfies the requested deliverable and success criteria.
The focused grounding audit is authoritative for factual grounding; the goal cannot
be satisfied until the Synthesizer revises grounding issues.
If the user asked to describe an action, it does NOT require the described real-world
action to have actually been performed. If the user asked Relay to perform an action,
mark it satisfied only when graph state contains evidence that the required action
actually occurred.
M3 may route additional work to scout, retriever, researcher, synthesizer, rag, or web.
Use rag for missing local/project evidence and web for missing current/external evidence.
Return only the required structured output."""


def make_goal_former_node(client: StructuredModelClient):
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
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "form_goal"],
        }

    return form_goal


def make_goal_evaluator_node(client: StructuredModelClient):
    def evaluate_goal(state: RelayState) -> dict:
        context = {
            "original_user_task": state["task"],
            "goal": state["goal"],
            "active_subgoal": state.get("active_subgoal"),
            "source_plan": state.get("source_plan"),
            "rag_results": state.get("rag_results", []),
            "web_results": state.get("web_results", []),
            "grounding_audit": state.get("grounding_audit"),
            "synthesizer_output": state.get("synthesizer_output"),
        }
        output = client.invoke(
            model=SYNTHESIZER_MODEL,
            system=EVALUATION_SYSTEM,
            prompt=json.dumps(context, ensure_ascii=False),
            output_type=GoalEvaluationOutput,
        )
        unsupported = state.get("grounding_audit", {}).get(
            "unsupported_claims",
            [],
        )
        if unsupported:
            output = GoalEvaluationOutput(
                status="missing_information",
                rationale=(
                    "Grounding audit found issues that must be revised "
                    "before completion."
                ),
                subgoal=(
                    "Revise the candidate so all grounding-audit issues are "
                    "removed or supported."
                ),
                route="synthesizer",
            )
        iteration = state.get("goal_iterations", 0) + 1
        updates: dict = {
            "goal_evaluation": output.model_dump(mode="json"),
            "goal_iterations": iteration,
            "model_trace": [*state.get("model_trace", []), SYNTHESIZER_MODEL],
            "visited_nodes": [*state.get("visited_nodes", []), "evaluate_goal"],
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
