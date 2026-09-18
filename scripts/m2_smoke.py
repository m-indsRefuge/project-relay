"""Live M2 smoke test against local Ollama models and goal routing."""

import argparse
import json
import tempfile
import uuid
from pathlib import Path

from relay.config import DEFAULT_CONFIG, RELAY_MODELS
from relay.graph import open_graph
from relay.models import OllamaClient
from relay.receipt import build_receipt

DEFAULT_TASK = (
    "A service stopped working immediately after a configuration change. "
    "Describe a cautious diagnostic approach using only the information supplied."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task passed through the live M2 goal-pursuit graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OllamaClient(
        host=DEFAULT_CONFIG.ollama_host,
        timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds,
    )

    client.ensure_models_available(RELAY_MODELS)

    with tempfile.TemporaryDirectory(prefix="relay-m2-") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoints.sqlite"

        with open_graph(
            checkpoint,
            model_client=client,
        ) as graph:
            result = graph.invoke(
                {
                    "episode_id": f"M2-SMOKE-{uuid.uuid4()}",
                    "task": args.task,
                    "should_interrupt": False,
                    "route_reason": None,
                    "terminal_status": "running",
                    "visited_nodes": [],
                    "model_trace": [],
                    "goal_iterations": 0,
                },
                {
                    "configurable": {
                        "thread_id": f"m2-smoke-{uuid.uuid4()}",
                    }
                },
            )

    if result["terminal_status"] not in {"completed", "failed"}:
        raise SystemExit(
            f"M2 smoke ended in unexpected state: {result['terminal_status']}"
        )

    if not result.get("goal"):
        raise SystemExit("M2 smoke did not form an explicit goal.")

    if result.get("goal_iterations", 0) < 1:
        raise SystemExit("M2 smoke did not evaluate the goal.")

    if result.get("goal_iterations", 0) > DEFAULT_CONFIG.max_goal_iterations:
        raise SystemExit("M2 smoke exceeded the bounded goal-iteration budget.")

    receipt = build_receipt(result)

    print(
        json.dumps(
            {
                "receipt": receipt,
                "grounding_audit": result.get("grounding_audit"),
                "goal_evaluation": result.get("goal_evaluation"),
                "active_subgoal": result.get("active_subgoal"),
                "scout_output": result.get("scout_output"),
                "retriever_output": result.get("retriever_output"),
                "researcher_output": result.get("researcher_output"),
                "synthesizer_output": result.get("synthesizer_output"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()