"""Live M1 smoke test against the four configured local Ollama models."""

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
        help="Task passed through the live four-model M1 graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OllamaClient(
        host=DEFAULT_CONFIG.ollama_host,
        timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds,
    )

    client.ensure_models_available(RELAY_MODELS)

    with tempfile.TemporaryDirectory(prefix="relay-m1-") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoints.sqlite"

        with open_graph(
            checkpoint,
            model_client=client,
        ) as graph:
            result = graph.invoke(
                {
                    "episode_id": f"M1-SMOKE-{uuid.uuid4()}",
                    "task": args.task,
                    "should_interrupt": False,
                    "route_reason": None,
                    "terminal_status": "running",
                    "visited_nodes": [],
                    "model_trace": [],
                },
                {
                    "configurable": {
                        "thread_id": f"m1-smoke-{uuid.uuid4()}",
                    }
                },
            )

    if result["terminal_status"] != "completed":
        raise SystemExit(f"M1 smoke did not complete: {result['terminal_status']}")

    if result["model_trace"] != list(RELAY_MODELS):
        raise SystemExit("M1 smoke did not execute the expected four-model sequence.")

    receipt = build_receipt(result)

    print(
        json.dumps(
            {
                "receipt": receipt,
                "scout_output": result["scout_output"],
                "retriever_output": result["retriever_output"],
                "researcher_output": result["researcher_output"],
                "synthesizer_output": result["synthesizer_output"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
