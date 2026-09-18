"""Live M3 smoke: local RAG + live web + provenance."""

import json
import tempfile
import uuid
from pathlib import Path

from relay.config import DEFAULT_CONFIG, EMBEDDING_MODEL, RELAY_MODELS
from relay.graph import open_graph
from relay.models import OllamaClient
from relay.receipt import build_receipt

TASK = """Use BOTH Relay's local knowledge base and a live web search.
Explain the difference between Relay's own use of LangGraph and the general
LangGraph workflow model described by current public documentation.
For Relay-specific claims, use local project knowledge.
For current LangGraph claims, use web evidence.
Preserve source provenance."""


def main() -> None:
    client = OllamaClient(
        host=DEFAULT_CONFIG.ollama_host,
        timeout_seconds=DEFAULT_CONFIG.ollama_timeout_seconds,
    )
    client.ensure_models_available((*RELAY_MODELS, EMBEDDING_MODEL))
    with tempfile.TemporaryDirectory(prefix="relay-m3-") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoints.sqlite"
        with open_graph(checkpoint, model_client=client) as graph:
            result = graph.invoke(
                {
                    "episode_id": f"M3-SMOKE-{uuid.uuid4()}",
                    "task": TASK,
                    "should_interrupt": False,
                    "route_reason": None,
                    "terminal_status": "running",
                    "visited_nodes": [],
                    "model_trace": [],
                    "source_trace": [],
                    "goal_iterations": 0,
                },
                {"configurable": {"thread_id": f"m3-smoke-{uuid.uuid4()}"}},
            )
    if result["terminal_status"] != "completed":
        raise SystemExit(f"M3 smoke did not complete: {result['terminal_status']}")
    plan = result.get("source_plan", {})
    if not plan.get("use_rag") or not plan.get("use_web"):
        raise SystemExit(f"M3 smoke required both RAG and web, but plan was: {plan}")
    if not result.get("rag_results"):
        raise SystemExit("M3 smoke returned no RAG evidence.")
    if not result.get("web_results"):
        raise SystemExit("M3 smoke returned no web evidence.")
    citations = set(result.get("synthesizer_output", {}).get("citations", []))
    evidence_ids = {
        record["source_id"]
        for record in [*result["rag_results"], *result["web_results"]]
    }
    if not citations:
        raise SystemExit("M3 smoke produced no source citations.")
    if not any(source_id.startswith("rag:") for source_id in citations):
        raise SystemExit("M3 smoke did not cite local RAG evidence.")
    if not any(source_id.startswith("web:") for source_id in citations):
        raise SystemExit("M3 smoke did not cite live web evidence.")
    if not citations.issubset(evidence_ids):
        raise SystemExit("M3 smoke produced citation IDs outside retrieved evidence.")
    if result.get("grounding_audit", {}).get("assessment") != "grounded":
        raise SystemExit("M3 smoke did not finish with grounded evidence.")
    print(
        json.dumps(
            {
                "receipt": build_receipt(result),
                "source_plan": result.get("source_plan"),
                "rag_results": result.get("rag_results"),
                "web_results": result.get("web_results"),
                "grounding_audit": result.get("grounding_audit"),
                "goal_evaluation": result.get("goal_evaluation"),
                "synthesizer_output": result.get("synthesizer_output"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
