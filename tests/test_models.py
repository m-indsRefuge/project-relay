"""Model-boundary and structured-output tests."""

import json

import httpx
import pytest
from pydantic import ValidationError

from relay.models import ModelInvocationError, OllamaClient
from relay.nodes.schemas import GoalEvaluationOutput, ScoutOutput


def test_ollama_structured_response_is_validated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"

        content = ScoutOutput(
            known=["Known item"],
            questions=["Question?"],
            handoff="Continue.",
        ).model_dump_json()

        return httpx.Response(
            200,
            json={"response": content},
        )

    client = OllamaClient(
        transport=httpx.MockTransport(handler),
    )

    result = client.invoke(
        model="test-model",
        system="system",
        prompt="prompt",
        output_type=ScoutOutput,
    )

    assert result.handoff == "Continue."


def test_ollama_malformed_json_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"response": "{not-json"},
        )

    client = OllamaClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelInvocationError, match="malformed JSON"):
        client.invoke(
            model="test-model",
            system="system",
            prompt="prompt",
            output_type=ScoutOutput,
        )


def test_ollama_schema_violation_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request

        return httpx.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "known": [],
                        "questions": [],
                    }
                )
            },
        )

    client = OllamaClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelInvocationError, match="violates the required schema"):
        client.invoke(
            model="test-model",
            system="system",
            prompt="prompt",
            output_type=ScoutOutput,
        )


def test_ollama_http_failure_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503)

    client = OllamaClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelInvocationError, match="Ollama request failed"):
        client.invoke(
            model="test-model",
            system="system",
            prompt="prompt",
            output_type=ScoutOutput,
        )


def test_missing_required_model_fails_before_graph_execution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"

        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "model-a"},
                ]
            },
        )

    client = OllamaClient(
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelInvocationError, match="model-b"):
        client.ensure_models_available(
            ("model-a", "model-b"),
        )


def test_missing_information_requires_subgoal_and_route() -> None:
    with pytest.raises(ValidationError):
        GoalEvaluationOutput(
            status="missing_information",
            rationale="More work is needed.",
        )


def test_satisfied_goal_rejects_spurious_route() -> None:
    with pytest.raises(ValidationError):
        GoalEvaluationOutput(
            status="satisfied",
            rationale="Done.",
            subgoal="Should not exist.",
            route="researcher",
        )
