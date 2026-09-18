"""M3 graph behaviour tests."""

from pathlib import Path

from conftest import FakeModelClient, FakeRagSearch, FakeWebSearch

from relay.graph import open_graph
from relay.receipt import build_receipt


def make_state(*, episode_id: str, task: str, should_interrupt: bool = False) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "should_interrupt": should_interrupt,
        "route_reason": None,
        "terminal_status": "running",
        "visited_nodes": [],
        "model_trace": [],
        "source_trace": [],
        "goal_iterations": 0,
    }


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_graph(
    tmp_path: Path,
    *,
    client: FakeModelClient,
    rag: FakeRagSearch | None = None,
    web: FakeWebSearch | None = None,
    thread_id: str = "thread",
) -> dict:
    checkpoint = tmp_path / "checkpoints.sqlite"
    with open_graph(
        checkpoint,
        model_client=client,
        rag_search=rag or FakeRagSearch(),
        web_search=web or FakeWebSearch(),
    ) as graph:
        return graph.invoke(
            make_state(episode_id="E-M3", task="Exercise Project Relay M3."), make_config(thread_id)
        )


def test_no_source_plan_skips_rag_and_web(tmp_path: Path) -> None:
    client = FakeModelClient()
    rag = FakeRagSearch()
    web = FakeWebSearch()
    result = run_graph(tmp_path, client=client, rag=rag, web=web)
    assert result["terminal_status"] == "completed"
    assert result["source_trace"] == []
    assert rag.calls == []
    assert web.calls == []
    assert "plan_sources" in result["visited_nodes"]


def test_rag_only_plan_routes_through_local_retrieval(tmp_path: Path) -> None:
    client = FakeModelClient(use_rag=True)
    rag = FakeRagSearch()
    web = FakeWebSearch()
    result = run_graph(tmp_path, client=client, rag=rag, web=web, thread_id="rag-only")
    assert result["source_trace"] == ["rag"]
    assert len(rag.calls) == 1
    assert web.calls == []
    assert result["synthesizer_output"]["citations"] == ["rag:fixture.md#chunk-000"]


def test_web_only_plan_routes_through_web(tmp_path: Path) -> None:
    client = FakeModelClient(use_web=True)
    rag = FakeRagSearch()
    web = FakeWebSearch()
    result = run_graph(tmp_path, client=client, rag=rag, web=web, thread_id="web-only")
    assert result["source_trace"] == ["web"]
    assert rag.calls == []
    assert len(web.calls) == 1
    assert result["synthesizer_output"]["citations"] == ["web:001"]


def test_both_sources_execute_rag_then_web(tmp_path: Path) -> None:
    result = run_graph(
        tmp_path, client=FakeModelClient(use_rag=True, use_web=True), thread_id="both"
    )
    assert result["source_trace"] == ["rag", "web"]
    assert result["visited_nodes"].index("rag_retrieve") < result["visited_nodes"].index(
        "web_search"
    )
    assert set(result["synthesizer_output"]["citations"]) == {"rag:fixture.md#chunk-000", "web:001"}


def test_goal_loop_can_request_new_rag_evidence(tmp_path: Path) -> None:
    client = FakeModelClient(
        evaluation_statuses=["missing_information", "satisfied"], gap_routes=["rag"]
    )
    result = run_graph(tmp_path, client=client, thread_id="rag-loop")
    assert result["terminal_status"] == "completed"
    assert result["goal_iterations"] == 2
    assert result["visited_nodes"].count("rag_retrieve") == 1
    assert result["visited_nodes"].count("synthesizer") == 2


def test_goal_loop_can_request_new_web_evidence(tmp_path: Path) -> None:
    client = FakeModelClient(
        evaluation_statuses=["missing_information", "satisfied"], gap_routes=["web"]
    )
    result = run_graph(tmp_path, client=client, thread_id="web-loop")
    assert result["terminal_status"] == "completed"
    assert result["goal_iterations"] == 2
    assert result["visited_nodes"].count("web_search") == 1
    assert result["visited_nodes"].count("synthesizer") == 2


def test_empty_task_fails_before_models_or_sources(tmp_path: Path) -> None:
    client = FakeModelClient(use_rag=True, use_web=True)
    rag = FakeRagSearch()
    web = FakeWebSearch()
    with open_graph(
        tmp_path / "checkpoints.sqlite", model_client=client, rag_search=rag, web_search=web
    ) as graph:
        result = graph.invoke(make_state(episode_id="E-empty", task="   "), make_config("empty"))
    assert result["terminal_status"] == "failed"
    assert client.calls == []
    assert rag.calls == []
    assert web.calls == []


def test_receipt_contains_source_provenance(tmp_path: Path) -> None:
    result = run_graph(
        tmp_path, client=FakeModelClient(use_rag=True, use_web=True), thread_id="receipt"
    )
    receipt = build_receipt(result)
    assert receipt["source_trace"] == ["rag", "web"]
    assert set(receipt["citations"]) == {"rag:fixture.md#chunk-000", "web:001"}
