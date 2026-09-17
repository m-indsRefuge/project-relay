# ================================================================
# Project Relay
# M0 - LangGraph Skeleton
# ================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = "C:\Users\nolan\AIProjects\relay"

function Set-Utf8File {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Content
    )

    $Parent = Split-Path -Parent $Path

    if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }

    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        [System.Text.UTF8Encoding]::new($false)
    )
}

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Relay repository not found: $Root"
}

Set-Location $Root

if (-not (Test-Path -LiteralPath ".git")) {
    throw "Relay is not initialized as a Git repository."
}

$Dirty = git status --porcelain

if ($Dirty) {
    Write-Host ""
    Write-Host "Current worktree changes:"
    Write-Host $Dirty
    Write-Host ""

    throw @"
Refusing to apply M0 to a dirty worktree.

Commit or stash the current work first, then run this script again.
"@
}

Write-Host ""
Write-Host "================================================"
Write-Host " Project Relay - M0 LangGraph Skeleton"
Write-Host "================================================"
Write-Host ""

Set-Utf8File -Path "$Root\pyproject.toml" -Content @'
[project]
name = "project-relay"
version = "0.1.0"
description = "A LangGraph learning and engineering experiment using four small local language models."
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }

dependencies = [
    "langgraph==1.2.11",
    "langgraph-checkpoint-sqlite==3.1.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0,<10",
    "pytest-cov>=7.0,<8",
    "ruff>=0.16,<0.17",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = [
    "E",
    "F",
    "I",
    "UP",
    "B",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/relay"]
'@

Set-Utf8File -Path "$Root\src\relay\state.py" -Content @'
"""Shared LangGraph state for Project Relay M0."""

from typing import Literal, NotRequired, TypedDict


TerminalStatus = Literal[
    "running",
    "completed",
    "failed",
]


class RelayState(TypedDict):
    """Minimal shared state required by the M0 graph."""

    episode_id: str
    task: str
    should_interrupt: bool
    route_reason: str | None
    terminal_status: TerminalStatus
    visited_nodes: list[str]
    resume_value: NotRequired[str]
'@

Set-Utf8File -Path "$Root\src\relay\graph.py" -Content @'
"""Project Relay M0 LangGraph."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from relay.state import RelayState


Route = Literal["complete", "fail", "pause_for_resume"]


def initialize_state(state: RelayState) -> dict:
    """Validate the minimal task and prepare deterministic routing."""

    visited = [*state.get("visited_nodes", []), "initialize_state"]
    task = state.get("task", "")

    if not task.strip():
        return {
            "route_reason": "task is empty",
            "terminal_status": "running",
            "visited_nodes": visited,
        }

    if state.get("should_interrupt", False):
        return {
            "route_reason": "resume probe requested",
            "terminal_status": "running",
            "visited_nodes": visited,
        }

    return {
        "route_reason": "task is valid",
        "terminal_status": "running",
        "visited_nodes": visited,
    }


def route_after_initialize(state: RelayState) -> Route:
    """Choose the next graph node from explicit state."""

    if not state.get("task", "").strip():
        return "fail"

    if state.get("should_interrupt", False):
        return "pause_for_resume"

    return "complete"


def pause_for_resume(state: RelayState) -> dict:
    """Pause graph execution and prove persisted resume behaviour."""

    resume_value = interrupt(
        {
            "kind": "m0_resume_probe",
            "episode_id": state["episode_id"],
            "message": "Resume Project Relay M0.",
        }
    )

    return {
        "resume_value": str(resume_value),
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "pause_for_resume",
        ],
        "route_reason": "checkpointed interrupt resumed",
    }


def complete(state: RelayState) -> dict:
    """Mark the M0 execution successful."""

    return {
        "terminal_status": "completed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "complete",
        ],
    }


def fail(state: RelayState) -> dict:
    """Terminate invalid M0 input visibly."""

    return {
        "terminal_status": "failed",
        "visited_nodes": [
            *state.get("visited_nodes", []),
            "fail",
        ],
    }


def build_graph(*, checkpointer: SqliteSaver):
    """Build and compile the M0 LangGraph."""

    builder = StateGraph(RelayState)

    builder.add_node("initialize_state", initialize_state)
    builder.add_node("pause_for_resume", pause_for_resume)
    builder.add_node("complete", complete)
    builder.add_node("fail", fail)

    builder.add_edge(START, "initialize_state")

    builder.add_conditional_edges(
        "initialize_state",
        route_after_initialize,
        {
            "complete": "complete",
            "fail": "fail",
            "pause_for_resume": "pause_for_resume",
        },
    )

    builder.add_edge("pause_for_resume", "complete")
    builder.add_edge("complete", END)
    builder.add_edge("fail", END)

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_graph(checkpoint_path: Path) -> Iterator:
    """Open a compiled Relay graph backed by a local SQLite checkpointer."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        yield build_graph(checkpointer=checkpointer)
'@

Set-Utf8File -Path "$Root\src\relay\receipt.py" -Content @'
"""Minimal deterministic M0 execution receipt."""

from typing import TypedDict

from relay.state import RelayState


class ExecutionReceipt(TypedDict):
    episode_id: str
    task: str
    terminal_status: str
    route_reason: str | None
    visited_nodes: list[str]


def build_receipt(state: RelayState) -> ExecutionReceipt:
    """Create the small operator-facing receipt required by M0."""

    return {
        "episode_id": state["episode_id"],
        "task": state["task"],
        "terminal_status": state["terminal_status"],
        "route_reason": state.get("route_reason"),
        "visited_nodes": list(state.get("visited_nodes", [])),
    }
'@

Set-Utf8File -Path "$Root\tests\test_graph.py" -Content @'
"""M0 graph behaviour tests."""

from pathlib import Path

from relay.graph import open_graph
from relay.receipt import build_receipt


def make_state(
    *,
    episode_id: str,
    task: str,
    should_interrupt: bool = False,
) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "should_interrupt": should_interrupt,
        "route_reason": None,
        "terminal_status": "running",
        "visited_nodes": [],
    }


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_valid_task_reaches_completed_terminal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-001",
                task="Prove the valid M0 route.",
            ),
            make_config("thread-valid"),
        )

    assert result["terminal_status"] == "completed"
    assert result["route_reason"] == "task is valid"
    assert result["visited_nodes"] == [
        "initialize_state",
        "complete",
    ]


def test_empty_task_reaches_failed_terminal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-002",
                task="   ",
            ),
            make_config("thread-invalid"),
        )

    assert result["terminal_status"] == "failed"
    assert result["route_reason"] == "task is empty"
    assert result["visited_nodes"] == [
        "initialize_state",
        "fail",
    ]


def test_execution_receipt_is_derived_from_terminal_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            make_state(
                episode_id="E-M0-003",
                task="Create an execution receipt.",
            ),
            make_config("thread-receipt"),
        )

    receipt = build_receipt(result)

    assert receipt == {
        "episode_id": "E-M0-003",
        "task": "Create an execution receipt.",
        "terminal_status": "completed",
        "route_reason": "task is valid",
        "visited_nodes": [
            "initialize_state",
            "complete",
        ],
    }
'@

Set-Utf8File -Path "$Root\tests\test_persistence.py" -Content @'
"""M0 checkpoint persistence and isolation tests."""

from pathlib import Path

from langgraph.types import Command

from relay.graph import open_graph


def make_state(
    *,
    episode_id: str,
    task: str,
    should_interrupt: bool = False,
) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "should_interrupt": should_interrupt,
        "route_reason": None,
        "terminal_status": "running",
        "visited_nodes": [],
    }


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_checkpoint_survives_database_reopen(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    config = make_config("persistent-thread")

    with open_graph(checkpoint) as graph:
        graph.invoke(
            make_state(
                episode_id="E-M0-010",
                task="Persist this state.",
            ),
            config,
        )

    with open_graph(checkpoint) as graph:
        snapshot = graph.get_state(config)

    assert snapshot.values["episode_id"] == "E-M0-010"
    assert snapshot.values["task"] == "Persist this state."
    assert snapshot.values["terminal_status"] == "completed"


def test_threads_do_not_contaminate_each_other(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"

    config_a = make_config("thread-a")
    config_b = make_config("thread-b")

    with open_graph(checkpoint) as graph:
        graph.invoke(
            make_state(
                episode_id="E-M0-A",
                task="State belonging to A.",
            ),
            config_a,
        )

        graph.invoke(
            make_state(
                episode_id="E-M0-B",
                task="State belonging to B.",
            ),
            config_b,
        )

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)

    assert state_a.values["episode_id"] == "E-M0-A"
    assert state_a.values["task"] == "State belonging to A."

    assert state_b.values["episode_id"] == "E-M0-B"
    assert state_b.values["task"] == "State belonging to B."


def test_interrupt_can_resume_after_database_reopen(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints.sqlite"
    config = make_config("interrupt-thread")

    with open_graph(checkpoint) as graph:
        events = list(
            graph.stream(
                make_state(
                    episode_id="E-M0-020",
                    task="Exercise checkpointed resume.",
                    should_interrupt=True,
                ),
                config,
            )
        )

        assert any("__interrupt__" in event for event in events)

        snapshot = graph.get_state(config)

        assert snapshot.next
        assert "pause_for_resume" in snapshot.next

    with open_graph(checkpoint) as graph:
        result = graph.invoke(
            Command(resume="M0 resume accepted"),
            config,
        )

    assert result["terminal_status"] == "completed"
    assert result["resume_value"] == "M0 resume accepted"
    assert result["route_reason"] == "checkpointed interrupt resumed"
    assert result["visited_nodes"] == [
        "initialize_state",
        "pause_for_resume",
        "complete",
    ]
'@

Set-Utf8File -Path "$Root\docs\FAILURE_MAP.md" -Content @'
# Project Relay — Failure Map

**Status:** Living operational document
**Milestone:** M0 — LangGraph Skeleton

## FM-001 — Invalid or empty task state

**Component:** M0 graph input / routing
**Code location:** `src/relay/graph.py::initialize_state`

### Observable symptoms

`terminal_status = "failed"`
`route_reason = "task is empty"`

### First check

Inspect terminal graph state or execution receipt.

### Recovery

Correct the task and start a new episode/thread.

### DO NOT

Do not edit checkpoint rows to force a completed state.

### Related tests

`test_empty_task_reaches_failed_terminal`

---

## FM-002 — SQLite checkpoint unavailable

**Component:** LangGraph persistence
**Code location:** `src/relay/graph.py::open_graph`

### Observable symptoms

Checkpoint creation, lookup, or resume raises a SQLite-related exception.

### First check

Preserve the original exception and verify the checkpoint path, permissions,
database ownership, and filesystem health.

### Recovery

Stop Relay and preserve the database before attempting repair.

### DO NOT

Do not delete or manually rewrite `checkpoints.sqlite` as a first response.

### Related tests

`test_checkpoint_survives_database_reopen`

---

## FM-003 — Wrong thread identity

**Component:** checkpoint routing

### Observable symptoms

The wrong episode appears to resume, or expected state cannot be found.

### First check

Inspect `config["configurable"]["thread_id"]`.

### Recovery

Stop execution and resume with the correct thread identity.

### DO NOT

Do not manually move checkpoint rows between thread IDs.

### Related tests

`test_threads_do_not_contaminate_each_other`

---

## FM-004 — Interrupted graph does not resume

**Component:** LangGraph interrupt/checkpoint lifecycle
**Code location:** `src/relay/graph.py::pause_for_resume`

### Observable symptoms

Execution remains paused, cannot locate its checkpoint, or restarts as a fresh
thread.

### First check

Using the exact original thread config, call `graph.get_state(config)` and
confirm a pending node exists.

### Recovery

Verify the thread ID and original checkpoint store before resuming.

### DO NOT

Do not add non-idempotent side effects before `interrupt()`. Interrupted
nodes restart from their beginning during resume.

### Related tests

`test_interrupt_can_resume_after_database_reopen`

---

## M0 Boundary

M0 contains no LLM, RAG, web, long-term memory, or autonomous goal-pursuit
logic. Failure entries for those systems are added only when those capabilities
are implemented.
'@

Write-Host "Synchronizing M0 dependencies..."
uv sync --extra dev

Write-Host ""
Write-Host "Running Ruff..."
uv run ruff check .

Write-Host ""
Write-Host "Running M0 test suite..."
uv run pytest

Write-Host ""
Write-Host "Running compileall..."
uv run python -m compileall -q src tests

Write-Host ""
Write-Host "Git status:"
git status --short

Write-Host ""
Write-Host "================================================"
Write-Host " M0 implementation applied."
Write-Host " Review the diff before committing."
Write-Host "================================================"
Write-Host ""
Write-Host "Suggested review commands:"
Write-Host "  git diff --check"
Write-Host "  git diff"
Write-Host ""
