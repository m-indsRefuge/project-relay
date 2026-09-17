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
---

## FM-005 — Validation command fails but milestone script continues

**Component:** Engineering / milestone validation
**Code location:** `m0-implement.ps1`
**Responsibility:** Prevent an invalid milestone from being reported as successfully applied.

### Failure point

A native validation command such as Ruff, pytest, compileall, `uv`, or Git
returns a non-zero process exit code, but PowerShell continues running because
the exit code was not explicitly converted into a terminating script error.

### Observable symptoms

A validation command reports failure, but later validation steps continue and
the script may still print a successful completion message.

### Likely causes

`$ErrorActionPreference = "Stop"` handles PowerShell terminating errors, but
native executable exit codes must still be checked explicitly by the script.

### First check

Inspect:

`$LASTEXITCODE`

immediately after the native command that reported failure.

### Evidence

The original M0 implementation run demonstrated this failure mode:

- Ruff returned two `I001` violations.
- pytest still ran and passed.
- compileall still ran.
- the script still printed `M0 implementation applied.`

### Propagation

This failure can create a false-positive milestone result: the software may
contain a failed validation gate even though the automation reports successful
completion.

### Recovery

1. Correct the underlying validation failure.
2. Add an explicit native exit-code guard.
3. Re-run every milestone validation gate from the beginning.
4. Do not accept or commit the milestone until all gates pass.

### Retry safety

Safe after correcting the original validation failure.

### Data risk

Low for validation-only commands, but operational risk is high because a bad
build could otherwise be accepted or released.

### DO NOT

Do not treat the script's final success message as authoritative when an
earlier validation command reported failure.

### Related protection

Critical native commands in milestone automation must be followed by:

`Assert-NativeSuccess "<operation>"`

### Related tests / verification

This failure is verified operationally by forcing or observing a native
command with a non-zero exit code and confirming the script terminates before
later gates execute.

---

## FM-006 — Ollama runtime unavailable

**Component:** Local model runtime
**Code location:** `src/relay/models/ollama.py`
**Responsibility:** Provide local structured inference to M1 model nodes.

### Observable symptoms

A model invocation or model-inventory request raises `ModelInvocationError`
before a valid model output reaches graph state.

### Likely causes

- Ollama is not running;
- the configured host is wrong;
- the local API is unreachable.

### First check

Verify the configured `OLLAMA_HOST` and query the Ollama model inventory.

### Propagation

The active model node cannot complete, so downstream model nodes must not run.

### Recovery

Restore the Ollama runtime and rerun the affected episode from a known-safe
checkpoint or a new thread as appropriate.

### DO NOT

Do not reinterpret a runtime connectivity failure as a model reasoning failure.

### Related tests

`test_ollama_http_failure_is_wrapped`

---

## FM-007 — Required Relay model missing

**Component:** Local model inventory
**Code location:** `src/relay/models/ollama.py::ensure_models_available`

### Observable symptoms

The live M1 smoke test stops before graph execution and names one or more
required models that are not visible to Ollama.

### First check

Run `ollama list` and compare exact model tags with `relay.config.RELAY_MODELS`.

### Recovery

Install or restore the exact required model tag, or deliberately amend the
design before substituting another model.

### DO NOT

Do not silently substitute a different model during the formal experiment.

### Related tests

`test_missing_required_model_fails_before_graph_execution`

---

## FM-008 — Model returns malformed structured output

**Component:** Model output boundary
**Code location:** `src/relay/models/ollama.py::invoke`

### Observable symptoms

A node raises `ModelInvocationError` because model output is empty, malformed
JSON, or incompatible with the required Pydantic schema.

### First check

Identify the model and output schema named by the original exception.

### Propagation

The current LangGraph node fails before invalid data is written into shared
state. Downstream nodes should not receive fabricated fallback content.

### Recovery

Preserve the failing prompt/output evidence, reproduce the failure, and decide
whether prompt/schema correction is required.

### DO NOT

Do not coerce malformed output into graph state by dropping required fields.

### Related tests

- `test_ollama_malformed_json_fails_closed`
- `test_ollama_schema_violation_fails_closed`

---

## FM-009 — Upstream specialist context missing

**Component:** Four-model state handoff
**Code locations:** `src/relay/nodes/`
**Responsibility:** Pass validated structured outputs through LangGraph state.

### Observable symptoms

A downstream node cannot access an expected upstream output such as
`scout_output` or `retriever_output`.

### Likely causes

- graph edge bypassed the expected node;
- upstream node failed before committing state;
- state schema or key changed incompatibly.

### First check

Inspect `visited_nodes`, `model_trace`, and the checkpoint immediately before
the failing node.

### Propagation

Later specialists may be unable to construct their prompt. This should be
treated as a state/graph failure rather than a model-quality problem.

### Recovery

Restore the expected graph path or state contract, then rerun regression tests.

### DO NOT

Do not fabricate a missing upstream output merely to allow later nodes to run.

### Related tests

`test_valid_task_runs_all_four_models`

---

## FM-010 — Four-model sequence differs from design

**Component:** LangGraph M1 topology
**Code location:** `src/relay/graph.py`
**Responsibility:** Execute Scout, Retriever, Researcher, and Synthesizer in
the frozen M1 sequence.

### Observable symptoms

`model_trace` is missing a model, contains a duplicate, contains an unexpected
model, or executes models in a different order.

### First check

Inspect `model_trace` and `visited_nodes` in the terminal receipt.

### Recovery

Review graph edges and node-model configuration before changing prompts or
model behaviour.

### DO NOT

Do not accept M1 based only on a plausible final answer. The expected graph
path itself is part of the milestone contract.

### Related tests

- `test_valid_task_runs_all_four_models`
- `test_interrupt_resumes_into_four_model_path_after_reopen`
