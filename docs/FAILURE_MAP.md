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
