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

---

## FM-011 — Goal declared satisfied prematurely

**Component:** Qwen goal evaluation
**Code location:** `src/relay/nodes/goal.py`
**Responsibility:** Decide whether the candidate answer meets the explicit
goal and success criteria.

### Observable symptoms

Relay completes even though the candidate answer visibly fails one or more
success criteria.

### First check

Inspect the stored `goal`, `goal_evaluation`, `synthesizer_output`, and
`goal_iterations` in the terminal checkpoint/receipt.

### Propagation

Premature completion prevents the graph from taking a useful corrective loop.

### Recovery

Reproduce the episode and review the goal-evaluation prompt/schema before
changing routing logic.

### DO NOT

Do not add unconditional extra loops merely to hide evaluator quality problems.

---

## FM-012 — Goal loop exhausts its iteration budget

**Component:** LangGraph M2 goal loop
**Code location:** `src/relay/graph.py::route_after_goal_evaluation`
**Responsibility:** Bound autonomous re-processing.

### Observable symptoms

`terminal_status = "failed"`

and:

`route_reason = "goal iteration budget exhausted"`

after three goal evaluations.

### First check

Inspect `goal_iterations`, `goal_evaluation`, `active_subgoal`, and
`visited_nodes`.

### Propagation

Relay stops rather than looping indefinitely.

### Recovery

Determine whether the task genuinely lacks information, the evaluator is
creating ineffective subgoals, or the specialist route is unproductive.

### DO NOT

Do not raise the iteration limit before diagnosing why the existing loops
failed.

### Related tests

`test_goal_iteration_budget_fails_closed`

---

## FM-013 — Invalid or non-actionable subgoal route

**Component:** Goal evaluation schema / LangGraph routing
**Code locations:** `src/relay/nodes/schemas.py`, `src/relay/graph.py`

### Observable symptoms

A `missing_information` decision cannot produce a valid next node or the model
returns a route outside Scout, Retriever, and Researcher.

### First check

Inspect the structured `GoalEvaluationOutput`.

### Recovery

Preserve the invalid model output and correct the evaluator prompt/schema
boundary. The graph must fail closed rather than invent a route.

### DO NOT

Do not route to RAG, web, or memory during M2; those subsystems do not exist yet.

### Related tests

`test_missing_information_requires_subgoal_and_route`

---

## FM-014 — Goal loop uses stale specialist context incorrectly

**Component:** M2 shared state during targeted rerouting
**Code locations:** `src/relay/nodes/`
**Responsibility:** Allow a targeted specialist pass to build on existing state
without pretending untouched evidence is new.

### Observable symptoms

A looped answer attributes old Scout/Retriever/Researcher output to the new
subgoal or appears to treat stale context as newly discovered evidence.

### First check

Inspect `active_subgoal`, `visited_nodes`, and each specialist output around
the rerouted pass.

### Recovery

Determine whether the targeted node prompt correctly distinguishes existing
context from the active subgoal.

### DO NOT

Do not erase all prior state on every loop; doing so would turn goal pursuit
into repeated stateless execution rather than LangGraph stateful refinement.
### Observed M2 live smoke goal-drift incident

The first M2 live smoke exposed a concrete version of this failure.

Original user task:

`Describe a cautious diagnostic approach ...`

Qwen formed the stronger objective:

`Identify the cause of the service failure ...`

The success criteria then required checking configuration state, service state,
and logs even though M2 had no tools capable of performing those checks.

The synthesizer correctly described how those checks should be performed, but
the evaluator declared the drifted goal satisfied without receiving the
original user task as part of its evaluation context.

### Root cause

The semantic authority chain was incomplete:

`original user task -> generated goal -> evaluation`

Goal evaluation received the generated goal but not the original task, so it
could not directly detect that goal formation had changed the requested
deliverable.

### Corrective action

- Goal formation explicitly preserves the requested deliverable.
- Goal evaluation receives the original task as authoritative context.
- The evaluator distinguishes describing/planning an action from actually
  performing that action.
- Regression tests protect both prompt contracts and evaluator context.

### Related tests

- `test_goal_contract_explicitly_preserves_requested_deliverable`
- `test_evaluator_contract_distinguishes_describing_from_performing`
- `test_goal_evaluator_receives_original_user_task`
---

## FM-015 — Inference promoted to supplied fact

**Component:** Multi-model evidence boundary
**Code locations:** `src/relay/nodes/scout.py`, `retriever.py`,
`researcher.py`, `synthesizer.py`, `goal.py`

### Observed symptom

During the M2.1 live smoke, the original task stated only that a service stopped
working immediately after a configuration change.

Scout added this to `known`:

`The change was the only recent event prior to the service failure.`

That exclusivity claim was not supplied by the user. The synthesizer later
repeated it as a supporting point.

### Root cause

The graph correctly preserved model outputs, but the semantic contracts did
not strongly distinguish:

- supplied fact;
- inference;
- hypothesis;
- question;
- recommendation.

Because downstream nodes consume upstream state, an unsupported inference can
gain apparent authority simply by being repeated across multiple model nodes.

### Propagation path

`unsupported Scout known -> Retriever context -> Researcher reasoning ->
Synthesizer supporting point -> possible evaluator acceptance`

### First diagnostics

Inspect:

- original `task`;
- `scout_output.known`;
- `retriever_output.organized_context`;
- `researcher_output.hypotheses`;
- `synthesizer_output.supporting_points`;
- `goal_evaluation`.

Compare every factual claim against the original task or later real evidence.

### Recovery

Strengthen the existing node contracts so supplied facts remain distinct from
hypotheses and recommendations. Re-run the original episode and verify that
unsupported exclusivity/causality claims disappear or remain explicitly
hypothetical.

### Data risk

Medium. Once long-term memory exists, an unsupported inference that is treated
as fact could later be persisted and amplified across episodes.

### DO NOT

Do not treat agreement among multiple Relay models as independent evidence.
They may all be reasoning from the same unsupported upstream statement.

### Related tests

- `test_scout_known_is_limited_to_supplied_facts`
- `test_retriever_must_not_promote_inference_to_fact`
- `test_researcher_keeps_hypotheses_nonfactual`
- `test_synthesizer_rejects_unsupported_supporting_points`
- `test_goal_evaluator_checks_candidate_against_original_task`

---

## FM-016 — Goal evaluator accepts a candidate despite unsupported claims

**Component:** M2 grounding / goal-evaluation boundary
**Code locations:** `src/relay/nodes/grounding.py`, `goal.py`,
`synthesizer.py`, `graph.py`

### Observed symptom

The M2.2 live smoke correctly grounded Scout, but Researcher produced an
overstated causal hypothesis and Synthesizer promoted it into
`supporting_points`:

`The service failure is directly attributable to the configuration change...`

Goal Evaluator nevertheless returned `satisfied` and claimed the answer
contained no unsupported factual claims.

### Root cause

Prompt-only grounding instructions were insufficient. Goal evaluation had two
jobs at once:

1. check factual grounding;
2. decide whether the goal was satisfied.

A single semantic decision could therefore miss a grounding error and still
authorize terminal completion.

### Corrective architecture

M2.3 separates those responsibilities:

`synthesis -> focused grounding audit -> goal evaluation`

The grounding audit uses Qwen but is a separate narrow LangGraph node.

If the audit reports unsupported claims, deterministic application logic
overrides any `satisfied` evaluator decision and creates a bounded revision
subgoal routed directly back to Synthesizer.

### Propagation control

An unsupported claim can no longer reach `complete` merely because Goal
Evaluator overlooks it. The graph requires a subsequent audit to return
`grounded`, or the existing goal-iteration budget eventually fails closed.

### DO NOT

Do not treat the grounding auditor as an external fact checker. In M2 it can
only compare candidate claims against the supplied user task.

### Related tests

- `test_grounding_audit_forces_synthesizer_revision`
- `test_grounded_audit_rejects_unsupported_claims`
- `test_revision_audit_requires_an_issue`

---

## FM-017 — Embedding model unavailable or embedding request fails

**Component:** Local RAG embedding boundary
**Code location:** `src/relay/rag/embeddings.py`

RAG retrieval fails before ranked knowledge reaches graph state. First confirm Ollama is running and
`embeddinggemma:300m-qat-q4_0` is visible. Restore the exact model/runtime and rerun retrieval.

**DO NOT:** silently replace semantic retrieval with arbitrary or fabricated chunks.

---

## FM-018 — Local RAG index has no knowledge documents

**Component:** M3 local knowledge index
**Code location:** `src/relay/rag/index.py`

`RagError` reports that no Markdown documents were found. Restore the version-controlled knowledge
documents. Do not create placeholder evidence merely so retrieval can continue.

---

## FM-019 — Live web search fails or returns no usable results

**Component:** M3 external web boundary
**Code location:** `src/relay/tools/web.py`

Network/provider failure or throttling may prevent live results. Preserve the failure and do not
substitute model knowledge while claiming that the web confirmed it.

---

## FM-020 — RAG and web evidence are conflated

**Component:** M3 source provenance

A local project document must not be presented as current external evidence, and a web snippet must
not be presented as controlled Relay documentation. Inspect `kind`, `source_id`, `source`, and
`retrieved_at`. Do not collapse the two evidence classes into an unlabeled blob.

---

## FM-021 — Synthesizer invents or omits source identifiers

**Component:** M3 provenance / grounding boundary
**Code location:** `src/relay/nodes/grounding.py`

Unknown citation IDs or missing citations when evidence was supplied force `needs_revision` before
terminal completion. Never accept plausible-looking citation strings that are absent from graph state.
