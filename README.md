# Project Relay

Project Relay is a LangGraph learning and engineering experiment.

Its primary purpose is to build and understand a compact but complete
LangGraph system through an end-to-end multi-agent workflow.

The secondary experiment investigates whether four small local language
models gain useful system-level capability when surrounded by a minimal
agentic harness.

## Models

- `phi4-mini:3.8b-q4_K_M` — Scout
- `ministral-3:8b` — Retriever
- `gemma4:e4b` — Researcher
- `qwen3:8b` — Goal formation, synthesis, goal evaluation

LangGraph is the orchestration authority. There is no LLM supervisor.

## Accepted Milestones

### M0 — LangGraph Skeleton

Proved state, routing, SQLite checkpointing, thread isolation,
interrupt/resume, receipts, and failure-aware documentation.

### M1 — Four-Model Graph

Proved the live model path:

`Phi -> Ministral -> Gemma -> Qwen`

through explicit LangGraph state.

## Current Milestone

### M2 — Goal Pursuit

M2 adds:

- explicit Qwen goal formation;
- observable success criteria;
- a focused Qwen grounding audit before goal evaluation;
- Qwen goal evaluation;
- one active subgoal at a time;
- conditional LangGraph routing back to Scout, Retriever, or Researcher;
- a maximum of three goal-evaluation iterations;
- explicit completion, failure, and budget-exhaustion terminals.

M2 still has no RAG, web, or long-term memory. Those capabilities are not
simulated.

## Engineering Standard

Every significant subsystem documents important failure boundaries in:

`docs/FAILURE_MAP.md`