# Project Relay

Project Relay is a LangGraph learning and engineering experiment.

Its primary purpose is to build and understand a compact but complete
LangGraph system through an end-to-end multi-agent workflow.

The secondary experiment investigates whether four small local language
models gain useful system-level capability when surrounded by a minimal
agentic harness.

## Models

Initial model set:

- `phi4-mini:3.8b-q4_K_M` — Scout
- `ministral-3:8b` — Retriever
- `gemma4:e4b` — Researcher
- `qwen3:8b` — Synthesizer and later integrative functions

LangGraph is the orchestration authority. There is no LLM supervisor.

## Accepted Milestone

### M0 — LangGraph Skeleton

M0 proved:

- explicit `StateGraph`;
- typed shared state;
- deterministic routing;
- explicit success/failure paths;
- SQLite checkpoint persistence;
- thread isolation;
- checkpointed interrupt/resume;
- execution receipts;
- failure-aware documentation.

## Current Milestone

### M1 — Four-Model Graph

M1 introduces exactly four local model-backed LangGraph nodes:

`Scout -> Retriever -> Researcher -> Synthesizer`

M1 deliberately does **not** introduce:

- RAG;
- web access;
- long-term memory;
- dynamic goals;
- iterative agent loops.

Those capabilities belong to later milestones.

## Engineering Standard

Every significant subsystem must document its known failure boundaries in:

`docs/FAILURE_MAP.md`

Failure documentation evolves alongside implementation and tests.