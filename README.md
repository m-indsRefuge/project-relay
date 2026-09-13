# Project Relay

Project Relay is a LangGraph learning and engineering experiment.

Its primary purpose is to build and understand a compact but complete
LangGraph system through an end-to-end multi-agent workflow.

The secondary experiment investigates whether four small local language
models gain useful system-level capability when surrounded by a minimal
agentic harness containing:

- explicit goals and bounded goal pursuit;
- structured shared state;
- conditional graph routing;
- RAG;
- external information access;
- working memory;
- episodic memory;
- learned semantic/procedural memory;
- memory consolidation;
- deliberate forgetting;
- checkpoint persistence;
- failure-aware engineering.

## Models

Initial model set:

- `phi4-mini:3.8b-q4_K_M` — Scout
- `ministral-3:8b` — Retriever
- `gemma4:e4b` — Researcher
- `qwen3:8b` — Goal formation, synthesis, evaluation, memory curation

LangGraph is the orchestration authority. There is no LLM supervisor.

## Current Milestone

### M0 — LangGraph Skeleton

M0 exists to prove:

- explicit `StateGraph`;
- typed graph state;
- deterministic nodes;
- conditional edges;
- success and failure terminals;
- SQLite checkpoint persistence;
- thread isolation;
- interruption/resume behaviour;
- execution receipts;
- failure-aware documentation.

No LLM, RAG, web, or long-term memory behaviour belongs in M0.

## Engineering Standard

Every significant subsystem must document its known failure boundaries in:

`docs/FAILURE_MAP.md`

Failure documentation evolves alongside implementation and tests.