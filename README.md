# Project Relay

Project Relay is a LangGraph learning and engineering experiment.
Its primary purpose is to build and understand a compact but complete LangGraph system.

## Generative Models
- `phi4-mini:3.8b-q4_K_M` — Scout
- `ministral-3:8b` — Retriever
- `gemma4:e4b` — Researcher
- `qwen3:8b` — goal formation, source planning, synthesis, grounding, evaluation

Relay also uses `embeddinggemma:300m-qat-q4_0` for local RAG embeddings. It is not a fifth generative agent.
LangGraph remains the orchestration authority.

## Accepted Milestones
- M0 — LangGraph Skeleton
- M1 — Four-Model Graph
- M2 — Goal Pursuit and Grounding

## Current Milestone: M3 — RAG + Web Evidence
M3 adds Qwen source planning, local Markdown RAG, Ollama embeddings, in-memory cosine retrieval,
live no-key web search, separate RAG/web evidence in graph state, source IDs/timestamps, explicit
citations, provenance-aware grounding checks, and RAG/web routes in the bounded goal loop.

M3 still has no long-term episodic or learned memory.

Important subsystem failure boundaries live in `docs/FAILURE_MAP.md`.