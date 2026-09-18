# Project Relay Architecture

Project Relay is primarily a LangGraph learning and engineering experiment.
LangGraph is Relay's orchestration authority. It owns graph state, node transitions,
conditional routing, checkpointed execution, and bounded loops. There is no LLM supervisor.

Relay has four generative roles: Phi-4 Mini Scout, Ministral Retriever, Gemma Researcher,
and Qwen3 integrative functions. The embedding model used by RAG is not a generative agent.

In M3, local RAG and web search are separate evidence classes. RAG answers questions about
version-controlled local project documents. Web search supplies external public information
that may change over time. A model claim is not evidence merely because another model repeats it.