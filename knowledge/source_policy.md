# Relay Source Policy

Relay keeps user facts, local RAG evidence, web evidence, and model reasoning distinct.
RAG is appropriate for project-specific documentation and controlled local reference material.
Web search is appropriate for current or external public information.

When RAG or web evidence materially contributes to an answer, Relay preserves source IDs.
A failed lookup remains visible; Relay must not fabricate replacement evidence.