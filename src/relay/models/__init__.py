"""Local model adapters for Project Relay."""

from relay.models.ollama import (
    ModelInvocationError,
    OllamaClient,
    StructuredModelClient,
)

__all__ = [
    "ModelInvocationError",
    "OllamaClient",
    "StructuredModelClient",
]