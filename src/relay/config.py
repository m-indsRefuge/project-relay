"""Central configuration for Project Relay."""

import os
from dataclasses import dataclass
from pathlib import Path

SCOUT_MODEL = "phi4-mini:3.8b-q4_K_M"
RETRIEVER_MODEL = "ministral-3:8b"
RESEARCHER_MODEL = "gemma4:e4b"
SYNTHESIZER_MODEL = "qwen3:8b"

RELAY_MODELS = (
    SCOUT_MODEL,
    RETRIEVER_MODEL,
    RESEARCHER_MODEL,
    SYNTHESIZER_MODEL,
)

MAX_GOAL_ITERATIONS = 3


@dataclass(frozen=True)
class RelayConfig:
    """Runtime configuration for Relay."""

    data_dir: Path = Path("data")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    memory_db: Path = Path("data/memory.sqlite")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_timeout_seconds: float = 180.0

    max_goal_iterations: int = MAX_GOAL_ITERATIONS


DEFAULT_CONFIG = RelayConfig()