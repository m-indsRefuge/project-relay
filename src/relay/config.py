"""Central configuration for Project Relay.

Configuration should remain explicit and small.

Values that are experimental parameters belong here rather than being
scattered through graph nodes.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RelayConfig:
    """Runtime configuration for Relay."""

    data_dir: Path = Path("data")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    memory_db: Path = Path("data/memory.sqlite")


DEFAULT_CONFIG = RelayConfig()