"""Configuration management for Minebot."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class LLaMAConfig:
    """Configuration for the LLaMA language model."""

    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.05


@dataclass
class MinecraftConfig:
    """Connection details for the Minecraft server."""

    host: str = "localhost"
    port: int = 25565
    username: str = "Minebot"
    password: Optional[str] = None
    version: Optional[str] = None


@dataclass
class PlannerConfig:
    """Configures task planning behaviour."""

    max_depth: int = 4
    max_branching: int = 3
    reconsider_interval: int = 30


@dataclass
class MinebotConfig:
    """Top-level configuration container."""

    llama: LLaMAConfig
    minecraft: MinecraftConfig = field(default_factory=MinecraftConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    action_timeout: int = 20
    memory_path: Path = Path("storage/memory.json")
    log_path: Path = Path("logs/minebot.log")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinebotConfig":
        llama = LLaMAConfig(**data["llama"])
        minecraft = MinecraftConfig(**data.get("minecraft", {}))
        planner = PlannerConfig(**data.get("planner", {}))
        return cls(
            llama=llama,
            minecraft=minecraft,
            planner=planner,
            action_timeout=data.get("action_timeout", 20),
            memory_path=Path(data.get("memory_path", "storage/memory.json")),
            log_path=Path(data.get("log_path", "logs/minebot.log")),
        )


def load_config(path: Path) -> MinebotConfig:
    """Load configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return MinebotConfig.from_dict(raw)


def default_config() -> MinebotConfig:
    """Create a default configuration skeleton."""

    return MinebotConfig(
        llama=LLaMAConfig(model_path="./models/llama-11b"),
    )


def ensure_directories(config: MinebotConfig) -> None:
    """Ensure directories for logs and memory exist."""

    for path in [config.memory_path.parent, config.log_path.parent]:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "LLaMAConfig",
    "MinecraftConfig",
    "PlannerConfig",
    "MinebotConfig",
    "default_config",
    "ensure_directories",
    "load_config",
]
