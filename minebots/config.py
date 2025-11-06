"""Configuration utilities for the MineBots agent."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
import json
import os

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass(slots=True)
class VisionConfig:
    """Settings related to the visual pipeline."""

    frame_store: Path = Path("./frames")
    max_history: int = 60
    downscale_factor: int = 2


@dataclass(slots=True)
class TransformerConfig:
    """Configuration for the transformer controller."""

    model_name: str = "microsoft/phi-2"
    device: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.2
    use_8bit: bool = False
    use_flash_attention: bool = False


@dataclass(slots=True)
class SimulationConfig:
    """Settings for the Juven simulation environment."""

    world_size: int = 16
    resource_clusters: int = 6
    max_cluster_size: int = 4
    alert_threshold: int = 5
    vision_resolution: tuple[int, int] = (192, 192)
    vision_noise: float = 8.0
    seed: int = 42


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for episodic memory."""

    max_events: int = 2048
    summary_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class AgentConfig:
    """High level configuration that ties together all components."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    goal: str = "Complete the Minecraft playthrough efficiently and safely."
    max_cycles: int = 50

    @classmethod
    def from_env(cls, base: Optional["AgentConfig"] = None) -> "AgentConfig":
        """Create a configuration object populated from environment variables."""

        def _json_env(name: str) -> Optional[dict[str, Any]]:
            value = os.getenv(name)
            if not value:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON for {name}: {value}") from exc

        base_cfg = base or cls()
        payload: dict[str, Any] = {}

        simulation_env = _json_env("MINEBOTS_SIMULATION")
        if simulation_env:
            payload["simulation"] = simulation_env

        vision_env = _json_env("MINEBOTS_VISION")
        if vision_env:
            payload["vision"] = vision_env

        transformer_env = _json_env("MINEBOTS_TRANSFORMER")
        if transformer_env:
            payload["transformer"] = transformer_env

        memory_env = _json_env("MINEBOTS_MEMORY")
        if memory_env:
            payload["memory"] = memory_env

        goal_env = os.getenv("MINEBOTS_GOAL")
        if goal_env:
            payload["goal"] = goal_env

        max_cycles_env = os.getenv("MINEBOTS_MAX_CYCLES")
        if max_cycles_env:
            payload["max_cycles"] = int(max_cycles_env)

        return cls._apply_mapping(base_cfg, payload)

    @classmethod
    def from_yaml(cls, path: Path | str, *, base: Optional["AgentConfig"] = None) -> "AgentConfig":
        """Load configuration overrides from a YAML file."""

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if yaml is None:  # pragma: no cover - requires optional dependency
            raise RuntimeError("pyyaml is required to load YAML configuration files")

        with config_path.open("r", encoding="utf-8") as handle:
            loaded: dict[str, Any] = yaml.safe_load(handle) or {}

        base_cfg = base or cls()
        return cls._apply_mapping(base_cfg, loaded)

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "AgentConfig":
        """Load configuration from YAML or environment variables."""

        if path:
            base_cfg = cls.from_yaml(path)
            return cls.from_env(base_cfg)
        return cls.from_env()

    @classmethod
    def _apply_mapping(cls, base_cfg: "AgentConfig", payload: dict[str, Any]) -> "AgentConfig":
        simulation_cfg = cls._update_simulation(base_cfg.simulation, payload.get("simulation", {}))
        vision_cfg = cls._update_vision(base_cfg.vision, payload.get("vision", {}))
        transformer_cfg = cls._update_transformer(base_cfg.transformer, payload.get("transformer", {}))
        memory_cfg = cls._update_memory(base_cfg.memory, payload.get("memory", {}))
        goal = payload.get("goal", base_cfg.goal)
        max_cycles = payload.get("max_cycles", base_cfg.max_cycles)

        return cls(
            simulation=simulation_cfg,
            vision=vision_cfg,
            transformer=transformer_cfg,
            memory=memory_cfg,
            goal=goal,
            max_cycles=max_cycles,
        )

    @staticmethod
    def _update_simulation(
        base_cfg: SimulationConfig, overrides: dict[str, Any]
    ) -> SimulationConfig:
        if not overrides:
            return base_cfg
        payload = dict(overrides)
        if "vision_resolution" in payload:
            payload["vision_resolution"] = tuple(payload["vision_resolution"])
        return replace(base_cfg, **payload)

    @staticmethod
    def _update_vision(base_cfg: VisionConfig, overrides: dict[str, Any]) -> VisionConfig:
        if not overrides:
            return base_cfg
        payload = dict(overrides)
        if "frame_store" in payload:
            payload["frame_store"] = Path(payload["frame_store"])
        return replace(base_cfg, **payload)

    @staticmethod
    def _update_transformer(
        base_cfg: TransformerConfig, overrides: dict[str, Any]
    ) -> TransformerConfig:
        if not overrides:
            return base_cfg
        return replace(base_cfg, **overrides)

    @staticmethod
    def _update_memory(base_cfg: MemoryConfig, overrides: dict[str, Any]) -> MemoryConfig:
        if not overrides:
            return base_cfg
        return replace(base_cfg, **overrides)
