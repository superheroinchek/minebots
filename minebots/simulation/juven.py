"""Juven simulation: lightweight Minecraft-inspired world with vision snapshots."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from ..config import SimulationConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


Coordinate = Tuple[int, int]


@dataclass(slots=True)
class SceneSnapshot:
    """Representation of the simulated world state."""

    frame: np.ndarray
    ascii_map: str
    description: str
    resources_remaining: int


class JuvenSimulator:
    """Grid-based simulation that mimics high-level Minecraft gameplay."""

    _DIRS = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }

    def __init__(self, config: SimulationConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)
        self._size = config.world_size
        self._agent: Coordinate = (self._size // 2, self._size // 2)
        self._resources = self._spawn_resources()
        self._tick = 0
        logger.info(
            "Initialized Juven simulation (size=%s, resources=%s)",
            self._size,
            len(self._resources),
        )

    async def connect(self) -> None:
        """Placeholder to keep API parity with the networked client."""

        await asyncio.sleep(0)
        logger.info("Juven simulation ready")

    async def disconnect(self) -> None:
        await asyncio.sleep(0)
        logger.info("Juven simulation shut down")

    async def observe(self, hint: str | None = None) -> SceneSnapshot:
        """Return the current snapshot of the simulated world."""

        self._tick += 1
        description = self._describe_state(hint)
        snapshot = self._build_snapshot(description)
        logger.debug("Observation generated: %s", description)
        return snapshot

    async def apply_action(self, action: str) -> SceneSnapshot:
        """Interpret the controller action and update the world state."""

        text = action.lower()
        logger.debug("Applying action to Juven simulation: %s", text)
        if any(keyword in text for keyword in ("move", "walk", "go")):
            description = self._handle_movement(text)
        elif any(keyword in text for keyword in ("gather", "mine", "collect")):
            description = self._handle_gathering()
        elif "build" in text or "craft" in text:
            description = self._handle_building(text)
        else:
            description = "Action had no clear effect; agent contemplates the surroundings."
        snapshot = self._build_snapshot(description)
        return snapshot

    def _handle_movement(self, text: str) -> str:
        for name, delta in self._movement_candidates(text):
            new_pos = self._clamp((self._agent[0] + delta[0], self._agent[1] + delta[1]))
            if new_pos != self._agent:
                self._agent = new_pos
                return f"Agent moved {name} to tile {self._agent}."
        return "Movement command unclear; agent stays put."

    def _movement_candidates(self, text: str) -> Iterable[Tuple[str, Coordinate]]:
        for name, delta in self._DIRS.items():
            if name in text:
                yield name, delta
        if "forward" in text or "ahead" in text:
            yield "north", self._DIRS["north"]
        if "back" in text or "backward" in text:
            yield "south", self._DIRS["south"]
        if "left" in text:
            yield "west", self._DIRS["west"]
        if "right" in text:
            yield "east", self._DIRS["east"]

    def _handle_gathering(self) -> str:
        if self._agent in self._resources:
            self._resources.remove(self._agent)
            return "Agent gathered nearby resources and cleared the tile."
        nearby = self._nearest_resource()
        if nearby is None:
            return "No resources remain in the Juven simulation."
        return f"No resources on current tile. Nearest resource located at {nearby}."

    def _handle_building(self, text: str) -> str:
        if "portal" in text:
            return "Agent assembles a mock Nether portal using collected resources."
        if "camp" in text or "shelter" in text:
            return "Agent constructs a temporary shelter with available materials."
        if "tool" in text or "weapon" in text:
            return "Agent crafts improved tools for future tasks."
        return "Agent experiments with crafting but produces nothing notable."

    def _describe_state(self, hint: str | None) -> str:
        resources = len(self._resources)
        base = (
            f"Tick {self._tick}: Agent at {self._agent}. Resources remaining: {resources}."
        )
        if hint:
            base += f" Focus: {hint}."
        if resources == 0:
            base += " World is cleared; agent may prepare for endgame."
        elif resources < self._cfg.alert_threshold:
            base += " Only a handful of resources remain nearby."
        return base

    def _build_snapshot(self, description: str) -> SceneSnapshot:
        ascii_map = self._ascii_overview()
        frame = self._render_frame()
        return SceneSnapshot(
            frame=frame,
            ascii_map=ascii_map,
            description=description,
            resources_remaining=len(self._resources),
        )

    def _spawn_resources(self) -> List[Coordinate]:
        resources: List[Coordinate] = []
        for _ in range(self._cfg.resource_clusters):
            center = self._rng.integers(0, self._size, size=2)
            cluster_size = self._rng.integers(2, self._cfg.max_cluster_size + 1)
            for _ in range(cluster_size):
                offset = self._rng.integers(-2, 3, size=2)
                pos = self._clamp((center[0] + offset[0], center[1] + offset[1]))
                if pos not in resources and pos != self._agent:
                    resources.append(pos)
        return resources

    def _nearest_resource(self) -> Coordinate | None:
        if not self._resources:
            return None
        distances = [
            (abs(r[0] - self._agent[0]) + abs(r[1] - self._agent[1]), r)
            for r in self._resources
        ]
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    def _ascii_overview(self) -> str:
        grid = [["." for _ in range(self._size)] for _ in range(self._size)]
        for x, y in self._resources:
            grid[x][y] = "R"
        ax, ay = self._agent
        grid[ax][ay] = "A"
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)

    def _render_frame(self) -> np.ndarray:
        height, width = self._cfg.vision_resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        ax, ay = self._agent
        scale_x = height / self._size
        scale_y = width / self._size
        agent_x = int(ax * scale_x)
        agent_y = int(ay * scale_y)
        cell_h = max(1, int(scale_x))
        cell_w = max(1, int(scale_y))
        frame[agent_x : agent_x + cell_h, agent_y : agent_y + cell_w, :] = [
            255,
            255,
            255,
        ]
        for rx, ry in self._resources:
            res_x = int(rx * scale_x)
            res_y = int(ry * scale_y)
            frame[res_x : res_x + cell_h, res_y : res_y + cell_w, :] = [
                255,
                165,
                0,
            ]
        noise = self._rng.normal(0, self._cfg.vision_noise, size=frame.shape)
        noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        return noisy_frame

    def _clamp(self, pos: Coordinate) -> Coordinate:
        return (
            max(0, min(self._size - 1, pos[0])),
            max(0, min(self._size - 1, pos[1])),
        )
