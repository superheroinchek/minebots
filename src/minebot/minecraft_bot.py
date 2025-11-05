"""High-level orchestrator for Minebot."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .actions import ActionContext, registry
from .config import MinebotConfig
from .llama_agent import LLaMAAgent
from .logging_config import logger
from .memory import LongTermMemory, MemoryItem, ShortTermMemory
from .minecraft_client import MinecraftClient
from .planner import Planner, Task


@dataclass
class Goal:
    """Represents a persistent objective."""

    name: str
    description: str


@dataclass
class MinecraftBot:
    """Combines perception, planning and action."""

    config: MinebotConfig
    goals: List[Goal] = field(default_factory=list)
    short_term_memory: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term_memory: Optional[LongTermMemory] = None
    llama: Optional[LLaMAAgent] = None
    planner: Optional[Planner] = None
    client: Optional[MinecraftClient] = None

    async def setup(self) -> None:
        logger.info("Setting up Minebot components")
        self.long_term_memory = LongTermMemory(self.config.memory_path)
        self.long_term_memory.load()
        self.llama = LLaMAAgent.load(self.config.llama)
        self.planner = Planner(self.config.planner, registry)
        self.client = MinecraftClient(self.config.minecraft)
        await self.client.connect()

    async def shutdown(self) -> None:
        if self.client:
            await self.client.disconnect()
        if self.long_term_memory:
            self.long_term_memory.save()

    async def loop(self) -> None:
        if not self.client or not self.llama:
            raise RuntimeError("Bot not set up")
        observation = await self.client.observe()
        obs_text = self._format_observation(observation)
        goals_text = [goal.description for goal in self.goals]
        action_description = registry.describe()
        suggestion = self.llama.suggest_action(
            observation=obs_text,
            goals=goals_text,
            short_term=self.short_term_memory,
            long_term=self.long_term_memory,
            actions=action_description,
        )
        action_name = suggestion.get("action")
        params = suggestion.get("params", {})
        if not isinstance(action_name, str):
            raise ValueError(f"Model suggested invalid action: {suggestion}")
        action = registry.get(action_name)
        context = ActionContext(client=self.client, description=obs_text)
        await action.run(context, params)
        self.short_term_memory.add(MemoryItem(type="action", content={"name": action_name, "params": params}))

    def _format_observation(self, observation: dict) -> str:
        position = observation.get("position", {})
        return (
            f"Position: x={position.get('x')}, y={position.get('y')}, z={position.get('z')}\n"
            f"Health: {observation.get('health')}"
        )

    def add_goal(self, goal: Goal) -> None:
        logger.info("Adding goal: %s", goal)
        self.goals.append(goal)

    def register_default_tasks(self) -> None:
        if not self.planner:
            raise RuntimeError("Planner not initialised")
        self.planner.register_task(
            Task(
                name="gather_wood",
                description="Collect wood by chopping logs.",
                required_actions=["move", "mine_block"],
            )
        )
        self.planner.register_task(
            Task(
                name="build_shelter",
                description="Construct a basic shelter using collected blocks.",
                required_actions=["move", "place_block"],
                prerequisites=["gather_wood"],
            )
        )

    async def execute_plan(self, goal_name: str) -> None:
        if not self.planner or not self.client:
            raise RuntimeError("Planner or client not initialised")
        actions = self.planner.build_plan(goal_name)
        context = ActionContext(client=self.client, description="Plan execution")
        for action in actions:
            await action.run(context, {})
            self.short_term_memory.add(MemoryItem(type="plan_step", content={"action": action.name}))


async def run_bot(config: MinebotConfig, goals: Iterable[Goal]) -> None:
    bot = MinecraftBot(config=config)
    for goal in goals:
        bot.add_goal(goal)
    bot.register_default_tasks()
    await bot.setup()
    try:
        while True:
            await bot.loop()
            await asyncio.sleep(2)
    finally:
        await bot.shutdown()


__all__ = ["MinecraftBot", "Goal", "run_bot"]
