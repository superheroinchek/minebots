"""High level planning utilities for the MineBots agent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

from ..memory.episodic import EpisodicMemory, MemoryEvent
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PlanStep:
    """A single actionable instruction for the bot."""

    description: str
    priority: int = 0


class Plan:
    """A mutable sequence of plan steps that can be adapted on the fly."""

    def __init__(self, steps: Iterable[PlanStep] | None = None) -> None:
        self.steps: List[PlanStep] = list(steps or [])

    def push(self, step: PlanStep) -> None:
        logger.debug("Adding plan step: %s", step)
        self.steps.append(step)

    def pop(self) -> PlanStep | None:
        if not self.steps:
            return None
        return self.steps.pop(0)

    def reprioritize(self) -> None:
        self.steps.sort(key=lambda step: step.priority, reverse=True)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.steps)


class Planner:
    """Planner that constructs and updates goals using episodic memory."""

    def __init__(self, memory: EpisodicMemory, default_goal: str) -> None:
        self._memory = memory
        self._default_goal = default_goal

    def build_initial_plan(self) -> Plan:
        logger.info("Building initial plan based on goal: %s", self._default_goal)
        steps = [
            PlanStep("Gather essential resources (wood, stone, food)", priority=10),
            PlanStep("Craft armor and weapons", priority=7),
            PlanStep("Locate and activate Nether portal", priority=5),
            PlanStep("Acquire Blaze rods and Ender pearls", priority=4),
            PlanStep("Locate End portal and defeat the Ender Dragon", priority=3),
        ]
        return Plan(steps)

    def update_plan(self, plan: Plan) -> Plan:
        summary = self._memory.summarize()
        logger.debug("Updating plan using memory summary: %s", summary)
        event = MemoryEvent(
            timestamp=datetime.utcnow(),
            observation="Replanning cycle",
            action="Recalculate priorities",
            outcome=summary,
        )
        self._memory.add_event(event)
        plan.reprioritize()
        return plan
