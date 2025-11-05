"""Task planning utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import networkx as nx
from loguru import logger

from .actions import Action, ActionRegistry
from .config import PlannerConfig


@dataclass
class Task:
    """Represents a task for the bot."""

    name: str
    description: str
    required_actions: List[str]
    prerequisites: List[str] = field(default_factory=list)


class Planner:
    """Simple task graph planner for Minebot."""

    def __init__(self, config: PlannerConfig, registry: ActionRegistry) -> None:
        self.config = config
        self.registry = registry
        self.graph = nx.DiGraph()

    def register_task(self, task: Task) -> None:
        self.graph.add_node(task.name, task=task)
        for prereq in task.prerequisites:
            self.graph.add_edge(prereq, task.name)
        logger.debug("Registered task %s", task.name)

    def build_plan(self, goal: str) -> List[Action]:
        if goal not in self.graph:
            raise KeyError(f"Unknown task {goal}")
        order = list(nx.algorithms.dag.topological_sort(self.graph.subgraph(nx.ancestors(self.graph, goal) | {goal})))
        actions: List[Action] = []
        for task_name in order:
            task: Task = self.graph.nodes[task_name]["task"]
            for action_name in task.required_actions:
                try:
                    actions.append(self.registry.get(action_name))
                except KeyError as exc:
                    raise KeyError(f"Task {task_name} references unknown action {action_name}") from exc
        return actions

    def describe(self) -> str:
        lines = ["Registered tasks:"]
        for name in self.graph.nodes:
            task: Task = self.graph.nodes[name]["task"]
            lines.append(f"- {name}: {task.description}")
        return "\n".join(lines)


__all__ = ["Planner", "Task"]
