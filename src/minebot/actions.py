"""Action primitives and skills for Minebot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from loguru import logger


class ActionError(RuntimeError):
    """Raised when an action cannot be executed."""


@dataclass
class ActionContext:
    """Context available to actions."""

    client: "MinecraftClient"
    description: str


ActionHandler = Callable[[ActionContext, Dict[str, object]], Awaitable[None]]


@dataclass
class Action:
    """Represents an executable action."""

    name: str
    handler: ActionHandler
    description: str

    async def run(self, context: ActionContext, params: Optional[Dict[str, object]] = None) -> None:
        params = params or {}
        logger.info("Executing action %s with params %s", self.name, params)
        await self.handler(context, params)


class ActionRegistry:
    """Registry of atomic and high-level skills."""

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if action.name in self._actions:
            raise ValueError(f"Action {action.name} already registered")
        self._actions[action.name] = action
        logger.debug("Registered action %s", action.name)

    def get(self, name: str) -> Action:
        if name not in self._actions:
            raise KeyError(f"Action {name} not found")
        return self._actions[name]

    def describe(self) -> str:
        return "\n".join(f"- {name}: {action.description}" for name, action in self._actions.items())


registry = ActionRegistry()


async def move_handler(context: ActionContext, params: Dict[str, object]) -> None:
    direction = params.get("direction", "forward")
    distance = float(params.get("distance", 1))
    await context.client.move(direction, distance)


async def mine_block_handler(context: ActionContext, params: Dict[str, object]) -> None:
    block = params.get("block")
    if not isinstance(block, str):
        raise ActionError("mine_block action requires a 'block' parameter")
    await context.client.mine_block(block)


async def place_block_handler(context: ActionContext, params: Dict[str, object]) -> None:
    block = params.get("block")
    await context.client.place_block(block)


async def craft_handler(context: ActionContext, params: Dict[str, object]) -> None:
    item = params.get("item")
    quantity = int(params.get("quantity", 1))
    if not isinstance(item, str):
        raise ActionError("craft action requires an 'item' parameter")
    await context.client.craft_item(item, quantity)


async def attack_handler(context: ActionContext, params: Dict[str, object]) -> None:
    target = params.get("target")
    await context.client.attack(target)


registry.register(
    Action(
        name="move",
        handler=move_handler,
        description="Move in a direction with an optional distance parameter.",
    )
)
registry.register(
    Action(
        name="mine_block",
        handler=mine_block_handler,
        description="Break the specified block type.",
    )
)
registry.register(
    Action(
        name="place_block",
        handler=place_block_handler,
        description="Place a block from inventory.",
    )
)
registry.register(
    Action(
        name="craft",
        handler=craft_handler,
        description="Craft an item given available resources.",
    )
)
registry.register(
    Action(
        name="attack",
        handler=attack_handler,
        description="Attack a target (mob or player).",
    )
)


__all__ = [
    "Action",
    "ActionContext",
    "ActionError",
    "ActionRegistry",
    "registry",
]
