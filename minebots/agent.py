"""Core orchestration logic for the miniature reasoning agent."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

from .tools import StylistInput, thinker_tool, stylist_tool


@dataclass
class AgentResponse:
    """Represents a single reply produced by :class:`MiniAgent`."""

    user_message: str
    reasoning: str
    final_answer: str


class MiniAgent:
    """A small, dependency-light agent that mimics tool orchestration.

    The agent always performs a reasoning pass with :func:`thinker_tool` and
    then refines the message using :func:`stylist_tool`. Past interactions are
    stored in a limited history so that the web interface can display previous
    turns if needed. The implementation is deterministic to keep tests fast.
    """

    def __init__(self, history_limit: int = 10):
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._history: Deque[AgentResponse] = deque(maxlen=history_limit)

    @property
    def history(self) -> List[AgentResponse]:
        """Return the stored conversation history."""

        return list(self._history)

    def respond(self, message: str) -> AgentResponse:
        """Process ``message`` and return an :class:`AgentResponse`.

        The logic mirrors the pseudo-agent from the original script but keeps
        the execution entirely CPU-bound and lightweight. This satisfies the
        requirement for "меньше нагрузка" while keeping behaviour predictable.
        """

        reasoning = thinker_tool(message)
        final_answer = stylist_tool(StylistInput(user_message=message, reasoning=reasoning))
        response = AgentResponse(user_message=message, reasoning=reasoning, final_answer=final_answer)
        self._history.append(response)
        return response

    def export_history(self) -> List[Tuple[str, str]]:
        """Return only the user and assistant messages for display purposes."""

        return [(item.user_message, item.final_answer) for item in self._history]


__all__ = ["MiniAgent", "AgentResponse"]
