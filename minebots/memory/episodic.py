"""Episodic memory implementation for MineBots."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Iterable, List

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MemoryEvent:
    """Representation of a high level event the agent should remember."""

    timestamp: datetime
    observation: str
    action: str
    outcome: str
    summary: str | None = None


class EpisodicMemory:
    """Fixed size memory that stores recent events for planning and recall."""

    def __init__(self, max_events: int) -> None:
        self._events: Deque[MemoryEvent] = deque(maxlen=max_events)

    def add_event(self, event: MemoryEvent) -> None:
        logger.debug("Recording memory event: %s", event)
        self._events.append(event)

    def summarize(self) -> str:
        """Return a compact textual summary of the stored events."""

        if not self._events:
            return "No significant events recorded yet."
        summaries: List[str] = []
        for event in self._events:
            summary = event.summary or f"Observed {event.observation}; action {event.action}; outcome {event.outcome}."
            summaries.append(summary)
        return " \n".join(summaries)

    def __iter__(self) -> Iterable[MemoryEvent]:
        return iter(self._events)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._events)
