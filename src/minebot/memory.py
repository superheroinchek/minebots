"""Memory subsystems for Minebot."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Deque, Dict, Iterable, List

from collections import deque

from loguru import logger


@dataclass
class MemoryItem:
    """Represents a single memory entry."""

    type: str
    content: Dict[str, object]


@dataclass
class ShortTermMemory:
    """Fixed-size buffer for recent events."""

    maxlen: int = 50
    _buffer: Deque[MemoryItem] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.maxlen)

    def add(self, item: MemoryItem) -> None:
        logger.debug("STM add: {}", item)
        self._buffer.append(item)

    def to_prompt(self) -> str:
        return "\n".join(f"[{item.type}] {item.content}" for item in self._buffer)

    def items(self) -> Iterable[MemoryItem]:
        return list(self._buffer)


@dataclass
class LongTermMemory:
    """Persistent store for goals, recipes and knowledge."""

    path: Path
    entries: List[MemoryItem] = field(default_factory=list)

    def load(self) -> None:
        if not self.path.exists():
            logger.warning("Long term memory file %s not found, starting fresh", self.path)
            self.entries = []
            return
        with self.path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.entries = [MemoryItem(**entry) for entry in payload]
        logger.info("Loaded %d memory items", len(self.entries))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump([item.__dict__ for item in self.entries], fh, ensure_ascii=False, indent=2)
        logger.debug("Persisted %d memory items", len(self.entries))

    def add(self, item: MemoryItem) -> None:
        logger.debug("LTM add: {}", item)
        self.entries.append(item)
        self.save()

    def search(self, keyword: str) -> List[MemoryItem]:
        keyword_lower = keyword.lower()
        return [entry for entry in self.entries if keyword_lower in str(entry.content).lower()]


__all__ = [
    "MemoryItem",
    "ShortTermMemory",
    "LongTermMemory",
]
