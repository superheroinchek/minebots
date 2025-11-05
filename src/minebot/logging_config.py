"""Logging utilities for Minebot."""
from __future__ import annotations

from pathlib import Path

from loguru import logger


def setup_logging(log_path: Path, level: str = "INFO") -> None:
    """Configure loguru to write both to stdout and a rotating log file."""

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level=level, rotation="10 MB", retention="10 days")


__all__ = ["logger", "setup_logging"]
