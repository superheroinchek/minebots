"""Logging helpers for MineBots."""
from __future__ import annotations

import logging
from logging import Logger
from typing import Optional


def configure_logging(level: int = logging.INFO) -> Logger:
    """Configure and return the root logger with a fast, structured formatter."""

    logger = logging.getLogger("minebots")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> Logger:
    """Return a module specific logger."""

    root = configure_logging()
    if name:
        return root.getChild(name)
    return root
