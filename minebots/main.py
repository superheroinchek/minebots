"""CLI entry point for MineBots."""
from __future__ import annotations

import argparse
import asyncio
import logging

from .agent import MineBotAgent
from .utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MineBots Juven simulation agent")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML configuration file with MineBots settings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=level)
    agent = MineBotAgent(config_path=args.config) if args.config else MineBotAgent()
    asyncio.run(agent.run())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
