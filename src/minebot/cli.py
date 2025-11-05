"""Command line interface for Minebot."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from .config import default_config, ensure_directories, load_config
from .logging_config import setup_logging
from .minecraft_bot import Goal, run_bot

app = typer.Typer(help="Minebot command line interface")


@app.command()
def init_config(path: Path = typer.Option(Path("minebot.yml"), help="Config file path")) -> None:
    """Generate a default configuration file."""

    config = default_config()
    ensure_directories(config)
    with path.open("w", encoding="utf-8") as fh:
        import yaml

        yaml.safe_dump(
            {
                "llama": {
                    "model_path": config.llama.model_path,
                    "tokenizer_path": config.llama.tokenizer_path,
                    "device": config.llama.device,
                    "max_new_tokens": config.llama.max_new_tokens,
                    "temperature": config.llama.temperature,
                    "top_p": config.llama.top_p,
                    "top_k": config.llama.top_k,
                    "repetition_penalty": config.llama.repetition_penalty,
                },
                "minecraft": {
                    "host": config.minecraft.host,
                    "port": config.minecraft.port,
                    "username": config.minecraft.username,
                },
            },
            fh,
        )
    typer.echo(f"Configuration skeleton written to {path}")


@app.command()
def start(
    config_path: Path = typer.Option(Path("minebot.yml"), help="Path to configuration file"),
    goal: Optional[str] = typer.Option(None, help="High level goal description"),
) -> None:
    """Start the Minebot loop."""

    config = load_config(config_path)
    ensure_directories(config)
    setup_logging(config.log_path)
    goals = [Goal(name="user_goal", description=goal)] if goal else []
    asyncio.run(run_bot(config, goals))


if __name__ == "__main__":
    app()
