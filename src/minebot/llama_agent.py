"""LLaMA 11B integration for action planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import LLaMAConfig
from .memory import LongTermMemory, ShortTermMemory


PROMPT_TEMPLATE = """You are Minebot, an advanced Minecraft assistant. You receive the player's observation, short-term memory and goals.\n""" \
    "Decide the next high-level action in JSON with keys action and params. Choose from actions: {actions}."""


@dataclass
class LLaMAAgent:
    """Wrapper around Hugging Face models for inference."""

    config: LLaMAConfig
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    generation_config: GenerationConfig

    @classmethod
    def load(cls, config: LLaMAConfig) -> "LLaMAAgent":
        tokenizer_path = config.tokenizer_path or config.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if config.device == "auto" else config.device,
        )
        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )
        return cls(config, tokenizer, model, generation_config)

    def _build_prompt(
        self,
        observation: str,
        goals: Iterable[str],
        short_term: ShortTermMemory,
        long_term: Optional[LongTermMemory],
        actions: str,
    ) -> str:
        goal_text = "\n".join(f"- {goal}" for goal in goals)
        ltm_text = "\n".join(f"[{item.type}] {item.content}" for item in (long_term.entries if long_term else []))
        prompt = (
            PROMPT_TEMPLATE.format(actions=actions)
            + f"\n## Observation\n{observation}\n"
            + f"\n## Current goals\n{goal_text}\n"
            + f"\n## Short term memory\n{short_term.to_prompt()}\n"
        )
        if ltm_text:
            prompt += f"\n## Knowledge base\n{ltm_text}\n"
        prompt += "\nRespond with a single JSON object like {\"action\": \"move\", \"params\": {\"direction\": \"forward\"}}."
        return prompt

    def suggest_action(
        self,
        observation: str,
        goals: Iterable[str],
        short_term: ShortTermMemory,
        long_term: Optional[LongTermMemory],
        actions: str,
    ) -> Dict[str, object]:
        prompt = self._build_prompt(observation, goals, short_term, long_term, actions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_json = response[len(prompt) :].strip()
        if not result_json:
            raise ValueError("Model produced empty response")
        try:
            import json

            parsed = json.loads(result_json.splitlines()[0])
        except Exception as exc:  # noqa: BLE001 - broad to emit debugging info
            raise ValueError(f"Failed to parse model output: {response}") from exc
        return parsed


__all__ = ["LLaMAAgent", "PROMPT_TEMPLATE"]
