"""Main orchestrator for the MineBots agent."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

from .config import AgentConfig
from .memory.episodic import EpisodicMemory, MemoryEvent
from .perception.vision import VisionObservation, VisionSystem
from .planning.planner import Plan, Planner
from .simulation import JuvenSimulator, SceneSnapshot
from .utils.logging import get_logger

logger = get_logger(__name__)


class TransformerController:
    """Wrapper around a HuggingFace transformer for fast action generation."""

    def __init__(self, config: AgentConfig) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers and torch are required for the controller")
        self._cfg = config.transformer
        logger.info("Loading transformer model %s", self._cfg.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = {"torch_dtype": dtype}
        if self._cfg.use_8bit:
            kwargs["load_in_8bit"] = True
        if self._cfg.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        device_map = "auto" if torch.cuda.is_available() else None
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._cfg.model_name,
            device_map=device_map,
            **kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def build_prompt(self, goal: str, plan: Plan, memory: str, observation: str) -> str:
        steps = "\n".join(f"- {step.description}" for step in plan.steps[:5])
        prompt = (
            "You are an elite Minecraft agent."
            f"\nPrimary goal: {goal}."
            "\nCurrent top plan steps:\n"
            f"{steps}"
            "\nRecent memory summary:\n"
            f"{memory}"
            "\nLatest observation:\n"
            f"{observation}"
            "\nRespond with the next high impact action in imperative form."
        )
        return prompt

    def choose_action(self, goal: str, plan: Plan, memory: str, observation: str) -> str:
        prompt = self.build_prompt(goal, plan, memory, observation)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            do_sample=self._cfg.temperature > 0.0,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded.strip().split("\n")[0]


class MineBotAgent:
    """High level agent that combines the Juven simulation and the controller."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        config_path: Optional[str] = None,
    ) -> None:
        if config and config_path:
            raise ValueError("Provide either a config object or a config path, not both")
        if config_path:
            self._cfg = AgentConfig.load(config_path)
        else:
            self._cfg = config or AgentConfig.from_env()
        self._memory = EpisodicMemory(self._cfg.memory.max_events)
        self._planner = Planner(self._memory, self._cfg.goal)
        self._vision = VisionSystem(self._cfg.vision)
        self._controller = TransformerController(self._cfg)
        self._simulator = JuvenSimulator(self._cfg.simulation)
        self._plan = self._planner.build_initial_plan()

    async def run(self) -> None:
        await self._simulator.connect()
        try:
            await self._control_loop()
        finally:
            await self._simulator.disconnect()

    async def _control_loop(self) -> None:
        for _ in range(self._cfg.max_cycles):
            if not self._plan:
                self._plan = self._planner.build_initial_plan()
            step = self._plan.pop()
            if step is None:
                await asyncio.sleep(0.1)
                continue
            observation = await self._gather_observation(step.description)
            memory_summary = self._memory.summarize()
            action = self._controller.choose_action(
                self._cfg.goal,
                self._plan,
                memory_summary,
                observation,
            )
            logger.info("Chosen action: %s", action)
            outcome_snapshot = await self._simulator.apply_action(action)
            self._record_snapshot(observation, action, outcome_snapshot)
            self._plan = self._planner.update_plan(self._plan)
            await asyncio.sleep(0.05)

    async def _gather_observation(self, hint: str) -> str:
        snapshot = await self._simulator.observe(hint)
        vision_obs = self._vision.ingest(snapshot.frame)
        observation = (
            f"Step hint: {hint}."
            f"\nScene description: {snapshot.description}"
            f"\nASCII overview:\n{snapshot.ascii_map}"
            f"\nStored frame: {vision_obs.filepath or 'not saved'}"
            f"\nResized frame shape: {vision_obs.resized.shape}"
        )
        return observation

    def _record_snapshot(self, observation: str, action: str, snapshot: SceneSnapshot) -> None:
        vision_obs: VisionObservation = self._vision.ingest(snapshot.frame)
        outcome = (
            f"{snapshot.description}"
            f"\nASCII overview after action:\n{snapshot.ascii_map}"
            f"\nFrame archived at {vision_obs.filepath}"
        )
        self._memory.add_event(
            MemoryEvent(
                timestamp=datetime.utcnow(),
                observation=observation,
                action=action,
                outcome=outcome,
            )
        )
