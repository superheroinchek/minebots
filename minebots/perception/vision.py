"""Computer vision pipeline for the MineBots agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

from ..config import VisionConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class VisionObservation:
    """A light-weight container for visual frames and derived metadata."""

    frame: np.ndarray
    resized: np.ndarray
    filepath: Optional[Path]


class VisionSystem:
    """Capture, normalize and persist visual information from the game."""

    def __init__(self, config: VisionConfig) -> None:
        self._cfg = config
        self._store = Path(self._cfg.frame_store)
        self._store.mkdir(parents=True, exist_ok=True)

    def ingest(self, frame: np.ndarray, *, save: bool = True) -> VisionObservation:
        """Ingest a raw RGB frame and return a normalized observation."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected RGB frame with shape (H, W, 3)")

        resized = frame[:: self._cfg.downscale_factor, :: self._cfg.downscale_factor, :]
        filepath: Optional[Path] = None
        if save:
            filepath = self._persist_frame(resized)
        logger.debug("Processed vision frame saved to %s", filepath)
        return VisionObservation(frame=frame, resized=resized, filepath=filepath)

    def _persist_frame(self, frame: np.ndarray) -> Path:
        existing = list(self._store.glob("frame_*"))
        idx = len(existing)
        path = self._store / f"frame_{idx:05d}.png"
        if Image is None:
            np.save(path.with_suffix(".npy"), frame)
            return path.with_suffix(".npy")
        image = Image.fromarray(frame.astype(np.uint8))
        image.save(path)
        return path
