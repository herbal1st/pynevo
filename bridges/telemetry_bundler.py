"""
Zero-allocation step logger and active step truncation bundler.
"""

import sys
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


class TelemetryBundler:
    """
    Manages pre-allocation, step logging, and truncation for telemetry.
    Forced to 1 step during headless training to keep archives extremely compact.
    """

    def __init__(self, max_steps: int, pop_size: int) -> None:
        self.max_steps: int = 1  # Force max_steps to 1 during training
        self.pop_size: int = pop_size
        self.channels: int = 8
        self._curr_buffer: Optional[NDArray[np.float32]] = None
        self._generations_telemetry: List[NDArray[np.float32]] = []
        self.allocate_generation_buffer()

    def allocate_generation_buffer(self) -> None:
        self._curr_buffer = np.zeros(
            (1, self.pop_size, self.channels),
            dtype=np.float32
        )

    def record_step_data(
        self,
        step_idx: int,
        cand_idx: int,
        x: float,
        y: float,
        heading: float,
        health: float,
        dist: float,
        hit_wall: bool,
        is_alive: bool,
        reached_exit: bool
    ) -> None:
        pass

    def finalize_generation(self, actual_steps: int) -> None:
        if self._curr_buffer is None:
            self.allocate_generation_buffer()

        trimmed: NDArray[np.float32] = self._curr_buffer[:1].copy()
        self._generations_telemetry.append(trimmed)
        self._curr_buffer = None

    def get_generation_telemetry(
        self,
        gen_idx: int
    ) -> NDArray[np.float32]:
        if not (0 <= gen_idx < len(self._generations_telemetry)):
            return np.zeros((1, self.pop_size, self.channels), dtype=np.float32)

        return self._generations_telemetry[gen_idx]

    @property
    def all_generations_telemetry(self) -> List[NDArray[np.float32]]:
        return self._generations_telemetry

    def clear_all(self) -> None:
        self._curr_buffer = None
        self._generations_telemetry.clear()
