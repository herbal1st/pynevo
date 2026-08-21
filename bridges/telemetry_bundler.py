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
    """

    def __init__(self, max_steps: int, pop_size: int) -> None:
        """
        Initializes zero-allocation single-generation telemetry buffer.
        """
        self.max_steps: int = max_steps
        self.pop_size: int = pop_size
        self.channels: int = 8
        self._curr_buffer: Optional[NDArray[np.float32]] = None
        self._generations_telemetry: List[NDArray[np.float32]] = []
        self.allocate_generation_buffer()

    def allocate_generation_buffer(self) -> None:
        """
        Pre-allocates flat float32 buffer for a single generation.
        """
        self._curr_buffer = np.zeros(
            (self.max_steps, self.pop_size, self.channels),
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
        """
        Writes candidate step outputs directly into pre-allocated buffer.
        """
        if self._curr_buffer is None:
            print("[Error] Telemetry buffer is not allocated!")
            sys.exit(1)

        if not (0 <= step_idx < self.max_steps):
            print(f"[Error] Step index {step_idx} out of bounds.")
            sys.exit(1)

        if not (0 <= cand_idx < self.pop_size):
            print(f"[Error] Candidate index {cand_idx} out of bounds.")
            sys.exit(1)

        self._curr_buffer[step_idx, cand_idx, 0] = x
        self._curr_buffer[step_idx, cand_idx, 1] = y
        self._curr_buffer[step_idx, cand_idx, 2] = heading
        self._curr_buffer[step_idx, cand_idx, 3] = health
        self._curr_buffer[step_idx, cand_idx, 4] = dist
        self._curr_buffer[step_idx, cand_idx, 5] = (
            1.0 if hit_wall else 0.0
        )
        self._curr_buffer[step_idx, cand_idx, 6] = (
            1.0 if is_alive else 0.0
        )
        self._curr_buffer[step_idx, cand_idx, 7] = (
            1.0 if reached_exit else 0.0
        )

    def finalize_generation(self, actual_steps: int) -> None:
        """
        Truncates trailing dead steps and stores trimmed generation array.
        """
        if self._curr_buffer is None:
            print("[Error] Telemetry buffer is not allocated!")
            sys.exit(1)

        clamped_steps: int = max(
            1, min(actual_steps, self.max_steps)
        )
        trimmed: NDArray[np.float32] = (
            self._curr_buffer[:clamped_steps].copy()
        )
        self._generations_telemetry.append(trimmed)
        self._curr_buffer = None

    def get_generation_telemetry(
        self,
        gen_idx: int
    ) -> NDArray[np.float32]:
        """
        Retrieves truncated telemetry matrix for specified generation.
        """
        if not (0 <= gen_idx < len(self._generations_telemetry)):
            print(
                f"[Error] Generation telemetry index {gen_idx} "
                f"out of bounds (0..{len(self._generations_telemetry) - 1})."
            )
            sys.exit(1)

        return self._generations_telemetry[gen_idx]

    @property
    def all_generations_telemetry(self) -> List[NDArray[np.float32]]:
        """
        Returns all recorded generation telemetry matrices.
        """
        return self._generations_telemetry

    def clear_all(self) -> None:
        """
        Clears current buffer and all recorded generation telemetry.
        """
        self._curr_buffer = None
        self._generations_telemetry.clear()
