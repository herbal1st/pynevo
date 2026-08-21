"""
Manages temporal observation frame stacking queues for candidates.
"""

from typing import Dict, List
import numpy as np
from numpy.typing import NDArray


class TemporalMemoryStacker:
    """
    Maintains temporal observation queues and concatenates history frames.
    """

    def __init__(self) -> None:
        """
        Initializes candidate temporal observation history dictionary.
        """
        self._history: Dict[int, List[NDArray[np.float32]]] = {}

    def reset_candidate_history(self, candidate_idx: int) -> None:
        """
        Clears temporal observation history buffer for candidate.
        """
        if candidate_idx in self._history:
            self._history[candidate_idx].clear()

    def stack_base_vector(
        self,
        candidate_idx: int,
        base_vector: NDArray[np.float32],
        memory_frames: int
    ) -> NDArray[np.float32]:
        """
        Appends base_vector to memory queue and returns concatenated array.
        """
        mem_k: int = max(0, memory_frames)
        total_frames_count: int = 1 + mem_k

        if total_frames_count <= 1:
            return base_vector

        if candidate_idx not in self._history:
            self._history[candidate_idx] = []

        cand_buf: List[NDArray[np.float32]] = self._history[candidate_idx]
        cand_buf.append(base_vector)

        while len(cand_buf) > total_frames_count:
            cand_buf.pop(0)

        while len(cand_buf) < total_frames_count:
            cand_buf.insert(0, base_vector.copy())

        return np.concatenate(cand_buf)
