"""
Pre-allocated array cache for candidate temporal observation queues.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class TemporalMemoryStacker:
    """
    Maintains temporal observation queues using pre-allocated array caches.
    """

    def __init__(
        self,
        max_candidates: int = 128,
        max_frames: int = 11,
        max_channels: int = 128
    ) -> None:
        """
        Initializes pre-allocated 3D array cache for candidate history.
        """
        self.max_candidates: int = max_candidates
        self.max_frames: int = max_frames
        self.max_channels: int = max_channels
        self._cache: NDArray[np.float32] = np.zeros(
            (max_candidates, max_frames, max_channels),
            dtype=np.float32
        )
        self._initialized: NDArray[np.bool_] = np.zeros(
            max_candidates, dtype=bool
        )

    def reset_candidate_history(self, candidate_idx: int) -> None:
        """
        Zeroes out candidate temporal observation history buffer slot.
        """
        self._ensure_capacity(candidate_idx, 1)
        self._cache[candidate_idx].fill(0.0)
        self._initialized[candidate_idx] = False

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

        base_len: int = int(base_vector.size)
        self._ensure_capacity(candidate_idx, base_len)

        if not self._initialized[candidate_idx]:
            for f_idx in range(total_frames_count):
                self._cache[candidate_idx, f_idx, :base_len] = base_vector
            self._initialized[candidate_idx] = True
        else:
            self._cache[
                candidate_idx, : total_frames_count - 1, :base_len
            ] = self._cache[
                candidate_idx, 1:total_frames_count, :base_len
            ]
            self._cache[
                candidate_idx, total_frames_count - 1, :base_len
            ] = base_vector

        stacked = self._cache[candidate_idx, :total_frames_count, :base_len]
        return stacked.flatten()

    def _ensure_capacity(
        self,
        candidate_idx: int,
        base_len: int
    ) -> None:
        """
        Expands pre-allocated cache if candidate index or channels grow.
        """
        need_cands: int = max(self.max_candidates, candidate_idx + 1)
        need_chans: int = max(self.max_channels, base_len)

        if (
            need_cands > self.max_candidates or
            need_chans > self.max_channels
        ):
            new_cache = np.zeros(
                (need_cands, self.max_frames, need_chans),
                dtype=np.float32
            )
            old_c, old_f, old_ch = self._cache.shape
            new_cache[:old_c, :old_f, :old_ch] = self._cache
            self._cache = new_cache

            new_init = np.zeros(need_cands, dtype=bool)
            new_init[: self._initialized.size] = self._initialized
            self._initialized = new_init

            self.max_candidates = need_cands
            self.max_channels = need_chans
