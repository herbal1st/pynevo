"""
Pre-allocated array cache for candidate temporal observation queues.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class TemporalMemoryStacker:
    """
    Maintains temporal observation queues using a zero-allocation flat buffer cache.
    """

    def __init__(
        self,
        max_candidates: int = 128,
        max_frames: int = 11,
        max_channels: int = 128
    ) -> None:
        self.max_candidates: int = max_candidates
        self.max_frames: int = max_frames
        self.max_channels: int = max_channels
        self._max_flat_dim: int = max_frames * max_channels

        # Flat 2D rolling cache: shape (max_candidates, max_frames * max_channels)
        self._flat_cache: NDArray[np.float32] = np.zeros(
            (max_candidates, self._max_flat_dim),
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
        self._flat_cache[candidate_idx].fill(0.0)
        self._initialized[candidate_idx] = False

    def stack_base_vector(
        self,
        candidate_idx: int,
        base_vector: NDArray[np.float32],
        memory_frames: int
    ) -> NDArray[np.float32]:
        """
        Shifts pre-allocated 1D buffer in-place and returns slice view without .flatten() allocations.
        """
        mem_k: int = memory_frames if memory_frames > 0 else 0
        total_frames_count: int = 1 + mem_k

        if total_frames_count <= 1:
            return base_vector

        base_len: int = int(base_vector.size)
        total_len: int = total_frames_count * base_len
        self._ensure_capacity(candidate_idx, base_len)

        buf = self._flat_cache[candidate_idx]

        if not self._initialized[candidate_idx]:
            # Populate all temporal slots with initial frame
            for f_idx in range(total_frames_count):
                st = f_idx * base_len
                buf[st : st + base_len] = base_vector
            self._initialized[candidate_idx] = True
        else:
            # In-place shift left by 1 frame length
            shift_len = total_len - base_len
            buf[:shift_len] = buf[base_len:total_len]
            # Write new frame at the tail
            buf[shift_len:total_len] = base_vector

        # Returns 1D contiguous view directly into cache with ZERO allocations
        return buf[:total_len]

    def _ensure_capacity(
        self,
        candidate_idx: int,
        base_len: int
    ) -> None:
        need_cands: int = self.max_candidates if self.max_candidates > (candidate_idx + 1) else (candidate_idx + 1)
        need_chans: int = self.max_channels if self.max_channels > base_len else base_len
        need_flat_dim: int = self.max_frames * need_chans

        if need_cands > self.max_candidates or need_flat_dim > self._max_flat_dim:
            new_cache = np.zeros((need_cands, need_flat_dim), dtype=np.float32)
            old_c, old_flat = self._flat_cache.shape
            copy_flat = old_flat if old_flat < need_flat_dim else need_flat_dim
            new_cache[:old_c, :copy_flat] = self._flat_cache[:, :copy_flat]
            self._flat_cache = new_cache

            new_init = np.zeros(need_cands, dtype=bool)
            new_init[: self._initialized.size] = self._initialized
            self._initialized = new_init

            self.max_candidates = need_cands
            self.max_channels = need_chans
            self._max_flat_dim = need_flat_dim