"""
Contiguous float16 tensor bundler for population neural weights.
Memory-safe: records champion & top elites to eliminate multi-gigabyte RAM spikes.
"""

import sys
from typing import List
import numpy as np
from numpy.typing import NDArray

from neural.network import NeuralNetwork


class WeightBundler:
    """
    Manages pre-allocation, ingestion, & retrieval of flat weight tensors.
    """

    def __init__(
        self,
        num_generations: int,
        pop_size: int,
        param_count: int
    ) -> None:
        self.num_generations: int = num_generations
        self.pop_size: int = pop_size
        self.param_count: int = param_count
        self._tensor: NDArray[np.float16] = np.zeros(
            (num_generations, pop_size, param_count),
            dtype=np.float16
        )

    @property
    def master_tensor(self) -> NDArray[np.float16]:
        return self._tensor

    def record_generation_weights(
        self,
        gen_idx: int,
        networks: List[NeuralNetwork]
    ) -> None:
        if not (0 <= gen_idx < self.num_generations):
            return

        n_to_record = min(len(networks), self.pop_size)
        for c_idx in range(n_to_record):
            flat_w = networks[c_idx].export_flat_weights()
            if flat_w.size == self.param_count:
                self._tensor[gen_idx, c_idx, :] = flat_w

    def get_candidate_weights(
        self,
        gen_idx: int,
        cand_idx: int
    ) -> NDArray[np.float16]:
        if not (0 <= gen_idx < self.num_generations):
            return np.zeros(self.param_count, dtype=np.float16)

        safe_cand = min(max(0, cand_idx), self.pop_size - 1)
        return self._tensor[gen_idx, safe_cand, :]

    def set_master_tensor(
        self,
        tensor: NDArray[np.float16]
    ) -> None:
        if tensor.ndim != 3:
            return

        self.num_generations = tensor.shape[0]
        self.pop_size = tensor.shape[1]
        self.param_count = tensor.shape[2]
        self._tensor = tensor.astype(np.float16)
