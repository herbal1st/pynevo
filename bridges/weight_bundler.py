"""
Contiguous float16 tensor bundler for population neural weights.
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
        """
        Initializes contiguous float16 master weight tensor buffer.
        """
        self.num_generations: int = num_generations
        self.pop_size: int = pop_size
        self.param_count: int = param_count
        self._tensor: NDArray[np.float16] = np.zeros(
            (num_generations, pop_size, param_count),
            dtype=np.float16
        )

    @property
    def master_tensor(self) -> NDArray[np.float16]:
        """
        Returns master 3D float16 weight tensor.
        """
        return self._tensor

    def record_generation_weights(
        self,
        gen_idx: int,
        networks: List[NeuralNetwork]
    ) -> None:
        """
        Exports and packs candidate network weights for a generation.
        """
        if not (0 <= gen_idx < self.num_generations):
            print(
                f"[Error] Generation index {gen_idx} out of bounds "
                f"(0..{self.num_generations - 1})."
            )
            sys.exit(1)

        if len(networks) != self.pop_size:
            print(
                f"[Error] Network count {len(networks)} does not match "
                f"population size {self.pop_size}."
            )
            sys.exit(1)

        for c_idx, net in enumerate(networks):
            flat_w = net.export_flat_weights()
            if flat_w.size != self.param_count:
                print(
                    f"[Error] Flat weight size {flat_w.size} does not "
                    f"match expected param_count {self.param_count}."
                )
                sys.exit(1)
            self._tensor[gen_idx, c_idx, :] = flat_w

    def get_candidate_weights(
        self,
        gen_idx: int,
        cand_idx: int
    ) -> NDArray[np.float16]:
        """
        Retrieves flat 1D float16 parameter vector for candidate.
        """
        if not (0 <= gen_idx < self.num_generations):
            print(
                f"[Error] Generation index {gen_idx} out of bounds "
                f"(0..{self.num_generations - 1})."
            )
            sys.exit(1)

        if not (0 <= cand_idx < self.pop_size):
            print(
                f"[Error] Candidate index {cand_idx} out of bounds "
                f"(0..{self.pop_size - 1})."
            )
            sys.exit(1)

        return self._tensor[gen_idx, cand_idx, :]

    def set_master_tensor(
        self,
        tensor: NDArray[np.float16]
    ) -> None:
        """
        Restores master weight tensor from loaded replay archive.
        """
        if tensor.ndim != 3:
            print(
                f"[Error] Master weight tensor must be 3D, "
                f"got {tensor.ndim}D."
            )
            sys.exit(1)

        self.num_generations = tensor.shape[0]
        self.pop_size = tensor.shape[1]
        self.param_count = tensor.shape[2]
        self._tensor = tensor.astype(np.float16)
