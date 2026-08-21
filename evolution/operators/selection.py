"""
Tournament selection operator for neuroevolution candidate selection.
"""

import random
from typing import List
from neural.network import NeuralNetwork


class TournamentSelection:
    """
    Selects top neural network candidate from k random tournament entries.
    """

    @staticmethod
    def select(
        networks: List[NeuralNetwork],
        scores: List[float],
        k: int = 3
    ) -> NeuralNetwork:
        """
        Selects candidate with highest fitness score from k random choices.
        """
        pop_size: int = len(networks)
        chosen_indices: List[int] = random.sample(
            range(pop_size), min(k, pop_size)
        )
        best_idx: int = max(chosen_indices, key=lambda idx: scores[idx])
        return networks[best_idx]
