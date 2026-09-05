"""
Gaussian noise mutation operator for neural network weight perturbation.
"""

import random
import numpy as np
from neural.network import NeuralNetwork


class GaussianMutation:
    """
    Applies Gaussian zero-mean noise mutations to network weight matrices.
    """

    @staticmethod
    def apply_mutation(
        child_net: NeuralNetwork,
        mutation_rate: float,
        mutation_scale: float
    ) -> None:
        """
        Applies Gaussian noise in-place to the contiguous parameter buffer.
        """
        if random.random() >= mutation_rate:
            return

        noise = np.random.normal(
            0.0, mutation_scale, size=child_net.param_buffer.shape
        )
        np.add(child_net.param_buffer, noise, out=child_net.param_buffer)