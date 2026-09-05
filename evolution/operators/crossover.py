"""
Uniform crossover operator combining parent neural network weight matrices.
"""

import numpy as np
from neural.network import NeuralNetwork


class UniformCrossover:
    """
    Applies element-wise uniform crossover between parent network weights.
    """

    @staticmethod
    def apply_crossover(
        parent_a: NeuralNetwork,
        parent_b: NeuralNetwork,
        child_net: NeuralNetwork
    ) -> None:
        """
        Combines parent weights and biases in-place using a vectorized array mask.
        """
        mask = np.random.random(child_net.param_buffer.shape) < 0.5
        np.copyto(
            child_net.param_buffer,
            np.where(mask, parent_a.param_buffer, parent_b.param_buffer)
        )