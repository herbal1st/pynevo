"""
Uniform crossover operator combining parent neural network weight matrices.
"""

import numpy as np
from numpy.typing import NDArray

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
        Combines parent weights and biases in-place into child network.
        """
        for i in range(len(parent_a.layers)):
            wa = parent_a.layers[i].weights
            wb = parent_b.layers[i].weights
            ba = parent_a.layers[i].biases
            bb = parent_b.layers[i].biases

            cw = child_net.layers[i].weights
            cb = child_net.layers[i].biases

            mask_w: NDArray[np.bool_] = (
                np.random.rand(*wa.shape) < 0.5
            )
            mask_b: NDArray[np.bool_] = (
                np.random.rand(*ba.shape) < 0.5
            )

            np.copyto(cw, np.where(mask_w, wa, wb))
            np.copyto(cb, np.where(mask_b, ba, bb))
