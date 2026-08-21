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
        Applies Gaussian noise in-place to weights based on mutation rate.
        """
        if random.random() >= mutation_rate:
            return

        for i in range(len(child_net.layers)):
            cw = child_net.layers[i].weights
            cb = child_net.layers[i].biases

            noise_w = np.random.normal(
                0.0, mutation_scale, size=cw.shape
            )
            noise_b = np.random.normal(
                0.0, mutation_scale, size=cb.shape
            )

            np.add(cw, noise_w, out=cw)
            np.add(cb, noise_b, out=cb)
