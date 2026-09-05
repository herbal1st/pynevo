"""
Neural weight matrix initializers (Xavier/Glorot, He, Gaussian).
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class WeightInitializer:
    """
    Generates layer weight and bias matrices using scaling strategies.
    """

    @staticmethod
    def initialize_layer_weights(
        input_count: int,
        neuron_count: int,
        style: str = "XAVIER"
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Creates C-contiguous weight matrix and zero bias array scaled by initialization style.
        """
        style_upper: str = style.upper()

        if style_upper == "HE":
            scale: float = np.sqrt(2.0 / float(input_count))
        elif style_upper == "GAUSSIAN":
            scale = 0.1
        else:
            scale = np.sqrt(2.0 / float(input_count + neuron_count))

        weights: NDArray[np.float64] = np.ascontiguousarray(
            scale * np.random.randn(input_count, neuron_count),
            dtype=np.float64
        )
        biases: NDArray[np.float64] = np.ascontiguousarray(
            np.zeros((1, neuron_count), dtype=np.float64),
            dtype=np.float64
        )

        return weights, biases