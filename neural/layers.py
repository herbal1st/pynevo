"""
Vectorized dense linear layer for neural network forward transformations.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from neural.weight_initializer import WeightInitializer


class NeuralDenseLayer:
    """
    Fully connected linear transformation layer (W * X + b).
    """

    def __init__(
        self,
        input_count: int,
        neuron_count: int,
        init_style: str = "XAVIER"
    ) -> None:
        """
        Initializes weight and bias matrices using WeightInitializer.
        """
        self.weights, self.biases = (
            WeightInitializer.initialize_layer_weights(
                input_count, neuron_count, init_style
            )
        )
        self.output: Optional[NDArray[np.float64]] = None

    @property
    def param_count(self) -> int:
        """
        Returns total number of scalar parameters in weights and biases.
        """
        return int(self.weights.size + self.biases.size)

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Executes linear projection forward calculation.
        """
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def export_flat_weights(self) -> NDArray[np.float16]:
        """
        Exports layer weights and biases as a single 1D float16 vector.
        """
        w_flat = self.weights.flatten().astype(np.float16)
        b_flat = self.biases.flatten().astype(np.float16)
        return np.concatenate([w_flat, b_flat])

    def import_flat_weights(self, flat_data: NDArray) -> None:
        """
        Imports flat vector in-place into layer weights and biases.
        """
        w_size: int = int(self.weights.size)
        w_slice = flat_data[:w_size].reshape(self.weights.shape)
        b_slice = flat_data[w_size:].reshape(self.biases.shape)

        np.copyto(self.weights, w_slice.astype(np.float64))
        np.copyto(self.biases, b_slice.astype(np.float64))
