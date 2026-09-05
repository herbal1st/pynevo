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
        init_style: str = "XAVIER",
        weights_buffer: Optional[NDArray[np.float64]] = None,
        biases_buffer: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Initializes weight and bias matrices using C-contiguous memory views.
        """
        if weights_buffer is not None and biases_buffer is not None:
            self.weights = weights_buffer
            self.biases = biases_buffer
        else:
            w, b = WeightInitializer.initialize_layer_weights(
                input_count, neuron_count, init_style
            )
            self.weights = np.ascontiguousarray(w, dtype=np.float64)
            self.biases = np.ascontiguousarray(b, dtype=np.float64)
        self.output: Optional[NDArray[np.float64]] = None

    @property
    def param_count(self) -> int:
        """
        Returns total number of scalar parameters in weights and biases.
        """
        return int(self.weights.size + self.biases.size)

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Executes linear projection forward calculation with contiguous arrays.
        """
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        input_c = np.ascontiguousarray(input_data, dtype=np.float64)
        self.output = np.dot(input_c, self.weights) + self.biases
        return self.output

    def export_flat_weights(self) -> NDArray[np.float16]:
        """
        Exports layer weights and biases as a single 1D float16 vector.
        """
        w_flat = np.ascontiguousarray(self.weights.flatten(), dtype=np.float16)
        b_flat = np.ascontiguousarray(self.biases.flatten(), dtype=np.float16)
        return np.concatenate([w_flat, b_flat])

    def import_flat_weights(self, flat_data: NDArray) -> None:
        """
        Imports flat vector in-place into layer weights and biases.
        """
        w_size: int = int(self.weights.size)
        w_slice = np.ascontiguousarray(
            flat_data[:w_size].reshape(self.weights.shape), dtype=np.float64
        )
        b_slice = np.ascontiguousarray(
            flat_data[w_size:].reshape(self.biases.shape), dtype=np.float64
        )

        np.copyto(self.weights, w_slice)
        np.copyto(self.biases, b_slice)