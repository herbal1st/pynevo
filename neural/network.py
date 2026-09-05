"""
Sequential multi-layer perceptron (MLP) architecture builder.
"""

import sys
from typing import List, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from neural.layers import NeuralDenseLayer
from neural.activations import ActivationReLU, ActivationSigmoid
from neural.weight_initializer import WeightInitializer


class NeuralNetwork:
    """
    Manages sequential data flow with a contiguous master parameter buffer.
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_layers: int = 3,
        neurons: int = 15,
        output_size: int = 4,
    ) -> None:
        """
        Constructs sequential MLP with layer weights referencing a single 1D buffer.
        """
        self.layers: List[NeuralDenseLayer] = []
        self.activations: List[Any] = []

        layer_specs: List[Tuple[int, int]] = [(input_size, neurons)]
        for _ in range(hidden_layers - 1):
            layer_specs.append((neurons, neurons))
        layer_specs.append((neurons, output_size))

        total_params: int = sum((in_c * out_c) + out_c for in_c, out_c in layer_specs)
        self.param_buffer: NDArray[np.float64] = np.zeros(
            total_params, dtype=np.float64
        )

        offset: int = 0
        for idx, (in_c, out_c) in enumerate(layer_specs):
            w_size: int = in_c * out_c
            b_size: int = out_c

            w_init, b_init = WeightInitializer.initialize_layer_weights(in_c, out_c)
            self.param_buffer[offset : offset + w_size] = w_init.flatten()
            self.param_buffer[offset + w_size : offset + w_size + b_size] = b_init.flatten()

            w_view = self.param_buffer[offset : offset + w_size].reshape((in_c, out_c))
            b_view = self.param_buffer[offset + w_size : offset + w_size + b_size].reshape((1, out_c))

            self.layers.append(
                NeuralDenseLayer(
                    in_c, out_c, weights_buffer=w_view, biases_buffer=b_view
                )
            )

            if idx < len(layer_specs) - 1:
                self.activations.append(ActivationReLU())
            else:
                self.activations.append(None)

            offset += w_size + b_size

        self.out_sigmoid: ActivationSigmoid = ActivationSigmoid()
        self.last_input_features: Optional[NDArray[np.float32]] = None

    @property
    def param_count(self) -> int:
        """
        Returns total number of scalar parameters in master parameter buffer.
        """
        return int(self.param_buffer.size)

    def copy_weights_from(self, source_net: "NeuralNetwork") -> None:
        """
        Copies all weights and biases in a single vectorized buffer copy.
        """
        np.copyto(self.param_buffer, source_net.param_buffer)

    def export_flat_weights(self) -> NDArray[np.float16]:
        """
        Exports all network parameters directly as a contiguous 1D float16 vector.
        """
        return np.ascontiguousarray(self.param_buffer, dtype=np.float16)

    def import_flat_weights(self, flat_data: NDArray) -> None:
        """
        Imports 1D parameter vector in-place in a single operation.
        """
        if flat_data.size != self.param_count:
            print(
                f"[Error] Parameter count mismatch in import_flat_weights! "
                f"Expected {self.param_count}, got {flat_data.size}."
            )
            sys.exit(1)

        np.copyto(self.param_buffer, flat_data.astype(np.float64))

    def forward(self, input_data: NDArray) -> NDArray[np.float64]:
        """
        Forward pass returning 4 wheel outputs (L-FWD, L-BWD, R-FWD, R-BWD).
        """
        self.last_input_features = np.ascontiguousarray(
            input_data.flatten(), dtype=np.float32
        )
        curr: NDArray[np.float64] = (
            np.ascontiguousarray(input_data, dtype=np.float64)
            if input_data.ndim == 2
            else np.ascontiguousarray(input_data[np.newaxis, :], dtype=np.float64)
        )

        for i in range(len(self.layers)):
            curr = self.layers[i].forward(curr)
            if self.activations[i] is not None:
                curr = self.activations[i].forward(curr)

        return self.out_sigmoid.forward(curr[:, 0:4])

    def export_live_activations(self) -> List[List[float]]:
        """
        Returns structured layer activation list from last live forward pass.
        """
        if self.last_input_features is not None:
            inp_list: List[float] = (
                self.last_input_features.astype(np.float64).tolist()
            )
        else:
            inp_list = [0.0] * self.layers[0].weights.shape[0]

        layer_list: List[List[float]] = [inp_list]

        for layer in self.layers[:-1]:
            if layer.output is not None:
                layer_list.append(layer.output.flatten().tolist())
            else:
                layer_list.append([0.0] * layer.weights.shape[1])

        if self.out_sigmoid.output is not None:
            out_vals: List[float] = (
                self.out_sigmoid.output.flatten().tolist()
            )
        else:
            out_vals = [0.0, 0.0, 0.0, 0.0]

        clamped_out: List[float] = [
            max(0.0, min(1.0, float(v))) for v in out_vals
        ]
        layer_list.append(clamped_out)

        return layer_list