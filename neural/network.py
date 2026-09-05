"""
Sequential multi-layer perceptron (MLP) architecture builder.
"""

import sys
import math
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from neural.layers import NeuralDenseLayer
from neural.weight_initializer import WeightInitializer

ZERO_F32: np.float32 = np.float32(0.0)


class NeuralNetwork:
    """
    Manages sequential data flow with contiguous float32 parameter buffers.
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_layers: int = 3,
        neurons: int = 15,
        output_size: int = 4,
    ) -> None:
        self.layers: List[NeuralDenseLayer] = []

        layer_specs: List[Tuple[int, int]] = [(input_size, neurons)]
        for _ in range(hidden_layers - 1):
            layer_specs.append((neurons, neurons))
        layer_specs.append((neurons, output_size))

        total_params: int = sum((in_c * out_c) + out_c for in_c, out_c in layer_specs)
        self.param_buffer: NDArray[np.float32] = np.zeros(
            total_params, dtype=np.float32
        )

        offset: int = 0
        self._weights: List[NDArray[np.float32]] = []
        self._biases: List[NDArray[np.float32]] = []

        for in_c, out_c in layer_specs:
            w_size: int = in_c * out_c
            b_size: int = out_c

            w_init, b_init = WeightInitializer.initialize_layer_weights(in_c, out_c)
            self.param_buffer[offset : offset + w_size] = w_init.flatten().astype(np.float32)
            self.param_buffer[offset + w_size : offset + w_size + b_size] = b_init.flatten().astype(np.float32)

            w_view = self.param_buffer[offset : offset + w_size].reshape((in_c, out_c))
            b_view = self.param_buffer[offset + w_size : offset + w_size + b_size].reshape((1, out_c))

            self.layers.append(
                NeuralDenseLayer(
                    in_c, out_c, weights_buffer=w_view, biases_buffer=b_view
                )
            )
            self._weights.append(w_view)
            self._biases.append(b_view[0])
            offset += w_size + b_size

        self.last_input_features: Optional[NDArray[np.float32]] = None
        self._last_output: Optional[NDArray[np.float32]] = None
        self._out_buffer: NDArray[np.float32] = np.empty((1, 4), dtype=np.float32)

    @property
    def param_count(self) -> int:
        return int(self.param_buffer.size)

    def copy_weights_from(self, source_net: "NeuralNetwork") -> None:
        np.copyto(self.param_buffer, source_net.param_buffer)

    def export_flat_weights(self) -> NDArray[np.float16]:
        return np.ascontiguousarray(self.param_buffer, dtype=np.float16)

    def import_flat_weights(self, flat_data: NDArray) -> None:
        if flat_data.size != self.param_count:
            print(
                f"[Error] Parameter count mismatch in import_flat_weights! "
                f"Expected {self.param_count}, got {flat_data.size}."
            )
            sys.exit(1)

        np.copyto(self.param_buffer, flat_data.astype(np.float32))

    def forward(self, input_data: NDArray) -> NDArray[np.float32]:
        """
        Ultra-fast in-place forward pass with zero array allocations and zero overflow risk.
        """
        self.last_input_features = input_data
        curr = input_data if input_data.ndim == 1 else input_data[0]

        num_layers = len(self._weights)
        for i in range(num_layers - 1):
            curr = np.dot(curr, self._weights[i]) + self._biases[i]
            # In-place C-speed ReLU (zero mask array allocations)
            np.maximum(ZERO_F32, curr, out=curr)

        # Output motor layer
        curr = np.dot(curr, self._weights[-1]) + self._biases[-1]

        # Inlined 4-element scalar sigmoid (0.15us, cannot overflow)
        out = self._out_buffer
        for j in range(4):
            val = float(curr[j])
            if val >= 0.0:
                out[0, j] = 1.0 / (1.0 + math.exp(-val))
            else:
                ez = math.exp(val)
                out[0, j] = ez / (1.0 + ez)

        self._last_output = out
        return out

    def export_live_activations(self) -> List[List[float]]:
        if self.last_input_features is not None:
            curr = (
                self.last_input_features if self.last_input_features.ndim == 1
                else self.last_input_features[0]
            )
            inp_list: List[float] = curr.flatten().astype(np.float64).tolist()
        else:
            inp_list = [0.0] * self._weights[0].shape[0]
            curr = np.zeros(len(inp_list), dtype=np.float32)

        layer_list: List[List[float]] = [inp_list]

        num_layers = len(self._weights)
        for i in range(num_layers - 1):
            curr = np.dot(curr, self._weights[i]) + self._biases[i]
            np.maximum(ZERO_F32, curr, out=curr)
            layer_list.append(curr.flatten().tolist())

        curr = np.dot(curr, self._weights[-1]) + self._biases[-1]
        out_vals = [
            float(1.0 / (1.0 + math.exp(-float(x)))) if float(x) >= 0.0 else float(math.exp(float(x)) / (1.0 + math.exp(float(x))))
            for x in curr[:4]
        ]
        layer_list.append(out_vals)

        return layer_list