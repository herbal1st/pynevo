"""
Sequential multi-layer perceptron (MLP) architecture builder.
"""

import sys
from typing import List, Any, Optional
import numpy as np
from numpy.typing import NDArray

from neural.layers import NeuralDenseLayer
from neural.activations import ActivationReLU, ActivationSigmoid


class NeuralNetwork:
    """
    Manages sequential data flow through dense layers and activation modules.
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_layers: int = 3,
        neurons: int = 15,
        output_size: int = 4,
    ) -> None:
        """
        Constructs sequential multi-layer MLP topology with 4 motor outputs.
        """
        self.layers: List[NeuralDenseLayer] = []
        self.activations: List[Any] = []

        self.layers.append(NeuralDenseLayer(input_size, neurons))
        self.activations.append(ActivationReLU())

        for _ in range(hidden_layers - 1):
            self.layers.append(NeuralDenseLayer(neurons, neurons))
            self.activations.append(ActivationReLU())

        self.layers.append(NeuralDenseLayer(neurons, output_size))
        self.activations.append(None)

        self.out_sigmoid: ActivationSigmoid = ActivationSigmoid()
        self.last_input_features: Optional[NDArray[np.float32]] = None

    @property
    def param_count(self) -> int:
        """
        Returns total number of scalar parameters across all dense layers.
        """
        return sum(layer.param_count for layer in self.layers)

    def copy_weights_from(self, source_net: "NeuralNetwork") -> None:
        """
        Copies weight and bias matrices in-place from source network.
        """
        for i in range(len(self.layers)):
            np.copyto(self.layers[i].weights, source_net.layers[i].weights)
            np.copyto(self.layers[i].biases, source_net.layers[i].biases)

    def export_flat_weights(self) -> NDArray[np.float16]:
        """
        Exports all network layer parameters as a single 1D float16 vector.
        """
        layer_vectors: List[NDArray[np.float16]] = [
            layer.export_flat_weights() for layer in self.layers
        ]
        return np.concatenate(layer_vectors)

    def import_flat_weights(self, flat_data: NDArray) -> None:
        """
        Imports a 1D float16 parameter vector in-place with shape check.
        """
        if flat_data.size != self.param_count:
            print(
                f"[Error] Parameter count mismatch in import_flat_weights! "
                f"Expected {self.param_count} parameters, got {flat_data.size}."
            )
            sys.exit(1)

        offset: int = 0
        for layer in self.layers:
            layer_params: int = layer.param_count
            layer_slice = flat_data[offset : offset + layer_params]
            layer.import_flat_weights(layer_slice)
            offset += layer_params

    def forward(self, input_data: NDArray) -> NDArray[np.float64]:
        """
        Forward pass returning 4 wheel outputs (L-FWD, L-BWD, R-FWD, R-BWD).
        """
        self.last_input_features = input_data.flatten().astype(np.float32)
        curr: NDArray[np.float64] = (
            input_data.astype(np.float64)
            if input_data.ndim == 2
            else input_data[np.newaxis, :].astype(np.float64)
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
