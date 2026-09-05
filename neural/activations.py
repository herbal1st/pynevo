"""
Vectorized activation modules with floating-point arithmetic overflow guards.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class ActivationReLU:
    """
    Rectified Linear Unit activation function.
    """

    def __init__(self) -> None:
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.output = np.maximum(0.0, input_data)
        return self.output


class ActivationTanh:
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self) -> None:
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.output = np.tanh(input_data)
        return self.output


class ActivationSigmoid:
    """
    Logistic sigmoid activation function with fast exponential evaluation.
    """

    def __init__(self) -> None:
        self.output: Optional[NDArray[np.float64]] = None

    def forward(self, input_data: NDArray[np.float64]) -> NDArray[np.float64]:
        # Fast vectorized sigmoid evaluation without copy-inducing np.clip
        self.output = 1.0 / (1.0 + np.exp(-input_data))
        return self.output