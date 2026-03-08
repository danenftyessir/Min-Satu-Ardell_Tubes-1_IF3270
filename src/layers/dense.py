"""
layer Dense - layer fully connected.
"""

import numpy as np
from typing import Dict, Tuple
from .base import BaseLayer


class Dense(BaseLayer):
    """layer Dense - setiap input terhubung ke setiap output neuron."""

    def __init__(self, input_dim: int, output_dim: int, initializer: str = 'uniform'):
        """inisialisasi layer Dense."""
        super().__init__(input_dim, output_dim)
        self.initializer = initializer
        self._initialize_parameters()

        # untuk propagasi mundur
        self.last_input = None
        self.last_output = None
        self.weight_gradient = None
        self.bias_gradient = None

    def _initialize_parameters(self) -> None:
        """inisialisasi bobot dan bias."""
        if self.initializer == 'zero':
            self.weights = np.zeros((self.input_dim, self.output_dim))
            self.biases = np.zeros(self.output_dim)
        elif self.initializer == 'uniform':
            self.weights = np.random.uniform(-0.5, 0.5, (self.input_dim, self.output_dim))
            self.biases = np.random.uniform(-0.5, 0.5, self.output_dim)
        elif self.initializer == 'normal':
            self.weights = np.random.normal(0, 0.1, (self.input_dim, self.output_dim))
            self.biases = np.random.normal(0, 0.1, self.output_dim)
        else:
            raise ValueError(f"initializer tidak dikenal: {self.initializer}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """propagasi maju: output = X @ W + b"""
        self.last_input = X
        output = np.dot(X, self.weights) + self.biases
        self.last_output = output
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """propagasi mundur - hitung gradien."""
        self.weight_gradient = np.dot(self.last_input.T, grad)
        self.bias_gradient = np.sum(grad, axis=0)
        grad_input = np.dot(grad, self.weights.T)
        return grad_input

    def get_params(self) -> Dict[str, np.ndarray]:
        """ambil parameter layer."""
        return {'weights': self.weights, 'biases': self.biases}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """atur parameter layer."""
        self.weights = params['weights']
        self.biases = params['biases']
