"""Softmax Activation - exp(x) / sum(exp(x))."""

import numpy as np
from .base import BaseActivation


class Softmax(BaseActivation):
    """Softmax: ubah menjadi probabilitas"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x_i) = exp(x_i) / sum(exp(x_j))"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Turunan softmax (biasanya digabung dengan loss)"""
        s = self.forward(x)
        return s * (1 - s)

    def backward_with_loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradien softmax dengan cross-entropy: dL/dx = S - y"""
        s = self.forward(x)
        return s - y
