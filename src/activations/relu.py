"""ReLU Activation - f(x) = max(0, x)."""

import numpy as np
from .base import BaseActivation


class ReLU(BaseActivation):
    """ReLU activation: f(x) = max(0, x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = max(0, x)"""
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 jika x > 0, else 0"""
        return (x > 0).astype(float)
