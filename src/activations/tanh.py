"""Tanh Activation - f(x) = tanh(x)."""

import numpy as np
from .base import BaseActivation


class Tanh(BaseActivation):
    """Hyperbolic tangent: f(x) = tanh(x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = tanh(x)"""
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 - f(x)^2"""
        tanh_val = self.forward(x)
        return 1 - tanh_val ** 2
