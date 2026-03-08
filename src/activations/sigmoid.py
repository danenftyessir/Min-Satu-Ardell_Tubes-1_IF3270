"""aktivasi sigmoid - f(x) = 1 / (1 + exp(-x))."""

import numpy as np
from .base import BaseActivation


class Sigmoid(BaseActivation):
    """aktivasi sigmoid: f(x) = 1 / (1 + exp(-x))"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = 1 / (1 + exp(-x))"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = f(x) * (1 - f(x))"""
        sig = self.forward(x)
        return sig * (1 - sig)
