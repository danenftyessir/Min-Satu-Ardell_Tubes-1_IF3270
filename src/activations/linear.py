"""aktivasi linear - f(x) = x."""

import numpy as np
from .base import BaseActivation


class Linear(BaseActivation):
    """aktivasi linear: f(x) = x"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x"""
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1"""
        return np.ones_like(x)
