import numpy as np
from .base import BaseActivation


class LeakyReLU(BaseActivation):
    """leaky relu: f(x) = max(alpha * x, x)"""

    def __init__(self, alpha: float = 0.01):
        """inisialisasi leakyrelu."""
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = max(alpha * x, x)"""
        return np.maximum(self.alpha * x, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = alpha jika x < 0, else 1"""
        return np.where(x > 0, 1.0, self.alpha)


class ELU(BaseActivation):
    """elu: f(x) = x jika x > 0, else alpha * (exp(x) - 1)"""

    def __init__(self, alpha: float = 1.0):
        """inisialisasi elu."""
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x jika x > 0, else alpha * (exp(x) - 1)"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 jika x > 0, else alpha * exp(x)"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
