"""
Bonus Activation Functions Module
==================================

Fungsi aktivasi tambahan untuk bonus (5%).
"""

import numpy as np
from .base import BaseActivation


class LeakyReLU(BaseActivation):
    """Leaky ReLU: f(x) = max(alpha * x, x)"""

    def __init__(self, alpha: float = 0.01):
        """Inisialisasi LeakyReLU."""
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = max(alpha * x, x)"""
        return np.maximum(self.alpha * x, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = alpha jika x < 0, else 1"""
        return np.where(x > 0, 1.0, self.alpha)


class ELU(BaseActivation):
    """ELU: f(x) = x jika x > 0, else alpha * (exp(x) - 1)"""

    def __init__(self, alpha: float = 1.0):
        """Inisialisasi ELU."""
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x jika x > 0, else alpha * (exp(x) - 1)"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 jika x > 0, else alpha * exp(x)"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))


class GELU(BaseActivation):
    """GELU: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Turunan GELU (numerical approximation)"""
        tanh_part = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
        sech_sq = 1 - tanh_part ** 2

        cdf = 0.5 * (1 + tanh_part)
        pdf = 0.5 * sech_sq * (np.sqrt(2 / np.pi) * (1 + 0.134145 * x ** 2))

        return cdf + x * pdf


class Swish(BaseActivation):
    """Swish: f(x) = x * sigmoid(x)"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x * sigmoid(x)"""
        x = np.clip(x, -500, 500)
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
        x = np.clip(x, -500, 500)
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)
