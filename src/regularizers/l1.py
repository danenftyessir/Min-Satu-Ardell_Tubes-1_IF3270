"""L1 Regularization."""

import numpy as np
from .base import BaseRegularizer


class L1Regularizer(BaseRegularizer):
    """L1 regularization: L = lambda * sum(|weights|)"""

    def __init__(self, lambda_param: float = 0.01):
        """Inisialisasi dengan lambda."""
        self.lambda_param = lambda_param

    def forward(self, weights: np.ndarray) -> float:
        """L1 = lambda * sum(|weights|)"""
        return self.lambda_param * np.sum(np.abs(weights))

    def backward(self, weights: np.ndarray) -> np.ndarray:
        """dL1/dw = lambda * sign(weights)"""
        return self.lambda_param * np.sign(weights)
