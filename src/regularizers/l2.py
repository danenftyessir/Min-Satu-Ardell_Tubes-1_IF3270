"""l2 regularization."""

import numpy as np
from .base import BaseRegularizer


class L2Regularizer(BaseRegularizer):
    """l2 regularization: L = (lambda/2) * sum(weights^2)"""

    def __init__(self, lambda_param: float = 0.01):
        """inisialisasi dengan lambda."""
        self.lambda_param = lambda_param

    def forward(self, weights: np.ndarray) -> float:
        """l2 = (lambda/2) * sum(weights^2)"""
        return (self.lambda_param / 2) * np.sum(weights ** 2)

    def backward(self, weights: np.ndarray) -> np.ndarray:
        """dl2/dw = lambda * weights"""
        return self.lambda_param * weights
