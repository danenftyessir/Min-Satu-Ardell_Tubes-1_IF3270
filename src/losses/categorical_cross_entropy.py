"""categorical cross-entropy loss."""

import numpy as np
from .base import BaseLoss


class CategoricalCrossEntropyLoss(BaseLoss):
    """categorical cross-entropy untuk multi-kelas."""

    def __init__(self, epsilon: float = 1e-15):
        """inisialisasi dengan epsilon untuk mencegah log(0)."""
        self.epsilon = epsilon

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """L = -(1/n) * sum(sum(y_true * log(y_pred)))"""
        y_pred = np.clip(y_pred, self.epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """dL/dy_pred = (1/n) * (y_pred - y_true)"""
        n = y_true.shape[0]
        return (1 / n) * (y_pred - y_true)
