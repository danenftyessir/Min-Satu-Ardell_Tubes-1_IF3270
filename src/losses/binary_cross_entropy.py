"""Binary Cross-Entropy Loss."""

import numpy as np
from .base import BaseLoss


class BinaryCrossEntropyLoss(BaseLoss):
    """Binary Cross-Entropy untuk klasifikasi biner."""

    def __init__(self, epsilon: float = 1e-15):
        """inisialisasi dengan epsilon untuk mencegah log(0)."""
        self.epsilon = epsilon

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))"""
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """dL/dy_pred = (1/n) * (y_pred - y_true) / (y_pred * (1 - y_pred))"""
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        n = y_true.shape[0]
        return (1 / n) * (y_pred - y_true) / (y_pred * (1 - y_pred))
