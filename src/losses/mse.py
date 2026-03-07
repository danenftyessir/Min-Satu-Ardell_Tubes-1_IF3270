"""MSE Loss - Mean Squared Error."""

import numpy as np
from .base import BaseLoss


class MSELoss(BaseLoss):
    """Mean Squared Error: L = (1/n) * sum((y_true - y_pred)^2)"""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """L = (1/n) * sum((y_true - y_pred)^2)"""
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """dL/dy_pred = (2/n) * (y_pred - y_true)"""
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)
