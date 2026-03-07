"""
Base Loss Function Module
==========================

This module provides the base class for all loss function implementations.
All loss functions should inherit from this base class.

Classes:
    BaseLoss: Abstract base class for all loss functions
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    """
    Abstract base class for all loss functions.

    This class defines the interface that all loss functions must implement.
    It provides common functionality and ensures consistent API across
    different loss implementations.

    Methods:
        __call__: Compute loss (alias for forward)
        forward: Compute loss value
        backward: Compute gradient of loss with respect to predictions
    """

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value (scalar)
        """
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of loss with respect to predictions.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Gradient array with same shape as y_pred
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute loss (convenience method).

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Loss value (scalar)
        """
        return self.forward(y_true, y_pred)
