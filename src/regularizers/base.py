"""
Base Regularizer Module
========================

This module provides the base class for all regularization methods.
All regularizers should inherit from this base class.

Classes:
    BaseRegularizer: Abstract base class for all regularizers
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseRegularizer(ABC):
    """
    Abstract base class for all regularizers.

    This class defines the interface that all regularizers must implement.
    Regularization helps prevent overfitting by adding a penalty term
    to the loss function.

    Methods:
        __call__: Compute regularization term (alias for forward)
        forward: Compute regularization term to add to loss
        backward: Compute gradient of regularization term
    """

    @abstractmethod
    def forward(self, weights: np.ndarray) -> float:
        """
        Compute the regularization term.

        Args:
            weights: Weight array

        Returns:
            Regularization term (scalar)
        """
        pass

    @abstractmethod
    def backward(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the regularization term.

        Args:
            weights: Weight array

        Returns:
            Gradient array with same shape as weights
        """
        pass

    def __call__(self, weights: np.ndarray) -> float:
        """
        Compute regularization term (convenience method).

        Args:
            weights: Weight array

        Returns:
            Regularization term (scalar)
        """
        return self.forward(weights)
