"""
Base Normalization Module
==========================

This module provides the base class for all normalization methods.
All normalization techniques should inherit from this base class.

Classes:
    BaseNormalization: Abstract base class for all normalization methods
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseNormalization(ABC):
    """
    Abstract base class for all normalization methods.

    This class defines the interface that all normalization methods must implement.
    Normalization helps stabilize and accelerate training by normalizing
    layer inputs.

    Methods:
        forward: Apply normalization
        backward: Compute gradients for normalization
    """

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply normalization to the input.

        Args:
            x: Input array
            training: Whether in training mode (affects running statistics)

        Returns:
            Normalized array
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Compute gradients for normalization.

        Args:
            grad: Gradient from the next layer

        Returns:
            Gradient array
        """
        pass
