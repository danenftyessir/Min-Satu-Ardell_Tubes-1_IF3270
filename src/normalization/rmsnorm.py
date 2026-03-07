"""
RMSNorm Normalization Module
=============================

This module provides the implementation of RMSNorm normalization.
RMSNorm is a simplified normalization technique that normalizes by the RMS of inputs.

Classes:
    RMSNorm: Root Mean Square Normalization
"""

import numpy as np
from .base import BaseNormalization


class RMSNorm(BaseNormalization):
    """
    Root Mean Square Normalization (RMSNorm).

    RMSNorm normalizes the input by dividing by the root mean square
    of the input values, then scales by a learnable gain parameter.

    Formula: output = gain * (x / sqrt(mean(x^2) + epsilon))

    RMSNorm is simpler than LayerNorm and BatchNorm as it doesn't
    center the data (subtract mean), making it computationally more efficient.

    Attributes:
        dim: Dimension of the input features
        epsilon: Small constant for numerical stability
        gain: Learnable scale parameter
        gain_gradient: Gradient for the gain parameter

    Example:
        >>> norm = RMSNorm(dim=512)
        >>> output = norm.forward(x)
        >>> grad = norm.backward(grad_output)
    """

    def __init__(self, dim: int, epsilon: float = 1e-8):
        """
        Initialize RMSNorm.

        Args:
            dim: Dimension of input features
            epsilon: Small constant for numerical stability
        """
        self.dim = dim
        self.epsilon = epsilon

        # Learnable gain parameter (initialized to 1)
        self.gain = np.ones(dim)
        self.gain_gradient = np.zeros(dim)

        # Store for backward pass
        self.last_input = None
        self.last_rms = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply RMSNorm normalization.

        output = gain * (x / sqrt(mean(x^2, axis=-1, keepdims=True) + epsilon))

        Args:
            x: Input array of shape (batch_size, dim)
            training: Whether in training mode (not used for RMSNorm but kept for consistency)

        Returns:
            Normalized array of same shape as input
        """
        # Store input for backward pass
        if training:
            self.last_input = x

        # Compute RMS (Root Mean Square)
        # RMS = sqrt(mean(x^2) + epsilon)
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(mean_square + self.epsilon)

        if training:
            self.last_rms = rms

        # Normalize and scale
        output = self.gain * (x / rms)

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Compute gradients for RMSNorm.

        Args:
            grad: Gradient from the next layer of shape (batch_size, dim)

        Returns:
            Gradient array with same shape as input
        """
        x = self.last_input
        rms = self.last_rms
        batch_size = x.shape[0]

        # Gradient with respect to gain
        # x_norm = x / rms
        x_norm = x / rms
        self.gain_gradient = np.sum(grad * x_norm, axis=0)

        # Gradient with respect to x
        # Using chain rule through the normalization operation
        # d/dx (gain * x / rms) = gain * (1/rms - x * x / (rms^3 * dim))

        term1 = grad * self.gain / rms
        term2 = grad * self.gain * x * x / (rms ** 3 * self.dim)

        grad_input = term1 - term2

        return grad_input
