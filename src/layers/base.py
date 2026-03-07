"""
Base Layer Module
=================

This module provides the base class for all layer implementations.
All layer classes should inherit from this base class.

Classes:
    BaseLayer: Abstract base class for all layers
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLayer(ABC):
    """
    Abstract base class for all layers.

    This class defines the interface that all layers must implement.
    It provides common functionality and ensures consistent API across
    different layer implementations.

    Attributes:
        input_dim: Dimension of input to the layer
        output_dim: Dimension of output from the layer

    Methods:
        forward: Forward pass through the layer
        backward: Backward pass to compute gradients
        get_params: Get layer parameters
        set_params: Set layer parameters
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the base layer.

        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the layer.

        Args:
            X: Input data of shape (batch_size, input_dim)

        Returns:
            Output of the layer of shape (batch_size, output_dim)
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Perform backward pass to compute gradients.

        Args:
            grad: Gradient from the next layer

        Returns:
            Gradient to pass to the previous layer
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Get layer parameters.

        Returns:
            Dictionary containing layer parameters (weights, biases, etc.)
        """
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """
        Set layer parameters.

        Args:
            params: Dictionary containing layer parameters
        """
        pass
