"""
Input Layer Module
==================

This module provides the implementation of the Input layer.
The input layer is responsible for receiving and validating input data.

Classes:
    InputLayer: Input layer implementation
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseLayer


class InputLayer(BaseLayer):
    """
    Input layer implementation.

    This layer serves as the entry point for data into the neural network.
    It validates input dimensions and can perform basic preprocessing.

    Attributes:
        input_dim: Expected dimension of input data
        output_dim: Dimension of output (same as input_dim)
        name: Optional name for the layer

    Example:
        >>> input_layer = InputLayer(input_dim=784)
        >>> output = input_layer.forward(X)
    """

    def __init__(self, input_dim: int, name: Optional[str] = None):
        """
        Initialize the Input layer.

        Args:
            input_dim: Expected dimension of input features
            name: Optional name for the layer
        """
        super().__init__(input_dim, input_dim)
        self.name = name or "input"

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Pass input through the input layer.

        This method validates the input dimension and returns the input
        without modification.

        Args:
            X: Input data of shape (batch_size, input_dim)

        Returns:
            Input data unchanged (batch_size, input_dim)

        Raises:
            ValueError: If input dimension doesn't match expected dimension
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, "
                f"got {X.shape[1]}"
            )

        return X

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Pass gradient through the input layer.

        Since the input layer doesn't transform the data, it simply
        passes the gradient through unchanged.

        Args:
            grad: Gradient from the next layer

        Returns:
            Gradient unchanged
        """
        return grad

    def get_params(self) -> dict:
        """
        Get layer parameters.

        Input layer has no learnable parameters.

        Returns:
            Empty dictionary
        """
        return {}

    def set_params(self, params: dict) -> None:
        """
        Set layer parameters.

        Input layer has no parameters to set.

        Args:
            params: Empty dictionary
        """
        pass
