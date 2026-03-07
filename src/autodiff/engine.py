"""
Automatic Differentiation - Computational Graph Engine
======================================================

This module provides the computational graph engine for automatic differentiation.
It manages the computational graph and handles gradient computation.

Classes:
    ComputationalGraph: Engine for managing computational graphs and automatic differentiation
"""

import numpy as np
from typing import List, Dict, Any
from .value import Value


class ComputationalGraph:
    """
    Computational Graph Engine for automatic differentiation.

    This class manages the creation and execution of computational graphs
    for neural networks with automatic differentiation capabilities.

    The engine tracks operations performed on Value objects and can
    automatically compute gradients through backpropagation.

    Attributes:
        values: List of Value objects in the graph
        operations: List of operations performed

    Example:
        >>> graph = ComputationalGraph()
        >>> x = graph.create_value(2.0, name='x')
        >>> y = graph.create_value(3.0, name='y')
        >>> z = x * y + x
        >>> z.backward()
        >>> print(x.grad, y.grad)
    """

    def __init__(self):
        """Initialize the computational graph engine."""
        self.values: List[Value] = []
        self.parameters: Dict[str, Value] = {}

    def create_value(self, data: float, name: str = None) -> Value:
        """
        Create a new Value in the graph.

        Args:
            data: The scalar value
            name: Optional name for the value

        Returns:
            New Value object
        """
        value = Value(data)
        self.values.append(value)

        if name is not None:
            self.parameters[name] = value

        return value

    def zero_grad(self) -> None:
        """
        Zero out all gradients in the graph.

        This should be called before each backward pass to ensure
        gradients are accumulated correctly.
        """
        for value in self.values:
            value.grad = 0.0

    def get_parameters(self) -> List[Value]:
        """
        Get all parameters (named values) in the graph.

        Returns:
            List of Value objects representing parameters
        """
        return list(self.parameters.values())

    def get_parameter_values(self) -> Dict[str, float]:
        """
        Get current values of all named parameters.

        Returns:
            Dictionary mapping parameter names to their values
        """
        return {name: value.data for name, value in self.parameters.items()}

    def get_parameter_gradients(self) -> Dict[str, float]:
        """
        Get gradients of all named parameters.

        Returns:
            Dictionary mapping parameter names to their gradients
        """
        return {name: value.grad for name, value in self.parameters.items()}

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update all parameters using gradient descent.

        Args:
            learning_rate: Learning rate for the update
        """
        for value in self.parameters.values():
            value.data -= learning_rate * value.grad
