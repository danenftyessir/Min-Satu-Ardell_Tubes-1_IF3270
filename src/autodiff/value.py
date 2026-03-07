"""
Automatic Differentiation - Value Module
=========================================

This module provides the Value class for automatic differentiation.
This implementation is inspired by micrograd for building computational graphs.

Classes:
    Value: Scalar value that supports automatic differentiation
"""

import numpy as np
from typing import Set, List, Tuple


class Value:
    """
    Value class for automatic differentiation.

    This class represents a scalar value that can track operations and
    automatically compute gradients using backpropagation. Inspired by
    the micrograd library.

    Each Value object stores:
    - Its data (the actual value)
    - Gradient (derivative with respect to some loss)
    - References to children (operands that created this value)
    - Operation that created this value
    - Backward function to compute local gradient

    Attributes:
        data: The scalar value
        grad: Gradient of this value
        _prev: Set of child values that created this value
        _op: Operation that created this value
        _backward: Function to compute local gradient

    Example:
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x
        >>> z.backward()
        >>> print(x.grad, y.grad)  # Should print 4.0 and 2.0
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = ''):
        """
        Initialize a Value object.

        Args:
            data: The scalar value
            _children: Tuple of child values (internal use)
            _op: Operation that created this value (internal use)
        """
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other: 'Value') -> 'Value':
        """
        Addition operation.

        Args:
            other: Another Value or scalar

        Returns:
            New Value representing the sum
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: 'Value') -> 'Value':
        """Reverse addition (other + self)."""
        return self.__add__(other)

    def __sub__(self, other: 'Value') -> 'Value':
        """
        Subtraction operation.

        Args:
            other: Another Value or scalar

        Returns:
            New Value representing the difference
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: 'Value') -> 'Value':
        """Reverse subtraction (other - self)."""
        return Value(other) - self

    def __mul__(self, other: 'Value') -> 'Value':
        """
        Multiplication operation.

        Args:
            other: Another Value or scalar

        Returns:
            New Value representing the product
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: 'Value') -> 'Value':
        """Reverse multiplication (other * self)."""
        return self.__mul__(other)

    def __truediv__(self, other: 'Value') -> 'Value':
        """
        Division operation.

        Args:
            other: Another Value or scalar

        Returns:
            New Value representing the quotient
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: 'Value') -> 'Value':
        """Reverse division (other / self)."""
        return Value(other) / self

    def __pow__(self, other: Union[float, int]) -> 'Value':
        """
        Power operation.

        Args:
            other: Exponent (scalar)

        Returns:
            New Value representing self^other
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        """
        ReLU activation function.

        Returns:
            New Value with ReLU applied
        """
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        """
        Sigmoid activation function.

        Returns:
            New Value with sigmoid applied
        """
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        """
        Hyperbolic tangent activation function.

        Returns:
            New Value with tanh applied
        """
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        """
        Exponential function.

        Returns:
            New Value with exp applied
        """
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients.

        This method computes the gradient of all nodes in the
        computational graph with respect to this node.
        """
        # Topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Go one variable at a time and apply the chain rule
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> 'Value':
        """Negation operation."""
        return self * -1

    def __repr__(self) -> str:
        """String representation of the Value."""
        return f"Value(data={self.data}, grad={self.grad})"
