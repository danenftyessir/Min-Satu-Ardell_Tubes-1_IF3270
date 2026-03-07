"""
Base Initializer Module
========================

This module provides the base class for all weight initialization methods.
All initializers should inherit from this base class.

Classes:
    BaseInitializer: Abstract base class for all initializers
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseInitializer(ABC):
    """
    Abstract base class for all weight initializers.

    This class defines the interface that all initializers must implement.
    It provides common functionality and ensures consistent API across
    different initialization methods.

    Methods:
        __call__: Initialize weights (alias for initialize)
        initialize: Initialize weights with the specific method
    """

    @abstractmethod
    def initialize(self, shape: tuple) -> np.ndarray:
        """
        Initialize weights with the specific method.

        Args:
            shape: Shape of the weight array to initialize

        Returns:
            Initialized weight array
        """
        pass

    def __call__(self, shape: tuple) -> np.ndarray:
        """
        Initialize weights (convenience method).

        Args:
            shape: Shape of the weight array to initialize

        Returns:
            Initialized weight array
        """
        return self.initialize(shape)
