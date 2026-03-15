"""
Dropout Layer Implementation.
Provides regularization by randomly dropping units during training.
"""
import numpy as np
from .base import BaseLayer


class DropoutLayer(BaseLayer):
    """
    Dropout Layer.

    Randomly sets a fraction of input units to zero during training,
    which helps prevent overfitting.

    During inference, all units are used (no dropout).

    Args:
        dropout_rate: Fraction of input units to drop (0.0 to 1.0)
    """

    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize Dropout Layer.

        Args:
            dropout_rate: Probability of dropping a unit (default: 0.5)
        """
        super().__init__(input_dim=None, output_dim=None)
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with dropout.

        Args:
            X: Input array
            training: Whether in training mode (applies dropout) or inference mode

        Returns:
            Output with dropout applied (training) or unchanged (inference)
        """
        self.training = training

        if training and self.dropout_rate > 0:
            # Create binary mask (keep units with probability 1 - dropout_rate)
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, X.shape)
            # Scale to maintain expected value
            return X * self.mask / (1 - self.dropout_rate)
        else:
            # During inference, return input unchanged
            self.mask = np.ones_like(X)
            return X

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass - propagate gradient only through kept units.

        Args:
            grad: Gradient from upper layer

        Returns:
            Gradient scaled by mask
        """
        if self.training and self.dropout_rate > 0:
            return grad * self.mask / (1 - self.dropout_rate)
        else:
            return grad

    def get_params(self) -> dict:
        """Get layer parameters."""
        return {
            'dropout_rate': self.dropout_rate
        }

    def set_params(self, params: dict) -> None:
        """Set layer parameters."""
        if 'dropout_rate' in params:
            self.dropout_rate = params['dropout_rate']
