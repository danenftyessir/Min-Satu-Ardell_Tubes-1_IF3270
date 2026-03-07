"""Gradient Descent Optimizer."""

import numpy as np
from typing import List
from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Standard Gradient Descent: w = w - lr * gradient."""

    def __init__(self, learning_rate: float = 0.01):
        """Inisialisasi dengan learning rate."""
        super().__init__(learning_rate)

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """Update parameter menggunakan gradient descent."""
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * weight_gradients[i]
            biases[i] -= self.learning_rate * bias_gradients[i]
