"""
Adam Optimizer Module
=====================

This module provides the implementation of Adam optimizer.
Adam is an adaptive learning rate optimization algorithm.

Classes:
    Adam: Adam (Adaptive Moment Estimation) optimizer
"""

import numpy as np
from typing import List
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam combines the advantages of two other extensions of stochastic
    gradient descent: AdaGrad and RMSProp.

    Adam computes adaptive learning rates for each parameter by maintaining
    estimates of first and second moments of the gradients.

    Attributes:
        learning_rate: Step size for weight updates
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability
        t: Time step counter

    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.update(weights, biases, weight_grads, bias_grads)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate: Step size for weight updates (default: 0.001)
            beta1: Exponential decay rate for first moment estimate (default: 0.9)
            beta2: Exponential decay rate for second moment estimate (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # Moment estimates for weights
        self.m_w = []  # First moment (mean)
        self.v_w = []  # Second moment (uncentered variance)

        # Moment estimates for biases
        self.m_b = []
        self.v_b = []

    def _initialize_moments(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ) -> None:
        """
        Initialize moment estimates if not already initialized.

        Args:
            weights: List of weight matrices
            biases: List of bias vectors
        """
        if not self.m_w:
            for w, b in zip(weights, biases):
                self.m_w.append(np.zeros_like(w))
                self.v_w.append(np.zeros_like(w))
                self.m_b.append(np.zeros_like(b))
                self.v_b.append(np.zeros_like(b))

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """
        Update parameters using Adam optimization.

        Adam update rule:
        1. Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
        2. Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
        3. Compute bias-corrected estimates: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        4. Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_gradients: List of weight gradients
            bias_gradients: List of bias gradients
        """
        self._initialize_moments(weights, biases)
        self.t += 1

        for i in range(len(weights)):
            # Update first moment (mean of gradients)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * weight_gradients[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * bias_gradients[i]

            # Update second moment (uncentered variance of gradients)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(weight_gradients[i])
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(bias_gradients[i])

            # Compute bias-corrected estimates
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
