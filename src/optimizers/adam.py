import numpy as np
from typing import List
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) Optimizer.

    Adam combines the benefits of RMSProp and Momentum:
    - Uses moving average of gradients (momentum)
    - Uses moving average of squared gradients (adaptive learning rate)
    - Bias correction for first moments

    Algorithm:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        w_t = w_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Inisialisasi ADAM optimizer.

        Argumen:
            learning_rate: Step size (default: 0.001)
            beta1: Exponential decay rate for first moment (default: 0.9)
            beta2: Exponential decay rate for second moment (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State variables untuk setiap parameter
        self.m_weights: List[np.ndarray] = []  # First moment for weights
        self.m_biases: List[np.ndarray] = []   # First moment for biases
        self.v_weights: List[np.ndarray] = []  # Second moment for weights
        self.v_biases: List[np.ndarray] = []   # Second moment for biases

        self.t = 0  # Time step counter

    def _initialize_moments(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ) -> None:
        """Inisialisasi momentum jika belum ada."""
        # Inisialisasi hanya jika jumlah parameter berubah
        if len(self.m_weights) != len(weights):
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_weights = [np.zeros_like(w) for w in weights]
            self.v_biases = [np.zeros_like(b) for b in biases]

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """
        Update parameter menggunakan ADAM algorithm.

        Argumen:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        # Inisialisasi momentum jika diperlukan
        self._initialize_moments(weights, biases)

        # Increment time step
        self.t += 1

        # Bias correction factors
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t
        m_hat_factor = 1 - beta1_t
        v_hat_factor = 1 - beta2_t

        # Update untuk setiap layer
        for i in range(len(weights)):
            # First moment (momentum): m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m_weights[i] = (
                self.beta1 * self.m_weights[i] +
                (1 - self.beta1) * weight_gradients[i]
            )
            self.m_biases[i] = (
                self.beta1 * self.m_biases[i] +
                (1 - self.beta1) * bias_gradients[i]
            )

            # Second moment (RMSProp-like): v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v_weights[i] = (
                self.beta2 * self.v_weights[i] +
                (1 - self.beta2) * (weight_gradients[i] ** 2)
            )
            self.v_biases[i] = (
                self.beta2 * self.v_biases[i] +
                (1 - self.beta2) * (bias_gradients[i] ** 2)
            )

            # Bias-corrected first moment estimate
            m_weights_hat = self.m_weights[i] / m_hat_factor
            m_biases_hat = self.m_biases[i] / m_hat_factor

            # Bias-corrected second moment estimate
            v_weights_hat = self.v_weights[i] / v_hat_factor
            v_biases_hat = self.v_biases[i] / v_hat_factor

            # Update weights: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
            weights[i] -= self.learning_rate * m_weights_hat / (
                np.sqrt(v_weights_hat) + self.epsilon
            )
            biases[i] -= self.learning_rate * m_biases_hat / (
                np.sqrt(v_biases_hat) + self.epsilon
            )

    def reset(self) -> None:
        """Reset optimizer state (untuk training ulang)."""
        self.m_weights = []
        self.m_biases = []
        self.v_weights = []
        self.v_biases = []
        self.t = 0
