from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class BaseOptimizer(ABC):
    """Kelas abstrak untuk optimizer."""

    def __init__(self, learning_rate: float = 0.01):
        """Inisialisasi optimizer."""
        self.learning_rate = learning_rate

    @abstractmethod
    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """Update parameter berdasarkan gradien."""
        pass
