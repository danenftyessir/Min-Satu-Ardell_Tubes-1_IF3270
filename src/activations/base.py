# Base Activation Module
from abc import ABC, abstractmethod
import numpy as np


class BaseActivation(ABC):
    """Kelas abstrak untuk fungsi aktivasi."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Terapkan fungsi aktivasi."""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Hitung turunan fungsi aktivasi."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply aktivasi (convenience method)."""
        return self.forward(x)
