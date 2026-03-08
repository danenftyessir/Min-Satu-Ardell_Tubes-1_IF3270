from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Kelas abstrak base untuk semua model."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation."""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> dict:
        """Backward propagation - hitung gradien."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              batch_size: int = 32, learning_rate: float = 0.01,
              epochs: int = 100, verbose: int = 1) -> dict:
        """Latih model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Lakukan prediksi."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Simpan model ke file."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model dari file."""
        pass
