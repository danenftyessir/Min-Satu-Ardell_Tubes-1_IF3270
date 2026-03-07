"""
Feedforward Neural Network (FFNN) - Implementasi lengkap.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import pickle
import os

from .base import BaseModel
from ..utils.plotting import plot_weight_distribution, plot_gradient_distribution


class FFNN(BaseModel):
    """
    Implementasi Feedforward Neural Network.

    Mendukung berbagai activation, loss function, initializer, dan regularizer.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss_function: str = 'mse',
        initializer: str = 'uniform',
        learning_rate: float = 0.01,
        regularizer: Optional[Dict] = None
    ):
        """Inisialisasi FFNN."""
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("Minimal 2 layer (input dan output)")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Jumlah aktivasi harus sama dengan jumlah layer - 1")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer

        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Inisialisasi bobot dan bias."""
        pass  # TODO: Implement

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation."""
        pass  # TODO: Implement

    def backward(self, grad: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """Backward propagation - hitung gradien."""
        pass  # TODO: Implement

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        batch_size: int = 32,
        learning_rate: float = None,
        epochs: int = 100,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Latih neural network."""
        pass  # TODO: Implement

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Lakukan prediksi."""
        pass  # TODO: Implement

    def save(self, filepath: str) -> None:
        """Simpan model ke file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss_function': self.loss_function,
            'initializer': self.initializer,
            'learning_rate': self.learning_rate,
            'regularizer': self.regularizer,
            'weights': self.weights,
            'biases': self.biases,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str) -> None:
        """Load model dari file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.layer_sizes = model_data['layer_sizes']
        self.activations = model_data['activations']
        self.loss_function = model_data['loss_function']
        self.initializer = model_data['initializer']
        self.learning_rate = model_data['learning_rate']
        self.regularizer = model_data['regularizer']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.is_fitted = model_data['is_fitted']

    def plot_weight_distribution(self, layers: List[int] = None) -> None:
        """
        Plot distribusi bobot untuk layer tertentu.

        Args:
            layers: List indeks layer (0-indexed). Jika None, plot semua.
        """
        plot_weight_distribution(self.weights, layers, title="Distribusi Bobot")

    def plot_gradient_distribution(self, layers: List[int] = None) -> None:
        """
        Plot distribusi gradien untuk layer tertentu.

        Args:
            layers: List indeks layer (0-indexed). Jika None, plot semua.
        """
        plot_gradient_distribution(self.weight_gradients, layers, title="Distribusi Gradien")
