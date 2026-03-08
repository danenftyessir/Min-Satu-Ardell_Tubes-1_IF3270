"""
Normalization Layers
====================

Layer normalization yang kompatibel dengan BaseLayer interface.
"""

import numpy as np
from .base import BaseLayer
from ..normalization import RMSNorm as BaseRMSNorm


class RMSNormLayer(BaseLayer):
    """
    RMSNorm Layer yang kompatibel dengan BaseLayer interface.

    Layer ini membungkus RMSNorm dari normalization module untuk
    digunakan dalam arsitektur neural network.

    Rumus: output = gain * (x / sqrt(mean(x^2) + epsilon))

    Argumen:
        dim: dimensi fitur input/output (sama untuk normalization)
        epsilon: konstanta kecil untuk stabilitas numerik
        gain_initial: nilai awal untuk parameter gain (default: 1.0)
    """

    def __init__(self, dim: int, epsilon: float = 1e-8, gain_initial: float = 1.0):
        """
        Inisialisasi RMSNorm Layer.

        Argumen:
            dim: dimensi fitur
            epsilon: konstanta untuk stabilitas numerik
            gain_initial: nilai awal gain parameter
        """
        super().__init__(input_dim=dim, output_dim=dim)

        # Gunakan RMSNorm base implementation
        self.norm = BaseRMSNorm(dim=dim, epsilon=epsilon)

        # Override gain initialization jika perlu
        if gain_initial != 1.0:
            self.norm.gain = np.full(dim, gain_initial)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass melalui RMSNorm.

        Argumen:
            X: input array dengan shape (batch_size, dim)
            training: apakah dalam mode training

        Returns:
            normalized output dengan shape yang sama
        """
        return self.norm.forward(X, training=training)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass untuk menghitung gradien.

        Argumen:
            grad: gradien dari layer berikutnya

        Returns:
            gradien untuk diteruskan ke layer sebelumnya
        """
        return self.norm.backward(grad)

    def get_params(self) -> dict:
        """
        Dapatkan parameter layer.

        Returns:
            dict dengan 'gain' parameter
        """
        return {
            'gain': self.norm.gain.copy(),
            'epsilon': self.norm.epsilon,
            'dim': self.norm.dim
        }

    def set_params(self, params: dict) -> None:
        """
        Atur parameter layer.

        Argumen:
            params: dict dengan 'gain' parameter
        """
        if 'gain' in params:
            self.norm.gain = params['gain'].copy()
        if 'epsilon' in params:
            self.norm.epsilon = params['epsilon']
        if 'dim' in params:
            self.norm.dim = params['dim']

    def get_gradients(self) -> dict:
        """
        Dapatkan gradien parameter.

        Returns:
            dict dengan gain gradient
        """
        return {
            'gain_gradient': self.norm.gain_gradient.copy()
        }


class LayerNormalization(BaseLayer):
    """
    Layer Normalization (LayerNorm).

    Normalisasi statistik (mean dan variance) di setiap sample
    dalam batch, bukan di seluruh batch seperti BatchNorm.

    Rumus: output = gamma * (x - mean) / sqrt(var + epsilon) + beta

    Argumen:
        dim: dimensi fitur input/output
        epsilon: konstanta kecil untuk stabilitas numerik
    """

    def __init__(self, dim: int, epsilon: float = 1e-5):
        """
        Inisialisasi LayerNorm.

        Argumen:
            dim: dimensi fitur
            epsilon: konstanta untuk stabilitas numerik
        """
        super().__init__(input_dim=dim, output_dim=dim)
        self.epsilon = epsilon

        # Parameter yang bisa dipelajari
        self.gamma = np.ones(dim)  # scale
        self.beta = np.zeros(dim)  # shift

        # Gradien
        self.gamma_gradient = np.zeros(dim)
        self.beta_gradient = np.zeros(dim)

        # Untuk backward pass
        self.last_input = None
        self.last_mean = None
        self.last_var = None
        self.last_normalized = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass LayerNorm.

        Argumen:
            X: input (batch_size, dim)
            training: mode training

        Returns:
            normalized output
        """
        self.last_input = X

        # Hitung mean dan variance per sample
        mean = np.mean(X, axis=-1, keepdims=True)  # (batch_size, 1)
        var = np.var(X, axis=-1, keepdims=True)    # (batch_size, 1)

        self.last_mean = mean
        self.last_var = var

        # Normalisasi
        normalized = (X - mean) / np.sqrt(var + self.epsilon)
        self.last_normalized = normalized

        # Scale dan shift
        output = self.gamma * normalized + self.beta

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass LayerNorm.

        Argumen:
            grad: gradien dari layer berikutnya (batch_size, dim)

        Returns:
            gradien untuk diteruskan
        """
        X = self.last_input
        normalized = self.last_normalized
        batch_size, dim = X.shape

        # Gradien terhadap gamma dan beta
        self.gamma_gradient = np.sum(grad * normalized, axis=0)
        self.beta_gradient = np.sum(grad, axis=0)

        # Gradien terhadap input
        # Menggunakan chain rule yang kompleks untuk layer norm
        std = np.sqrt(self.last_var + self.epsilon)

        # Term 1: d/dx dari (gamma * norm + beta)
        grad_normalized = grad * self.gamma

        # Term 2: d/dx dari normalization (x - mean) / std
        # Ini turunan parsial yang kompleks
        grad_x = (grad_normalized -
                 np.mean(grad_normalized, axis=-1, keepdims=True) -
                 normalized * np.mean(grad_normalized * normalized, axis=-1, keepdims=True)
                 ) / std

        return grad_x

    def get_params(self) -> dict:
        """
        Dapatkan parameter layer.
        """
        return {
            'gamma': self.gamma.copy(),
            'beta': self.beta.copy(),
            'epsilon': self.epsilon,
            'dim': self.input_dim
        }

    def set_params(self, params: dict) -> None:
        """
        Atur parameter layer.
        """
        if 'gamma' in params:
            self.gamma = params['gamma'].copy()
        if 'beta' in params:
            self.beta = params['beta'].copy()
        if 'epsilon' in params:
            self.epsilon = params['epsilon']

    def get_gradients(self) -> dict:
        """
        Dapatkan gradien parameter.
        """
        return {
            'gamma_gradient': self.gamma_gradient.copy(),
            'beta_gradient': self.beta_gradient.copy()
        }
