import numpy as np
from .base import BaseNormalization


class RMSNorm(BaseNormalization):
    """
    rmsnorm menormalisasi input dengan membagi dengan root mean square
    dari nilai input, lalu menskalakan dengan parameter gain yang dapat dipelajarin

    rumus: output = gain * (x / sqrt(mean(x^2) + epsilon))

    attributes:
        dim: dimensi dari fitur input
        epsilon: konstanta kecil untuk stabilitas numerik
        gain: parameter skala yang dapat dipelajari
        gain_gradient: gradien untuk parameter gain
    """

    def __init__(self, dim: int, epsilon: float = 1e-8):
        """
        inisialisasi rmsnorm
        dim: dimensi dari fitur input
        epsilon: konstanta kecil untuk stabilitas numerik
        """
        self.dim = dim
        self.epsilon = epsilon

        # parameter gain yang dapat dipelajari (diinisialisasi ke 1)
        self.gain = np.ones(dim)
        self.gain_gradient = np.zeros(dim)

        # simpan untuk backward pass
        self.last_input = None
        self.last_rms = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        output = gain * (x / sqrt(mean(x^2, axis=-1, keepdims=true) + epsilon))
        x: array input dengan shape (batch_size, dim)
        training: apakah dalam mode training (tidak digunakan untuk rmsnorm tapi dipertahankan untuk konsistensi)
        """
        # simpan input untuk backward pass
        if training:
            self.last_input = x

        # hitung rms (root mean square)
        # rms = sqrt(mean(x^2) + epsilon)
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(mean_square + self.epsilon)

        if training:
            self.last_rms = rms

        # normalisasi dan skala
        output = self.gain * (x / rms)

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # hitung gradien untuk rmsnorm
        x = self.last_input
        rms = self.last_rms
        batch_size = x.shape[0]

        # gradien terhadap gain
        # x_norm = x / rms
        x_norm = x / rms
        self.gain_gradient = np.sum(grad * x_norm, axis=0)

        # gradien terhadap x
        # menggunakan chain rule melalui operasi normalisasi
        # d/dx (gain * x / rms) = gain * (1/rms - x * x / (rms^3 * dim))

        term1 = grad * self.gain / rms
        term2 = grad * self.gain * x * x / (rms ** 3 * self.dim)

        grad_input = term1 - term2

        return grad_input
