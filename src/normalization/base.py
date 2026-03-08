from abc import ABC, abstractmethod
import numpy as np


class BaseNormalization(ABC):
    """
    kelas abstrak dasar untuk semua metode normalisasi.
    metode:
        forward: terapkan normalisasi
        backward: hitung gradien untuk normalisasi
    """

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        terapkan normalisasi ke input.

        argumen:
            x: array input
            training: apakah dalam mode training (mempengaruhi statistik running)
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        hitung gradien untuk normalisasi.

        argumen:
            grad: gradien dari layer selanjutnya
        """
        pass
