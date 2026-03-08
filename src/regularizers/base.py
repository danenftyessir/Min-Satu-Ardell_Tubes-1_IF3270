"""
modul kelas dasar regularizer
=============================

modul ini menyediakan kelas dasar untuk semua metode regularisasi.
semua regularizer harus mewarisi dari kelas dasar ini.

kelas:
    baseregularizer: kelas abstrak dasar untuk semua regularizer
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseRegularizer(ABC):
    """
    kelas abstrak dasar untuk semua regularizer.

    kelas ini mendefinisikan interface yang harus diimplementasikan semua regularizer.
    regularisasi membantu mencegah overfitting dengan menambahkan penalti
    ke fungsi loss.

    metode:
        __call__: hitung regularisasi (alias untuk forward)
        forward: hitung regularisasi untuk ditambahkan ke loss
        backward: hitung gradien dari regularisasi
    """

    @abstractmethod
    def forward(self, weights: np.ndarray) -> float:
        """
        hitung regularisasi.

        argumen:
            weights: array bobot

        kembali:
            regularisasi (skalar)
        """
        pass

    @abstractmethod
    def backward(self, weights: np.ndarray) -> np.ndarray:
        """
        hitung gradien dari regularisasi.

        argumen:
            weights: array bobot

        kembali:
            array gradien dengan bentuk yang sama seperti weights
        """
        pass

    def __call__(self, weights: np.ndarray) -> float:
        """
        hitung regularisasi (metode kemudahan).

        argumen:
            weights: array bobot

        kembali:
            regularisasi (skalar)
        """
        return self.forward(weights)
