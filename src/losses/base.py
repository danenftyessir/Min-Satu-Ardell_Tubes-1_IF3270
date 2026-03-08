"""
modul kelas dasar fungsi loss
=============================

modul ini menyediakan kelas dasar untuk semua implementasi fungsi loss.
semua fungsi loss harus mewarisi dari kelas dasar ini.

kelas:
    baseloss: kelas abstrak dasar untuk semua fungsi loss
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    """
    kelas abstrak dasar untuk semua fungsi loss.

    kelas ini mendefinisikan interface yang harus diimplementasikan semua fungsi loss.
    kelas ini menyediakan fungsionalitas umum dan memastikan api yang konsisten
    di berbagai implementasi loss.

    metode:
        __call__: hitung loss (alias untuk forward)
        forward: hitung nilai loss
        backward: hitung gradien loss terhadap prediksi
    """

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        hitung nilai loss.

        argumen:
            y_true: nilai target sebenarnya
            y_pred: nilai prediksi

        kembali:
            nilai loss (skalar)
        """
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        hitung gradien loss terhadap prediksi.

        argumen:
            y_true: nilai target sebenarnya
            y_pred: nilai prediksi

        kembali:
            array gradien dengan bentuk yang sama seperti y_pred
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        hitung loss (metode kemudahan).

        argumen:
            y_true: nilai target sebenarnya
            y_pred: nilai prediksi

        kembali:
            nilai loss (skalar)
        """
        return self.forward(y_true, y_pred)
