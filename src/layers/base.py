"""
modul kelas dasar layer
======================

modul ini menyediakan kelas dasar untuk semua implementasi layer.
semua kelas layer harus mewarisi dari kelas dasar ini.

kelas:
    baselayer: kelas abstrak dasar untuk semua layer
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLayer(ABC):
    """
    kelas abstrak dasar untuk semua layer.

    kelas ini mendefinisikan interface yang harus diimplementasikan semua layer.
    kelas ini menyediakan fungsionalitas umum dan memastikan api yang konsisten
    di berbagai implementasi layer.

    atribut:
        input_dim: dimensi input ke layer
        output_dim: dimensi output dari layer

    metode:
        forward: propagasi maju melalui layer
        backward: propagasi mundur untuk menghitung gradien
        get_params: dapatkan parameter layer
        set_params: atur parameter layer
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        inisialisasi layer dasar.

        argumen:
            input_dim: jumlah fitur input
            output_dim: jumlah fitur output
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        lakukan propagasi maju melalui layer.

        argumen:
            X: data input dengan bentuk (batch_size, input_dim)

        kembali:
            output dari layer dengan bentuk (batch_size, output_dim)
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        lakukan propagasi mundur untuk menghitung gradien.

        argumen:
            grad: gradien dari layer selanjutnya

        kembali:
            gradien untuk diteruskan ke layer sebelumnya
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        dapatkan parameter layer.

        kembali:
            dictionary yang berisi parameter layer (bobot, bias, dll)
        """
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """
        atur parameter layer.

        argumen:
            params: dictionary yang berisi parameter layer
        """
        pass
