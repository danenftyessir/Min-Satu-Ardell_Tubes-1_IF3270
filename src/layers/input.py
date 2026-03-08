"""
modul layer input
==================

modul ini menyediakan implementasi dari layer input.
layer input bertanggung jawab untuk menerima dan memvalidasi data input.

kelas:
    InputLayer: implementasi layer input
"""

import numpy as np
from typing import Tuple, Optional
from .base import BaseLayer


class InputLayer(BaseLayer):
    """
    implementasi layer input.

    layer ini berfungsi sebagai titik masuk data ke dalam jaringan saraf.
    layer ini memvalidasi dimensi input dan dapat melakukan preprocessing dasar.

    atribut:
        input_dim: dimensi yang diharapkan dari data input
        output_dim: dimensi output (sama dengan input_dim)
        name: nama opsional untuk layer

    contoh:
        >>> input_layer = InputLayer(input_dim=784)
        >>> output = input_layer.forward(X)
    """

    def __init__(self, input_dim: int, name: Optional[str] = None):
        """
        inisialisasi layer input.

        argumen:
            input_dim: dimensi yang diharapkan dari fitur input
            name: nama opsional untuk layer
        """
        super().__init__(input_dim, input_dim)
        self.name = name or "input"

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        lewatkan input melalui layer input.

        metode ini memvalidasi dimensi input dan mengembalikan input
        tanpa modifikasi.

        argumen:
            X: data input dengan bentuk (batch_size, input_dim)

        kembali:
            data input tidak berubah (batch_size, input_dim)

        Raises:
            ValueError: jika dimensi input tidak sesuai dengan dimensi yang diharapkan
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"ketidakcocokan dimensi input. diharapkan {self.input_dim}, "
                f"didapat {X.shape[1]}"
            )

        return X

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        lewatkan gradien melalui layer input.

        karena layer input tidak mengubah data, layer ini hanya
        meneruskan gradien tanpa perubahan.

        argumen:
            grad: gradien dari layer berikutnya

        kembali:
            gradien tidak berubah
        """
        return grad

    def get_params(self) -> dict:
        """
        dapatkan parameter layer.

        layer input tidak memiliki parameter yang dapat dipelajari.

        kembali:
            kamus kosong
        """
        return {}

    def set_params(self, params: dict) -> None:
        """
        atur parameter layer.

        layer input tidak memiliki parameter untuk diatur.

        argumen:
            params: kamus kosong
        """
        pass
