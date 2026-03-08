from abc import ABC, abstractmethod
import numpy as np


class BaseInitializer(ABC):
    """
    kelas abstrak dasar untuk semua inisialisasi bobot.

    kelas ini mendefinisikan interface yang harus diimplementasikan semua initializer.
    kelas ini menyediakan fungsionalitas umum dan memastikan api yang konsisten
    di berbagai metode inisialisasi.

    metode:
        __call__: inisialisasi bobot (alias untuk initialize)
        initialize: inisialisasi bobot dengan metode spesifik
    """

    @abstractmethod
    def initialize(self, shape: tuple) -> np.ndarray:
        """
        inisialisasi bobot dengan metode spesifik.

        argumen:
            shape: bentuk array bobot yang akan diinisialisasi

        kembali:
            array bobot yang sudah diinisialisasi
        """
        pass

    def __call__(self, shape: tuple) -> np.ndarray:
        """
        inisialisasi bobot (metode kemudahan).

        argumen:
            shape: bentuk array bobot yang akan diinisialisasi

        kembali:
            array bobot yang sudah diinisialisasi
        """
        return self.initialize(shape)
