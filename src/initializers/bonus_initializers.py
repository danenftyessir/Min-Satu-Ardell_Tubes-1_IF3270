import numpy as np
from .base import BaseInitializer


class XavierInitializer(BaseInitializer):
    """
    inisialisasi bobot dari distribusi uniform dengan batas berdasarkan
    jumlah unit input dan output.

    untuk distribusi uniform: bounds = sqrt(6 / (fan_in + fan_out))
    untuk distribusi normal: std = sqrt(2 / (fan_in + fan_out))
    attributes:
        uniform: apakah menggunakan uniform (true) atau normal (false)
        seed: random seed untuk reproduktibilitas
    """

    def __init__(self, uniform: bool = True, seed: int = None):
        """
        inisialisasi xavier.

        argumen:
            uniform: gunakan distribusi uniform jika true, distribusi normal jika false
            seed: random seed untuk reproduktibilitas
        """
        self.uniform = uniform
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def initialize(self, shape: tuple) -> np.ndarray:
        """
        inisialisasi bobot menggunakan inisialisasi xavier.

        argumen:
            shape: shape dari array bobot (fan_in, fan_out)
        """
        fan_in, fan_out = shape[0], shape[1]

        if self.uniform:
            # distribusi uniform
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, size=shape)
        else:
            # distribusi normal
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0, std, size=shape)


class HeInitializer(BaseInitializer):
    """
    inisialisasi he (kaiming).

    inisialisasi bobot dari distribusi normal dengan variansi berdasarkan
    jumlah unit input.

    untuk distribusi normal: std = sqrt(2 / fan_in)

    untuk distribusi uniform: bounds = sqrt(6 / fan_in)

    inisialisasi he dirancang khusus untuk relu dan variannya.
    memperhitungkan fakta bahwa relu menghapus setengah dari aktivasi.
    """

    def __init__(self, uniform: bool = False, seed: int = None):
        """
        inisialisasi he.

        argumen:
            uniform: gunakan distribusi uniform jika true, distribusi normal jika false
            seed: random seed untuk reproduktibilitas
        """
        self.uniform = uniform
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def initialize(self, shape: tuple) -> np.ndarray:
        """
        inisialisasi bobot menggunakan inisialisasi he.

        argumen:
            shape: shape dari array bobot (fan_in, fan_out)
        """
        fan_in = shape[0]

        if self.uniform:
            # distribusi uniform
            limit = np.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, size=shape)
        else:
            # distribusi normal (default dan paling umum)
            std = np.sqrt(2.0 / fan_in)
            return np.random.normal(0, std, size=shape)
