"""Zero Initializer."""

import numpy as np
from .base import BaseInitializer


class ZeroInitializer(BaseInitializer):
    """Inisialisasi semua bobot dengan nol."""

    def initialize(self, shape: tuple) -> np.ndarray:
        """Return array of zeros."""
        return np.zeros(shape)
