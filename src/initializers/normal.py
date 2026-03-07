"""Normal Initializer."""

import numpy as np
from .base import BaseInitializer


class NormalInitializer(BaseInitializer):
    """Inisialisasi bobot dari distribusi normal."""

    def __init__(self, mean: float = 0.0, variance: float = 0.1, seed: int = None):
        """Inisialisasi dengan mean dan variance."""
        self.mean = mean
        self.variance = variance
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def initialize(self, shape: tuple) -> np.ndarray:
        """Return array dengan nilai normal random."""
        std = np.sqrt(self.variance)
        return np.random.normal(self.mean, std, size=shape)
