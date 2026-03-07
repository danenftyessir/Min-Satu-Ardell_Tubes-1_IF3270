"""Uniform Initializer."""

import numpy as np
from .base import BaseInitializer


class UniformInitializer(BaseInitializer):
    """Inisialisasi bobot dari distribusi uniform."""

    def __init__(self, lower_bound: float = -0.5, upper_bound: float = 0.5, seed: int = None):
        """Inisialisasi dengan batas bawah dan atas."""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def initialize(self, shape: tuple) -> np.ndarray:
        """Return array dengan nilai uniform random."""
        return np.random.uniform(self.lower_bound, self.upper_bound, size=shape)
