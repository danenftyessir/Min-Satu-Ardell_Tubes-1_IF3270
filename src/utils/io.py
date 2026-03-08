import pickle
import numpy as np
from typing import Any
import os


def save_weights(weights: list, biases: list, filepath: str) -> None:
    """Simpan bobot dan bias ke file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_data = {'weights': weights, 'biases': biases}
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Bobot disimpan ke {filepath}")


def load_weights(filepath: str) -> tuple:
    """Load bobot dan bias dari file."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Bobot dimuat dari {filepath}")
    return model_data['weights'], model_data['biases']


def save_training_history(history: dict, filepath: str) -> None:
    """Simpan history training ke file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"History training disimpan ke {filepath}")


def load_training_history(filepath: str) -> dict:
    """Load history training dari file."""
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    print(f"History training dimuat dari {filepath}")
    return history
