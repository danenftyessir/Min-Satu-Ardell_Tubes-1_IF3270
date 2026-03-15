import pickle
import numpy as np
from typing import Any
import os
import csv
from datetime import datetime


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


def save_training_history_to_csv(history: dict, output_dir: str = 'data') -> str:
    """
    Simpan training history ke file CSV.

    argumen:
        history: Dictionary berisi 'train_loss', 'val_loss', dan opsional 'learning_rate'
        output_dir: Directory untuk menyimpan file CSV

    return:
        Path ke file CSV yang disimpan
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp untuk reproducibility
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'training_history_{timestamp}.csv')

    epochs = len(history['train_loss'])
    has_lr = 'learning_rate' in history and history['learning_rate'] is not None

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        if has_lr:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
        else:
            writer.writerow(['epoch', 'train_loss', 'val_loss'])

        # Data
        for i in range(epochs):
            row = [i + 1, history['train_loss'][i], history['val_loss'][i]]
            if has_lr:
                row.append(history['learning_rate'][i])
            writer.writerow(row)

    print(f"Training history CSV disimpan ke {filepath}")
    return filepath


def save_predictions_to_csv(
    predictions: np.ndarray,
    actual_labels: np.ndarray,
    output_dir: str = 'data',
    model_name: str = 'model'
) -> str:
    """
    Simpan prediksi model ke file CSV.

    argumen:
        predictions: Array prediksi model (bisa probabilities atau class labels)
        actual_labels: Array label aktual
        output_dir: Directory untuk menyimpan file CSV
        model_name: Nama model untuk nama file

    return:
        Path ke file CSV yang disimpan
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp untuk reproducibility
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'{model_name}_predictions_{timestamp}.csv')

    # Convert predictions ke class labels jika berupa probabilities
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Multi-class: ambil argmax
        pred_labels = np.argmax(predictions, axis=1)
    else:
        # Binary: threshold 0.5
        pred_labels = (predictions.flatten() >= 0.5).astype(int)

    # Convert actual labels ke 1D jika one-hot encoded
    if len(actual_labels.shape) > 1 and actual_labels.shape[1] > 1:
        actual_labels = np.argmax(actual_labels, axis=1)
    else:
        actual_labels = actual_labels.flatten().astype(int)

    # Check correctness
    correct = (pred_labels == actual_labels)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'actual', 'predicted', 'correct'])

        for i in range(len(pred_labels)):
            writer.writerow([i, actual_labels[i], pred_labels[i], correct[i]])

    print(f"Predictions CSV disimpan ke {filepath}")
    return filepath
