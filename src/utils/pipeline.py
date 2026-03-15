"""
Shared pipeline utilities for main.py and train.py

Module ini berisi fungsi-fungsi bersama yang digunakan oleh
berbagai entry point untuk mengurangi redundansi kode.
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

# Setup project path untuk imports di module ini
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def setup_project_path() -> str:
    """
    Setup project path untuk imports.

    Menambahkan parent directory (project root) ke sys.path
    sehingga import dari src dapat bekerja dengan benar.

    Returns:
        str: Path ke project root directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Hanya tambahkan jika belum ada
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    return parent_dir


def prepare_dataset(
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    visualize: bool = True
) -> Tuple:
    """
    Common data preparation workflow.

    Melakukan loading, exploration, visualization, dan preprocessing
    dataset. Digunakan oleh baik main.py maupun train.py.

    Args:
        data_path: Path ke file dataset CSV
        output_dir: Directory untuk menyimpan output
        test_size: Proporsi data untuk testing
        val_size: Proporsi data training untuk validasi
        random_state: Random seed untuk reproduktibilitas
        visualize: Apakah membuat visualisasi EDA

    Returns:
        tuple: (preprocessor, X_train, X_val, X_test, y_train, y_val, y_test, info)
    """
    from src.utils.preprocessing import DataPreprocessor

    print("\n" + "="*70)
    print("PREPARING DATASET")
    print("="*70)

    # Inisialisasi dan load data
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.explore_data()

    # Visualisasi jika diminta
    if visualize:
        viz_path = os.path.join(output_dir, 'eda_visualization.png')
        preprocessor.visualize_data(save_path=viz_path)

    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Dapatkan info dataset
    info = preprocessor.get_data_info()

    # Simpan processed data
    preprocessor.save_processed_data(
        output_dir=os.path.join(output_dir, 'processed_data')
    )

    return (preprocessor, X_train, X_val, X_test,
            y_train, y_val, y_test, info)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    metrics_func: Optional[Dict] = None
) -> Dict:
    """
    Common model evaluation workflow.

    Melakukan evaluasi model pada train, validation, dan test set
    dengan berbagai metrik.

    Args:
        model: Model yang sudah dilatih
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        metrics_func: Optional custom metrics functions

    Returns:
        dict: Dictionary berisi semua evaluation metrics
    """
    from src.utils.metrics import (accuracy, precision, recall,
                                    f1_score, confusion_matrix)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Convert predictions from one-hot to class indices if needed
    if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] > 1:
        y_train_pred = np.argmax(y_train_pred, axis=1)
    if len(y_val_pred.shape) > 1 and y_val_pred.shape[1] > 1:
        y_val_pred = np.argmax(y_val_pred, axis=1)
    if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] > 1:
        y_test_pred = np.argmax(y_test_pred, axis=1)

    # Convert y_true to 1D if needed
    if len(y_train.shape) > 1:
        y_train_idx = np.argmax(y_train, axis=1)
    else:
        y_train_idx = y_train
    if len(y_val.shape) > 1:
        y_val_idx = np.argmax(y_val, axis=1)
    else:
        y_val_idx = y_val
    if len(y_test.shape) > 1:
        y_test_idx = np.argmax(y_test, axis=1)
    else:
        y_test_idx = y_test

    # Basic accuracy metrics
    results = {
        'train_accuracy': accuracy(y_train_idx, y_train_pred),
        'val_accuracy': accuracy(y_val_idx, y_val_pred),
        'test_accuracy': accuracy(y_test_idx, y_test_pred),
        'train_pred': y_train_pred,
        'val_pred': y_val_pred,
        'test_pred': y_test_pred
    }

    # Additional metrics for test set
    results['precision'] = precision(y_test_idx, y_test_pred)
    results['recall'] = recall(y_test_idx, y_test_pred)
    results['f1_score'] = f1_score(y_test_idx, y_test_pred)
    results['confusion_matrix'] = confusion_matrix(y_test_idx, y_test_pred)

    return results


def save_training_artifacts(
    model,
    history: Dict,
    output_dir: str,
    model_name: str
) -> Tuple[str, str]:
    """
    Common save workflow untuk model dan training history.

    Args:
        model: Model yang sudah dilatih
        history: Training history dictionary
        output_dir: Output directory
        model_name: Nama untuk file yang disimpan

    Returns:
        tuple: (model_path, history_path)
    """
    from src.utils.io import save_training_history as save_history

    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pkl')
    model.save(model_path)

    # Save training history
    history_path = os.path.join(output_dir, f'{model_name}_history.pkl')
    save_history(history, history_path)

    return model_path, history_path
