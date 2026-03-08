import numpy as np
from typing import Union


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung skor akurasi

    accuracy = (number of correct predictions) / (total predictions)

    argumen:
        y_true: label sebenarnya
        y_pred: label prediksi
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung skor precision.

    precision = TP / (TP + FP)

    argumen:
        y_true: label sebenarnya (biner: 0 atau 1)
        y_pred: label prediksi (biner: 0 atau 1)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung skor recall.

    recall = TP / (TP + FN)

    argumen:
        y_true: label sebenarnya (biner: 0 atau 1)
        y_pred: label prediksi (biner: 0 atau 1)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung skor f1.

    f1 = 2 * (precision * recall) / (precision + recall)

    argumen:
        y_true: label sebenarnya (biner: 0 atau 1)
        y_pred: label prediksi (biner: 0 atau 1)
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    hitung confusion matrix.

    untuk klasifikasi biner:
    [[TN, FP],
     [FN, TP]]

    argumen:
        y_true: label sebenarnya
        y_pred: label prediksi
    """
    # dapatkan kelas unik
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # buat confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return cm


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung mean squared error.

    MSE = (1/n) * sum((y_true - y_pred)^2)

    argumen:
        y_true: nilai sebenarnya
        y_pred: nilai prediksi
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung mean absolute error.

    MAE = (1/n) * sum(|y_true - y_pred|)

    argumen:
        y_true: nilai sebenarnya
        y_pred: nilai prediksi
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    hitung r**2 (koefisien determinasi) score.

    r**2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

    argumen:
        y_true: nilai sebenarnya
        y_pred: nilai prediksi
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
