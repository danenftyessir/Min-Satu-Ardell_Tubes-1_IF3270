"""
Metrics Utilities Module
========================

This module provides utility functions for computing various metrics.

Functions:
    accuracy: Compute accuracy score
    precision: Compute precision score
    recall: Compute recall score
    f1_score: Compute F1 score
    confusion_matrix: Compute confusion matrix
"""

import numpy as np
from typing import Union


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score.

    Accuracy = (number of correct predictions) / (total predictions)

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1

    Example:
        >>> acc = accuracy(y_true, y_pred)
        >>> print(f"Accuracy: {acc:.4f}")
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision score.

    Precision = TP / (TP + FP)

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)

    Returns:
        Precision score

    Example:
        >>> prec = precision(y_true, y_pred)
        >>> print(f"Precision: {prec:.4f}")
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall score.

    Recall = TP / (TP + FN)

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)

    Returns:
        Recall score

    Example:
        >>> rec = recall(y_true, y_pred)
        >>> print(f"Recall: {rec:.4f}")
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)

    Returns:
        F1 score

    Example:
        >>> f1 = f1_score(y_true, y_pred)
        >>> print(f"F1 Score: {f1:.4f}")
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    For binary classification:
    [[TN, FP],
     [FN, TP]]

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as numpy array

    Example:
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> print(cm)
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Create confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return cm


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    MSE = (1/n) * sum((y_true - y_pred)^2)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value

    Example:
        >>> mse = mean_squared_error(y_true, y_pred)
        >>> print(f"MSE: {mse:.4f}")
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    MAE = (1/n) * sum(|y_true - y_pred|)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value

    Example:
        >>> mae = mean_absolute_error(y_true, y_pred)
        >>> print(f"MAE: {mae:.4f}")
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.

    R² = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score

    Example:
        >>> r2 = r2_score(y_true, y_pred)
        >>> print(f"R² Score: {r2:.4f}")
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
