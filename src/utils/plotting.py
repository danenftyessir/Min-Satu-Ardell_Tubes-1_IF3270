"""
Plotting Utilities Module
==========================

This module provides utility functions for plotting weight and gradient distributions.

Functions:
    plot_weight_distribution: Plot distribution of layer weights
    plot_gradient_distribution: Plot distribution of layer gradients
    plot_training_history: Plot training and validation loss curves
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict


def plot_weight_distribution(
    weights: List[np.ndarray],
    layer_indices: Optional[List[int]] = None,
    title: str = "Weight Distribution",
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot the distribution of weights for specified layers.

    Args:
        weights: List of weight matrices for each layer
        layer_indices: List of layer indices to plot (0-indexed). If None, plots all layers.
        title: Title for the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_weight_distribution(model.weights, layers=[0, 1, 2])
        >>> plt.show()
    """
    if layer_indices is None:
        layer_indices = list(range(len(weights)))

    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)

    if n_layers == 1:
        axes = [axes]

    for i, layer_idx in enumerate(layer_indices):
        weight = weights[layer_idx]
        weight_flat = weight.flatten()

        axes[i].hist(weight_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Layer {layer_idx} Weights')
        axes[i].set_xlabel('Weight Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_gradient_distribution(
    gradients: List[np.ndarray],
    layer_indices: Optional[List[int]] = None,
    title: str = "Gradient Distribution",
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot the distribution of gradients for specified layers.

    Args:
        gradients: List of gradient arrays for each layer
        layer_indices: List of layer indices to plot (0-indexed). If None, plots all layers.
        title: Title for the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_gradient_distribution(model.weight_gradients, layers=[0, 1, 2])
        >>> plt.show()
    """
    if layer_indices is None:
        layer_indices = list(range(len(gradients)))

    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)

    if n_layers == 1:
        axes = [axes]

    for i, layer_idx in enumerate(layer_indices):
        grad = gradients[layer_idx]
        grad_flat = grad.flatten()

        axes[i].hist(grad_flat, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[i].set_title(f'Layer {layer_idx} Gradients')
        axes[i].set_xlabel('Gradient Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary containing training history with keys:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch (optional)
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> history = model.train(X_train, y_train, X_val, y_val, epochs=100)
        >>> fig = plot_training_history(history)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training loss
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    # Plot validation loss if available
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_multiple_training_histories(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = 'train_loss',
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot multiple training histories for comparison.

    Args:
        histories: Dictionary where keys are model names and values are history dictionaries
        metric: Which metric to plot ('train_loss' or 'val_loss')
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> histories = {
        ...     'Model 1': history1,
        ...     'Model 2': history2,
        ...     'Model 3': history3
        ... }
        >>> fig = plot_multiple_training_histories(histories)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']

    for i, (model_name, history) in enumerate(histories.items()):
        if metric in history and history[metric]:
            epochs = range(1, len(history[metric]) + 1)
            color = colors[i % len(colors)]
            ax.plot(epochs, history[metric], color=color, label=model_name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss' if 'loss' in metric else 'Metric')
    ax.set_title(f'Training Comparison - {metric}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
