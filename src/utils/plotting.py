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
    plot distribusi bobot untuk layer yang ditentukan.

    argumen:
        weights: daftar matriks bobot untuk setiap layer
        layer_indices: daftar indeks layer yang akan diplot (0-indexed). jika None, plot semua layer.
        title: judul untuk plot
        figsize: ukuran figure (lebar, tinggi)
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
    plot distribusi gradien untuk layer yang ditentukan.

    argumen:
        gradients: daftar array gradien untuk setiap layer
        layer_indices: daftar indeks layer yang akan diplot (0-indexed). jika None, plot semua layer.
        title: judul untuk plot
        figsize: ukuran figure (lebar, tinggi)
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
    plot kurva loss training dan validasi.

    argumen:
        history: dictionary yang berisi history training dengan kunci:
                - 'train_loss': daftar loss training per epoch
                - 'val_loss': daftar loss validasi per epoch (opsional)
        figsize: ukuran figure (lebar, tinggi)
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # plot training loss
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    # plot validation loss jika tersedia
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
    plot beberapa history training untuk perbandingan.

    argumen:
        histories: dictionary di mana kunci adalah nama model dan nilai adalah dictionary history
        metric: metrik yang akan diplot ('train_loss' atau 'val_loss')
        figsize: ukuran figure (lebar, tinggi)
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
