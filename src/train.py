"""
Training for FFNN
"""

import numpy as np
import json
import os
import sys

# Setup project path untuk imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.ffnn import FFNN
from src.utils.plotting import plot_training_history
from src.utils.pipeline import prepare_dataset, evaluate_model, save_training_artifacts
import matplotlib.pyplot as plt


class FFNNTrainer:
    """
    Helper class untuk training FFNN dengan berbagai konfigurasi.
    """

    def __init__(self, data_path='data/datasetml_2026.csv', output_dir='training_results'):
        """
        Inisialisasi trainer.

        argumen:
            data_path: Path ke dataset
            output_dir: Directory untuk menyimpan hasil training
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.preprocessor = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.info = None

        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Preprocess dan split dataset.

        argumen:
            test_size: Proporsi test data
            val_size: Proporsi validation data dari train
            random_state: Random seed
        """
        print("\n" + "="*70)
        print("PREPARING DATA")
        print("="*70)

        (self.preprocessor, self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test, self.info) = prepare_dataset(
            data_path=self.data_path,
            output_dir=self.output_dir,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            visualize=True
        )

        print(f"\nDataset prepared:")
        print(f"  Features: {self.info['n_features']}")
        print(f"  Classes: {self.info['n_classes']}")
        print(f"  Train: {self.info['n_train_samples']}")
        print(f"  Val: {self.info['n_val_samples']}")
        print(f"  Test: {self.info['n_test_samples']}")

    def train_model(
        self,
        model_name,
        layer_sizes,
        activations,
        loss_function='categorical_cross_entropy',
        initializer='uniform',
        learning_rate=0.01,
        batch_size=32,
        epochs=100,
        regularizer=None,
        verbose=1
    ):
        """
        Train model dengan konfigurasi tertentu.

        argumen:
            model_name: Nama untuk model
            layer_sizes: List ukuran layer
            activations: List fungsi aktivasi
            loss_function: Fungsi loss
            initializer: Metode inisialisasi
            learning_rate: Learning rate
            batch_size: Ukuran batch
            epochs: Jumlah epoch
            regularizer: Regularizer config
            verbose: Level verbosity
        """
        print("\n" + "="*70)
        print(f"TRAINING MODEL: {model_name}")
        print("="*70)

        print(f"\nConfiguration:")
        print(f"  Layer sizes: {layer_sizes}")
        print(f"  Activations: {activations}")
        print(f"  Loss: {loss_function}")
        print(f"  Initializer: {initializer}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        if regularizer:
            print(f"  Regularizer: {regularizer['type']} (lambda={regularizer['lambda_param']})")

        # Buat model
        model = FFNN(
            layer_sizes=layer_sizes,
            activations=activations,
            loss_function=loss_function,
            initializer=initializer,
            learning_rate=learning_rate,
            regularizer=regularizer
        )

        # Training
        history = model.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            verbose=verbose
        )

        # Evaluasi
        eval_results = evaluate_model(
            model, self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test
        )

        train_acc = eval_results['train_accuracy']
        val_acc = eval_results['val_accuracy']
        test_acc = eval_results['test_accuracy']

        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")

        # Simpan model dan history
        model_path, history_path = save_training_artifacts(
            model=model,
            history=history,
            output_dir=self.output_dir,
            model_name=model_name
        )

        # Plot training history
        plot_path = os.path.join(self.output_dir, f'{model_name}_training.png')
        fig = plot_training_history(history)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nSaved:")
        print(f"  Model: {model_path}")
        print(f"  History: {history_path}")
        print(f"  Plot: {plot_path}")

        return {
            'name': model_name,
            'model': model,
            'history': history,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'config': {
                'layer_sizes': layer_sizes,
                'activations': activations,
                'loss_function': loss_function,
                'initializer': initializer,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'regularizer': regularizer
            }
        }

    def compare_models(self, results_dict, metric='test_acc'):
        """
        Bandingkan beberapa model.

        argumen:
            results_dict: Dictionary hasil training
            metric: Metric untuk dibandingkan
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        # Sort by metric
        sorted_results = sorted(
            results_dict.items(),
            key=lambda x: x[1][metric],
            reverse=True
        )

        print(f"\nRanking by {metric}:")
        for rank, (name, result) in enumerate(sorted_results, 1):
            print(f"  {rank}. {name}:")
            print(f"       Train Acc: {result['train_acc']:.4f}")
            print(f"       Val Acc:   {result['val_acc']:.4f}")
            print(f"       Test Acc:  {result['test_acc']:.4f}")
            print(f"       Train Loss: {result['final_train_loss']:.4f}")
            print(f"       Val Loss:   {result['final_val_loss']:.4f}")

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy comparison
        names = list(results_dict.keys())
        train_accs = [results_dict[name]['train_acc'] for name in names]
        val_accs = [results_dict[name]['val_acc'] for name in names]
        test_accs = [results_dict[name]['test_acc'] for name in names]

        x = np.arange(len(names))
        width = 0.25

        axes[0].bar(x - width, train_accs, width, label='Train', alpha=0.8)
        axes[0].bar(x, val_accs, width, label='Val', alpha=0.8)
        axes[0].bar(x + width, test_accs, width, label='Test', alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss comparison
        train_losses = [results_dict[name]['final_train_loss'] for name in names]
        val_losses = [results_dict[name]['final_val_loss'] for name in names]

        axes[1].bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
        axes[1].bar(x + width/2, val_losses, width, label='Val Loss', alpha=0.8)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Final Loss Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nComparison plot saved to: {comparison_path}")

    def save_summary(self, results_dict):
        """
        Simpan summary hasil eksperimen ke file JSON.

        argumen:
            results_dict: Dictionary hasil training
        """
        summary = {}

        for name, result in results_dict.items():
            summary[name] = {
                'train_accuracy': float(result['train_acc']),
                'val_accuracy': float(result['val_acc']),
                'test_accuracy': float(result['test_acc']),
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'config': result['config']
            }

        summary_path = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")


def example_basic_training():
    """Contoh: Training dasar."""
    trainer = FFNNTrainer(output_dir='results_basic')
    trainer.prepare_data()

    n_features = trainer.info['n_features']
    n_classes = trainer.info['n_classes']

    result = trainer.train_model(
        model_name='basic_ffnn',
        layer_sizes=[n_features, 32, 16, n_classes],
        activations=['relu', 'relu', 'softmax'],
        learning_rate=0.01,
        epochs=50
    )

    return result


def example_width_comparison():
    """Contoh: Membandingkan berbagai width."""
    trainer = FFNNTrainer(output_dir='results_width')
    trainer.prepare_data()

    n_features = trainer.info['n_features']
    n_classes = trainer.info['n_classes']

    widths = [8, 16, 32]
    results = {}

    for width in widths:
        name = f'width_{width}'
        result = trainer.train_model(
            model_name=name,
            layer_sizes=[n_features, width, width, n_classes],
            activations=['relu', 'relu', 'softmax'],
            learning_rate=0.01,
            epochs=50,
            verbose=0
        )
        results[name] = result

    trainer.compare_models(results)
    trainer.save_summary(results)

    return results


def example_activation_comparison():
    """Contoh: Membandingkan berbagai activation."""
    trainer = FFNNTrainer(output_dir='results_activation')
    trainer.prepare_data()

    n_features = trainer.info['n_features']
    n_classes = trainer.info['n_classes']

    activations = ['linear', 'relu', 'sigmoid', 'tanh']
    results = {}

    for activation in activations:
        name = f'activ_{activation}'
        result = trainer.train_model(
            model_name=name,
            layer_sizes=[n_features, 32, 16, n_classes],
            activations=[activation, activation, 'softmax'],
            learning_rate=0.01,
            epochs=50,
            verbose=0
        )
        results[name] = result

    trainer.compare_models(results)
    trainer.save_summary(results)

    return results


def example_learning_rate_comparison():
    """Contoh: Membandingkan berbagai learning rate."""
    trainer = FFNNTrainer(output_dir='results_lr')
    trainer.prepare_data()

    n_features = trainer.info['n_features']
    n_classes = trainer.info['n_classes']

    learning_rates = [0.001, 0.01, 0.1]
    results = {}

    for lr in learning_rates:
        name = f'lr_{lr}'
        result = trainer.train_model(
            model_name=name,
            layer_sizes=[n_features, 32, 16, n_classes],
            activations=['relu', 'relu', 'softmax'],
            learning_rate=lr,
            epochs=50,
            verbose=0
        )
        results[name] = result

    trainer.compare_models(results)
    trainer.save_summary(results)

    return results


def example_regularization_comparison():
    """Contoh: Membandingkan regularisasi."""
    trainer = FFNNTrainer(output_dir='results_regularization')
    trainer.prepare_data()

    n_features = trainer.info['n_features']
    n_classes = trainer.info['n_classes']

    regs = [
        None,
        {'type': 'l1', 'lambda_param': 0.01},
        {'type': 'l2', 'lambda_param': 0.01}
    ]
    reg_names = ['none', 'l1', 'l2']
    results = {}

    for reg_name, reg in zip(reg_names, regs):
        name = f'reg_{reg_name}'
        result = trainer.train_model(
            model_name=name,
            layer_sizes=[n_features, 32, 16, n_classes],
            activations=['relu', 'relu', 'softmax'],
            learning_rate=0.01,
            epochs=50,
            regularizer=reg,
            verbose=0
        )
        results[name] = result

    trainer.compare_models(results)
    trainer.save_summary(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train FFNN with various configurations')
    parser.add_argument('--example', type=str, default='basic',
                        choices=['basic', 'width', 'activation', 'lr', 'regularization'],
                        help='Example to run')

    args = parser.parse_args()

    print("="*70)
    print("FFNN TRAINING SCRIPT")
    print("="*70)
    print(f"\nRunning example: {args.example}")

    if args.example == 'basic':
        example_basic_training()
    elif args.example == 'width':
        example_width_comparison()
    elif args.example == 'activation':
        example_activation_comparison()
    elif args.example == 'lr':
        example_learning_rate_comparison()
    elif args.example == 'regularization':
        example_regularization_comparison()

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)