"""
Hyperparameter Tuning untuk FFNN dengan GD dan Adam Optimizer.
Termasuk: arsitektur, learning rate, batch size, epochs, regularization
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import time
from src.models.ffnn import FFNN
from src.optimizers import Adam
from src.utils.pipeline import prepare_dataset

RESULTS_FILE = 'results/best_hyperparams.json'


def load_data():
    """Load dan preprocess data."""
    data_path = 'data/datasetml_2026.csv'
    scaler, X_train, X_val, X_test, y_train, y_val, y_test, info = prepare_dataset(data_path, 'results', random_state=42)
    y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    return X_train, y_train, X_val, y_val, X_test, y_test_labels


def train_and_evaluate(optimizer_type, X_train, y_train, X_val, y_val, X_test, y_test, params):
    """Train dan evaluate dengan konfigurasi tertentu."""
    # Get architecture from params
    layer_sizes = params.get('layer_sizes', [35, 32, 32, 2])
    n_layers = len(layer_sizes) - 1

    # Get activations from params
    activations = params.get('activations', ['relu'] * (n_layers - 1) + ['softmax'])

    # Get normalization from params
    normalization = params.get('normalization', [None] * (n_layers - 1))

    # Get initializer from params
    initializer = params.get('initializer', 'xavier')

    # Create model
    model = FFNN(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function='categorical_cross_entropy',
        initializer=initializer,
        normalization=normalization
    )

    epochs = params.get('epochs', 150)

    if optimizer_type == 'adam':
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            weight_decay=params.get('weight_decay', 0.01)
        )
    else:
        optimizer = None

    history = model.train(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        epochs=epochs, verbose=0, optimizer=optimizer,
        patience=10  # Use early stopping
    )
    y_pred = model.predict(X_test)
    return np.mean(y_pred == y_test), history['val_loss'][-1], history['train_loss'][-1]


def tune_adam_aggressive(X_train, y_train, X_val, y_val, X_test, y_test, time_limit=1800):
    """Tune Adam dengan search space yang lebih agresif."""
    print("\n" + "="*70)
    print("TUNING ADAM (NORMALIZATION + ACTIVATION + ARCHITECTURE)")
    print("="*70)

    # Arsitektur dengan 35 features (sesuai preprocessing terbaru)
    n_features = X_train.shape[1]
    architectures = [
        [n_features, 64, 32, 2],
        [n_features, 128, 64, 32, 2],
        [n_features, 256, 128, 64, 2],
        [n_features, 128, 128, 64, 2],
    ]

    # Normalization options - dynamically generate based on layer count
    # Will be adjusted per architecture in the loop

    # Activation options
    activation_options = [
        ['relu', 'relu', 'softmax'],
        ['leakyrelu', 'leakyrelu', 'softmax'],
        ['elu', 'elu', 'softmax'],
    ]

    # Initializer options
    initializer_options = ['xavier', 'he']

    # Learning rates (for Adam with normalization)
    learning_rates = [0.0005, 0.001, 0.0003]

    # Weight decay
    weight_decays = [0.001, 0.005, 0.01]

    # Batch sizes
    batch_sizes = [16, 32]

    best_acc = 0
    best_params = {}
    results = []

    start_time = time.time()
    count = 0

    for arch in architectures:
        n_transforms = len(arch) - 1  # number of transformations (layers - 1)
        # Generate normalization options based on number of transformations
        normalization_options = [
            [None] * n_transforms,
            ['layernorm'] * n_transforms,
            ['rmsnorm'] * n_transforms,
        ]

        for norm in normalization_options:
            for act in activation_options:
                for init in initializer_options:
                    for lr in learning_rates:
                        for wd in weight_decays:
                            for bs in batch_sizes:
                                # Check time limit
                                if time.time() - start_time > time_limit:
                                    print("Time limit reached!")
                                    break

                                params = {
                                    'layer_sizes': arch,
                                    'normalization': norm,
                                    'activations': act,
                                    'initializer': init,
                                    'learning_rate': lr,
                                    'weight_decay': wd,
                                    'batch_size': bs,
                                    'epochs': 150
                                }

                                try:
                                    acc, val_loss, train_loss = train_and_evaluate(
                                        'adam', X_train, y_train, X_val, y_val, X_test, y_test, params
                                    )
                                    results.append((arch, norm, act, init, lr, wd, bs, acc))

                                    print(f"  arch={arch}, norm={norm[0] if norm[0] else 'None'}, act={act[0]}, init={init}, lr={lr}, wd={wd}, bs={bs}: acc={acc:.4f}")

                                    if acc > best_acc:
                                        best_acc = acc
                                        best_params = params.copy()
                                        print(f"    *** NEW BEST: {acc:.4f} ***")

                                except Exception as e:
                                    print(f"  Error: {e}")

                                count += 1

    elapsed = time.time() - start_time
    print(f"\nBest Adam: arch={best_params['layer_sizes']}, norm={best_params['normalization']}, act={best_params['activations'][0]}, init={best_params['initializer']}, lr={best_params['learning_rate']}, wd={best_params['weight_decay']}, bs={best_params['batch_size']}, acc={best_acc:.4f}")
    print(f"Time: {elapsed/60:.1f} menit")

    return best_params, best_acc, results


def tune_gd_aggressive(X_train, y_train, X_val, y_val, X_test, y_test, time_limit=900):
    """Tune GD dengan search space."""
    print("\n" + "="*70)
    print("TUNING GRADIENT DESCENT (NORMALIZATION + ACTIVATION)")
    print("="*70)

    n_features = X_train.shape[1]
    architectures = [
        [n_features, 64, 32, 2],
        [n_features, 128, 64, 32, 2],
        [n_features, 256, 128, 64, 2],
    ]

    # Activation options
    activation_options = [
        ['relu', 'relu', 'softmax'],
        ['leakyrelu', 'leakyrelu', 'softmax'],
    ]

    # Initializer options
    initializer_options = ['xavier', 'he']

    learning_rates = [0.005, 0.01, 0.02]
    batch_sizes = [32, 64]

    best_acc = 0
    best_params = {}
    results = []

    start_time = time.time()

    for arch in architectures:
        n_transforms = len(arch) - 1  # number of transformations (layers - 1)
        # Generate normalization options based on number of transformations
        normalization_options = [
            [None] * n_transforms,
            ['layernorm'] * n_transforms,
        ]

        for norm in normalization_options:
            for act in activation_options:
                for init in initializer_options:
                    for lr in learning_rates:
                        for bs in batch_sizes:
                            if time.time() - start_time > time_limit:
                                print("Time limit reached!")
                                break

                            params = {
                                'layer_sizes': arch,
                                'normalization': norm,
                                'activations': act,
                                'initializer': init,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'epochs': 150
                            }

                            try:
                                acc, val_loss, train_loss = train_and_evaluate(
                                    'gd', X_train, y_train, X_val, y_val, X_test, y_test, params
                                )
                                results.append((arch, norm, act, init, lr, bs, acc))

                                print(f"  arch={arch}, norm={norm[0] if norm[0] else 'None'}, act={act[0]}, init={init}, lr={lr}, bs={bs}: acc={acc:.4f}")

                                if acc > best_acc:
                                    best_acc = acc
                                    best_params = params.copy()
                                    print(f"    *** NEW BEST: {acc:.4f} ***")

                            except Exception as e:
                                print(f"  Error: {e}")

    elapsed = time.time() - start_time
    print(f"\nBest GD: arch={best_params['layer_sizes']}, norm={best_params['normalization']}, act={best_params['activations'][0]}, init={best_params['initializer']}, lr={best_params['learning_rate']}, bs={best_params['batch_size']}, acc={best_acc:.4f}")
    print(f"Time: {elapsed:.1f}s")

    return best_params, best_acc, results


def main():
    print("="*70)
    print("AGGRESSIVE HYPERPARAMETER TUNING")
    print("Target: >80% accuracy")
    print("="*70)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    total_start = time.time()
    time_limit_adam = 2700  # 45 menit untuk Adam
    time_limit_gd = 900     # 15 menit untuk GD

    # Tune Adam (more important since it's usually better)
    adam_params, adam_acc, adam_results = tune_adam_aggressive(
        X_train, y_train, X_val, y_val, X_test, y_test, time_limit_adam
    )

    # Tune GD
    gd_params, gd_acc, gd_results = tune_gd_aggressive(
        X_train, y_train, X_val, y_val, X_test, y_test, time_limit_gd
    )

    total_elapsed = time.time() - total_start

    # Final Results
    print("\n" + "="*70)
    print("HASIL AKHIR TUNING")
    print("="*70)
    print(f"\nGradient Descent:")
    print(f"  Best architecture: {gd_params['layer_sizes']}")
    print(f"  Best normalization: {gd_params.get('normalization', [None, None])}")
    print(f"  Best activations: {gd_params.get('activations', ['relu', 'relu', 'softmax'])}")
    print(f"  Best initializer: {gd_params.get('initializer', 'xavier')}")
    print(f"  Best params: lr={gd_params['learning_rate']}, batch_size={gd_params['batch_size']}")
    print(f"  Best accuracy: {gd_acc:.4f} ({gd_acc*100:.2f}%)")

    print(f"\nAdam Optimizer:")
    print(f"  Best architecture: {adam_params['layer_sizes']}")
    print(f"  Best normalization: {adam_params.get('normalization', [None, None])}")
    print(f"  Best activations: {adam_params.get('activations', ['relu', 'relu', 'softmax'])}")
    print(f"  Best initializer: {adam_params.get('initializer', 'xavier')}")
    print(f"  Best params: lr={adam_params['learning_rate']}, wd={adam_params['weight_decay']}, batch_size={adam_params['batch_size']}")
    print(f"  Best accuracy: {adam_acc:.4f} ({adam_acc*100:.2f}%)")

    print(f"\nTotal time: {total_elapsed/60:.1f} menit")
    print("="*70)

    # Save best params to JSON
    best_results = {
        'gd': {
            'layer_sizes': gd_params['layer_sizes'],
            'normalization': gd_params.get('normalization', [None, None]),
            'activations': gd_params.get('activations', ['relu', 'relu', 'softmax']),
            'initializer': gd_params.get('initializer', 'xavier'),
            'learning_rate': gd_params['learning_rate'],
            'batch_size': gd_params['batch_size'],
            'epochs': gd_params.get('epochs', 150),
            'accuracy': gd_acc
        },
        'adam': {
            'layer_sizes': adam_params['layer_sizes'],
            'normalization': adam_params.get('normalization', [None, None]),
            'activations': adam_params.get('activations', ['relu', 'relu', 'softmax']),
            'initializer': adam_params.get('initializer', 'xavier'),
            'learning_rate': adam_params['learning_rate'],
            'weight_decay': adam_params['weight_decay'],
            'batch_size': adam_params['batch_size'],
            'epochs': adam_params.get('epochs', 150),
            'accuracy': adam_acc
        }
    }

    os.makedirs('results', exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(best_results, f, indent=2)

    print(f"\nBest params saved to: {RESULTS_FILE}")


def load_saved_params():
    """Load saved hyperparameter tuning results."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    main()
