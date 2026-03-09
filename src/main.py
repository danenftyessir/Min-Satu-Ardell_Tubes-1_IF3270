import argparse
import os
import sys
import numpy as np

# setup project path untuk imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.ffnn import FFNN
from src.models.autodiff_ffnn import AutodiffFFNN
from src.utils.plotting import plot_training_history
from src.utils.pipeline import prepare_dataset, evaluate_model, save_training_artifacts
import matplotlib.pyplot as plt


def parse_args():
    """parse argumen baris perintah."""
    parser = argparse.ArgumentParser(description='implementasi Feedforward Neural Network')

    # argumen data
    parser.add_argument('--data', type=str, default='data/datasetml_2026.csv',
                        help='path ke file dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='proporsi data untuk testing')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='proporsi data training untuk validasi')

    # arsitektur model
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='ukuran layer (contoh: --layers 64 32 16 2 untuk 3 hidden layers)')
    parser.add_argument('--activations', type=str, nargs='+', default=None,
                        help='fungsi aktivasi untuk setiap layer (contoh: --activations relu relu softmax)')

    # parameter training
    parser.add_argument('--loss', type=str, default='categorical_cross_entropy',
                        choices=['mse', 'binary_cross_entropy', 'categorical_cross_entropy'],
                        help='fungsi loss')
    parser.add_argument('--initializer', type=str, default='uniform',
                        choices=['zero', 'uniform', 'normal', 'xavier', 'he'],
                        help='metode inisialisasi bobot')
    parser.add_argument('--optimizer', type=str, default='gd',
                        choices=['gd'],
                        help='optimizer (gd=gradient descent)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size untuk training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='jumlah training epochs')
    parser.add_argument('--verbose', type=int, default=1,
                        choices=[0, 1],
                        help='tingkat verbositas (0=silent, 1=progress)')

    # regularisasi
    parser.add_argument('--regularizer', type=str, default=None,
                        choices=['l1', 'l2'],
                        help='tipe regularisasi')
    parser.add_argument('--lambda', type=float, default=0.01,
                        dest='lambda_param',
                        help='parameter lambda regularisasi')

    # normalization
    parser.add_argument('--normalization', type=str, nargs='+', default=None,
                        choices=['none', 'rmsnorm', 'layernorm'],
                        help='normalization untuk setiap layer (contoh: --normalization rmsnorm layernorm none)')

    # output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='direktori untuk menyimpan hasil')
    parser.add_argument('--model-name', type=str, default='ffnn_model',
                        help='nama untuk menyimpan model')

    # aksi
    parser.add_argument('--skip-train', action='store_true',
                        help='lewati training dan hanya evaluasi')
    parser.add_argument('--model-path', type=str, default=None,
                        help='path untuk memuat model yang sudah dilatih')
    parser.add_argument('--use-autodiff', action='store_true',
                        help='gunakan autodiff.Value untuk automatic differentiation')

    return parser.parse_args()


def main():
    """fungsi utama."""
    args = parse_args()

    # buat direktori output
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("FEEDFORWARD NEURAL NETWORK - PROGRAM UTAMA")
    print("="*70)

    # 1. preprocessing
    print("\n[1] PREPROCESSING DATA")
    print("-" * 70)
    _, X_train, X_val, X_test, y_train, y_val, y_test, info = prepare_dataset(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )

    print(f"\ninfo dataset:")
    print(f"  fitur: {info['n_features']}")
    print(f"  kelas: {info['n_classes']}")
    print(f"  train: {info['n_train_samples']}, val: {info['n_val_samples']}, test: {info['n_test_samples']}")

    # 2. bangun atau muat model
    print("\n[2] SETUP MODEL")
    print("-" * 70)

    if args.model_path and args.skip_train:
        # muat model yang sudah ada
        print(f"memuat model dari: {args.model_path}")
        model = FFNN(
            layer_sizes=[],
            activations=[],
            loss_function=args.loss,
            initializer=args.initializer
        )
        model.load(args.model_path)
    elif args.use_autodiff:
        # gunakan AutodiffFFNN (40% bonus)
        print("menggunakan AutodiffFFNN dengan autodiff.Value")

        # cek keterbatasan autodiff
        if args.regularizer:
            print("[WARNING] Regularizer tidak didukung dalam mode autodiff, diabaikan.")
        if args.normalization and any(n is not None for n in args.normalization):
            print("[WARNING] Normalization tidak didukung dalam mode autodiff, diabaikan.")

        # ukuran layer default jika tidak ditentukan
        if args.layers is None:
            n_hidden = 2
            hidden_neurons = 32
            args.layers = [info['n_features']] + [hidden_neurons] * n_hidden + [info['n_classes']]

        # aktivasi default jika tidak ditentukan
        if args.activations is None:
            n_activations = len(args.layers) - 1
            args.activations = ['relu'] * (n_activations - 1) + ['softmax']

        print(f"arsitektur:")
        print(f"  ukuran layer: {args.layers}")
        print(f"  aktivasi: {args.activations}")
        print(f"  fungsi loss: {args.loss}")
        print(f"  inisialisasi: {args.initializer}")
        print(f"  mode: AUTODIFF (40% bonus)")

        model = AutodiffFFNN(
            layer_sizes=args.layers,
            activations=args.activations,
            loss_function=args.loss,
            initializer=args.initializer,
            learning_rate=args.learning_rate,
            use_autodiff=True
        )
    else:
        # buat model baru
        # ukuran layer default jika tidak ditentukan
        if args.layers is None:
            n_hidden = 2
            hidden_neurons = 32
            args.layers = [info['n_features']] + [hidden_neurons] * n_hidden + [info['n_classes']]

        # aktivasi default jika tidak ditentukan
        if args.activations is None:
            n_activations = len(args.layers) - 1
            args.activations = ['relu'] * (n_activations - 1) + ['softmax']

        print(f"arsitektur:")
        print(f"  ukuran layer: {args.layers}")
        print(f"  aktivasi: {args.activations}")
        print(f"  fungsi loss: {args.loss}")
        print(f"  inisialisasi: {args.initializer}")
        print(f"  optimizer: {args.optimizer}")

        # setup regularizer
        regularizer = None
        if args.regularizer:
            regularizer = {
                'type': args.regularizer,
                'lambda_param': args.lambda_param
            }
            print(f"  regularizer: {args.regularizer} (lambda={args.lambda_param})")

        # setup normalization
        n_layers = len(args.layers) - 1  # Jumlah hidden + output layer
        if args.normalization is None:
            # Default: tidak ada normalization
            normalization = [None] * n_layers
        else:
            # Konversi 'none' string ke None
            normalization = []
            for i in range(n_layers):
                if i < len(args.normalization):
                    norm_type = args.normalization[i]
                    normalization.append(None if norm_type == 'none' else norm_type)
                else:
                    # Jika kurang, gunakan normalization terakhir
                    normalization.append(args.normalization[-1] if args.normalization[-1] != 'none' else None)

            print(f"  normalization: {normalization}")

        model = FFNN(
            layer_sizes=args.layers,
            activations=args.activations,
            loss_function=args.loss,
            initializer=args.initializer,
            learning_rate=args.learning_rate,
            regularizer=regularizer,
            normalization=normalization
        )

    # 3. training
    if not args.skip_train:
        print("\n[3] TRAINING")
        print("-" * 70)
        print(f"konfigurasi training:")
        print(f"  epochs: {args.epochs}")
        print(f"  batch size: {args.batch_size}")
        print(f"  learning rate: {args.learning_rate}")
        print(f"  verbose: {args.verbose}")

        # training berbeda untuk AutodiffFFNN vs FFNN
        if args.use_autodiff:
            # training loop untuk AutodiffFFNN
            print("\n menggunakan training loop autodiff (lebih lambat tapi edukatif)...")

            history = {
                'train_loss': [],
                'val_loss': [],
                'train_accuracy': [],
                'val_accuracy': []
            }

            n_samples = X_train.shape[0]
            n_batches = max(1, n_samples // args.batch_size)

            for epoch in range(args.epochs):
                epoch_train_loss = 0.0

                # mini-batch training
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * args.batch_size
                    end_idx = min(start_idx + args.batch_size, n_samples)

                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]

                    # training step
                    result = model.train_step(X_batch, y_batch, X_val, y_val)
                    epoch_train_loss += result['loss']

                # average metrics
                epoch_train_loss /= n_batches

                # validation
                val_pred = model.forward(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred + 1e-15))

                # accuracy
                train_pred = model.forward(X_train)
                train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
                val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(val_loss)
                history['train_accuracy'].append(train_acc)
                history['val_accuracy'].append(val_acc)

                if args.verbose == 1 and (epoch % 10 == 0 or epoch == args.epochs - 1):
                    print(f"epoch {epoch+1}/{args.epochs} - "
                          f"loss: {epoch_train_loss:.4f} - val_loss: {val_loss:.4f} - "
                          f"train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
        else:
            # training loop standard untuk FFNN
            history = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                verbose=args.verbose
            )

        # plot training history
        fig = plot_training_history(history)
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        print(f"\ntraining history disimpan ke: {os.path.join(args.output_dir, 'training_history.png')}")

        # simpan model dan training history
        if args.use_autodiff:
            # AutodiffFFNN menggunakan save() method
            model_path = os.path.join(args.output_dir, f'{args.model_name}.pkl')
            model.save(model_path)

            # save history separately
            import pickle
            history_path = os.path.join(args.output_dir, f'{args.model_name}_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)

            print(f"model disimpan ke: {model_path}")
            print(f"training history disimpan ke: {history_path}")
        else:
            # FFNN menggunakan utility function
            model_path, history_path = save_training_artifacts(
                model=model,
                history=history,
                output_dir=args.output_dir,
                model_name=args.model_name
            )
            print(f"model disimpan ke: {model_path}")
            print(f"training history disimpan ke: {history_path}")

    # 4. evaluasi
    print("\n[4] EVALUASI")
    print("-" * 70)

    # evaluasi model
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    train_acc = results['train_accuracy']
    val_acc = results['val_accuracy']
    test_acc = results['test_accuracy']

    print(f"train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    # metrik tambahan
    print(f"\nmetrik tambahan (test set):")
    print(f"  precision: {results['precision']:.4f}")
    print(f"  recall:    {results['recall']:.4f}")
    print(f"  f1 score:  {results['f1_score']:.4f}")

    # confusion matrix
    print(f"\nconfusion matrix:")
    print(results['confusion_matrix'])

    # 5. visualisasi
    print("\n[5] VISUALISASI")
    print("-" * 70)

    # plot distribusi bobot
    try:
        model.plot_weight_distribution()
        plt.savefig(os.path.join(args.output_dir, 'weight_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"distribusi bobot disimpan")
    except Exception as e:
        print(f"tidak bisa plot distribusi bobot: {e}")

    # plot distribusi gradien
    try:
        if hasattr(model, 'weight_gradients') and model.weight_gradients:
            model.plot_gradient_distribution()
            plt.savefig(os.path.join(args.output_dir, 'gradient_distribution.png'), dpi=300, bbox_inches='tight')
            print(f"distribusi gradien disimpan")
    except Exception as e:
        print(f"tidak bisa plot distribusi gradien: {e}")

    # 6. ringkasan
    print("\n" + "="*70)
    print("RINGKASAN")
    print("="*70)
    print(f"\nhasil disimpan ke: {args.output_dir}")
    print(f"test accuracy akhir: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nfile yang dihasilkan:")
    print(f"  - eda_visualization.png")
    print(f"  - training_history.png")
    print(f"  - weight_distribution.png")
    print(f"  - gradient_distribution.png")
    if not args.skip_train:
        print(f"  - {args.model_name}.pkl (model)")
        print(f"  - {args.model_name}_history.pkl (training history)")

if __name__ == "__main__":
    main()