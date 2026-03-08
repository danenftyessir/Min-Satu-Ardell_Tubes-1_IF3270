"""
titik masuk utama untuk implementasi FFNN
=========================================

script ini adalah titik masuk utama untuk menjalankan Feedforward Neural Network.
"""

import argparse
import sys
import os
import numpy as np

# tambahkan direktori parent ke path untuk import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.preprocessing import DataPreprocessor
from src.models.ffnn import FFNN
from src.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from src.utils.plotting import plot_training_history, plot_weight_distribution, plot_gradient_distribution
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
                        choices=['gd', 'adam'],
                        help='optimizer (gd=gradient descent, adam)')
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
    preprocessor = DataPreprocessor(args.data)
    preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.visualize_data(save_path=os.path.join(args.output_dir, 'eda_visualization.png'))

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )

    info = preprocessor.get_data_info()
    print(f"\ninfo dataset:")
    print(f"  fitur: {info['n_features']}")
    print(f"  kelas: {info['n_classes']}")
    print(f"  train: {info['n_train_samples']}, val: {info['n_val_samples']}, test: {info['n_test_samples']}")

    # simpan data yang sudah diproses
    preprocessor.save_processed_data(output_dir=os.path.join(args.output_dir, 'processed_data'))

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

        model = FFNN(
            layer_sizes=args.layers,
            activations=args.activations,
            loss_function=args.loss,
            initializer=args.initializer,
            learning_rate=args.learning_rate,
            regularizer=regularizer
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

        # simpan model
        model_path = os.path.join(args.output_dir, f'{args.model_name}.pkl')
        model.save(model_path)
        print(f"model disimpan ke: {model_path}")

        # simpan training history
        from utils.io import save_training_history
        history_path = os.path.join(args.output_dir, f'{args.model_name}_history.pkl')
        save_training_history(history, history_path)
        print(f"training history disimpan ke: {history_path}")

    # 4. evaluasi
    print("\n[4] EVALUASI")
    print("-" * 70)

    # evaluasi set train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy(y_train, y_train_pred)
    print(f"train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

    # evaluasi set validasi
    y_val_pred = model.predict(X_val)
    val_acc = accuracy(y_val, y_val_pred)
    print(f"val accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)")

    # evaluasi set test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy(y_test, y_test_pred)
    print(f"test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

    # metrik tambahan
    print(f"\nmetrik tambahan (test set):")
    print(f"  precision: {precision(y_test, y_test_pred):.4f}")
    print(f"  recall:    {recall(y_test, y_test_pred):.4f}")
    print(f"  f1 score:  {f1_score(y_test, y_test_pred):.4f}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nconfusion matrix:")
    print(cm)

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

    print("\n" + "="*70)
    print("PROGRAM SELESAI DENGAN BERHASIL!")
    print("="*70)


if __name__ == "__main__":
    main()