"""
script test untuk implementasi FFNN
===================================

script ini untuk menguji implementasi Feedforward Neural Network
dengan dataset yang sudah dipreprocess.
"""

import numpy as np
import sys
import os

# tambahkan src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.ffnn import FFNN
from src.preprocessing import DataPreprocessor
from src.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def test_preprocessing():
    """test preprocessing dataset."""
    print("\n" + "="*60)
    print("TEST 1: PREPROCESSING DATASET")
    print("="*60)

    data_path = 'data/datasetml_2026.csv'
    preprocessor = DataPreprocessor(data_path)

    # muat dan eksplorasi data
    preprocessor.load_data()
    preprocessor.explore_data()

    # visualisasi data
    preprocessor.visualize_data(save_path='data/eda_visualization.png')

    # preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data()

    # tampilkan info
    info = preprocessor.get_data_info()
    print(f"\njumlah fitur: {info['n_features']}")
    print(f"jumlah kelas: {info['n_classes']}")
    print(f"distribusi kelas (train): {info['class_distribution']['train']}")

    # simpan data yang sudah diproses
    preprocessor.save_processed_data()

    return X_train, X_val, X_test, y_train, y_val, y_test, info


def test_ffnn_basic(X_train, y_train, X_val, y_val, n_features, n_classes):
    """test FFNN dengan konfigurasi dasar."""
    print("\n" + "="*60)
    print("TEST 2: FFNN BASIC FUNCTIONALITY")
    print("="*60)

    # tentukan ukuran layer
    # input layer: n_features
    # hidden layer: 16 neurons
    # output layer: n_classes (untuk klasifikasi)
    layer_sizes = [n_features, 16, n_classes]
    activations = ['relu', 'softmax']  # ReLU untuk hidden, Softmax untuk output

    print(f"\nkonfigurasi FFNN:")
    print(f"  ukuran layer: {layer_sizes}")
    print(f"  aktivasi: {activations}")
    print(f"  fungsi loss: categorical_cross_entropy")
    print(f"  inisialisasi: uniform")

    # buat model
    model = FFNN(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function='categorical_cross_entropy',
        initializer='uniform',
        learning_rate=0.01
    )

    print(f"\nmodel berhasil dibuat!")
    print(f"jumlah layer: {len(model.weights)}")

    # cek shape bobot
    print(f"\nshape bobot dan bias:")
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        print(f"  layer {i}: weights={w.shape}, biases={b.shape}")

    # test forward pass
    print(f"\ntest forward pass...")
    X_sample = X_train[:5]  # ambil 5 sample
    output = model.forward(X_sample)
    print(f"  input shape: {X_sample.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  sample output (prediksi pertama): {output[0]}")

    return model


def test_training(model, X_train, y_train, X_val, y_val, epochs=10):
    """test proses training."""
    print("\n" + "="*60)
    print("TEST 3: TRAINING FFNN")
    print("="*60)

    print(f"\nkonfigurasi training:")
    print(f"  epochs: {epochs}")
    print(f"  batch size: 32")
    print(f"  learning rate: 0.01")

    # latih model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=32,
        learning_rate=0.01,
        epochs=epochs,
        verbose=1
    )

    print(f"\ntraining selesai!")
    print(f"  final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  final val loss: {history['val_loss'][-1]:.4f}")

    # plot training history
    from src.utils.plotting import plot_training_history
    fig = plot_training_history(history)
    plt.savefig('data/training_history_test.png', dpi=300, bbox_inches='tight')
    print(f"\ntraining history plot disimpan ke: data/training_history_test.png")

    return model, history


def test_prediction(model, X_test, y_test):
    """test prediksi."""
    print("\n" + "="*60)
    print("TEST 4: PREDICTION")
    print("="*60)

    # lakukan prediksi
    y_pred = model.predict(X_test)

    print(f"\nhasil prediksi:")
    print(f"  jumlah test sample: {len(y_test)}")
    print(f"  sample prediksi (10 pertama): {y_pred[:10]}")
    print(f"  sample true label (10 pertama): {y_test[:10]}")

    # hitung metrik
    acc = accuracy(y_test, y_pred)
    print(f"\naccuracy: {acc:.4f} ({acc*100:.2f}%)")

    return y_pred


def test_weight_distribution(model):
    """test plotting distribusi bobot."""
    print("\n" + "="*60)
    print("TEST 5: WEIGHT DISTRIBUTION")
    print("="*60)

    print(f"\nplotting distribusi bobot untuk semua layer...")
    model.plot_weight_distribution()
    plt.savefig('data/weight_distribution_test.png', dpi=300, bbox_inches='tight')
    print(f"weight distribution plot disimpan ke: data/weight_distribution_test.png")


def test_save_load(model, X_test):
    """test save dan load model."""
    print("\n" + "="*60)
    print("TEST 6: SAVE & LOAD MODEL")
    print("="*60)

    # simpan model
    model_path = 'data/test_model.pkl'
    print(f"\nmenyimpan model ke: {model_path}")
    model.save(model_path)
    print(f"model berhasil disimpan!")

    # muat model
    print(f"\nmemuat model dari: {model_path}")
    loaded_model = FFNN(
        layer_sizes=model.layer_sizes,
        activations=model.activations,
        loss_function=model.loss_function,
        initializer=model.initializer
    )
    loaded_model.load(model_path)
    print(f"model berhasil dimuat!")

    # test prediksi dengan loaded model
    print(f"\ntest prediksi dengan loaded model...")
    pred_original = model.predict(X_test[:5])
    pred_loaded = loaded_model.predict(X_test[:5])

    if np.allclose(pred_original, pred_loaded):
        print(f"✓ prediksi sama! save & load berhasil.")
    else:
        print(f"✗ prediksi berbeda! ada masalah dengan save/load.")


def test_different_configurations(X_train, y_train, X_val, y_val, n_features, n_classes):
    """test berbagai konfigurasi FFNN."""
    print("\n" + "="*60)
    print("TEST 7: DIFFERENT CONFIGURATIONS")
    print("="*60)

    configs = [
        {
            'name': 'Shallow Network (2 hidden layers)',
            'layer_sizes': [n_features, 8, 8, n_classes],
            'activations': ['relu', 'relu', 'softmax']
        },
        {
            'name': 'Deep Network (4 hidden layers)',
            'layer_sizes': [n_features, 32, 16, 8, 4, n_classes],
            'activations': ['relu', 'relu', 'relu', 'relu', 'softmax']
        },
        {
            'name': 'Wide Network (wide hidden layer)',
            'layer_sizes': [n_features, 64, n_classes],
            'activations': ['relu', 'softmax']
        }
    ]

    results = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"ukuran layer: {config['layer_sizes']}")
        print(f"aktivasi: {config['activations']}")

        model = FFNN(
            layer_sizes=config['layer_sizes'],
            activations=config['activations'],
            loss_function='categorical_cross_entropy',
            initializer='uniform',
            learning_rate=0.01
        )

        # latih untuk beberapa epoch
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=32,
            learning_rate=0.01,
            epochs=5,
            verbose=0
        )

        results[config['name']] = {
            'model': model,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }

        print(f"final train loss: {history['train_loss'][-1]:.4f}")
        print(f"final val loss: {history['val_loss'][-1]:.4f}")

    return results


def main():
    """fungsi test utama."""
    print("\n" + "="*60)
    print("FFNN IMPLEMENTATION TEST SUITE")
    print("="*60)

    try:
        # test 1: preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test, info = test_preprocessing()

        # test 2: FFNN basic
        model = test_ffnn_basic(X_train, y_train, X_val, y_val, info['n_features'], info['n_classes'])

        # test 3: training
        model, history = test_training(model, X_train, y_train, X_val, y_val, epochs=20)

        # test 4: prediction
        y_pred = test_prediction(model, X_test, y_test)

        # test 5: weight distribution
        test_weight_distribution(model)

        # test 6: save & load
        test_save_load(model, X_test)

        # test 7: different configurations
        results = test_different_configurations(X_train, y_train, X_val, y_val, info['n_features'], info['n_classes'])

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nimplementasi FFNN sudah lengkap dan berfungsi dengan baik.")
        print("silakan lanjutkan dengan eksperimen sesuai spesifikasi tugas besar.")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)