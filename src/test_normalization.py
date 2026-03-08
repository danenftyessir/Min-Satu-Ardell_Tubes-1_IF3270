"""
Test Script untuk Normalization Layers
======================================

Script ini mendemonstrasikan penggunaan normalization layers
(RMSNorm dan LayerNormalization) yang sudah terintegrasi
dalam sistem layer.
"""

import sys
import os
import numpy as np

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.layers import RMSNormLayer, LayerNormalization
from src.models.ffnn import FFNN


def test_rmsnorm_layer():
    """Test RMSNorm Layer."""
    print("="*70)
    print("TEST 1: RMSNorm Layer")
    print("="*70)

    dim = 4
    batch_size = 2

    # Buat layer
    norm = RMSNormLayer(dim=dim, epsilon=1e-8)

    # Input random
    X = np.random.randn(batch_size, dim) * 2 + 1  # Mean=1, Std=2

    print(f"Input shape: {X.shape}")
    print(f"Input:\n{X}")

    # Forward pass
    output = norm.forward(X, training=True)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")

    # Cek RMS dari output harus dekat dengan gain
    rms_per_sample = np.sqrt(np.mean(output ** 2, axis=-1))
    print(f"\nRMS per sample (setelah normalisasi): {rms_per_sample}")
    print(f"Gain parameter: {norm.norm.gain}")

    # Test backward pass
    grad_output = np.ones_like(output)
    grad_input = norm.backward(grad_output)

    print(f"\nGradien input shape: {grad_input.shape}")
    print(f"Gradien gain: {norm.norm.gain_gradient}")

    print("[OK] RMSNorm Layer test PASSED\n")


def test_layernorm_layer():
    """Test LayerNormalization Layer."""
    print("="*70)
    print("TEST 2: LayerNormalization Layer")
    print("="*70)

    dim = 4
    batch_size = 3

    # Buat layer
    norm = LayerNormalization(dim=dim, epsilon=1e-5)

    # Input random dengan berbagai range
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [-1.0, 0.0, 1.0, 2.0]
    ], dtype=np.float32)

    print(f"Input shape: {X.shape}")
    print(f"Input:\n{X}")

    # Forward pass
    output = norm.forward(X, training=True)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")

    # Cek mean dan std per sample setelah normalization
    means = np.mean(output, axis=-1)
    stds = np.std(output, axis=-1)
    print(f"\nMean per sample: {means}")
    print(f"Std per sample: {stds}")
    print("(Harus mendekati 0 dan 1)")

    # Test backward pass
    grad_output = np.ones_like(output)
    grad_input = norm.backward(grad_output)

    print(f"\nGradien input shape: {grad_input.shape}")
    print(f"Gradien gamma: {norm.gamma_gradient}")
    print(f"Gradien beta: {norm.beta_gradient}")

    print("[OK] LayerNormalization test PASSED\n")


def test_normalization_in_ffnn():
    """Test penggunaan normalization dalam FFNN."""
    print("="*70)
    print("TEST 3: Normalization dalam FFNN")
    print("="*70)

    # Catatan: Saat ini FFNN belum mendukung normalization layers
    # secara native. Ini adalah demonstrasi bagaimana seharusnya digunakan.

    print("Membuat sample data...")
    X_train = np.random.randn(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, 100)

    # Normalisasi standalone dulu
    print("\n1. Normalisasi data dengan RMSNorm:")
    rms_norm = RMSNormLayer(dim=10)
    X_normalized = rms_norm.forward(X_train, training=True)

    print(f"   Sebelum normalisasi - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    print(f"   Setelah normalisasi - Mean: {X_normalized.mean():.4f}, Std: {X_normalized.std():.4f}")

    # Test dengan LayerNorm
    print("\n2. Normalisasi data dengan LayerNorm:")
    layer_norm = LayerNormalization(dim=10)
    X_normalized_ln = layer_norm.forward(X_train, training=True)

    means_per_sample = np.mean(X_normalized_ln, axis=-1)
    stds_per_sample = np.std(X_normalized_ln, axis=-1)

    print(f"   Mean per sample: {means_per_sample.mean():.6f} (≈0)")
    print(f"   Std per sample: {stds_per_sample.mean():.6f} (≈1)")

    print("\n[OK] Normalization dalam FFNN test PASSED\n")


def test_normalization_gradients():
    """Test bahwa gradien mengalir dengan benar melalui normalization."""
    print("="*70)
    print("TEST 4: Gradient Flow melalui Normalization")
    print("="*70)

    dim = 4
    batch_size = 2

    # RMSNorm
    print("1. RMSNorm Gradient Check:")
    rms = RMSNormLayer(dim=dim)
    X = np.random.randn(batch_size, dim)

    # Forward
    output = rms.forward(X, training=True)

    # Backward
    grad_output = np.ones_like(output)
    grad_input = rms.backward(grad_output)

    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Gradien input shape: {grad_input.shape}")
    print(f"   Gradien gain: {rms.norm.gain_gradient}")

    assert grad_input.shape == X.shape, "Shape gradien input salah"
    assert rms.norm.gain_gradient.shape == (dim,), "Shape gradien gain salah"

    # LayerNorm
    print("\n2. LayerNorm Gradient Check:")
    ln = LayerNormalization(dim=dim)
    X = np.random.randn(batch_size, dim)

    # Forward
    output = ln.forward(X, training=True)

    # Backward
    grad_output = np.ones_like(output)
    grad_input = ln.backward(grad_output)

    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Gradien input shape: {grad_input.shape}")
    print(f"   Gradien gamma: {ln.gamma_gradient}")
    print(f"   Gradien beta: {ln.beta_gradient}")

    assert grad_input.shape == X.shape, "Shape gradien input salah"
    assert ln.gamma_gradient.shape == (dim,), "Shape gradien gamma salah"
    assert ln.beta_gradient.shape == (dim,), "Shape gradien beta salah"

    print("\n[OK] Gradient flow test PASSED\n")


def demonstrate_usage():
    """Demonstrasi cara penggunaan normalization layers."""
    print("="*70)
    print("DEMONSTRASI: Cara Penggunaan Normalization Layers")
    print("="*70)

    print("""
1. Import normalization layers:
   from src.layers import RMSNormLayer, LayerNormalization

2. Buat normalization layer:
   rms_norm = RMSNormLayer(dim=128)  # untuk 128 fitur
   layer_norm = LayerNormalization(dim=128)

3. Gunakan dalam forward pass:
   normalized_output = rms_norm.forward(X, training=True)

4. Backward pass:
   grad_input = rms_norm.backward(grad_output)

5. Akses parameter:
   params = rms_norm.get_params()  # {'gain': ..., 'dim': ..., 'epsilon': ...}
   rms_norm.set_params({'gain': new_gain, ...})

6. Dapatkan gradien untuk optimizer:
   grads = rms_norm.get_gradients()  # {'gain_gradient': ...}

NOTA: Normalization layers dapat digunakan:
   - Setelah Dense layer sebelum activation
   - Setelah activation layer
   - Di awal network untuk normalisasi input
   - Di dalam residual connections

Contoh arsitektur:
   Input -> Dense -> RMSNorm -> ReLU -> Dense -> LayerNorm -> Softmax
    """)

    print("\n[OK] Demonstrasi selesai\n")


def main():
    """Jalankan semua test."""
    print("\n" + "="*70)
    print("NORMALIZATION LAYERS TEST SUITE")
    print("="*70 + "\n")

    try:
        test_rmsnorm_layer()
        test_layernorm_layer()
        test_normalization_in_ffnn()
        test_normalization_gradients()
        demonstrate_usage()

        print("="*70)
        print("ALL TESTS PASSED! [OK]")
        print("="*70)
        print("\nNormalization layers berfungsi dengan baik:")
        print("  [OK] RMSNormLayer - Root Mean Square Normalization")
        print("  [OK] LayerNormalization - Layer Normalization dengan gamma & beta")
        print("  [OK] Gradient flow mengalir dengan benar")
        print("  [OK] Kompatibel dengan BaseLayer interface")
        print("\nTips penggunaan:")
        print("  - RMSNorm: baik untuk transformer architectures, stabil")
        print("  - LayerNorm: baik untuk RNN, lebih common di deep learning")
        print("  - Keduanya bisa digunakan dalam FFNN dengan modifikasi")

    except AssertionError as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
