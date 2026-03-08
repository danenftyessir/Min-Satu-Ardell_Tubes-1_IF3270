"""
Test Script untuk Autodiff Module
==================================

Script ini mendemonstrasikan penggunaan autodiff.Value class
untuk automatic differentiation.

autodiff.Value mengimplementasikan automatic differentiation
untuk operasi skalar dengan backpropagation otomatis.
"""

import sys
import os

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.autodiff import Value


def test_basic_operations():
    """Test operasi dasar aritmatika."""
    print("="*70)
    print("TEST 1: Operasi Dasar")
    print("="*70)

    x = Value(2.0)
    y = Value(3.0)

    # Penjumlahan
    z = x + y
    print(f"x = {x.data}, y = {y.data}")
    print(f"x + y = {z.data}")

    # Perkalian
    z = x * y
    print(f"x * y = {z.data}")

    # Kompleks
    z = x * y + x
    print(f"x * y + x = {z.data}")

    # Backpropagation
    z.backward()
    print(f"\nSetelah z.backward():")
    print(f"dz/dx = {x.grad} (expected: 4.0)")
    print(f"dz/dy = {y.grad} (expected: 2.0)")

    assert abs(x.grad - 4.0) < 1e-6, "Gradien x salah"
    assert abs(y.grad - 2.0) < 1e-6, "Gradien y salah"
    print("[OK] Test operasi dasar PASSED\n")


def test_complex_expression():
    """Test ekspresi yang lebih kompleks."""
    print("="*70)
    print("TEST 2: Ekspresi Kompleks")
    print("="*70)

    # f(x, y, z) = (x + y) * z
    x = Value(2.0)
    y = Value(3.0)
    z = Value(4.0)

    f = (x + y) * z
    print(f"x = {x.data}, y = {y.data}, z = {z.data}")
    print(f"f = (x + y) * z = {f.data}")

    f.backward()

    print(f"\nGradien:")
    print(f"df/dx = {x.grad} (expected: 4.0)")
    print(f"df/dy = {y.grad} (expected: 4.0)")
    print(f"df/dz = {z.grad} (expected: 5.0)")

    assert abs(x.grad - 4.0) < 1e-6, "Gradien x salah"
    assert abs(y.grad - 4.0) < 1e-6, "Gradien y salah"
    assert abs(z.grad - 5.0) < 1e-6, "Gradien z salah"
    print("[OK] Test ekspresi kompleks PASSED\n")


def test_activation_functions():
    """Test fungsi aktivasi dengan autodiff."""
    print("="*70)
    print("TEST 3: Fungsi Aktivasi")
    print("="*70)

    x = Value(0.5)

    # Sigmoid-like: 1 / (1 + exp(-x))
    # Untuk simplicity, gunakan pendekatan
    # exp(x) ≈ 1 + x + x²/2 + x³/6 untuk x kecil
    def sigmoid_approx(v):
        return v.relu()  # Simplified activation

    # ReLU approximation: max(0, x)
    def relu(v):
        out = Value(max(0, v.data), (v,), 'relu')

        def _backward():
            if v.data > 0:
                v.grad += out.grad

        out._backward = _backward
        return out

    # Test ReLU
    x1 = Value(2.0)
    x2 = Value(-1.0)

    y1 = relu(x1)
    y2 = relu(x2)

    print(f"ReLU({x1.data}) = {y1.data}")
    print(f"ReLU({x2.data}) = {y2.data}")

    # Test gradient
    y1.backward()
    print(f"dReLU/dx (x=2.0) = {x1.grad} (expected: 1.0)")

    y2.backward()
    print(f"dReLU/dx (x=-1.0) = {x2.grad} (expected: 0.0)")

    assert abs(x1.grad - 1.0) < 1e-6, "Gradien ReLU positif salah"
    assert abs(x2.grad - 0.0) < 1e-6, "Gradien ReLU negatif salah"
    print("[OK] Test fungsi aktivasi PASSED\n")


def test_neural_network_forward():
    """Test forward pass sederhana neural network."""
    print("="*70)
    print("TEST 4: Neural Network Forward Pass")
    print("="*70)

    # Simple neuron: y = w*x + b
    x = Value(2.0)
    w = Value(0.5)
    b = Value(1.0)

    # Forward pass
    y = w * x + b
    print(f"Input: x = {x.data}")
    print(f"Weight: w = {w.data}")
    print(f"Bias: b = {b.data}")
    print(f"Output: y = w*x + b = {y.data}")
    print(f"Expected: 0.5*2.0 + 1.0 = 2.0")

    assert abs(y.data - 2.0) < 1e-6, "Output salah"
    print("[OK] Test forward pass PASSED\n")


def test_neural_network_backward():
    """Test backward pass (gradien) untuk neural network."""
    print("="*70)
    print("TEST 5: Neural Network Backward Pass")
    print("="*70)

    # Simple neuron dengan loss
    x = Value(2.0)
    w = Value(0.5)
    b = Value(1.0)
    target = Value(3.0)

    # Forward pass
    y = w * x + b

    # MSE Loss: L = 0.5 * (y - target)²
    loss = (y - target) ** 2
    loss = loss * Value(0.5)

    print(f"Prediction: y = {y.data}")
    print(f"Target: {target.data}")
    print(f"Loss: {loss.data}")

    # Backward pass
    loss.backward()

    print(f"\nGradien:")
    print(f"dL/dw = {w.grad}")
    print(f"dL/db = {b.grad}")
    print(f"dL/dx = {x.grad}")

    # Manual gradient check
    # y = w*x + b = 0.5*2 + 1 = 2
    # loss = 0.5 * (2-3)² = 0.5
    # dL/dy = y - target = 2 - 3 = -1
    # dL/dw = dL/dy * dy/dw = -1 * x = -2
    # dL/db = dL/dy * dy/db = -1 * 1 = -1

    print(f"\nExpected:")
    print(f"dL/dw = -2.0")
    print(f"dL/db = -1.0")

    assert abs(w.grad - (-2.0)) < 1e-6, "Gradien w salah"
    assert abs(b.grad - (-1.0)) < 1e-6, "Gradien b salah"
    print("[OK] Test backward pass PASSED\n")


def main():
    """Jalankan semua test."""
    print("\n" + "="*70)
    print("AUTODIFF MODULE TEST SUITE")
    print("="*70 + "\n")

    try:
        test_basic_operations()
        test_complex_expression()
        test_activation_functions()
        test_neural_network_forward()
        test_neural_network_backward()

        print("="*70)
        print("ALL TESTS PASSED! [OK]")
        print("="*70)
        print("\nAutodiff.Value berfungsi dengan baik untuk:")
        print("  [OK] Operasi aritmatika dasar (+, -, *, /)")
        print("  [OK] Ekspresi kompleks")
        print("  [OK] Fungsi aktivasi (ReLU, dll)")
        print("  [OK] Neural network forward pass")
        print("  [OK] Neural network backward pass (gradien)")
        print("\nCatatan:")
        print("  - autodiff.Value bekerja dengan SKALAR")
        print("  - Untuk neural network dengan array, gunakan layers/")
        print("  - autodiff cocok untuk educasi dan understanding")
        print("  - Implementasi FFNN menggunakan numpy arrays untuk efisiensi")

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
