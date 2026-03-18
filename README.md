# Feedforward Neural Network from Scratch

Implementasi lengkap Feedforward Neural Network (FFNN) dari nol tanpa menggunakan library machine learning external seperti TensorFlow atau PyTorch. Proyek ini dibuat untuk memenuhi Tugas Besar 1 IF3270 Pembelajaran Mesin 2025/2026.

Kelompok: Min Satu Ardell

13523124 - Muhammad Raihaan Perdana

13523136 - Danendra Shafi Athallah

13523155 - M. Abizzar Gamadrian

## Table of Contents

- [Deskripsi Proyek](#deskripsi-proyek)
- [Struktur Proyek](#struktur-proyek)
- [Fitur](#fitur)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Arsitektur](#arsitektur)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Lisensi](#lisensi)

## Deskripsi Proyek

Proyek ini mengimplementasikan Feedforward Neural Network dengan berbagai fitur:

- **Manual Backpropagation**: Implementasi backpropagation manual untuk menghitung gradien
- **Automatic Differentiation**: Dukungan untuk automatic differentiation menggunakan struktur data `autodiff.Value`
- **Flexible Architecture**: Mendukung berbagai arsitektur neural network dengan konfigurasi yang dapat disesuaikan
- **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, Linear, dan Softmax
- **Various Optimizers**: Gradient Descent dan Adam
- **Regularization**: L1, L2, dan Dropout
- **Normalization**: RMSNorm dan Layer Normalization
- **Comparison**: Perbandingan dengan scikit-learn MLPClassifier

## Struktur Proyek

```
Min-Satu-Ardell_Tubes-1_IF3270/
├── src/                          # Source code utama
│   ├── activations/              # Fungsi aktivasi
│   │   ├── base.py
│   │   ├── sigmoid.py
│   │   ├── relu.py
│   │   ├── tanh.py
│   │   ├── linear.py
│   │   ├── softmax.py
│   │   └── bonus_activations.py
│   ├── autodiff/               # Automatic differentiation engine
│   │   ├── engine.py
│   │   └── value.py
│   ├── initializers/           # Weight initialization
│   │   ├── base.py
│   │   ├── zero.py
│   │   ├── uniform.py
│   │   ├── normal.py
│   │   └── bonus_initializers.py (Xavier, He)
│   ├── layers/                 # Layer neural network
│   │   ├── base.py
│   │   ├── dense.py
│   │   ├── input.py
│   │   ├── dropout.py
│   │   └── normalization.py
│   ├── losses/                 # Loss functions
│   │   ├── base.py
│   │   ├── mse.py
│   │   ├── binary_cross_entropy.py
│   │   └── categorical_cross_entropy.py
│   ├── models/                 # Model neural network
│   │   ├── base.py
│   │   ├── ffnn.py            # FFNN dengan manual backprop
│   │   └── autodiff_ffnn.py   # FFNN dengan autodiff
│   ├── normalization/         # Normalization layers
│   │   ├── base.py
│   │   └── rmsnorm.py
│   ├── optimizers/            # Optimizers
│   │   ├── base.py
│   │   ├── gradient_descent.py
│   │   └── adam.py
│   ├── regularizers/          # Regularization
│   │   ├── base.py
│   │   ├── l1.py
│   │   └── l2.py
│   ├── utils/                 # Utility functions
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── preprocessing.py
│   │   ├── pipeline.py
│   │   ├── io.py
│   │   └── tune_hyperparams.py
│   ├── main.py                # Main CLI interface
│   └── train.py              # Training script
├── notebooks/                  # Jupyter notebooks
│   ├── testing.ipynb          # Eksperimen dan testing
│   └── compare_ffnn_sklearn.py
├── data/                       # Dataset
├── docs/                       # Dokumentasi
│   └── Spesifikasi Tugas Besar 1 IF3270 Pembelajaran Mesin 2025_2026.docx.pdf
├── results/                    # Hasil training
└── README.md
```

## Fitur

### Activation Functions
- Sigmoid
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Linear
- Softmax

### Initializers
- Zero Initializer
- Uniform Initializer
- Normal Initializer
- Xavier (Glorot) Initializer
- He Initializer

### Loss Functions
- Mean Squared Error (MSE)
- Binary Cross Entropy
- Categorical Cross Entropy

### Optimizers
- Gradient Descent
- Adam (Adaptive Moment Estimation)

### Regularization
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Dropout

### Normalization
- RMSNorm
- Layer Normalization

### CLI Features
- Konfigurasi arsitektur melalui command line
- Custom learning rate, batch size, epochs
- Load dan save model
- Visualisasi training history

## Instalasi

### Prerequisites
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Cara Install

1. Clone repository:
```bash
git clone <repository-url>
cd Min-Satu-Ardell_Tubes-1_IF3270
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Penggunaan

### CLI Basic Usage

```bash
cd src

# Training dengan konfigurasi default
python main.py --data ../data/datasetml_2026.csv

# Training dengan custom arsitektur
python main.py --data ../data/datasetml_2026.csv --layers 64 32 16 --activations relu relu softmax

# Training dengan custom hyperparameters
python main.py --data ../data/datasetml_2026.csv --learning-rate 0.01 --epochs 100 --batch-size 32

# Training dengan regularization
python main.py --data ../data/datasetml_2026.csv --regularizer l2 --lambda 0.001

# Training dengan normalization
python main.py --data ../data/datasetml_2026.csv --normalization rmsnorm

# Training dengan autodiff
python main.py --data ../data/datasetml_2026.csv --use-autodiff

# Load dan evaluasi model
python main.py --model-path ../results/ffnn_model.pkl --skip-train
```

### Parameter CLI Lengkap

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--data` | data/datasetml_2026.csv | Path ke dataset |
| `--layers` | None | Ukuran hidden layers |
| `--activations` | None | Activation functions |
| `--loss` | categorical_cross_entropy | Loss function |
| `--initializer` | uniform | Weight initializer |
| `--optimizer` | gd | Optimizer (gd/sgd) |
| `--learning-rate` | 0.01 | Learning rate |
| `--batch-size` | 32 | Batch size |
| `--epochs` | 100 | Jumlah epoch |
| `--regularizer` | None | Regularization type (l1/l2) |
| `--lambda` | 0.01 | Regularization lambda |
| `--normalization` | None | Normalization type |
| `--use-autodiff` | False | Gunakan autodiff engine |

### Jupyter Notebook

Buka `notebooks/testing.ipynb` untuk melihat eksperimen lengkap dan cara penggunaan dalam notebook.

## Arsitektur

### Neural Network Structure

```
Input Layer → Hidden Layers → Output Layer
     ↓            ↓              ↓
  Dense      Dense/           Softmax/
  Layer     Normalization     Sigmoid
```

### Backpropagation

Implementasi backpropagation manual menghitung gradien menggunakan chain rule:

1. **Forward Pass**: Input → Hidden Layers → Output
2. **Compute Loss**: Hitung error antara prediksi dan target
3. **Backward Pass**: Hitung gradien dari output ke input
4. **Update Weights**: Update bobot menggunakan optimizer

### Automatic Differentiation

Dukungan untuk automatic differentiation menggunakan `autodiff.Value`:

- Setiap operasi matematik membangun computational graph
- Gradien dihitung secara otomatis dengan backpropagation
- Mendukung operasi dasar: +, -, *, /, **, log, exp, dll.

## Hasil Eksperimen

### Perbandingan Hyperparameters

| Eksperimen | Parameter Terbaik | Accuracy |
|------------|-------------------|----------|
| Activation Functions | Linear | **0.7665** |
| Learning Rate | 0.1 | **0.7630** |
| Initializer | Xavier | **0.7640** |
| Optimizer | Adam (lr=0.01) | **0.7605** |
| Regularization | L2 (λ=0.001) | **0.7655** |
| Batch Size | 8 | 0.7535 |
| Depth | 4 layers | 0.7545 |
| Width | 8 neurons | 0.7520 |
| Normalization | None | **0.7665** |
| LR Scheduler | Plateau (factor=0.3) | **0.7620** |

### Perbandingan dengan scikit-learn

| Model | Accuracy |
|-------|----------|
| Custom FFNN (Manual Backprop) | **0.7605** |
| sklearn MLPClassifier | **0.7615** |

**Kesimpulan**: Perbedaan hanya **0.1%** menunjukkan bahwa implementasiFFNN dari scratch memiliki performa yang kompetitif dengan library machine learning standar.

### Hyperparameters Optimal

Berdasarkan eksperimen, hyperparameters optimal untuk dataset ini adalah:

- **Activation**: Linear
- **Learning Rate**: 0.1
- **Batch Size**: 8-16
- **Initializer**: Xavier
- **Regularization**: L2 (λ=0.001)
- **Depth**: 4 hidden layers
- **Width**: 8 neurons per layer
- **Optimizer**: Adam atau SGD

### Insight Penting

1. **Linear Activation** memberikan performa terbaik, kemungkinan karena dataset yang digunakan relatif sederhana
2. **Xavier Initialization** penting untuk mencegah vanishing/exploding gradients
3. **L2 Regularization** membantu mencegah overfitting
4. **Learning Rate tinggi (0.1)** lebih efektif untuk dataset ini
5. **Tanpa Normalization** memberikan hasil terbaik - kemungkinan karena data sudah cukup bersih

Proyek ini dibuat untuk tugas akademik IF3270 Pembelajaran Mesin 2025/2026.
