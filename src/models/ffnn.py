import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import pickle
import os

from .base import BaseModel
from ..utils.plotting import plot_weight_distribution, plot_gradient_distribution


class FFNN(BaseModel):
    """
    Implementasi Feedforward Neural Network.

    Mendukung berbagai activation, loss function, initializer, dan regularizer.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss_function: str = 'mse',
        initializer: str = 'uniform',
        learning_rate: float = 0.01,
        regularizer: Optional[Dict] = None
    ):
        """Inisialisasi FFNN."""
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("Minimal 2 layer (input dan output)")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Jumlah aktivasi harus sama dengan jumlah layer - 1")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer

        self.weights = []
        self.biases = []
        self.weight_gradients = []
        self.bias_gradients = []

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Inisialisasi bobot dan bias."""
        from ..initializers import ZeroInitializer, UniformInitializer, NormalInitializer

        self.weights = []
        self.biases = []

        # Pilih initializer yang sesuai
        if self.initializer == 'zero':
            init = ZeroInitializer()
        elif self.initializer == 'uniform':
            init = UniformInitializer(lower_bound=-0.5, upper_bound=0.5)
        elif self.initializer == 'normal':
            init = NormalInitializer(mean=0.0, variance=0.1)
        else:
            raise ValueError(f"Initializer tidak dikenal: {self.initializer}")

        # Inisialisasi bobot untuk setiap layer
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            # Inisialisasi bobot dengan shape (input_size, output_size)
            weight = init.initialize((input_size, output_size))
            bias = np.zeros(output_size)  # Bias selalu diinisialisasi dengan 0

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        argumen:
            X: Input data dengan shape (batch_size, input_dim)

        kembali:
            Output dari neural network
        """
        from ..activations import Linear, ReLU, Sigmoid, Tanh, Softmax

        # Simpan intermediate values untuk backward pass
        self.layer_inputs = [X]  # Input untuk setiap layer
        self.layer_outputs = []  # Output sebelum activation
        self.activations_outputs = []  # Output setelah activation

        current_output = X

        # Forward pass melalui setiap layer
        for i in range(len(self.weights)):
            # Linear transformation: z = X @ W + b
            z = np.dot(current_output, self.weights[i]) + self.biases[i]
            self.layer_outputs.append(z)

            # Apply activation function
            activation_name = self.activations[i].lower()

            if activation_name == 'linear':
                activation = Linear()
            elif activation_name == 'relu':
                activation = ReLU()
            elif activation_name == 'sigmoid':
                activation = Sigmoid()
            elif activation_name == 'tanh':
                activation = Tanh()
            elif activation_name == 'softmax':
                activation = Softmax()
            else:
                raise ValueError(f"Activation function tidak dikenal: {activation_name}")

            # Apply activation
            current_output = activation.forward(z)
            self.activations_outputs.append(current_output)
            self.layer_inputs.append(current_output)

        return current_output

    def backward(self, grad: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Backward propagation - hitung gradien menggunakan chain rule.

        argumen:
            grad: Gradient dari loss function terhadap output

        kembali:
            Dictionary berisi gradien bobot dan bias
        """
        from ..activations import Linear, ReLU, Sigmoid, Tanh, Softmax

        self.weight_gradients = []
        self.bias_gradients = []

        # Gradient dari layer terakhir
        current_grad = grad

        # Backward pass melalui setiap layer (dari belakang ke depan)
        for i in range(len(self.weights) - 1, -1, -1):
            # Dapatkan activation function untuk layer ini
            activation_name = self.activations[i].lower()

            if activation_name == 'linear':
                activation = Linear()
            elif activation_name == 'relu':
                activation = ReLU()
            elif activation_name == 'sigmoid':
                activation = Sigmoid()
            elif activation_name == 'tanh':
                activation = Tanh()
            elif activation_name == 'softmax':
                activation = Softmax()
            else:
                raise ValueError(f"Activation function tidak dikenal: {activation_name}")

            # Hitung gradient dari activation function
            # Gradient sebelum activation = gradient setelah activation * turunan activation
            z = self.layer_outputs[i]  # Output sebelum activation

            if activation_name == 'softmax':
                # Untuk softmax dengan cross-entropy, gradient = s - y
                # Ini sudah diterima sebagai input grad
                activation_grad = current_grad
            else:
                # Untuk activation function lain
                activation_grad = current_grad * activation.backward(z)

            # Input ke layer ini
            layer_input = self.layer_inputs[i]

            # Hitung gradient terhadap bobot dan bias
            # dL/dW = input.T @ dL/dz
            weight_grad = np.dot(layer_input.T, activation_grad)
            bias_grad = np.sum(activation_grad, axis=0)

            # Simpan gradien
            self.weight_gradients.insert(0, weight_grad)
            self.bias_gradients.insert(0, bias_grad)

            # Hitung gradient untuk layer sebelumnya
            # dL/dX = dL/dz @ W.T
            current_grad = np.dot(activation_grad, self.weights[i].T)

        # Tambah regularizer gradient jika ada
        if self.regularizer is not None:
            from ..regularizers import L1Regularizer, L2Regularizer

            reg_type = self.regularizer.get('type', '').lower()

            if reg_type == 'l1':
                reg = L1Regularizer(lambda_param=self.regularizer.get('lambda_param', 0.01))
                for i in range(len(self.weight_gradients)):
                    reg_grad = reg.backward(self.weights[i])
                    self.weight_gradients[i] += reg_grad

            elif reg_type == 'l2':
                reg = L2Regularizer(lambda_param=self.regularizer.get('lambda_param', 0.01))
                for i in range(len(self.weight_gradients)):
                    reg_grad = reg.backward(self.weights[i])
                    self.weight_gradients[i] += reg_grad

        return {
            'weight_gradients': self.weight_gradients,
            'bias_gradients': self.bias_gradients
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        batch_size: int = 32,
        learning_rate: float = None,
        epochs: int = 100,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Latih neural network dengan mini-batch gradient descent.

        argumen:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            batch_size: Ukuran batch untuk mini-batch gradient descent
            learning_rate: Learning rate untuk update bobot
            epochs: Jumlah epoch training
            verbose: Level verbosity (0: silent, 1: progress bar)

        kembali:
            Dictionary berisi training history
        """
        from ..losses import MSELoss, BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss

        # Gunakan learning rate dari constructor jika tidak disediakan
        if learning_rate is None:
            learning_rate = self.learning_rate

        # Pilih loss function
        loss_name = self.loss_function.lower()
        if loss_name == 'mse':
            loss_fn = MSELoss()
        elif loss_name in ['binary_cross_entropy', 'bce', 'binary_crossentropy']:
            loss_fn = BinaryCrossEntropyLoss()
        elif loss_name in ['categorical_cross_entropy', 'cce', 'categorical_crossentropy']:
            loss_fn = CategoricalCrossEntropyLoss()
        else:
            raise ValueError(f"Loss function tidak dikenal: {self.loss_function}")

        # Siapkan target untuk multi-class classification jika perlu
        if loss_name in ['categorical_cross_entropy', 'cce', 'categorical_crossentropy']:
            if y_train.ndim == 1:
                # One-hot encoding
                n_classes = len(np.unique(y_train))
                y_train_encoded = np.zeros((y_train.shape[0], n_classes))
                y_train_encoded[np.arange(y_train.shape[0]), y_train] = 1
                y_train = y_train_encoded

                if y_val is not None and y_val is not None:
                    y_val_encoded = np.zeros((y_val.shape[0], n_classes))
                    y_val_encoded[np.arange(y_val.shape[0]), y_val] = 1
                    y_val = y_val_encoded

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        n_samples = X_train.shape[0]

        # Training loop
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            n_batches = 0

            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Hitung loss
                loss = loss_fn.forward(y_batch, y_pred)
                epoch_train_loss += loss
                n_batches += 1

                # Backward pass
                grad = loss_fn.backward(y_batch, y_pred)
                self.backward(grad)

                # Update bobot
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * self.weight_gradients[i]
                    self.biases[i] -= learning_rate * self.bias_gradients[i]

            # Rata-rata training loss
            avg_train_loss = epoch_train_loss / n_batches
            history['train_loss'].append(avg_train_loss)

            # Hitung validation loss jika ada validation data
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = loss_fn.forward(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
            else:
                history['val_loss'].append(0.0)

            # Print progress
            if verbose == 1:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {history['val_loss'][-1]:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}")

        self.is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Lakukan prediksi menggunakan trained model.

        argumen:
            X: Input data dengan shape (n_samples, input_dim)

        kembali:
            Predictions dengan shape (n_samples, n_output)
        """
        if not self.is_fitted:
            raise ValueError("Model belum dilatih. Panggil train() terlebih dahulu.")

        # Forward pass untuk mendapatkan predictions
        predictions = self.forward(X)

        # Untuk klasifikasi, return class labels
        loss_name = self.loss_function.lower()
        if loss_name in ['mse']:
            # Untuk regression, return raw predictions
            return predictions
        elif loss_name in ['binary_cross_entropy', 'bce', 'binary_crossentropy']:
            # Untuk binary classification, return class 0 atau 1
            return (predictions > 0.5).astype(int).flatten()
        elif loss_name in ['categorical_cross_entropy', 'cce', 'categorical_crossentropy']:
            # Untuk multi-class classification, return class dengan probabilitas tertinggi
            return np.argmax(predictions, axis=1)
        else:
            return predictions

    def save(self, filepath: str) -> None:
        """Simpan model ke file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss_function': self.loss_function,
            'initializer': self.initializer,
            'learning_rate': self.learning_rate,
            'regularizer': self.regularizer,
            'weights': self.weights,
            'biases': self.biases,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str) -> None:
        """Load model dari file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.layer_sizes = model_data['layer_sizes']
        self.activations = model_data['activations']
        self.loss_function = model_data['loss_function']
        self.initializer = model_data['initializer']
        self.learning_rate = model_data['learning_rate']
        self.regularizer = model_data['regularizer']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.is_fitted = model_data['is_fitted']

    def plot_weight_distribution(self, layers: List[int] = None) -> None:
        """
        Plot distribusi bobot untuk layer tertentu.

        argumen:
            layers: List indeks layer (0-indexed). Jika None, plot semua.
        """
        plot_weight_distribution(self.weights, layers, title="Distribusi Bobot")

    def plot_gradient_distribution(self, layers: List[int] = None) -> None:
        """
        Plot distribusi gradien untuk layer tertentu.

        argumen:
            layers: List indeks layer (0-indexed). Jika None, plot semua.
        """
        plot_gradient_distribution(self.weight_gradients, layers, title="Distribusi Gradien")
