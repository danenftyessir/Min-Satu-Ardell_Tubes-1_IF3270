import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from ..autodiff import Value


class AutodiffFFNN:
    """
    FFNN dengan automatic differentiation menggunakan autodiff.Value class.

    Setiap parameter (weights, biases) menjadi Value objects yang dapat melacak
    computational graph dan menghitung gradien secara otomatis.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss_function: str = 'mse',
        initializer: str = 'uniform',
        learning_rate: float = 0.01,
        use_autodiff: bool = True
    ):
        """
        Inisialisasi AutodiffFFNN.

        Args:
            layer_sizes: List ukuran setiap layer
            activations: List fungsi aktivasi
            loss_function: Fungsi loss ('mse', 'binary_cross_entropy', 'categorical_cross_entropy')
            initializer: Metode inisialisasi
            learning_rate: Learning rate
            use_autodiff: Jika True, gunakan autodiff.Value untuk parameters
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.use_autodiff = use_autodiff

        self.weights = []
        self.biases = []
        self.is_fitted = False

        if use_autodiff:
            self._initialize_autodiff_params()
        else:
            self._initialize_numpy_params()

    def _initialize_autodiff_params(self) -> None:
        """Inisialisasi parameters sebagai Value objects untuk autodiff."""
        from ..initializers import (ZeroInitializer, UniformInitializer, NormalInitializer,
                                     XavierInitializer, HeInitializer)

        # Pilih initializer
        if self.initializer == 'zero':
            init = ZeroInitializer()
        elif self.initializer == 'uniform':
            init = UniformInitializer(-0.5, 0.5)
        elif self.initializer == 'normal':
            init = NormalInitializer(0.0, 0.1)
        elif self.initializer == 'xavier':
            init = XavierInitializer(uniform=True)
        elif self.initializer == 'he':
            init = HeInitializer(uniform=False)
        else:
            raise ValueError(f"Initializer tidak dikenal: {self.initializer}")

        # Inisialisasi weights dan biases sebagai Value objects
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            # Initialize weights
            weight_data = init.initialize((input_size, output_size))
            weight_values = [[Value(weight_data[r, c])
                             for c in range(output_size)]
                             for r in range(input_size)]
            self.weights.append(weight_values)

            # Initialize biases
            bias_values = [Value(0.0) for _ in range(output_size)]
            self.biases.append(bias_values)

    def _initialize_numpy_params(self) -> None:
        """Inisialisasi parameters sebagai numpy arrays (fallback)."""
        from ..initializers import (ZeroInitializer, UniformInitializer, NormalInitializer,
                                     XavierInitializer, HeInitializer)

        if self.initializer == 'zero':
            init = ZeroInitializer()
        elif self.initializer == 'uniform':
            init = UniformInitializer(-0.5, 0.5)
        elif self.initializer == 'normal':
            init = NormalInitializer(0.0, 0.1)
        elif self.initializer == 'xavier':
            init = XavierInitializer(uniform=True)
        elif self.initializer == 'he':
            init = HeInitializer(uniform=False)
        else:
            raise ValueError(f"Initializer tidak dikenal: {self.initializer}")

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            weight = init.initialize((input_size, output_size))
            bias = np.zeros(output_size)

            self.weights.append(weight)
            self.biases.append(bias)

    def _linear_autodiff(self, X: np.ndarray, W: List[List[Value]], b: List[Value]) -> List[List[Value]]:
        """
        Linear transformation dengan autodiff: Z = X @ W + b

        Args:
            X: Input numpy array (batch_size, input_dim)
            W: Weights sebagai Value objects (input_dim, output_dim)
            b: Biases sebagai Value objects (output_dim,)

        Returns:
            Z: Output sebagai Value objects (batch_size, output_dim)
        """
        batch_size, input_dim = X.shape
        output_dim = len(b)

        Z = []
        for i in range(batch_size):
            row = []
            for j in range(output_dim):
                # Compute z[i,j] = sum(X[i,k] * W[k,j]) + b[j]
                result = Value(0.0)
                for k in range(input_dim):
                    result = result + Value(X[i, k]) * W[k][j]
                result = result + b[j]
                row.append(result)
            Z.append(row)

        return Z

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation dengan autodiff.

        Args:
            X: Input data (batch_size, input_dim)

        Returns:
            Output predictions (batch_size, output_dim)
        """
        from ..activations import Linear, ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU

        # Simpan intermediate values untuk backward pass
        self.layer_inputs = [X]
        self.layer_outputs = []  # Sebelum activation
        self.activations_outputs = []  # Setelah activation

        current_output = X

        # Forward through each layer
        for i in range(len(self.weights)):
            # Linear transformation
            if self.use_autodiff:
                Z = self._linear_autodiff(current_output, self.weights[i], self.biases[i])
            else:
                Z = np.dot(current_output, self.weights[i]) + self.biases[i]

            self.layer_outputs.append(Z)

            # Activation function
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
            elif activation_name == 'leakyrelu':
                activation = LeakyReLU()
            elif activation_name == 'elu':
                activation = ELU()
            else:
                raise ValueError(f"Activation tidak dikenal: {activation_name}")

            # Apply activation
            if self.use_autodiff:
                # Convert Value objects back to numpy for activation
                Z_np = np.array([[z.data for z in row] for row in Z])
                current_output = activation.forward(Z_np)

                # Store gradients if needed for backward
                self.activations_outputs.append(current_output)
                self.layer_inputs.append(current_output)
            else:
                current_output = activation.forward(Z)
                self.activations_outputs.append(current_output)
                self.layer_inputs.append(current_output)

        self.is_fitted = True
        return current_output

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Value:
        """
        Compute loss sebagai Value object untuk autodiff.

        Args:
            y_true: True labels
            y_pred: Predictions

        Returns:
            Loss as Value object
        """
        # Ensure y_true and y_pred are 2D arrays
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        batch_size = y_true.shape[0]
        n_classes = y_true.shape[1]

        if self.loss_function == 'mse':
            # MSE = 0.5 * mean((y_true - y_pred)^2)
            loss = Value(0.0)
            for i in range(batch_size):
                for j in range(n_classes):
                    diff = Value(y_true[i, j]) - Value(y_pred[i, j])
                    loss = loss + diff * diff * Value(0.5)
            loss = loss / Value(batch_size)

        elif self.loss_function == 'categorical_cross_entropy':
            # Cross entropy untuk multi-class
            # CCE = -sum(y_true * log(y_pred))
            loss = Value(0.0)
            for i in range(batch_size):
                for j in range(n_classes):
                    # Clip untuk log stability
                    y_pred_clipped = max(y_pred[i, j], 1e-15)
                    loss = loss - Value(y_true[i, j]) * Value(np.log(y_pred_clipped))
            loss = loss / Value(batch_size)

        elif self.loss_function == 'binary_cross_entropy':
            # Binary cross entropy
            loss = Value(0.0)
            for i in range(batch_size):
                y_pred_clipped = max(y_pred[i, 0], 1e-15)
                loss = loss - Value(y_true[i, 0]) * Value(np.log(y_pred_clipped))
                loss = loss - Value(1 - y_true[i, 0]) * Value(np.log(1.0001 - y_pred_clipped))
            loss = loss / Value(batch_size)

        return loss

    def backward(self) -> Dict[str, List]:
        """
        Backward propagation dengan autodiff - OTOMATIS!

        Hanya memanggil loss.backward() dan gradien akan mengalir ke semua parameters.
        """
        if not self.use_autodiff:
            raise NotImplementedError("Backward manual harus diimplementasi untuk non-autodiff mode")

        # Gradien sudah otomatis dihitung saat loss.backward()
        # Extract gradients dari Value objects
        weight_grads = []
        bias_grads = []

        for i in range(len(self.weights)):
            w_grad = np.array([[w.grad for w in row] for row in self.weights[i]])
            b_grad = np.array([b.grad for b in self.biases[i]])

            weight_grads.append(w_grad)
            bias_grads.append(b_grad)

            # Zero gradients for next iteration
            for row in self.weights[i]:
                for w in row:
                    w.grad = 0.0
            for b in self.biases[i]:
                b.grad = 0.0

        return {
            'weights': weight_grads,
            'biases': bias_grads
        }

    def train_step(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict:
        """
        Satu step training: forward, compute loss, backward, update parameters.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)

        Returns:
            Dict dengan metrics
        """
        # Forward pass
        y_pred = self.forward(X_train)

        # Compute loss
        if self.use_autodiff:
            loss = self._compute_loss(y_train, y_pred)
        else:
            # Fallback ke numpy computation
            if self.loss_function == 'mse':
                loss = np.mean(0.5 * (y_train - y_pred) ** 2)
            elif self.loss_function == 'categorical_cross_entropy':
                loss = -np.mean(y_train * np.log(y_pred + 1e-15))
            else:
                raise NotImplementedError(f"Loss {self.loss_function} belum diimplementasi")

        # Backward pass (autodiff magic!)
        if self.use_autodiff:
            loss.backward()
            gradients = self.backward()
        else:
            raise NotImplementedError("Manual backward belum diimplementasi")

        # Update parameters
        for i in range(len(self.weights)):
            # Update weights - create new Value objects with updated values
            self.weights[i] = [[Value(w.data - self.learning_rate * gradients['weights'][i][r, c])
                                    for c, w in enumerate(row)]
                                   for r, row in enumerate(self.weights[i])]
            # Update biases - create new Value objects with updated values
            self.biases[i] = [Value(b.data - self.learning_rate * gradients['biases'][i][j])
                                for j, b in enumerate(self.biases[i])]

        # Compute metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.forward(X_val)

            # Ensure y_val has the same shape as y_val_pred
            if len(y_val.shape) == 1:
                # Convert 1D to 2D if needed
                y_val_reshaped = y_val.reshape(-1, 1)
                # For binary classification with 2 outputs, convert to one-hot
                if y_val_pred.shape[1] == 2:
                    y_val_onehot = np.zeros((y_val.shape[0], 2))
                    for i in range(y_val.shape[0]):
                        y_val_onehot[i, int(y_val[i])] = 1
                    y_val = y_val_onehot

            val_loss = np.mean(0.5 * (y_val - y_val_pred) ** 2)
        else:
            val_loss = None

        return {
            'loss': loss.data if self.use_autodiff else loss,
            'val_loss': val_loss
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)

    def save(self, filepath: str) -> None:
        """Save model ke file."""
        import pickle

        # Convert Value objects back to numpy for saving
        weights_numpy = []
        biases_numpy = []

        for i in range(len(self.weights)):
            if self.use_autodiff:
                w_np = np.array([[w.data for w in row] for row in self.weights[i]])
                b_np = np.array([b.data for b in self.biases[i]])
            else:
                w_np = self.weights[i]
                b_np = self.biases[i]

            weights_numpy.append(w_np)
            biases_numpy.append(b_np)

        model_data = {
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'loss_function': self.loss_function,
            'initializer': self.initializer,
            'learning_rate': self.learning_rate,
            'weights': weights_numpy,
            'biases': biases_numpy,
            'use_autodiff': self.use_autodiff,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str) -> None:
        """Load model dari file."""
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.layer_sizes = model_data['layer_sizes']
        self.activations = model_data['activations']
        self.loss_function = model_data['loss_function']
        self.initializer = model_data['initializer']
        self.learning_rate = model_data['learning_rate']
        self.use_autodiff = model_data.get('use_autodiff', True)
        self.is_fitted = model_data['is_fitted']

        # Restore weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            w = model_data['weights'][i]
            b = model_data['biases'][i]

            if self.use_autodiff:
                # Convert back to Value objects
                w_values = [[Value(w[r, c]) for c in range(w.shape[1])]
                            for r in range(w.shape[0])]
                b_values = [Value(val) for val in b]
                self.weights.append(w_values)
                self.biases.append(b_values)
            else:
                self.weights.append(w)
                self.biases.append(b)
