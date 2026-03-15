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
        Uses optimized computation while maintaining gradient connections.

        Args:
            X: Input numpy array (batch_size, input_dim)
            W: Weights sebagai Value objects (input_dim, output_dim)
            b: Biases sebagai Value objects (output_dim,)

        Returns:
            Z: Output sebagai Value objects (batch_size, output_dim)
        """
        batch_size, input_dim = X.shape
        output_dim = len(b)

        # Convert weights to numpy for efficient matrix multiplication
        W_np = np.array([[w.data for w in row] for row in W])
        b_np = np.array([bias.data for bias in b])

        # Vectorized matrix multiplication (fast!)
        Z_data = np.dot(X, W_np) + b_np  # (batch_size, output_dim)

        # Create Value objects - for backward pass, we'll compute gradients manually
        Z = [[Value(z) for z in row] for row in Z_data]

        # Store raw data for manual gradient computation in backward()
        self._last_Z_data = Z_data
        self._last_X = X

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

        # Compute scalar loss value
        if self.loss_function == 'mse':
            # MSE = 0.5 * mean((y_true - y_pred)^2)
            loss_value = 0.5 * np.mean((y_true - y_pred) ** 2)

        elif self.loss_function == 'categorical_cross_entropy':
            # Cross entropy untuk multi-class: CCE = -sum(y_true * log(y_pred))
            y_pred_clipped = np.clip(y_pred, 1e-15, 1.0)
            loss_value = -np.mean(y_true * np.log(y_pred_clipped))

        elif self.loss_function == 'binary_cross_entropy':
            # Binary cross entropy
            y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
            loss_value = -np.mean(y_true * np.log(y_pred_clipped) +
                                 (1 - y_true) * np.log(1 - y_pred_clipped))

        # Create a Value object for the loss (used for display)
        loss = Value(float(loss_value))

        return loss

    def backward(self, y_true: np.ndarray = None, y_pred: np.ndarray = None) -> Dict[str, List]:
        """
        Backward propagation - computes gradients manually for efficiency.

        Args:
            y_true: True labels (optional, used for gradient computation)
            y_pred: Predicted labels (optional, used for gradient computation)
        """
        if not self.use_autodiff:
            raise NotImplementedError("Backward manual harus diimplementasi untuk non-autodiff mode")

        # Use stored values from forward pass if not provided
        if y_true is None:
            y_true = getattr(self, '_last_y_train', None)
        if y_pred is None:
            y_pred = getattr(self, '_last_y_pred', None)

        # Use stored X from forward pass
        X = self.layer_inputs[0] if self.layer_inputs else None

        # Convert y_true to one-hot if needed
        if y_true is not None and len(y_true.shape) == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((y_true.shape[0], n_classes))
            y_true_onehot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1
            y_true = y_true_onehot

        # Get stored intermediate values
        if not hasattr(self, '_last_layer_outputs') or self._last_layer_outputs is None:
            # Fallback: compute gradients using stored data
            pass

        # Compute gradient of loss w.r.t. output (dL/dY)
        if y_true is not None and y_pred is not None:
            if self.loss_function == 'mse':
                # dL/dY = (Y_pred - Y_true) / n
                dL_dY = (y_pred - y_true) / y_true.shape[0]
            elif self.loss_function == 'categorical_cross_entropy':
                # For softmax + cross-entropy: dL/dY = Y_pred - Y_true
                # (this is a special case - no division by n needed)
                dL_dY = y_pred - y_true
            elif self.loss_function == 'binary_cross_entropy':
                # dL/dY = (Y_pred - Y_true) / (Y_pred * (1 - Y_pred) * n)
                dL_dY = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15) / y_true.shape[0]
            else:
                dL_dY = np.zeros_like(y_pred)
        else:
            dL_dY = np.ones_like(self.layer_outputs[-1]) / y_true.shape[0] if y_true is not None else np.ones((1, self.layer_sizes[-1]))

        # Compute gradients through layers in reverse order
        weight_grads = []
        bias_grads = []

        for i in reversed(range(len(self.weights))):
            # Get stored values for this layer
            # For now, use stored data or compute fresh
            W = self.weights[i]
            b = self.biases[i]

            # Get input to this layer
            if i == 0:
                X_layer = X
            else:
                X_layer = self.activations_outputs[i-1] if i-1 < len(self.activations_outputs) else self.layer_inputs[i]

            # Convert Value weights to numpy for gradient computation
            W_np = np.array([[w.data for w in row] for row in W])
            b_np = np.array([bias.data for bias in b])

            # Compute dL/dZ (gradient through linear transformation)
            # For ReLU: dL/dZ = dL/dY * ReLU'(Z)
            # For softmax: handled in loss gradient
            activation_name = self.activations[i].lower()

            # Get pre-activation values (Z)
            Z_np = np.dot(X_layer, W_np) + b_np

            # Apply activation derivative
            if activation_name == 'relu':
                dZ = (Z_np > 0).astype(float)
                dL_dZ = dL_dY * dZ
            elif activation_name == 'sigmoid':
                sig = 1 / (1 + np.exp(-Z_np))
                dZ = sig * (1 - sig)
                dL_dZ = dL_dY * dZ
            elif activation_name == 'tanh':
                dZ = 1 - np.tanh(Z_np) ** 2
                dL_dZ = dL_dY * dZ
            elif activation_name == 'softmax':
                # For softmax + cross-entropy, this is already handled
                dL_dZ = dL_dY
            elif activation_name == 'linear':
                dL_dZ = dL_dY
            else:
                dL_dZ = dL_dY

            # Compute gradients: dL/dW = X.T @ dL/dZ
            dL_dW = np.dot(X_layer.T, dL_dZ)
            dL_db = np.sum(dL_dZ, axis=0)

            weight_grads.insert(0, dL_dW)
            bias_grads.insert(0, dL_db)

            # Propagate gradient to previous layer
            # dL/dX = dL/dZ @ W.T
            dL_dY = np.dot(dL_dZ, W_np.T)

        # Store gradients in the Value objects for compatibility
        for i, (w_grad, b_grad) in enumerate(zip(weight_grads, bias_grads)):
            for r in range(len(self.weights[i])):
                for c in range(len(self.weights[i][r])):
                    self.weights[i][r][c].grad = w_grad[r, c]
            for j in range(len(self.biases[i])):
                self.biases[i][j].grad = b_grad[j]

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
            # Store predictions for gradient computation
            self._last_y_pred = y_pred
            self._last_y_train = y_train
            # Use manual gradient computation (faster than Value.backward())
            gradients = self.backward(y_train, y_pred)
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
