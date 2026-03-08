import numpy as np
from typing import List
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    optimizer Adam (Adaptive Moment Estimation).

    Adam menggabungkan kelebihan dari dua ekstensi lain dari stochastic
    gradient descent: AdaGrad dan RMSProp.

    Adam menghitung learning rate adaptif untuk setiap parameter dengan
    mempertahankan estimasi momen pertama dan kedua dari gradien.

    atribut:
        learning_rate: ukuran langkah untuk update bobot
        beta1: laju peluruhan eksponensial untuk momen pertama
        beta2: laju peluruhan eksponensial untuk momen kedua
        epsilon: konstanta kecil untuk stabilitas numerik
        t: penghitung langkah waktu

    contoh:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.update(weights, biases, weight_grads, bias_grads)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        inisialisasi optimizer Adam.

        argumen:
            learning_rate: ukuran langkah untuk update bobot (default: 0.001)
            beta1: laju peluruhan eksponensial untuk estimasi momen pertama (default: 0.9)
            beta2: laju peluruhan eksponensial untuk estimasi momen kedua (default: 0.999)
            epsilon: konstanta kecil untuk stabilitas numerik (default: 1e-8)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # estimasi momen untuk bobot
        self.m_w = []  # momen pertama (mean)
        self.v_w = []  # momen kedua (varians tanpa pusat)

        # estimasi momen untuk bias
        self.m_b = []
        self.v_b = []

    def _initialize_moments(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ) -> None:
        """
        inisialisasi estimasi momen jika belum diinisialisasi.

        argumen:
            weights: daftar matriks bobot
            biases: daftar vektor bias
        """
        if not self.m_w:
            for w, b in zip(weights, biases):
                self.m_w.append(np.zeros_like(w))
                self.v_w.append(np.zeros_like(w))
                self.m_b.append(np.zeros_like(b))
                self.v_b.append(np.zeros_like(b))

    def update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ) -> None:
        """
        update parameter menggunakan optimasi Adam.

        aturan update Adam:
        1. update estimasi momen pertama yang bias: m = beta1 * m + (1 - beta1) * g
        2. update estimasi momen kedua yang bias: v = beta2 * v + (1 - beta2) * g^2
        3. hitung estimasi yang sudah dikoreksi bias: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        4. update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

        argumen:
            weights: daftar matriks bobot
            biases: daftar vektor bias
            weight_gradients: daftar gradien bobot
            bias_gradients: daftar gradien bias
        """
        self._initialize_moments(weights, biases)
        self.t += 1

        for i in range(len(weights)):
            # update momen pertama (mean dari gradien)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * weight_gradients[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * bias_gradients[i]

            # update momen kedua (varians tanpa pusat dari gradien)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(weight_gradients[i])
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(bias_gradients[i])

            # hitung estimasi yang sudah dikoreksi bias
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # update parameter
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
