import numpy as np
from typing import List, Dict, Any
from .value import Value


class ComputationalGraph:
    """
    mesin graf komputasi untuk automatic differentiation.

    kelas ini mengelola pembuatan dan eksekusi graf komputasi
    untuk jaringan saraf dengan kemampuan automatic differentiation.

    mesin melacak operasi yang dilakukan pada objek nilai dan dapat
    secara otomatis menghitung gradien melalui backpropagation.

    atribut:
        values: daftar objek nilai dalam graf
        operations: daftar operasi yang dilakukan

    contoh:
        >>> graph = ComputationalGraph()
        >>> x = graph.create_value(2.0, name='x')
        >>> y = graph.create_value(3.0, name='y')
        >>> z = x * y + x
        >>> z.backward()
        >>> print(x.grad, y.grad)
    """

    def __init__(self):
        """inisialisasi mesin graf komputasi."""
        self.values: List[Value] = []
        self.parameters: Dict[str, Value] = {}

    def create_value(self, data: float, name: str = None) -> Value:
        """
        buat nilai baru dalam graf.

        argumen:
            data: nilai skalar
            name: nama opsional untuk nilai

        kembali:
            objek nilai baru
        """
        value = Value(data)
        self.values.append(value)

        if name is not None:
            self.parameters[name] = value

        return value

    def zero_grad(self) -> None:
        """
        nol-kan semua gradien dalam graf.

        ini harus dipanggil sebelum setiap propagasi mundur untuk memastikan
        gradien terakumulasi dengan benar.
        """
        for value in self.values:
            value.grad = 0.0

    def get_parameters(self) -> List[Value]:
        """
        dapatkan semua parameter (nilai bernama) dalam graf.

        kembali:
            daftar objek nilai yang merepresentasikan parameter
        """
        return list(self.parameters.values())

    def get_parameter_values(self) -> Dict[str, float]:
        """
        dapatkan nilai saat ini dari semua parameter bernama.

        kembali:
            kamus yang memetakan nama parameter ke nilainya
        """
        return {name: value.data for name, value in self.parameters.items()}

    def get_parameter_gradients(self) -> Dict[str, float]:
        """
        dapatkan gradien dari semua parameter bernama.

        kembali:
            kamus yang memetakan nama parameter ke gradiennya
        """
        return {name: value.grad for name, value in self.parameters.items()}

    def update_parameters(self, learning_rate: float) -> None:
        """
        update semua parameter menggunakan gradient descent.

        argumen:
            learning_rate: learning rate untuk update
        """
        for value in self.parameters.values():
            value.data -= learning_rate * value.grad
