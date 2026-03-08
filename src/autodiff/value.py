import numpy as np
from typing import Set, List, Tuple


class Value:
    """
    kelas nilai untuk automatic differentiation.

    kelas ini merepresentasikan nilai skalar yang dapat melacak operasi dan
    secara otomatis menghitung gradien menggunakan backpropagation. terinspirasi oleh
    pustaka micrograd.

    setiap objek nilai menyimpan:
    - data (nilai sebenarnya)
    - gradien (turunan terhadap suatu loss)
    - referensi ke children (operan yang membuat nilai ini)
    - operasi yang membuat nilai ini
    - fungsi backward untuk menghitung gradien lokal

    atribut:
        data: nilai skalar
        grad: gradien dari nilai ini
        _prev: himpunan nilai anak yang membuat nilai ini
        _op: operasi yang membuat nilai ini
        _backward: fungsi untuk menghitung gradien lokal

    contoh:
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x
        >>> z.backward()
        >>> print(x.grad, y.grad)  # seharusnya mencetak 4.0 dan 2.0
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = ''):
        """
        inisialisasi objek nilai.

        argumen:
            data: nilai skalar
            _children: tupel dari nilai anak (penggunaan internal)
            _op: operasi yang membuat nilai ini (penggunaan internal)
        """
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other: 'Value') -> 'Value':
        """
        operasi penjumlahan.

        argumen:
            other: nilai atau skalar lain

        kembali:
            nilai baru yang merepresentasikan penjumlahan
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: 'Value') -> 'Value':
        """penjumlahan terbalik (other + self)."""
        return self.__add__(other)

    def __sub__(self, other: 'Value') -> 'Value':
        """
        operasi pengurangan.

        argumen:
            other: nilai atau skalar lain

        kembali:
            nilai baru yang merepresentasikan selisih
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: 'Value') -> 'Value':
        """pengurangan terbalik (other - self)."""
        return Value(other) - self

    def __mul__(self, other: 'Value') -> 'Value':
        """
        operasi perkalian.

        argumen:
            other: nilai atau skalar lain

        kembali:
            nilai baru yang merepresentasikan hasil kali
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: 'Value') -> 'Value':
        """perkalian terbalik (other * self)."""
        return self.__mul__(other)

    def __truediv__(self, other: 'Value') -> 'Value':
        """
        operasi pembagian.

        argumen:
            other: nilai atau skalar lain

        kembali:
            nilai baru yang merepresentasikan hasil bagi
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: 'Value') -> 'Value':
        """pembagian terbalik (other / self)."""
        return Value(other) / self

    def __pow__(self, other: Union[float, int]) -> 'Value':
        """
        operasi pangkat.

        argumen:
            other: eksponen (skalar)

        kembali:
            nilai baru yang merepresentasikan self^other
        """
        assert isinstance(other, (int, float)), "hanya mendukung pangkat int/float untuk saat ini"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        """
        fungsi aktivasi reLU.

        kembali:
            nilai baru dengan reLU diterapkan
        """
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        """
        fungsi aktivasi sigmoid.

        kembali:
            nilai baru dengan sigmoid diterapkan
        """
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        """
        fungsi aktivasi hiperbolik tangen.

        kembali:
            nilai baru dengan tanh diterapkan
        """
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        """
        fungsi eksponensial.

        kembali:
            nilai baru dengan exp diterapkan
        """
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        lakukan backpropagation untuk menghitung gradien.

        metode ini menghitung gradien dari semua node dalam
        graf komputasi terhadap node ini.
        """
        # urutkan secara topologis semua anak dalam graf
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # satu variabel pada satu waktu dan terapkan aturan rantai
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> 'Value':
        """operasi negasi."""
        return self * -1

    def __repr__(self) -> str:
        """representasi string dari nilai."""
        return f"Value(data={self.data}, grad={self.grad})"
