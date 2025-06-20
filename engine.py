import math as m

class Value:
    def __init__(self, _data, _children=(), _op="", _label=""):
        self._data = _data
        self._prev = set(_children)
        self._op = _op
        self._label = _label
        self._grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self._data}, grad={self._grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._data + other._data, (self, other), "+")

        def _backward():
            self._grad += 1.0 * out._grad
            other._grad += 1.0 * out._grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._data * other._data, (self, other), "*")

        def _backward():
            self._grad += other._data * out._grad
            other._grad += self._data * out._grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return Value(other) * self**-1

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float powers supported"
        out = Value(self._data ** other, (self,), f"**{other}")

        def _backward():
            self._grad += other * (self._data ** (other - 1)) * out._grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self._data
        _tanh = (m.exp(2 * x) - 1) / (m.exp(2 * x) + 1)
        out = Value(_tanh, (self,), "tanh")

        def _backward():
            self._grad += (1 - _tanh ** 2) * out._grad
        out._backward = _backward

        return out

    def exp(self):
        x = self._data
        out = Value(m.exp(x), (self,), "exp")

        def _backward():
            self._grad += out._data * out._grad  # use cached value
        out._backward = _backward

        return out
