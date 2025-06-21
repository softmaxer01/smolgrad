import math as m
import random

class Value:
    def __init__(self, _data, _children=(), _op="", _label=""):
        self._data = _data
        self._grad = 0.0
        self._op = _op
        self._label = _label
        self._prev = set(_children)
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

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return Value(other) * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports int/float powers"
        out = Value(self._data ** other, (self,), f"**{other}")

        def _backward():
            self._grad += other * (self._data ** (other - 1)) * out._grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self._data
        t = (m.exp(2 * x) - 1) / (m.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self._grad += (1 - t ** 2) * out._grad
        out._backward = _backward
        return out

    def exp(self):
        x = self._data
        out = Value(m.exp(x), (self,), "exp")

        def _backward():
            self._grad += out._data * out._grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self._grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, num_inp):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inp)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act += wi * xi
        return act.tanh()

    def params(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, num_inp, num_out):
        self.neurons = [Neuron(num_inp) for _ in range(num_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def params(self):
        return [p for neuron in self.neurons for p in neuron.params()]


class MLP:
    def __init__(self, num_inp, num_outs):
        sz = [num_inp] + num_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(num_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x if isinstance(x, list) else [x])
        return x

    def params(self):
        return [p for layer in self.layers for p in layer.params()]
