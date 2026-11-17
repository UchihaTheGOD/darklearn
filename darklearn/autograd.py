"""darklearn.autograd — backward helpers and a scalar Value node.

The Tensor class in :mod:`darklearn.tensor` already builds the graph
(add / mul / matmul / relu / tanh / sum / mean / reshape). This module
provides the public entry points and a numeric grad-check, plus a tiny
scalar ``Value`` (micrograd-style) for teaching / debugging.
"""

from __future__ import annotations

import math

from .tensor import Tensor


def backward(tensor: Tensor) -> None:
    """Run reverse-mode autodiff from a scalar ``tensor``."""
    tensor.backward()


def gradcheck(func, tensor: Tensor, eps: float = 1e-3, tol: float = 1e-2) -> bool:
    """Compare analytic grad of ``func(tensor)`` against numeric grad.

    ``func`` must return a scalar Tensor. Returns True when the max
    absolute error is below ``tol``. Raises on non-scalar output.
    Only supports 1-d input tensors on Day 1.
    """
    out = func(tensor)
    if out.numel != 1:
        raise ValueError("gradcheck needs func to return a scalar Tensor")
    tensor.zero_grad()
    out.backward()
    analytic = [float(x) for x in tensor.grad._data] if tensor.grad is not None else [0.0] * tensor.numel
    numeric = []
    orig = [float(x) for x in tensor._data]
    for i in range(tensor.numel):
        tensor._data[i] = orig[i] + eps
        plus = func(tensor).item()
        tensor._data[i] = orig[i] - eps
        minus = func(tensor).item()
        tensor._data[i] = orig[i]
        numeric.append((plus - minus) / (2.0 * eps))
    tensor.zero_grad()
    err = max(abs(a - n) for a, n in zip(analytic, numeric)) if analytic else 0.0
    return err < tol


class Value:
    """Scalar autograd node (micrograd-style), for small graphs / demos.

    Supports +, *, relu, tanh, and .backward(). Not used by Tensor —
    it exists so the grad idea can be tested in isolation.
    """

    def __init__(self, data: float, _prev=(), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _bw():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _bw():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _bw
        return out

    __rmul__ = __mul__

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _bw():
            self.grad += out.grad if self.data > 0 else 0.0
        out._backward = _bw
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _bw():
            self.grad += (1.0 - out.data ** 2) * out.grad
        out._backward = _bw
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


__all__ = ["backward", "gradcheck", "Value", "Tensor"]
