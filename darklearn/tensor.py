"""darklearn.tensor — minimal Tensor, pure Python stdlib only.

Storage is a flat ``array('f')`` in row-major order plus a ``shape`` tuple.
No numpy, no external dependency. Ops are deliberately small and correct;
speed comes from flat storage, tight loops, and (later) an optional C core.

Supported: 0-d scalars, 1-d vectors, 2-d matrices.
Elementwise ops support same-shape, scalar, and (m, n) + (n,) row broadcast.
"""

from __future__ import annotations

import math
import random
from array import array
from typing import Iterable, Sequence, Union

Number = Union[int, float]


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _flatten(data) -> tuple[list[float], tuple[int, ...]]:
    """Flatten nested lists / scalars into (flat_list, shape)."""
    if isinstance(data, (int, float)):
        return [float(data)], ()
    if isinstance(data, array):
        return [float(x) for x in data], None  # shape supplied by caller
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], (0,)
        if isinstance(data[0], (list, tuple)):
            # nested: assume rectangular
            rows = len(data)
            inner, inner_shape = _flatten(data[0])
            cols = len(inner)
            flat: list[float] = []
            for row in data:
                f, _ = _flatten(row)
                flat.extend(f)
            return flat, (rows,) + inner_shape
        return [float(x) for x in data], (len(data),)
    raise TypeError(f"cannot build Tensor from {type(data)}")


def _infer_shape(data) -> tuple[int, ...]:
    _, shape = _flatten(data)
    return shape  # type: ignore[return-value]


class Tensor:
    """A small autograd-capable tensor backed by ``array('f')``.

    Attributes:
        data: flat ``array('f')`` storage, row-major.
        shape: tuple of ints.
        requires_grad: whether to track ops for backward().
        grad: accumulated gradient Tensor (same shape) or None.
    """

    def __init__(self, data=None, shape=None, requires_grad: bool = False,
                 _prev=(), _op: str = "", _backward=None):
        if data is None:
            data = 0.0
        if isinstance(data, Tensor):
            flat = list(data._data)
            inferred = data.shape
        elif isinstance(data, array):
            flat = [float(x) for x in data]
            inferred = shape if shape is not None else (len(flat),)
        else:
            flat, inferred = _flatten(data)
        if shape is not None:
            shape_t = tuple(shape)
            if _numel(shape_t) != len(flat):
                raise ValueError(
                    f"shape {shape_t} needs {_numel(shape_t)} elements, got {len(flat)}")
            inferred = shape_t
        self._data: array = array("f", flat)
        self.shape: tuple[int, ...] = tuple(inferred)
        self.requires_grad = bool(requires_grad)
        self.grad: Tensor | None = None
        self._prev: tuple[Tensor, ...] = tuple(_prev)
        self._op = _op
        self._backward = _backward or (lambda: None)

    # -- construction helpers -------------------------------------------------
    @classmethod
    def zeros(cls, shape, requires_grad: bool = False) -> "Tensor":
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls([0.0] * _numel(shape), shape=shape, requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, requires_grad: bool = False) -> "Tensor":
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls([1.0] * _numel(shape), shape=shape, requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape, seed: int | None = None,
              requires_grad: bool = False) -> "Tensor":
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        rng = random.Random(seed)
        return cls([rng.gauss(0.0, 1.0) for _ in range(_numel(shape))],
                   shape=shape, requires_grad=requires_grad)

    @classmethod
    def rand(cls, shape, seed: int | None = None,
             requires_grad: bool = False) -> "Tensor":
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        rng = random.Random(seed)
        return cls([rng.random() for _ in range(_numel(shape))],
                   shape=shape, requires_grad=requires_grad)

    @classmethod
    def arange(cls, n: int) -> "Tensor":
        return cls([float(i) for i in range(n)], shape=(n,))

    # -- basic properties -----------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return _numel(self.shape)

    def tolist(self) -> list:
        """Return nested Python lists matching shape (flat list for 1-d)."""
        if self.shape == ():
            return float(self._data[0])
        if len(self.shape) == 1:
            return [float(x) for x in self._data]
        if len(self.shape) == 2:
            rows, cols = self.shape
            return [[float(self._data[r * cols + c]) for c in range(cols)]
                    for r in range(rows)]
        # generic fallback: flat list
        return [float(x) for x in self._data]

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError(f"item() needs 1 element, got {self.numel}")
        return float(self._data[0])

    def detach(self) -> "Tensor":
        return Tensor(list(self._data), shape=self.shape, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def transpose(self) -> "Tensor":
        if len(self.shape) != 2:
            raise ValueError("transpose() only supports 2-d tensors on Day 1")
        rows, cols = self.shape
        out = [0.0] * (rows * cols)
        for r in range(rows):
            for c in range(cols):
                out[c * rows + r] = float(self._data[r * cols + c])
        result = Tensor(out, shape=(cols, rows),
                        requires_grad=self.requires_grad,
                        _prev=(self,), _op="transpose")

        def _bw():
            if self.requires_grad and result.grad is not None:
                g = result.grad
                # grad of transpose is transpose of grad
                grow, gcol = g.shape
                back = [0.0] * (rows * cols)
                for r in range(grow):
                    for c in range(gcol):
                        back[c * grow + r] = float(g._data[r * gcol + c])
                self._accum(Tensor(back, shape=self.shape))
        result._backward = _bw
        return result

    # -- internal helpers -----------------------------------------------------
    def _accum(self, grad: "Tensor") -> None:
        if self.grad is None:
            self.grad = grad.detach()
        else:
            assert self.grad.shape == grad.shape, \
                f"grad shape mismatch {self.grad.shape} vs {grad.shape}"
            for i in range(self.grad.numel):
                self.grad._data[i] += grad._data[i]

    def _new(self, flat, shape, parents, op, backward_fn) -> "Tensor":
        requires = any(getattr(p, "requires_grad", False) for p in parents)
        out = Tensor(flat, shape=shape, requires_grad=requires,
                     _prev=tuple(parents), _op=op)
        out._backward = backward_fn
        return out

    @staticmethod
    def _as_tensor(x) -> "Tensor":
        if isinstance(x, Tensor):
            return x
        return Tensor(x)

    def _broadcast_add(self, other: "Tensor"):
        """Return (a_flat, b_flat_expanded, out_shape) for add/sub."""
        a, b = self, other
        if a.shape == b.shape:
            return list(a._data), list(b._data), a.shape
        if b.shape == ():
            return list(a._data), [float(b._data[0])] * a.numel, a.shape
        if a.shape == ():
            return [float(a._data[0])] * b.numel, list(b._data), b.shape
        # (m, n) + (n,) row broadcast
        if len(a.shape) == 2 and len(b.shape) == 1 and a.shape[1] == b.shape[0]:
            rows, cols = a.shape
            b_exp = [float(b._data[c]) for _ in range(rows) for c in range(cols)]
            return list(a._data), b_exp, a.shape
        if len(b.shape) == 2 and len(a.shape) == 1 and b.shape[1] == a.shape[0]:
            rows, cols = b.shape
            a_exp = [float(a._data[c]) for _ in range(rows) for c in range(cols)]
            return a_exp, list(b._data), b.shape
        raise ValueError(f"cannot broadcast {a.shape} and {b.shape}")

    # -- elementwise ops ------------------------------------------------------
    def __add__(self, other):
        other = self._as_tensor(other)
        a_flat, b_flat, shape = self._broadcast_add(other)
        out_flat = [a + b for a, b in zip(a_flat, b_flat)]
        out = self._new(out_flat, shape, (self, other), "add", lambda: None)

        def _bw():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                self._accum(self._reduce_grad(g, self.shape))
            if other.requires_grad:
                other._accum(self._reduce_grad(g, other.shape))
        out._backward = _bw
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._as_tensor(other)
        a_flat, b_flat, shape = self._broadcast_add(other)
        out_flat = [a - b for a, b in zip(a_flat, b_flat)]
        out = self._new(out_flat, shape, (self, other), "sub", lambda: None)

        def _bw():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                self._accum(self._reduce_grad(g, self.shape))
            if other.requires_grad:
                # grad of -b
                neg = Tensor([-float(x) for x in self._reduce_grad(g, other.shape)._data],
                             shape=other.shape)
                other._accum(neg)
        out._backward = _bw
        return out

    def __rsub__(self, other):
        return self._as_tensor(other).__sub__(self)

    def __mul__(self, other):
        other = self._as_tensor(other)
        a_flat, b_flat, shape = self._broadcast_add(other)
        out_flat = [a * b for a, b in zip(a_flat, b_flat)]
        out = self._new(out_flat, shape, (self, other), "mul", lambda: None)

        def _bw():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                ga = Tensor([float(gv) * bv for gv, bv in zip(g._data, b_flat)],
                            shape=shape)
                self._accum(self._reduce_grad(ga, self.shape))
            if other.requires_grad:
                gb = Tensor([float(gv) * av for gv, av in zip(g._data, a_flat)],
                            shape=shape)
                other._accum(self._reduce_grad(gb, other.shape))
        out._backward = _bw
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._as_tensor(other)
        a_flat, b_flat, shape = self._broadcast_add(other)
        out_flat = [a / b for a, b in zip(a_flat, b_flat)]
        out = self._new(out_flat, shape, (self, other), "div", lambda: None)

        def _bw():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                ga = Tensor([float(gv) / bv for gv, bv in zip(g._data, b_flat)],
                            shape=shape)
                self._accum(self._reduce_grad(ga, self.shape))
            if other.requires_grad:
                gb = Tensor([-float(gv) * av / (bv * bv)
                             for gv, av, bv in zip(g._data, a_flat, b_flat)],
                            shape=shape)
                other._accum(self._reduce_grad(gb, other.shape))
        out._backward = _bw
        return out

    def __neg__(self):
        out = self._new([-float(x) for x in self._data], self.shape,
                        (self,), "neg", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                self._accum(Tensor([-float(x) for x in out.grad._data],
                                   shape=self.shape))
        out._backward = _bw
        return out

    def _reduce_grad(self, grad: "Tensor", target_shape) -> "Tensor":
        """Reduce a broadcast grad back to target_shape."""
        if grad.shape == target_shape:
            return grad.detach()
        if target_shape == ():
            return Tensor([sum(float(x) for x in grad._data)], shape=())
        if len(grad.shape) == 2 and len(target_shape) == 1:
            rows, cols = grad.shape
            summed = [0.0] * cols
            for r in range(rows):
                for c in range(cols):
                    summed[c] += float(grad._data[r * cols + c])
            return Tensor(summed, shape=target_shape)
        raise ValueError(f"cannot reduce grad {grad.shape} to {target_shape}")

    # -- matmul ---------------------------------------------------------------
    def __matmul__(self, other):
        other = self._as_tensor(other)
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul only supports 2-d tensors on Day 1")
        m, k1 = self.shape
        k2, n = other.shape
        if k1 != k2:
            raise ValueError(f"matmul shape mismatch {self.shape} @ {other.shape}")
        out_flat = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for k in range(k1):
                    s += float(self._data[i * k1 + k]) * float(other._data[k * n + j])
                out_flat[i * n + j] = s
        out = self._new(out_flat, (m, n), (self, other), "matmul", lambda: None)

        def _bw():
            if out.grad is None:
                return
            g = out.grad  # (m, n)
            if self.requires_grad:
                # grad_A = G @ B.T  (m, n) @ (n, k) -> (m, k)
                ga = [0.0] * (m * k1)
                for i in range(m):
                    for k in range(k1):
                        s = 0.0
                        for j in range(n):
                            s += float(g._data[i * n + j]) * float(other._data[k * n + j])
                        ga[i * k1 + k] = s
                self._accum(Tensor(ga, shape=self.shape))
            if other.requires_grad:
                # grad_B = A.T @ G  (k, m) @ (m, n) -> (k, n)
                gb = [0.0] * (k2 * n)
                for k in range(k2):
                    for j in range(n):
                        s = 0.0
                        for i in range(m):
                            s += float(self._data[i * k1 + k]) * float(g._data[i * n + j])
                        gb[k * n + j] = s
                other._accum(Tensor(gb, shape=other.shape))
        out._backward = _bw
        return out

    # -- activations / reductions --------------------------------------------
    def relu(self) -> "Tensor":
        out_flat = [x if x > 0.0 else 0.0 for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "relu", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) if xv > 0.0 else 0.0
                     for gv, xv in zip(out.grad._data, self._data)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        out_flat = [math.tanh(float(x)) for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "tanh", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) * (1.0 - ov * ov)
                     for gv, ov in zip(out.grad._data, out_flat)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def exp(self) -> "Tensor":
        out_flat = [math.exp(float(x)) for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "exp", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) * ov for gv, ov in zip(out.grad._data, out_flat)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def log(self) -> "Tensor":
        out_flat = [math.log(float(x)) for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "log", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) / float(xv) for gv, xv in zip(out.grad._data, self._data)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        # Numerically stable: 1/(1+exp(-x)) for x>=0, exp(x)/(1+exp(x)) for x<0.
        out_flat = []
        for x in self._data:
            x = float(x)
            if x >= 0.0:
                out_flat.append(1.0 / (1.0 + math.exp(-x)))
            else:
                ex = math.exp(x)
                out_flat.append(ex / (1.0 + ex))
        out = self._new(out_flat, self.shape, (self,), "sigmoid", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) * ov * (1.0 - ov)
                     for gv, ov in zip(out.grad._data, out_flat)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def pow(self, exponent: Number) -> "Tensor":
        """Elementwise power with a scalar exponent: ``x ** p``."""
        p = float(exponent)
        out_flat = [float(x) ** p for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "pow", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) * p * (float(xv) ** (p - 1.0))
                     for gv, xv in zip(out.grad._data, self._data)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def __pow__(self, exponent: Number) -> "Tensor":
        return self.pow(exponent)

    def max(self) -> "Tensor":
        """Scalar Tensor holding the largest element.

        Gradient flows to the first argmax position (ties: first wins).
        """
        idx = max(range(self.numel), key=lambda i: float(self._data[i]))
        out = self._new([float(self._data[idx])], (), (self,), "max", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [0.0] * self.numel
                g[idx] = float(out.grad._data[0])
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def clamp(self, min_val: Number, max_val: Number) -> "Tensor":
        """Elementwise clamp to ``[min_val, max_val]``.

        Gradient passes through only where the input was inside the
        range (0 where it was clamped).
        """
        lo, hi = float(min_val), float(max_val)
        out_flat = [min(hi, max(lo, float(x))) for x in self._data]
        out = self._new(out_flat, self.shape, (self,), "clamp", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [float(gv) if lo <= float(xv) <= hi else 0.0
                     for gv, xv in zip(out.grad._data, self._data)]
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def argmax(self, dim: int | None = None):
        """Index of the largest element (first wins on ties).

        ``dim=None`` returns a flat int. For a 2-d tensor, ``dim=-1`` (or 1)
        returns one index per row, ``dim=0`` one per column.
        No gradient — an index is not differentiable.
        """
        if dim is None:
            return max(range(self.numel), key=lambda i: float(self._data[i]))
        if len(self.shape) != 2:
            raise ValueError("argmax(dim=...) needs a 2-d tensor")
        rows, cols = self.shape
        d = dim % 2
        if d == 1:
            return [max(range(cols), key=lambda c: float(self._data[r * cols + c]))
                    for r in range(rows)]
        return [max(range(rows), key=lambda r: float(self._data[r * cols + c]))
                for c in range(cols)]

    def softmax(self, dim: int = -1) -> "Tensor":
        """Softmax along ``dim`` (default: last). Numerically stable
        (max-subtraction per slice). Gradient is the softmax Jacobian
        applied to the upstream grad, per slice:
        ``grad_i = s_i * (g_i - sum_j g_j * s_j)``.
        """
        if len(self.shape) == 1:
            slices = [list(range(self.numel))]
        elif len(self.shape) == 2:
            rows, cols = self.shape
            d = dim % 2
            if d == 1:
                slices = [list(range(r * cols, (r + 1) * cols))
                          for r in range(rows)]
            else:
                slices = [[r * cols + c for r in range(rows)]
                          for c in range(cols)]
        else:
            raise ValueError("softmax only supports 1-d / 2-d tensors")
        out_flat = [0.0] * self.numel
        for sl in slices:
            m = max(float(self._data[i]) for i in sl)
            exps = [math.exp(float(self._data[i]) - m) for i in sl]
            total = sum(exps)
            for i, e in zip(sl, exps):
                out_flat[i] = e / total
        out = self._new(out_flat, self.shape, (self,), "softmax", lambda: None)

        def _bw():
            if out.grad is None or not self.requires_grad:
                return
            g_flat = [0.0] * self.numel
            for sl in slices:
                s = [float(out._data[i]) for i in sl]
                g = [float(out.grad._data[i]) for i in sl]
                dot = sum(gi * si for gi, si in zip(g, s))
                for i, gi, si in zip(sl, g, s):
                    g_flat[i] = si * (gi - dot)
            self._accum(Tensor(g_flat, shape=self.shape))
        out._backward = _bw
        return out

    def layer_norm(self, eps: float = 1e-5) -> "Tensor":
        """LayerNorm over the last dim (per row for 2-d, whole tensor
        for 1-d): ``(x - mean) / sqrt(var + eps)`` per row.

        No learnable parameters here — the ``nn.LayerNorm`` module adds
        gamma / beta on top. Gradient is the standard formula, using
        the saved normalised values and per-row inverse std:
        ``dx = inv * (dy - mean(dy) - x_hat * mean(dy * x_hat))``.
        """
        if len(self.shape) == 1:
            slices = [(0, self.numel)]
        elif len(self.shape) == 2:
            rows, cols = self.shape
            slices = [(r * cols, (r + 1) * cols) for r in range(rows)]
        else:
            raise ValueError("layer_norm only supports 1-d / 2-d tensors")
        out_flat = [0.0] * self.numel
        x_hat = [0.0] * self.numel
        inv_std: list[float] = []
        for s, e in slices:
            n = e - s
            mean = sum(float(self._data[i]) for i in range(s, e)) / n
            var = sum((float(self._data[i]) - mean) ** 2
                      for i in range(s, e)) / n
            inv = 1.0 / math.sqrt(var + eps)
            inv_std.append(inv)
            for i in range(s, e):
                x_hat[i] = (float(self._data[i]) - mean) * inv
                out_flat[i] = x_hat[i]
        out = self._new(out_flat, self.shape, (self,), "layer_norm",
                        lambda: None)

        def _bw():
            if out.grad is None or not self.requires_grad:
                return
            g_flat = [0.0] * self.numel
            for k, (s, e) in enumerate(slices):
                n = e - s
                inv = inv_std[k]
                dy = [float(out.grad._data[i]) for i in range(s, e)]
                xh = x_hat[s:e]
                mean_dy = sum(dy) / n
                mean_dy_xh = sum(d * xx for d, xx in zip(dy, xh)) / n
                for j, i in enumerate(range(s, e)):
                    g_flat[i] = inv * (dy[j] - mean_dy - xh[j] * mean_dy_xh)
            self._accum(Tensor(g_flat, shape=self.shape))
        out._backward = _bw
        return out

    def sum(self) -> "Tensor":
        total = sum(float(x) for x in self._data)
        out = self._new([total], (), (self,), "sum", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                gv = float(out.grad._data[0])
                self._accum(Tensor([gv] * self.numel, shape=self.shape))
        out._backward = _bw
        return out

    def mean(self) -> "Tensor":
        return self.sum() / float(self.numel)

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        if _numel(shape) != self.numel:
            raise ValueError(f"reshape {self.shape} to {shape}: size mismatch")
        out = self._new(list(self._data), shape, (self,), "reshape", lambda: None)
        orig = self.shape

        def _bw():
            if out.grad is not None and self.requires_grad:
                self._accum(Tensor(list(out.grad._data), shape=orig))
        out._backward = _bw
        return out

    # -- backward ---------------------------------------------------------------
    def backward(self, grad: "Tensor | None" = None) -> None:
        if self.numel != 1 and grad is None:
            raise ValueError("backward() needs a scalar tensor or explicit grad")
        if grad is None:
            grad = Tensor([1.0], shape=())
        self.grad = grad.detach() if self.grad is None else self.grad
        # topological order (post-order DFS)
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: "Tensor"):
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node._prev:
                build(p)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            node._backward()

    # -- dunder helpers ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.tolist()}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("scalar Tensor has no len()")
        return self.shape[0]

    def __getitem__(self, idx):
        # Row indexing for 2-d, element indexing for 1-d. Returns a Tensor.
        # Preserves the autograd graph: grad flows back to the sliced row.
        if len(self.shape) == 1:
            i = idx if idx >= 0 else idx + self.shape[0]
            out = self._new([float(self._data[i])], (), (self,), "getitem",
                            lambda: None)

            def _bw1():
                if out.grad is not None and self.requires_grad:
                    g = [0.0] * self.numel
                    g[i] = float(out.grad._data[0])
                    self._accum(Tensor(g, shape=self.shape))
            out._backward = _bw1
            return out
        if len(self.shape) == 2:
            rows, cols = self.shape
            r = idx if idx >= 0 else idx + rows
            start = r * cols
            out = self._new(list(self._data[start:start + cols]), (cols,),
                            (self,), "getitem", lambda: None)

            def _bw2():
                if out.grad is not None and self.requires_grad:
                    g = [0.0] * self.numel
                    for c in range(cols):
                        g[start + c] = float(out.grad._data[c])
                    self._accum(Tensor(g, shape=self.shape))
            out._backward = _bw2
            return out
        raise ValueError("getitem only supports 1-d / 2-d on Day 1")

    def gather_rows(self, indices) -> "Tensor":
        """Gather rows by index: ``out[i] = self[indices[i]]``.

        Only 2-d. ``indices`` is a list of ints in ``[0, rows)``.
        Gradient scatter-adds into the parent rows, so a row used
        twice gets the sum of both grads (what ``nn.Embedding`` needs).
        """
        if len(self.shape) != 2:
            raise ValueError("gather_rows only supports 2-d tensors")
        rows, cols = self.shape
        idxs = [int(i) for i in indices]
        for i in idxs:
            if not 0 <= i < rows:
                raise IndexError(f"gather_rows index {i} out of range "
                                 f"for {rows} rows")
        out_flat: list[float] = []
        for i in idxs:
            start = i * cols
            out_flat.extend(float(x) for x in self._data[start:start + cols])
        out = self._new(out_flat, (len(idxs), cols), (self,),
                        "gather_rows", lambda: None)

        def _bw():
            if out.grad is not None and self.requires_grad:
                g = [0.0] * self.numel
                for r, i in enumerate(idxs):
                    for c in range(cols):
                        g[i * cols + c] += float(out.grad._data[r * cols + c])
                self._accum(Tensor(g, shape=self.shape))
        out._backward = _bw
        return out

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        """Slice a contiguous block along ``dim``: ``out`` is
        ``self`` with ``length`` entries starting at ``start``.

        1-d: ``dim`` must be 0. 2-d: ``dim`` 0 slices rows, 1 slices
        columns. Gradient scatters back into the parent (zeros
        outside the block) — what splits a Q/K/V projection into
        attention heads, and a weight matrix into per-head blocks.
        """
        if len(self.shape) == 1:
            if dim != 0:
                raise ValueError(f"narrow dim {dim} out of range for 1-d")
            n = self.shape[0]
            if not 0 <= start and start + length <= n and length >= 0:
                raise IndexError(f"narrow(0, {start}, {length}) out of "
                                 f"range for length {n}")
            out_flat = [float(x) for x in self._data[start:start + length]]
            out = self._new(out_flat, (length,), (self,), "narrow",
                            lambda: None)

            def _bw1():
                if out.grad is not None and self.requires_grad:
                    g = [0.0] * self.numel
                    for i in range(length):
                        g[start + i] = float(out.grad._data[i])
                    self._accum(Tensor(g, shape=self.shape))
            out._backward = _bw1
            return out
        if len(self.shape) == 2:
            rows, cols = self.shape
            n = rows if dim == 0 else cols if dim == 1 else -1
            if n < 0:
                raise ValueError(f"narrow dim {dim} out of range for 2-d")
            if not 0 <= start and start + length <= n and length >= 0:
                raise IndexError(f"narrow({dim}, {start}, {length}) out "
                                 f"of range for size {n}")
            if dim == 0:
                out_flat = []
                for r in range(start, start + length):
                    out_flat.extend(float(x) for x in
                                    self._data[r * cols:(r + 1) * cols])
                out_shape = (length, cols)
            else:
                out_flat = []
                for r in range(rows):
                    base = r * cols
                    out_flat.extend(float(self._data[base + c])
                                    for c in range(start, start + length))
                out_shape = (rows, length)
            out = self._new(out_flat, out_shape, (self,), "narrow",
                            lambda: None)

            def _bw2():
                if out.grad is not None and self.requires_grad:
                    g = [0.0] * self.numel
                    if dim == 0:
                        for r in range(length):
                            for c in range(cols):
                                g[(start + r) * cols + c] = float(
                                    out.grad._data[r * cols + c])
                    else:
                        for r in range(rows):
                            for c in range(length):
                                g[r * cols + start + c] = float(
                                    out.grad._data[r * length + c])
                    self._accum(Tensor(g, shape=self.shape))
            out._backward = _bw2
            return out
        raise ValueError("narrow only supports 1-d / 2-d tensors")
