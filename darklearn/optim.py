"""darklearn.optim — SGD (with momentum) and Adam (pure stdlib)."""

from __future__ import annotations

import math

from .tensor import Tensor


class Optimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self):  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum and weight decay."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._vel: dict[int, list[float]] = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = [float(x) for x in p.grad._data]
            if self.weight_decay:
                g = [gi + self.weight_decay * float(pi)
                     for gi, pi in zip(g, p._data)]
            if self.momentum:
                v = self._vel.get(id(p))
                if v is None:
                    v = [0.0] * len(g)
                    self._vel[id(p)] = v
                v[:] = [self.momentum * vi + gi for vi, gi in zip(v, g)]
                update = v
            else:
                update = g
            for i in range(p.numel):
                p._data[i] -= self.lr * update[i]


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2014)."""

    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self._m: dict[int, list[float]] = {}
        self._v: dict[int, list[float]] = {}

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for p in self.params:
            if p.grad is None:
                continue
            m = self._m.setdefault(id(p), [0.0] * p.numel)
            v = self._v.setdefault(id(p), [0.0] * p.numel)
            for i in range(p.numel):
                gi = float(p.grad._data[i])
                if self.weight_decay:
                    gi += self.weight_decay * float(p._data[i])
                m[i] = b1 * m[i] + (1.0 - b1) * gi
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2
                p._data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


__all__ = ["Optimizer", "SGD", "Adam"]
