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

    def state_dict(self) -> dict:
        """Plain-data snapshot of the optimizer (lr + per-param state),
        keyed by param index so it survives a save/load round-trip."""
        return {"lr": self.lr}

    def load_state_dict(self, state: dict) -> None:
        """Restore a snapshot from ``state_dict()``. The optimizer
        must already hold the same params in the same order."""
        if "lr" in state:
            self.lr = float(state["lr"])


def _indexed_state(vecs: dict, params) -> dict:
    """Per-param lists keyed by param index (``vecs`` is keyed by
    ``id(param)`` internally)."""
    return {i: list(vecs.get(id(p), [0.0] * p.numel))
            for i, p in enumerate(params)}


def _load_indexed(vecs: dict, state: dict, key: str, params) -> None:
    """Restore per-param lists from ``state[key]`` into ``vecs``."""
    saved = state.get(key, {})
    for i, values in saved.items():
        i = int(i)
        if not 0 <= i < len(params):
            raise ValueError(f"optimizer state key {key}[{i}] has no "
                             f"matching param ({len(params)} params)")
        if len(values) != params[i].numel:
            raise ValueError(f"optimizer state {key}[{i}] has "
                             f"{len(values)} values, param has "
                             f"{params[i].numel}")
        vecs[id(params[i])] = [float(x) for x in values]


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum and weight decay."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._vel: dict[int, list[float]] = {}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["vel"] = _indexed_state(self._vel, self.params)
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        _load_indexed(self._vel, state, "vel", self.params)

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

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["t"] = self.t
        state["m"] = _indexed_state(self._m, self.params)
        state["v"] = _indexed_state(self._v, self.params)
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "t" in state:
            self.t = int(state["t"])
        _load_indexed(self._m, state, "m", self.params)
        _load_indexed(self._v, state, "v", self.params)

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


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

    Unlike ``Adam(weight_decay=...)``, which adds ``wd * p`` to the
    gradient (so the decay is rescaled by Adam's adaptive step), AdamW
    decays the weights directly: ``p -= lr * wd * p`` before the Adam
    update. Same betas/eps/bias-correction as Adam.
    """

    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self._m: dict[int, list[float]] = {}
        self._v: dict[int, list[float]] = {}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["t"] = self.t
        state["m"] = _indexed_state(self._m, self.params)
        state["v"] = _indexed_state(self._v, self.params)
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "t" in state:
            self.t = int(state["t"])
        _load_indexed(self._m, state, "m", self.params)
        _load_indexed(self._v, state, "v", self.params)

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for p in self.params:
            if p.grad is None:
                continue
            # decoupled decay first: plain SGD-style shrink, NOT in the grad
            if self.weight_decay:
                for i in range(p.numel):
                    p._data[i] -= self.lr * self.weight_decay * float(p._data[i])
            m = self._m.setdefault(id(p), [0.0] * p.numel)
            v = self._v.setdefault(id(p), [0.0] * p.numel)
            for i in range(p.numel):
                gi = float(p.grad._data[i])
                m[i] = b1 * m[i] + (1.0 - b1) * gi
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2
                p._data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "StepLR", "CosineAnnealingLR",
           "clip_grad_norm_"]


# -- learning-rate schedulers -------------------------------------------------

class StepLR:
    """Multiply the optimizer lr by ``gamma`` every ``step_size`` steps."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self._count = 0

    def step(self) -> None:
        self._count += 1
        if self._count % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def get_lr(self) -> float:
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine-decay the lr from its starting value to ``eta_min`` over ``T_max`` steps."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self._t = 0

    def step(self) -> None:
        self._t += 1
        t = min(self._t, self.T_max)
        cos = math.cos(math.pi * t / self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos)

    def get_lr(self) -> float:
        return self.optimizer.lr


# -- gradient clipping ---------------------------------------------------------

def clip_grad_norm_(params, max_norm: float) -> float:
    """Scale all grads so their total L2 norm is at most ``max_norm``.

    Returns the total norm before clipping. No-op when no grads exist
    or the norm is already within budget.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = math.sqrt(sum(float(x) * float(x)
                          for g in grads for x in g._data))
    if total <= max_norm or total == 0.0:
        return total
    scale = max_norm / total
    for g in grads:
        for i in range(g.numel):
            g._data[i] *= scale
    return total
