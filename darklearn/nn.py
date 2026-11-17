"""darklearn.nn — modules, layers and losses (pure stdlib)."""

from __future__ import annotations

import math

from .tensor import Tensor


class Module:
    """Base class for all layers / models."""

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for v in vars(self).values():
            if isinstance(v, Tensor):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor):
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class Linear(Module):
    """Fully-connected layer: y = x @ W.T + b. x is (batch, in_features)."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = math.sqrt(2.0 / max(1, in_features))
        w = Tensor.randn((out_features, in_features), seed=seed,
                         requires_grad=True)
        # scale in place (keeps requires_grad)
        for i in range(w.numel):
            w._data[i] *= scale
        self.weight = w
        self.bias = Tensor.zeros((out_features,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x (batch, in) @ W.T (in, out) -> (batch, out), then + bias (out,)
        return x @ self.weight.T + self.bias

    def __repr__(self):
        return f"Linear({self.in_features} -> {self.out_features})"


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        inner = ", ".join(repr(l) for l in self.layers)
        return f"Sequential({inner})"


class MSELoss(Module):
    """Mean squared error: mean((pred - target) ** 2)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(pred, target)


class CrossEntropyLoss(Module):
    """Mean negative log-likelihood over logits (batch, classes).

    Targets are a list of class indices. Implemented from Tensor ops
    (max-subtraction, exp, sum, log) so autograd flows through it.
    """

    def forward(self, logits: Tensor, targets) -> Tensor:
        if len(logits.shape) != 2:
            raise ValueError("CrossEntropyLoss needs 2-d logits")
        batch, classes = logits.shape
        if len(targets) != batch:
            raise ValueError("targets length must match batch size")
        total = None
        for i in range(batch):
            row = logits[i]  # (classes,)
            # stable log-softmax for one row, built from Tensor ops
            m = max(float(x) for x in row._data)
            shifted = row - Tensor([m], shape=())
            exp_row = shifted.exp()
            log_sum = exp_row.sum().log()
            log_prob = shifted - log_sum
            # pick target class: log_prob[target] as a scalar Tensor
            # Build via sum with a one-hot mask (no grad through mask).
            mask = [0.0] * classes
            mask[int(targets[i])] = 1.0
            picked = (log_prob * Tensor(mask, shape=(classes,))).sum()
            nll = -picked
            total = nll if total is None else total + nll
        return total / float(batch)


__all__ = ["Module", "Linear", "ReLU", "Tanh", "Sequential",
           "MSELoss", "mse_loss", "CrossEntropyLoss"]
