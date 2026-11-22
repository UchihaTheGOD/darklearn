"""darklearn.nn — modules, layers and losses (pure stdlib)."""

from __future__ import annotations

import math
import random

from .tensor import Tensor


class Module:
    """Base class for all layers / models.

    ``training`` is True by default; ``eval()`` switches a module (and
    all its children) to inference mode, ``train()`` switches back.
    Layers like Dropout behave differently in the two modes.
    """

    def __init__(self):
        self.training = True

    def train(self, mode: bool = True) -> "Module":
        """Set training mode (recursively) and return self."""
        self.training = bool(mode)
        for child in self._children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        """Switch to evaluation mode (recursively) and return self."""
        return self.train(False)

    def _children(self) -> list["Module"]:
        children: list[Module] = []
        for v in vars(self).values():
            if isinstance(v, Module):
                children.append(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        children.append(item)
        return children

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
    """Fully-connected layer: y = x @ W.T + b. x is (batch, in_features).

    ``init`` picks the weight scale: ``"he"`` (default, std sqrt(2/fan_in),
    good with ReLU) or ``"xavier"`` / ``"glorot"`` (std sqrt(2/(fan_in +
    fan_out)), good with tanh/sigmoid).
    """

    def __init__(self, in_features: int, out_features: int,
                 seed: int | None = None, init: str = "he"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if init == "he":
            scale = math.sqrt(2.0 / max(1, in_features))
        elif init in ("xavier", "glorot"):
            scale = math.sqrt(2.0 / max(1, in_features + out_features))
        else:
            raise ValueError(f"unknown init {init!r}: use 'he' or 'xavier'")
        self.init = init
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


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class GELU(Module):
    """Gaussian Error Linear Unit (exact tanh approximation).

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Built from Tensor ops, so autograd flows through it.
    """

    def forward(self, x: Tensor) -> Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + (c * (x + 0.044715 * (x ** 3))).tanh())


class Dropout(Module):
    """Randomly zero elements with probability ``p`` during training
    (inverted dropout: survivors are scaled by ``1/(1-p)`` so the
    expected output is unchanged). In eval mode it is the identity.

    The mask is a plain constant Tensor, so autograd flows through the
    surviving elements (scaled) and is zero where dropped.
    """

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = float(p)
        self._rng = random.Random(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        scale = 1.0 / keep
        mask = [scale if self._rng.random() < keep else 0.0
                for _ in range(x.numel)]
        return x * Tensor(mask, shape=x.shape)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Embedding(Module):
    """Token embedding: a lookup table of ``num_embeddings`` rows.

    ``forward(ids)`` takes one token id (int) or a flat list of ids and
    returns a Tensor of shape ``(len(ids), embedding_dim)`` — one row
    per token. Gradients flow back to the rows actually used, and a
    token used twice in one batch gets the sum of both grads.
    Weights are N(0, 1/sqrt(embedding_dim)) at init.

    Note: darklearn Tensors are 1-d / 2-d, so a (batch, seq) batch is
    passed in flattened and comes back as (batch*seq, embedding_dim);
    reshape it on the caller side once 3-d support lands.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 seed: int | None = None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding sizes must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = Tensor.randn((num_embeddings, embedding_dim), seed=seed,
                         requires_grad=True)
        scale = 1.0 / math.sqrt(embedding_dim)
        for i in range(w.numel):
            w._data[i] *= scale
        self.weight = w

    def forward(self, ids) -> Tensor:
        if isinstance(ids, int):
            ids = [ids]
        ids = [int(i) for i in ids]
        return self.weight.gather_rows(ids)

    def __repr__(self):
        return (f"Embedding({self.num_embeddings}, "
                f"{self.embedding_dim})")


class LayerNorm(Module):
    """Layer normalisation over the last dim of a 2-d input (per row).

    ``y = gamma * (x - mean) / sqrt(var + eps) + beta``, with
    learnable per-feature ``gamma`` (init 1) and ``beta`` (init 0).
    Built on ``Tensor.layer_norm`` plus the existing broadcast ops,
    so autograd flows to x, gamma and beta.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("LayerNorm normalized_shape must be positive")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.gamma = Tensor.ones((normalized_shape,), requires_grad=True)
        self.beta = Tensor.zeros((normalized_shape,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 2 or x.shape[1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm({self.normalized_shape}) needs 2-d input with "
                f"{self.normalized_shape} features, got shape {x.shape}")
        return x.layer_norm(self.eps) * self.gamma + self.beta

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> Tensor:
    """Sinusoidal positional encoding (Vaswani et al., 2017), no params.

    ``PE[pos, 2i] = sin(pos / 10000^(2i/d_model))`` and
    ``PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))``.
    Returns a (seq_len, d_model) Tensor; add it to token embeddings
    (``emb + sinusoidal_positional_encoding(len, d)``).
    """
    if seq_len <= 0 or d_model <= 0:
        raise ValueError("sinusoidal_positional_encoding sizes must be positive")
    flat: list[float] = []
    for pos in range(seq_len):
        for i in range(d_model):
            denom = 10000.0 ** (2 * (i // 2) / d_model)
            angle = pos / denom
            flat.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
    return Tensor(flat, shape=(seq_len, d_model))


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


class BCELoss(Module):
    """Binary cross-entropy, mean over all elements:
    ``-(y*log(p) + (1-y)*log(1-p))``.

    ``pred`` holds probabilities in [0, 1] (typically a sigmoid output),
    ``target`` holds 0/1 labels of the same shape. Probabilities are
    clamped to [1e-7, 1-1e-7] so log(0) can never produce -inf.
    Built from Tensor ops, so autograd flows through it.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"BCELoss needs matching shapes, "
                             f"got {pred.shape} vs {target.shape}")
        eps = 1e-7
        p = pred.clamp(eps, 1.0 - eps)
        one = Tensor(1.0, shape=())
        loss = -(target * p.log() + (one - target) * (one - p).log())
        return loss.mean()


__all__ = ["Module", "Linear", "ReLU", "Tanh", "Sigmoid", "GELU", "Dropout",
           "Embedding", "LayerNorm", "sinusoidal_positional_encoding",
           "Sequential", "MSELoss", "mse_loss", "CrossEntropyLoss", "BCELoss"]
