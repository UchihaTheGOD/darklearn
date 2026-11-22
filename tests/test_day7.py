"""Day 7 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day7.py
    python3 -m pytest tests/test_day7.py -q

Covers: Embedding (lookup + grads), LayerNorm (forward, gradcheck,
gamma/beta), sinusoidal positional encoding, and a tiny end-to-end
embedding training run.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, Embedding, LayerNorm, Linear, MSELoss,
    sinusoidal_positional_encoding,
)
from darklearn.autograd import gradcheck
from darklearn.tensor import Tensor


def _close(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_embedding_forward():
    emb = Embedding(5, 4, seed=7)
    out = emb([0, 2, 2, 4])
    assert out.shape == (4, 4)
    # Each output row is the matching weight row.
    for row, tok in enumerate([0, 2, 2, 4]):
        assert out.tolist()[row] == emb.weight.tolist()[tok]
    # A single int id gives one row.
    assert emb(3).shape == (1, 4)
    assert emb(3).tolist()[0] == emb.weight.tolist()[3]
    # Out-of-range ids are rejected.
    try:
        emb([5])
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for id 5")
    try:
        emb([-1])
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for id -1")


def test_embedding_grad():
    emb = Embedding(4, 3, seed=1)
    out = emb([1, 2, 2, 3])
    out.sum().backward()
    assert emb.weight.grad is not None
    g = emb.weight.grad.tolist()
    # Rows used once get grad 1.0 per element; row 2 was used twice.
    assert all(_close(v, 1.0) for v in g[1])
    assert all(_close(v, 2.0) for v in g[2])
    assert all(_close(v, 1.0) for v in g[3])
    # Unused row gets no grad at all.
    assert all(v == 0.0 for v in g[0])


def test_layernorm_forward():
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]],
               requires_grad=True)
    y = ln(x)
    assert y.shape == (2, 4)
    rows = y.tolist()
    # Row 0 has mean ~0 and variance ~1 (gamma=1, beta=0 by default).
    mean = sum(rows[0]) / len(rows[0])
    var = sum((v - mean) ** 2 for v in rows[0]) / len(rows[0])
    assert _close(mean, 0.0, tol=1e-5)
    assert _close(var, 1.0, tol=1e-4)
    # A constant row normalises to (numerically) zero, not unit variance.
    assert all(_close(v, 0.0, tol=1e-5) for v in rows[1])


def test_layernorm_gradcheck():
    # gradcheck only supports 1-d inputs: check Tensor.layer_norm on one row.
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    weights = Tensor([1.0, 2.0, 3.0, 4.0])
    # Weighted sum of the normalised row: non-degenerate grads.
    assert gradcheck(lambda t: (t.layer_norm() * weights).sum(), x)


def test_layernorm_gamma_beta():
    ln = LayerNorm(3)
    for i in range(3):
        ln.gamma._data[i] = 2.0
        ln.beta._data[i] = 0.5
    x = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
    y = ln(x)
    x_hat = x.layer_norm().tolist()
    for r in range(2):
        for c in range(3):
            assert _close(y.tolist()[r][c], 2.0 * x_hat[r][c] + 0.5)
    y.sum().backward()
    # dL/dbeta = ones summed over rows; dL/dgamma = x_hat summed over rows.
    assert ln.beta.grad is not None
    assert all(_close(v, 2.0) for v in ln.beta.grad.tolist())
    assert ln.gamma.grad is not None
    expect = [sum(x_hat[r][c] for r in range(2)) for c in range(3)]
    for got, want in zip(ln.gamma.grad.tolist(), expect):
        assert _close(got, want)
    # Input grad: rows are identical up to a shift, so grads match.
    assert x.grad is not None
    assert x.grad.tolist()[0] == x.grad.tolist()[1]


def test_positional_encoding():
    pe = sinusoidal_positional_encoding(4, 6)
    assert pe.shape == (4, 6)
    rows = pe.tolist()
    # Position 0: sin(0)=0, cos(0)=1 alternating.
    assert rows[0] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    # Position 1, first pair: sin(1), cos(1).
    assert _close(rows[1][0], math.sin(1.0))
    assert _close(rows[1][1], math.cos(1.0))
    # Position 2, first pair: sin(2), cos(2).
    assert _close(rows[2][0], math.sin(2.0))
    assert _close(rows[2][1], math.cos(2.0))
    # Higher pairs use a larger denominator (slower oscillation).
    assert rows[1][2] != rows[1][0]
    # Deterministic: two calls give the same table.
    pe2 = sinusoidal_positional_encoding(4, 6)
    assert pe2.tolist() == rows
    # Bad sizes are rejected.
    try:
        sinusoidal_positional_encoding(0, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for seq_len 0")


def test_embedding_trains():
    # Toy task: learn an embedding + linear head that predicts the
    # parity of each token id (even -> 1, odd -> 0).
    emb = Embedding(4, 4, seed=3)
    head = Linear(4, 1, seed=4)
    opt = Adam(list(emb.parameters()) + list(head.parameters()), lr=0.05)
    loss_fn = MSELoss()
    ids = [0, 1, 2, 3]
    target = Tensor([[1.0], [0.0], [1.0], [0.0]])
    first = None
    for step in range(150):
        opt.zero_grad()
        pred = head(emb(ids))
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert first is not None
    assert loss.item() < first * 0.5
    assert loss.item() < 0.05
    # Embeddings for same-parity tokens end up closer than cross-parity.
    rows = emb.weight.tolist()
    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    assert dist(rows[0], rows[2]) < dist(rows[0], rows[1])


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 7 tests passed.")
