"""Day 1 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day1.py
    python3 -m pytest tests/test_day1.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (Tensor, Linear, Sequential, ReLU, SGD, Adam,
                       MSELoss, TensorDataset, DataLoader)
from darklearn.autograd import gradcheck, Value


def test_tensor_add_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    assert (a + b).tolist() == [[6.0, 8.0], [10.0, 12.0]]
    assert (a * 2.0).tolist() == [[2.0, 4.0], [6.0, 8.0]]
    c = a @ Tensor([[1.0, 0.0], [0.0, 1.0]])
    assert c.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    d = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) @ Tensor([[1.0], [1.0], [1.0]])
    assert d.tolist() == [[6.0], [15.0]]
    assert Tensor([1.0, 2.0, 3.0]).sum().item() == 6.0
    assert abs(Tensor([1.0, 2.0, 3.0]).mean().item() - 2.0) < 1e-6
    assert Tensor([-1.0, 2.0]).relu().tolist() == [0.0, 2.0]


def test_autograd_grad_check():
    # d/dx sum((x * 2)^2) at x=[1,2] is [8, 16]
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = ((x * 2.0) * (x * 2.0)).sum()
    y.backward()
    assert x.grad is not None
    assert abs(x.grad.tolist()[0] - 8.0) < 1e-5
    assert abs(x.grad.tolist()[1] - 16.0) < 1e-5
    # numeric gradcheck on a small function
    z = Tensor([0.5, -1.0], requires_grad=True)
    assert gradcheck(lambda t: (t * t).sum() + t.sum(), z)
    # scalar Value node
    a = Value(2.0)
    b = Value(3.0)
    c = (a * b).relu()
    c.backward()
    assert abs(a.grad - 3.0) < 1e-9 and abs(b.grad - 2.0) < 1e-9


def test_autograd_matmul_grad():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    y = (a @ w).sum()
    y.backward()
    assert a.grad.tolist() == [[3.0, 4.0]]
    assert w.grad.tolist() == [[1.0], [2.0]]


def test_linear_forward():
    lin = Linear(2, 3, seed=0)
    x = Tensor([[1.0, 0.0], [0.0, 1.0]])
    out = lin(x)
    assert out.shape == (2, 3)
    assert len(lin.parameters()) == 2  # weight + bias


def test_sgd_step_reduces_loss():
    import random
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(32)]
    ys = [x[0] + 2 * x[1] for x in xs]
    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()
    xt = Tensor(xs, shape=(32, 2))
    yt = Tensor(ys, shape=(32, 1))
    first = loss_fn(model(xt), yt).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    last = loss_fn(model(xt), yt).item()
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert last < 0.5, f"loss still high: {last}"


def test_dataloader_batching():
    ds = TensorDataset([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
                       [1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(ds) == 5
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (2, 2)
    assert batches[0][1].shape == (2, 1)
    assert batches[-1][0].shape == (1, 2)
    # seeded shuffle is reproducible
    l1 = [b.tolist() for _, b in DataLoader(ds, batch_size=2, shuffle=True, seed=7)]
    l2 = [b.tolist() for _, b in DataLoader(ds, batch_size=2, shuffle=True, seed=7)]
    assert l1 == l2


def test_cross_entropy_runs():
    from darklearn import CrossEntropyLoss
    logits = Tensor([[2.0, 0.5], [0.1, 1.5]], requires_grad=True)
    loss = CrossEntropyLoss()(logits, [0, 1])
    assert 0.0 < loss.item() < 2.0
    loss.backward()
    assert logits.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 1 tests passed.")
