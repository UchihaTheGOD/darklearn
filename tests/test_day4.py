"""Day 4 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day4.py
    python3 -m pytest tests/test_day4.py -q

Covers: Tensor.argmax / softmax, BCELoss, AdamW, Xavier init, Dropout.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn.autograd import gradcheck


def test_tensor_argmax():
    x = Tensor([1.0, 5.0, 3.0, 5.0])
    assert x.argmax() == 1  # first max wins on ties
    assert Tensor([-2.0]).argmax() == 0
    m = Tensor([[1.0, 4.0, 2.0],
                [9.0, 0.0, 3.0]])
    assert m.argmax(dim=-1) == [1, 0]
    assert m.argmax(dim=1) == [1, 0]
    assert m.argmax(dim=0) == [1, 0, 1]
    try:
        Tensor([1.0, 2.0]).argmax(dim=0)
    except ValueError:
        pass
    else:
        raise AssertionError("argmax(dim=...) on 1-d should raise")


def test_tensor_softmax():
    x = Tensor([1.0, 2.0, 3.0])
    s = x.softmax().tolist()
    exps = [math.exp(v - 3.0) for v in (1.0, 2.0, 3.0)]
    total = sum(exps)
    for got, e in zip(s, exps):
        assert abs(got - e / total) < 1e-6
    assert abs(sum(s) - 1.0) < 1e-6
    # large values stay stable (no overflow)
    big = Tensor([1000.0, 1001.0, 999.0]).softmax().tolist()
    assert abs(sum(big) - 1.0) < 1e-6
    assert big[1] > big[0] > big[2]
    # 2-d row-wise: each row sums to 1
    m = Tensor([[1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0]])
    rows = m.softmax(dim=-1).tolist()
    assert abs(sum(rows[0]) - 1.0) < 1e-6
    assert abs(sum(rows[1]) - 1.0) < 1e-6
    assert all(abs(v - 1.0 / 3.0) < 1e-6 for v in rows[1])
    # 2-d column-wise: each column sums to 1
    cols = m.softmax(dim=0).tolist()
    assert abs(cols[0][0] + cols[1][0] - 1.0) < 1e-6
    # gradcheck on 1-d and 2-d (row-wise)
    z = Tensor([0.3, -0.7, 1.2], requires_grad=True)
    assert gradcheck(lambda t: t.softmax().sum(), z)
    z2 = Tensor([[0.3, -0.7], [1.2, 0.5]], requires_grad=True)
    assert gradcheck(lambda t: t.softmax(dim=-1).sum(), z2)
    # gradcheck on a non-trivial scalar: sum of squares of softmax
    z3 = Tensor([0.5, -1.0, 2.0], requires_grad=True)
    assert gradcheck(lambda t: (t.softmax() * t.softmax()).sum(), z3)


def test_tensor_clamp():
    x = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
    c = x.clamp(0.0, 1.0)
    assert c.tolist() == [0.0, 0.5, 1.0]
    c.sum().backward()
    # grad passes through only where unclamped
    assert x.grad.tolist() == [0.0, 1.0, 0.0]


def test_bce_loss():
    from darklearn import BCELoss, Sigmoid
    loss_fn = BCELoss()
    # hand-computed: p=0.5, y=1 -> -log(0.5); p=0.5, y=0 -> -log(0.5)
    p = Tensor([0.5, 0.5])
    y = Tensor([1.0, 0.0])
    assert abs(loss_fn(p, y).item() - 0.6931471805599453) < 1e-6
    # confident correct prediction -> small loss; confident wrong -> large
    assert loss_fn(Tensor([0.9]), Tensor([1.0])).item() < 0.11
    assert loss_fn(Tensor([0.9]), Tensor([0.0])).item() > 2.3
    # extreme probabilities never produce inf/nan (clamped)
    extreme = loss_fn(Tensor([0.0, 1.0]), Tensor([1.0, 0.0])).item()
    assert math.isfinite(extreme) and extreme > 10.0
    # grad flows: d/dp of BCE at p=0.5, y=1 is -1/p = -2 (mean over 2 items: -1)
    z = Tensor([0.5, 0.5], requires_grad=True)
    loss_fn(z, y).backward()
    assert abs(z.grad.tolist()[0] - (-1.0)) < 1e-5
    assert abs(z.grad.tolist()[1] - 1.0) < 1e-5
    # end-to-end: sigmoid outputs + BCE trains a tiny classifier down
    from darklearn import Sequential, Linear, Adam
    import random
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(32)]
    ys = [1.0 if x[0] + x[1] > 0 else 0.0 for x in xs]
    model = Sequential(Linear(2, 4, seed=1), Sigmoid(), Linear(4, 1, seed=2),
                       Sigmoid())
    opt = Adam(model.parameters(), lr=0.05)
    first = None
    for epoch in range(30):
        opt.zero_grad()
        pred = model(Tensor(xs))
        loss = loss_fn(pred.reshape(32), Tensor(ys))
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first, f"BCE loss did not decrease: {first} -> {loss.item()}"
    # shape mismatch raises
    try:
        loss_fn(Tensor([0.5]), Tensor([1.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("BCELoss shape mismatch should raise")


def test_adamw():
    from darklearn import Adam, AdamW
    # one hand-computed step: p=1, grad=1, lr=0.1, wd=0.1
    p = Tensor([1.0], requires_grad=True)
    p.grad = Tensor([1.0])
    AdamW([p], lr=0.1, weight_decay=0.1).step()
    # decay first: 1 - 0.1*0.1*1 = 0.99, then Adam step ~0.1 -> 0.89
    assert abs(p.tolist()[0] - 0.89) < 1e-6, p.tolist()
    # Adam with L2 in the grad gives a DIFFERENT answer (0.9): decay is
    # decoupled in AdamW, not rescaled by the adaptive step
    q = Tensor([1.0], requires_grad=True)
    q.grad = Tensor([1.0])
    Adam([q], lr=0.1, weight_decay=0.1).step()
    assert abs(q.tolist()[0] - 0.9) < 1e-6, q.tolist()
    assert abs(p.tolist()[0] - q.tolist()[0]) > 1e-3
    # wd=0: AdamW is exactly Adam
    r = Tensor([2.0, -1.0], requires_grad=True)
    r.grad = Tensor([0.5, -0.25])
    s = Tensor([2.0, -1.0], requires_grad=True)
    s.grad = Tensor([0.5, -0.25])
    AdamW([r], lr=0.01, weight_decay=0.0).step()
    Adam([s], lr=0.01).step()
    assert r.tolist() == s.tolist()
    # AdamW trains the MLP loss down
    from darklearn import Sequential, Linear, ReLU, MSELoss
    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    xs = Tensor([[0.5, -0.5], [1.0, 0.0], [-1.0, 1.0], [0.0, 0.5]])
    ys = Tensor([[x[0] + 2.0 * x[1]] for x in xs.tolist()])
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = MSELoss()(model(xs), ys)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first, f"AdamW loss did not decrease: {first} -> {loss.item()}"


def test_xavier_init():
    from darklearn import Linear
    # Xavier std is sqrt(2/(fan_in+fan_out)): 100->100 gives 0.1;
    # He std is sqrt(2/fan_in): 100->100 gives ~0.141. With 10k samples
    # the measured std should sit close to the target.
    for init, target in (("xavier", 0.1), ("glorot", 0.1), ("he", 0.1414)):
        layer = Linear(100, 100, seed=7, init=init)
        vals = layer.weight.tolist()
        flat = [v for row in vals for v in row]
        mean = sum(flat) / len(flat)
        var = sum((v - mean) ** 2 for v in flat) / len(flat)
        std = math.sqrt(var)
        assert abs(std - target) < 0.02, f"{init}: std {std}, expected ~{target}"
        assert abs(mean) < 0.02
    # default is still He
    default = Linear(100, 100, seed=7)
    assert default.init == "he"
    assert default.weight.tolist() == Linear(100, 100, seed=7).weight.tolist()
    # unknown init raises
    try:
        Linear(4, 4, init="orthogonal")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown init should raise ValueError")


def test_dropout():
    from darklearn import Dropout, Sequential, Linear, ReLU
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    # eval mode: identity, exactly
    d = Dropout(0.5, seed=0)
    d.eval()
    assert d(x).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert d.training is False
    # train mode: outputs are 0 or x/(1-p) = 2x, never anything else
    d.train()
    assert d.training is True
    out = d(x).tolist()
    assert all(v == 0.0 or abs(v - 2.0 * xv) < 1e-6
               for v, xv in zip(out, [1.0, 2.0, 3.0, 4.0]))
    assert any(v == 0.0 for v in out)  # some dropped with this seed
    assert any(v != 0.0 for v in out)  # some kept
    # grad flows: 0 where dropped, 2x-scale where kept
    z = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    d2 = Dropout(0.5, seed=0)
    d2(z).sum().backward()
    assert z.grad.tolist() == [0.0 if v == 0.0 else 2.0 for v in out]
    # p=0 is identity even in train mode
    assert Dropout(0.0)(x).tolist() == [1.0, 2.0, 3.0, 4.0]
    # keep rate is ~1-p over many elements
    big = Tensor([1.0] * 2000)
    kept = sum(1 for v in Dropout(0.5, seed=1)(big).tolist() if v != 0.0)
    assert 900 < kept < 1100, f"kept {kept}/2000, expected ~1000"
    # train()/eval() recurse into children (Sequential -> Dropout)
    model = Sequential(Linear(2, 4, seed=1), ReLU(), Dropout(0.5, seed=2),
                       Linear(4, 1, seed=3))
    model.eval()
    assert model.training is False
    assert model.layers[2].training is False
    assert model.layers[0].training is False
    model.train()
    assert model.layers[2].training is True
    # bad p raises
    try:
        Dropout(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Dropout(1.0) should raise ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 4 tests passed.")
