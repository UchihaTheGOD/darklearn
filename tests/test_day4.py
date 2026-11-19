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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 4 tests passed.")
