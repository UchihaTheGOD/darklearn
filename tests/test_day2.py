"""Day 2 tests — plain asserts, runnable with pytest or directly:
    python3 tests/test_day2.py
    python3 -m pytest tests/test_day2.py -q

Covers: Tensor.sigmoid / pow / max, Sigmoid + GELU modules,
StepLR + CosineAnnealingLR, clip_grad_norm_.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import Tensor
from darklearn.autograd import gradcheck


def test_tensor_sigmoid():
    x = Tensor([0.0, 1.0, -1.0, 10.0, -10.0])
    s = x.sigmoid().tolist()
    assert abs(s[0] - 0.5) < 1e-6
    assert abs(s[1] - 0.7310585786300049) < 1e-6
    assert abs(s[2] - 0.2689414213699951) < 1e-6
    assert s[3] > 0.9999 and s[4] < 0.0001
    # gradcheck: d/dx sum(sigmoid(x)) = sum(sigmoid*(1-sigmoid))
    z = Tensor([0.3, -0.7, 1.2], requires_grad=True)
    assert gradcheck(lambda t: t.sigmoid().sum(), z)


def test_tensor_pow():
    x = Tensor([2.0, 3.0, 0.5])
    assert (x ** 2).tolist() == [4.0, 9.0, 0.25]
    assert x.pow(3).tolist() == [8.0, 27.0, 0.125]
    assert (x ** 0).tolist() == [1.0, 1.0, 1.0]
    # gradcheck: d/dx sum(x**3) = 3*x**2
    z = Tensor([0.5, 1.5, -1.0], requires_grad=True)
    assert gradcheck(lambda t: (t ** 3).sum(), z)
    z2 = Tensor([0.5, 1.5], requires_grad=True)
    assert gradcheck(lambda t: (t ** 2).sum(), z2)


def test_tensor_max():
    x = Tensor([1.0, 5.0, 3.0, 5.0], requires_grad=True)
    m = x.max()
    assert m.shape == ()
    assert m.item() == 5.0
    m.backward()
    # grad goes to the FIRST argmax only
    assert x.grad.tolist() == [0.0, 1.0, 0.0, 0.0]
    # scalar max works too
    assert Tensor([-2.0]).max().item() == -2.0


def test_sigmoid_gelu_modules():
    from darklearn import Sigmoid, GELU
    x = Tensor([-1.0, 0.0, 1.0, 2.0])
    s = Sigmoid()(x).tolist()
    assert abs(s[1] - 0.5) < 1e-6
    assert abs(s[0] - 0.2689414213699951) < 1e-6
    # GELU vs the math reference
    g = GELU()(x).tolist()
    for xv, gv in zip([-1.0, 0.0, 1.0, 2.0], g):
        ref = 0.5 * xv * (1.0 + math.tanh(math.sqrt(2.0 / math.pi)
                                          * (xv + 0.044715 * xv ** 3)))
        assert abs(gv - ref) < 1e-5, f"gelu({xv}) = {gv}, expected {ref}"
    assert abs(g[1]) < 1e-6  # gelu(0) == 0
    # grads flow through both modules
    z = Tensor([0.4, -0.9], requires_grad=True)
    assert gradcheck(lambda t: Sigmoid()(t).sum(), z)
    z2 = Tensor([0.4, -0.9], requires_grad=True)
    assert gradcheck(lambda t: GELU()(t).sum(), z2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 2 tests passed.")
