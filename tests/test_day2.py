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


def test_lr_schedulers():
    from darklearn import SGD, StepLR, CosineAnnealingLR
    p = Tensor([1.0], requires_grad=True)
    opt = SGD([p], lr=0.1)
    sched = StepLR(opt, step_size=2, gamma=0.1)
    assert sched.get_lr() == 0.1
    sched.step()
    assert sched.get_lr() == 0.1
    sched.step()
    assert abs(sched.get_lr() - 0.01) < 1e-12
    sched.step()
    sched.step()
    assert abs(sched.get_lr() - 0.001) < 1e-12

    opt2 = SGD([p], lr=0.1)
    cos = CosineAnnealingLR(opt2, T_max=4, eta_min=0.0)
    cos.step()
    assert abs(cos.get_lr() - 0.08535533905932738) < 1e-9
    cos.step()
    assert abs(cos.get_lr() - 0.05) < 1e-9
    cos.step()
    cos.step()
    assert abs(cos.get_lr() - 0.0) < 1e-12
    cos.step()  # stays at eta_min past T_max
    assert abs(cos.get_lr() - 0.0) < 1e-12


def test_clip_grad_norm():
    from darklearn import clip_grad_norm_
    a = Tensor([3.0, 4.0], requires_grad=True)
    b = Tensor([0.0], requires_grad=True)
    # fake grads with total norm 5.0
    a.grad = Tensor([3.0, 4.0])
    b.grad = Tensor([0.0])
    total = clip_grad_norm_([a, b], max_norm=2.5)
    assert abs(total - 5.0) < 1e-9
    assert abs(a.grad.tolist()[0] - 1.5) < 1e-6
    assert abs(a.grad.tolist()[1] - 2.0) < 1e-6
    # already within budget: untouched
    total2 = clip_grad_norm_([a, b], max_norm=10.0)
    assert abs(total2 - 2.5) < 1e-9
    assert abs(a.grad.tolist()[0] - 1.5) < 1e-6
    # no grads at all: returns 0.0, no crash
    c = Tensor([1.0], requires_grad=True)
    assert clip_grad_norm_([c], max_norm=1.0) == 0.0


def test_trainer_lite():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import trainer_lite
    from darklearn import (Sequential, Linear, ReLU, Adam, MSELoss,
                           TensorDataset, DataLoader, StepLR)
    import random
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(32)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    loader = DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=True, seed=0)
    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = Adam(model.parameters(), lr=0.05)
    sched = StepLR(opt, step_size=10, gamma=0.5)
    trainer = trainer_lite.Trainer(model, MSELoss(), opt, scheduler=sched,
                                   max_grad_norm=1.0)
    history = trainer.fit(loader, epochs=5, log_every=100)
    assert len(history) == 5
    assert history[-1] < history[0], f"loss did not decrease: {history}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All Day 2 tests passed.")
