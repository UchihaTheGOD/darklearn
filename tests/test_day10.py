"""Day 10 tests — Month 5: Trainer, checkpoints, grad accumulation,
early stopping (stdlib only, plain `python3` runner)."""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, CrossEntropyLoss, DataLoader, Linear, MSELoss, ReLU, SGD,
    Sequential, Tensor, TensorDataset,
)
from darklearn.autograd import gradcheck
from darklearn.checkpoint import (
    load_checkpoint, load_state_dict, save_checkpoint, state_dict,
)
from darklearn.nn import Module
from darklearn.trainer import EarlyStopping, Trainer


def test_tensor_cat():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0]])
    c = Tensor.cat([a, b], dim=0)
    assert c.shape == (3, 2)
    assert c.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    d = Tensor.cat([a, b.T], dim=1)
    assert d.shape == (2, 3)
    assert d.tolist() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]
    e = Tensor.cat([Tensor([1.0, 2.0]), Tensor([3.0])])
    assert e.shape == (3,) and e.tolist() == [1.0, 2.0, 3.0]
    # errors
    for bad in (lambda: Tensor.cat([]),
                lambda: Tensor.cat([a, Tensor([1.0])]),
                lambda: Tensor.cat([a, Tensor([[1.0, 2.0, 3.0]])], dim=0),
                lambda: Tensor.cat([a, Tensor([[1.0]])], dim=1),
                lambda: Tensor.cat([a], dim=2)):
        try:
            bad()
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("cat should have raised")


def test_tensor_cat_grad():
    # gradcheck the cat op end-to-end via a small function.
    def f(x):
        a = x.reshape((2, 2))
        b = Tensor([[1.0, 1.0]], requires_grad=False)
        return Tensor.cat([a, b], dim=0).sum()

    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    assert gradcheck(f, x), "cat gradcheck failed"
    # backward slices grads back to each input
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)
    c = Tensor.cat([a, b], dim=1)
    c.sum().backward()
    assert a.grad.tolist() == [[1.0, 1.0]]
    assert b.grad.tolist() == [[1.0, 1.0]]


def _mlp():
    return Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))


def _mlp_data(n=64, seed=0):
    import random
    rng = random.Random(seed)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    return TensorDataset(xs, ys)


def test_named_parameters():
    class TwoLayer(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(2, 3, seed=1)
            self.layers = [Linear(3, 1, seed=2)]

        def forward(self, x):
            return self.layers[0](self.fc1(x))

    m = TwoLayer()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["fc1.weight", "fc1.bias",
                     "layers.0.weight", "layers.0.bias"], names
    sd = state_dict(m)
    assert set(sd) == set(names)
    assert sd["fc1.weight"]["shape"] == (3, 2)
    assert len(sd["fc1.weight"]["data"]) == 6


def test_trainer_fit_and_eval():
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, MSELoss(), opt)
    history = trainer.fit(loader, epochs=10, log_every=10)
    assert len(history["train_loss"]) == 10
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_loss"][-1] < 0.05
    val = trainer.evaluate(loader)
    assert abs(val - history["train_loss"][-1]) < 0.05
    # perplexity is exp(mean loss) for a log-loss; here it is just
    # exp of the eval loss, so it must match exactly.
    assert abs(trainer.perplexity(loader) - math.exp(val)) < 1e-6
    # eval mode is restored to train mode after evaluate()
    assert model.training is True


def test_trainer_grad_accumulation():
    class CountingAdam(Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = 0

        def step(self):
            self.steps += 1
            super().step()

    model = _mlp()
    loader = DataLoader(_mlp_data(32), batch_size=8)  # 4 batches/epoch
    opt = CountingAdam(model.parameters(), lr=0.05)
    trainer = Trainer(model, MSELoss(), opt, accum_steps=2)
    trainer.fit(loader, epochs=2, log_every=99)
    # 4 micro-batches per epoch / 2 = 2 optimizer steps per epoch.
    assert opt.steps == 4, f"expected 4 optimizer steps, got {opt.steps}"
    # loss still goes down with accumulation
    model2 = _mlp()
    loader2 = DataLoader(_mlp_data(32), batch_size=8)
    opt2 = Adam(model2.parameters(), lr=0.05)
    t2 = Trainer(model2, MSELoss(), opt2, accum_steps=2)
    h = t2.fit(loader2, epochs=5, log_every=99)
    assert h["train_loss"][-1] < h["train_loss"][0]


def test_trainer_early_stopping():
    # Val loss that never improves after epoch 1: stop after patience.
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.0)  # frozen weights: no learning
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    trainer = Trainer(model, MSELoss(), opt, early_stopping=stopper)
    history = trainer.fit(loader, epochs=20, val_loader=loader, log_every=99)
    assert len(history["train_loss"]) == 3, \
        f"expected stop after 3 epochs, got {len(history['train_loss'])}"
    assert stopper.stopped is True
    assert stopper.best_value is not None
    # restore_best_weights keeps the best-seen weights
    best = {k: list(v["data"]) for k, v in state_dict(model).items()}
    stopper.restore_best_weights(model)
    after = {k: list(v["data"]) for k, v in state_dict(model).items()}
    assert after == best


def test_checkpoint_save_load():
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, MSELoss(), opt)
    trainer.fit(loader, epochs=3, log_every=99)
    before = state_dict(model)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, model, optimizer=opt, epoch=3,
                        extra={"note": "day10"})
        # perturb the weights, then reload: outputs must match again
        for p in model.parameters():
            for i in range(p.numel):
                p._data[i] = 0.0
        meta = load_checkpoint(path, model=model, optimizer=opt)
        assert meta["epoch"] == 3
        assert meta["extra"] == {"note": "day10"}
        after = state_dict(model)
        for k in before:
            assert before[k]["shape"] == after[k]["shape"]
            assert before[k]["data"] == after[k]["data"], k
        # optimizer state round-trips: Adam step count restored
        assert opt.t == 3 * len(loader), f"Adam t={opt.t}"
        assert len(opt._m) == len(model.parameters())
    # errors: missing file, bad shape
    try:
        load_checkpoint("/nonexistent/ckpt.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint should raise")
    model2 = _mlp()
    bad = {"fc.weight": {"shape": (5, 5), "data": [0.0] * 25}}
    try:
        load_state_dict(model2, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown key should raise")


def test_checkpoint_resume_matches():
    # Training 6 epochs straight == 3 epochs, save, reload, 3 more.
    def run(resume_from=None):
        model = _mlp()
        loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
        opt = Adam(model.parameters(), lr=0.05)
        if resume_from is not None:
            load_checkpoint(resume_from, model=model, optimizer=opt)
            start = 3
        else:
            start = 0
        trainer = Trainer(model, MSELoss(), opt)
        h = trainer.fit(loader, epochs=6 - start, log_every=99)
        return h["train_loss"][-1], state_dict(model)

    loss_full, sd_full = run()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        model = _mlp()
        loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
        opt = Adam(model.parameters(), lr=0.05)
        Trainer(model, MSELoss(), opt).fit(loader, epochs=3, log_every=99)
        save_checkpoint(path, model, optimizer=opt, epoch=3)
        loss_resumed, sd_resumed = run(resume_from=path)
    assert abs(loss_full - loss_resumed) < 1e-6, \
        f"resumed loss {loss_resumed} != full {loss_full}"
    for k in sd_full:
        assert sd_full[k]["data"] == sd_resumed[k]["data"], k


TESTS = [
    test_tensor_cat,
    test_tensor_cat_grad,
    test_named_parameters,
    test_trainer_fit_and_eval,
    test_trainer_grad_accumulation,
    test_trainer_early_stopping,
    test_checkpoint_save_load,
    test_checkpoint_resume_matches,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
        else:
            print(f"PASS {t.__name__}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
