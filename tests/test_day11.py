"""Day 11 tests — Month 5: Trainer checkpoint callback + LR scheduler
with the Trainer (stdlib only, plain `python3` runner)."""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, CosineAnnealingLR, CrossEntropyLoss, DataLoader, Linear,
    MSELoss, ReLU, Sequential, TensorDataset,
)
from darklearn.checkpoint import load_checkpoint
from darklearn.trainer import Trainer


def _mlp():
    return Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))


def _mlp_data(n=64, seed=0):
    import random
    rng = random.Random(seed)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    return TensorDataset(xs, ys)


def _split(data, seed=0):
    from darklearn import split_dataset
    return split_dataset(data, val_fraction=0.25, seed=seed)


def test_trainer_checkpoint_callback():
    model = _mlp()
    train_ds, val_ds = _split(_mlp_data())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, seed=0)
    val_loader = DataLoader(val_ds, batch_size=16)
    opt = Adam(model.parameters(), lr=0.05)
    with tempfile.TemporaryDirectory() as d:
        # nested path on purpose: the callback must create the folder.
        path = os.path.join(d, "nested", "best.pt")
        trainer = Trainer(model, MSELoss(), opt, checkpoint_path=path)
        history = trainer.fit(train_loader, epochs=6, val_loader=val_loader,
                              log_every=99)
        assert os.path.isfile(path), "checkpoint file was not created"
        # best_epoch is the argmin of the val-loss history: the file
        # was only written on improvement, never on a worse epoch.
        best_i = min(range(len(history["val_loss"])),
                     key=lambda i: history["val_loss"][i])
        assert trainer.best_epoch == best_i + 1, \
            f"best_epoch {trainer.best_epoch} != argmin {best_i + 1}"
        assert abs(trainer.best_loss - history["val_loss"][best_i]) < 1e-9
        meta = load_checkpoint(path, model=None)
        assert meta["epoch"] == trainer.best_epoch, \
            f"file holds epoch {meta['epoch']}, best is {trainer.best_epoch}"
        assert abs(meta["extra"]["val_loss"] - trainer.best_loss) < 1e-9
        # reloading the file into a fresh model reproduces the best
        # val loss exactly (same weights, same deterministic loader).
        model2 = _mlp()
        load_checkpoint(path, model=model2)
        trainer2 = Trainer(model2, MSELoss(), Adam(model2.parameters()))
        assert abs(trainer2.evaluate(val_loader) - trainer.best_loss) < 1e-6


def test_trainer_checkpoint_no_val_loader():
    # Without a val loader the callback watches the train loss.
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.05)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pt")
        trainer = Trainer(model, MSELoss(), opt, checkpoint_path=path)
        history = trainer.fit(loader, epochs=4, log_every=99)
        assert os.path.isfile(path)
        assert trainer.best_epoch is not None
        assert abs(trainer.best_loss - min(history["train_loss"])) < 1e-9
        meta = load_checkpoint(path, model=None)
        assert "train_loss" in meta["extra"]
        assert abs(meta["extra"]["train_loss"] - trainer.best_loss) < 1e-9


def test_trainer_checkpoint_disabled():
    # No checkpoint_path: nothing is tracked, nothing is written.
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, MSELoss(), opt)
    trainer.fit(loader, epochs=2, log_every=99)
    assert trainer.best_loss is None
    assert trainer.best_epoch is None
    assert trainer.checkpoint_path is None


def test_trainer_cosine_scheduler():
    # CosineAnnealingLR stepped once per epoch by the Trainer: the
    # lr must decay every epoch and land exactly on eta_min at T_max,
    # then hold there. Training still reduces the loss while it does.
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.1)
    sched = CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
    trainer = Trainer(model, MSELoss(), opt, scheduler=sched)
    lrs = []
    for _ in range(6):
        history = trainer.fit(loader, epochs=1, log_every=99)
        lrs.append(opt.lr)
    # step 1 of the cosine: 0.05 * (1 + cos(pi/4)) = 0.0854...
    assert abs(lrs[0] - 0.05 * (1.0 + math.cos(math.pi / 4))) < 1e-9, lrs
    assert lrs[0] > lrs[1] > lrs[2] > lrs[3], lrs
    assert lrs[3] == 0.0, f"lr at T_max should be eta_min, got {lrs[3]}"
    assert lrs[4] == 0.0 and lrs[5] == 0.0, "lr must hold at eta_min"
    assert history["train_loss"][-1] < 0.05, \
        f"loss did not fall with the scheduler: {history['train_loss']}"


TESTS = [
    test_trainer_checkpoint_callback,
    test_trainer_checkpoint_no_val_loader,
    test_trainer_checkpoint_disabled,
    test_trainer_cosine_scheduler,
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
