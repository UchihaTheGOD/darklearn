"""Trainer-lite: a minimal training loop over a DataLoader.

A small preview of the full Trainer planned for Month 5: epochs, batches,
loss, optimiser, optional LR scheduler (stepped per epoch) and gradient
clipping. Kept in examples/ for now — the library-level Trainer
(``darklearn.trainer.Trainer``, with eval loop, perplexity, gradient
accumulation and early stopping) landed on Day 10.

Run:  python3 examples/trainer_lite.py   (from repo root)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (Tensor, Sequential, Linear, ReLU, Adam, MSELoss,
                       TensorDataset, DataLoader, CosineAnnealingLR,
                       clip_grad_norm_)


class Trainer:
    """Train ``model`` on ``loader`` for ``epochs`` epochs.

    Args:
        model: a darklearn Module.
        loss_fn: a loss Module, called as loss_fn(pred, target).
        optimizer: an Optimizer over model.parameters().
        scheduler: optional LR scheduler, stepped once per epoch.
        max_grad_norm: optional float; grads are clipped to this total norm.
    """

    def __init__(self, model, loss_fn, optimizer, scheduler=None,
                 max_grad_norm: float | None = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

    def fit(self, loader, epochs: int = 1, log_every: int = 10) -> list[float]:
        """Run training; returns the mean loss of each epoch."""
        history: list[float] = []
        for epoch in range(1, epochs + 1):
            total = 0.0
            batches = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += loss.item()
                batches += 1
            if self.scheduler is not None:
                self.scheduler.step()
            mean_loss = total / max(1, batches)
            history.append(mean_loss)
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(f"epoch {epoch:<4} loss={mean_loss:.4f}")
        return history


def main() -> list[float]:
    rng = random.Random(0)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(64)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=16, shuffle=True, seed=0)

    model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
    opt = Adam(model.parameters(), lr=0.05)
    sched = CosineAnnealingLR(opt, T_max=30, eta_min=0.001)
    trainer = Trainer(model, MSELoss(), opt, scheduler=sched, max_grad_norm=1.0)
    history = trainer.fit(loader, epochs=30, log_every=10)
    assert history[-1] < history[0], "loss did not decrease"
    assert history[-1] < 0.05, f"final loss too high: {history[-1]}"
    print(f"OK — loss {history[0]:.4f} -> {history[-1]:.4f}.")
    return history


if __name__ == "__main__":
    main()
