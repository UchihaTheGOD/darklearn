"""darklearn.trainer — the library-level Trainer (pure stdlib).

Month 5: epochs over a DataLoader, an eval loop, perplexity, gradient
accumulation and early stopping. This supersedes the minimal
``examples/trainer_lite.py`` (kept as a small standalone demo).

A batch from the loader is ``(x, y)`` by default: the model is called
as ``model(x)`` and the loss as ``loss_fn(pred, y)``. Pass
``batch_fn`` to reshape a batch first — e.g. for a language model,
turn the (batch, seq) id Tensor into per-row inputs and the (batch, 1)
target Tensor into a list of ints (see ``examples/train_tiny_lm.py``).
"""

from __future__ import annotations

import math

from .checkpoint import load_state_dict, state_dict
from .optim import clip_grad_norm_


class EarlyStopping:
    """Stop training when a monitored value stops improving.

    ``patience`` is how many epochs without improvement to tolerate;
    ``min_delta`` is the smallest change that counts as an improvement
    (0.0 = any improvement counts). ``mode="min"`` watches a loss
    (lower is better); ``mode="max"`` watches a score. The best-seen
    weights are snapshotted in memory, and
    ``restore_best_weights(model)`` puts them back — so the model you
    keep is the best one, not the last one.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = "min", restore_best_weights: bool = True):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights_flag = bool(restore_best_weights)
        self.best_value: float | None = None
        self.best_state: dict | None = None
        self.bad_epochs = 0
        self.stopped = False

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value: float, model=None) -> bool:
        """Feed one epoch's monitored value; returns True if training
        should stop. ``model`` is needed to snapshot the best weights
        (pass it when ``restore_best_weights`` is on)."""
        if self._is_better(value):
            self.best_value = float(value)
            self.bad_epochs = 0
            if self.restore_best_weights_flag and model is not None:
                self.best_state = state_dict(model)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped

    def restore_best_weights(self, model) -> None:
        """Copy the best-seen weights back into ``model`` (no-op when
        nothing was snapshotted)."""
        if self.best_state is not None:
            load_state_dict(model, self.best_state)


class Trainer:
    """Train ``model`` on a DataLoader.

    Args:
        model: a darklearn Module.
        loss_fn: a loss Module, called as ``loss_fn(pred, target)``.
        optimizer: an Optimizer over ``model.parameters()``.
        scheduler: optional LR scheduler, stepped once per epoch.
        max_grad_norm: optional float; grads are clipped to this total
            L2 norm just before each optimizer step.
        accum_steps: gradient accumulation — the optimizer steps once
            every ``accum_steps`` micro-batches, so a large effective
            batch fits in memory a small batch at a time. The loss is
            divided by ``accum_steps`` before backward so the gradient
            is the mean over the accumulated batches.
        early_stopping: optional ``EarlyStopping``; it watches the val
            loss when a val loader is given, else the train loss.
        batch_fn: optional ``batch -> (model_input, target)``; the
            default passes the loader's ``(x, y)`` straight through.
    """

    def __init__(self, model, loss_fn, optimizer, scheduler=None,
                 max_grad_norm: float | None = None, accum_steps: int = 1,
                 early_stopping: EarlyStopping | None = None,
                 batch_fn=None):
        if accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.accum_steps = int(accum_steps)
        self.early_stopping = early_stopping
        self.batch_fn = batch_fn

    def _run_batches(self, loader, training: bool) -> float:
        """One pass over ``loader``; returns the mean loss. When
        ``training`` is True, steps the optimizer (with accumulation);
        otherwise just evaluates."""
        total = 0.0
        batches = 0
        if training:
            self.model.train()
            self.optimizer.zero_grad()
            micro = 0
            try:
                n_batches = len(loader)
            except TypeError:
                n_batches = None
            for xb, yb in loader:
                x, y = (self.batch_fn((xb, yb)) if self.batch_fn
                        else (xb, yb))
                loss = self.loss_fn(self.model(x), y)
                (loss / float(self.accum_steps)).backward()
                total += loss.item()
                batches += 1
                micro += 1
                last = n_batches is not None and batches == n_batches
                if micro % self.accum_steps == 0 or last:
                    if self.max_grad_norm is not None:
                        clip_grad_norm_(self.model.parameters(),
                                        self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    micro = 0
            if micro:  # loader without __len__ and a leftover cycle
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(),
                                    self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            was_training = self.model.training
            self.model.eval()
            for xb, yb in loader:
                x, y = (self.batch_fn((xb, yb)) if self.batch_fn
                        else (xb, yb))
                total += self.loss_fn(self.model(x), y).item()
                batches += 1
            self.model.train(was_training)
        if batches == 0:
            raise ValueError("cannot train/evaluate on an empty loader")
        return total / batches

    def fit(self, train_loader, epochs: int = 1, val_loader=None,
            log_every: int = 1) -> dict:
        """Run training; returns a history dict with ``train_loss``
        (one mean loss per epoch) and, when ``val_loader`` is given,
        ``val_loss`` and ``val_perplexity`` per epoch."""
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_perplexity": [],
        }
        for epoch in range(1, epochs + 1):
            train_loss = self._run_batches(train_loader, training=True)
            history["train_loss"].append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            msg = f"epoch {epoch:<4} train={train_loss:.4f}"
            if val_loader is not None:
                val_loss = self._run_batches(val_loader, training=False)
                history["val_loss"].append(val_loss)
                history["val_perplexity"].append(math.exp(val_loss))
                msg += (f" val={val_loss:.4f} "
                        f"ppl={math.exp(val_loss):.2f}")
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(msg)
            if self.early_stopping is not None:
                watched = (history["val_loss"][-1] if val_loader is not None
                           else train_loss)
                if self.early_stopping.step(watched, model=self.model):
                    print(f"early stopping at epoch {epoch} "
                          f"(best={self.early_stopping.best_value:.4f})")
                    if self.early_stopping.restore_best_weights_flag:
                        self.early_stopping.restore_best_weights(self.model)
                    break
        return history

    def evaluate(self, loader) -> float:
        """Mean loss over ``loader`` (eval mode, no optimizer steps)."""
        return self._run_batches(loader, training=False)

    def perplexity(self, loader) -> float:
        """``exp(mean loss)`` over ``loader`` — the standard language
        modelling metric when the loss is per-token cross-entropy in
        nats (a perplexity of N means the model is as surprised as if
        it chose uniformly among N tokens)."""
        return math.exp(self.evaluate(loader))


__all__ = ["Trainer", "EarlyStopping"]
