# darklearn

A pure-stdlib Python library for training ML and LLM models — like PyTorch / scikit-learn, but with **zero external dependencies**.

Everything is hardcoded and optimised in the core: tensors, autograd, layers, optimisers, data loading. No numpy, no torch, no pip installs beyond Python itself. Python ≥ 3.10.

> Vision: a small, readable, fast-enough library you can actually understand end-to-end — and grow into a real LLM trainer over 6–7 months of small daily steps.

## Quickstart

```python
from darklearn import Tensor, Sequential, Linear, ReLU, Adam, MSELoss

# synthetic data: y = x0 + 2*x1
xs = [[0.5, -1.0], [1.0, 0.5]]
ys = [-1.5, 2.0]
x = Tensor(xs, shape=(2, 2))
y = Tensor(ys, shape=(2, 1))

model = Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))
opt = Adam(model.parameters(), lr=0.05)
loss_fn = MSELoss()

for step in range(200):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

print("loss:", loss_fn(model(x), y).item())
```

Run the example:

```bash
python3 examples/train_mlp.py
python3 tests/test_day1.py
```

## Train with the Trainer + checkpoints

For real training loops, use the library `Trainer` instead of a
hand-written loop. It handles epochs, eval, perplexity, gradient
accumulation, LR scheduling, early stopping, and — with
`checkpoint_path` — saving the best model to disk automatically
whenever the val loss improves.

```python
from darklearn import (
    Adam, CosineAnnealingLR, CrossEntropyLoss, DataLoader, Trainer,
    EarlyStopping,
)
from darklearn.checkpoint import load_checkpoint

opt = Adam(model.parameters(), lr=0.05)
sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.005)
trainer = Trainer(
    model, CrossEntropyLoss(), opt,
    scheduler=sched,
    early_stopping=EarlyStopping(patience=3),
    checkpoint_path="best.pt",   # best val-loss weights, saved on disk
)
history = trainer.fit(train_loader, epochs=10, val_loader=val_loader)
print("val perplexity:", trainer.perplexity(val_loader))
print("best epoch:", trainer.best_epoch)

# Later, or in a fresh process: reload the best weights.
meta = load_checkpoint("best.pt", model=model, optimizer=opt)
print("restored epoch", meta["epoch"], "val_loss", meta["extra"]["val_loss"])
```

See `examples/train_tiny_lm.py` (tiny LM + checkpoint) and
`examples/train_tiny_lm_long.py` (longer corpus, cosine schedule,
checkpoint callback) for full working versions.

## What exists on Day 1 (v0.1.0-day1)

- `darklearn.tensor.Tensor` — flat `array('f')` storage, add/sub/mul/div, matmul, relu/tanh/exp/log, sum/mean, reshape, transpose
- `darklearn.autograd` — reverse-mode backward, numeric gradcheck, scalar `Value`
- `darklearn.nn` — `Module`, `Linear`, `ReLU`, `Tanh`, `Sequential`, `MSELoss`, `CrossEntropyLoss`
- `darklearn.optim` — `SGD` (momentum) and `Adam`
- `darklearn.data` — `Dataset`, `TensorDataset`, `DataLoader` (seeded shuffle)

## Design rules

1. **Stdlib only.** If it needs a pip install, it doesn't go in the core.
2. **Correct before fast.** Every op ships with a test and a gradcheck.
3. **Small daily steps.** 5–8 small commits a day, each one real and tested.
4. **Later: optional C core.** A C accelerator may be added as an optional speed-up with a pure-Python fallback — never as a hard requirement.

## Roadmap

6–7 month plan in [`docs/ROADMAP.md`](docs/ROADMAP.md) — tensor/autograd → nn/optim → data/tokenizer → transformer/attention → LLM training loop + checkpointing → optimisation + optional C core → polish and 1.0 release.

## License

MIT — see [LICENSE](LICENSE). Developed by Gaurav Singh.
