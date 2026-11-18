# darklearn — Progress Log

## Day 1 — 2026-11-18 — v0.1.0-day1 (foundation)

### Built
- Scaffold: `README.md`, `LICENSE` (MIT, Gaurav Singh), `.gitignore`,
  `pyproject.toml` (name `darklearn`, Python >= 3.10, **zero dependencies**).
- `darklearn/tensor.py` (~528 lines): Tensor backed by flat `array('f')`
  row-major storage. Ops: add/sub/mul/div (same-shape, scalar, (m,n)+(n,)
  broadcast), matmul (2-d), relu/tanh/exp/log, sum/mean, reshape,
  transpose, row/element `__getitem__` with autograd preserved.
- `darklearn/autograd.py`: `backward()` entry point, numeric `gradcheck()`,
  scalar `Value` node (micrograd-style) for demos/debugging.
- `darklearn/nn.py`: `Module`, `Linear` (He init), `ReLU`, `Tanh`,
  `Sequential`, `MSELoss`, `CrossEntropyLoss` (stable log-softmax from ops).
- `darklearn/optim.py`: `SGD` (momentum + weight decay), `Adam`.
- `darklearn/data.py`: `Dataset`, `TensorDataset`, `DataLoader`
  (batching, seeded shuffle, drop_last).
- `darklearn/__init__.py`: exposes Tensor, Linear, Sequential, SGD, Adam,
  MSELoss, Dataset, DataLoader, `__version__ = "0.1.0-day1"`.
- `tests/test_day1.py`: 7 tests, runnable with plain `python3` (no pytest needed).
- `examples/train_mlp.py`: tiny MLP 2->8->1 on synthetic `y = x0 + 2*x1`.
- `docs/ROADMAP.md`: 6–7 month daily plan (see below).

### Test results (2026-11-18, Python 3.12.3, local VM)
- `python3 -m py_compile darklearn/*.py tests/test_day1.py examples/train_mlp.py` — OK
- `python3 tests/test_day1.py` — **7/7 PASS**:
  `test_tensor_add_matmul`, `test_autograd_grad_check`,
  `test_autograd_matmul_grad`, `test_linear_forward`,
  `test_sgd_step_reduces_loss`, `test_dataloader_batching`,
  `test_cross_entropy_runs`
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000 in 200 steps, <1s**
  (step 50: 0.0015, step 100: 0.0001, final 0.0000). Asserts pass.
- Note: `pytest` is not installed in this VM, so tests are written to run
  with plain `python3` as well as pytest. No external dependency was added.

### Bugs fixed during Day 1
1. Test bug: `b[1]` on a 1-row final DataLoader batch — changed the
   seeded-shuffle check to compare whole-batch `tolist()` output.
2. Real bug: `Tensor.__getitem__` detached rows from the graph, so
   `CrossEntropyLoss` grad was `None`. Fixed: `__getitem__` now creates a
   graph node whose backward scatters the row grad back into the parent.

### Not done / not pushed
- **Not pushed to GitHub.** Repo creation via the GitHub connector failed
  with `403 Resource not accessible by integration` — the Muse GitHub App
  cannot create a user repo. The user must create an empty repo named
  `darklearn` on github.com and grant the app access; then the local
  files here can be pushed in small commits.
- Nothing in `~/workspace/sentientai*` or the PYTHON repo was touched.

### Day 2 — next tasks (Month 1, Week 2)
1. Add `Sigmoid` + `GELU` activations (with gradchecks).
2. Add `Tensor.sigmoid()`, `Tensor.pow()` / elementwise `**`, and `max()` helper.
3. Add LR schedulers: `StepLR` + `CosineAnnealingLR` in `optim.py`.
4. Add gradient clipping (`clip_grad_norm_`) and a `Trainer`-lite loop in `examples/`.
5. Add a benchmark script `examples/bench_matmul.py` (pure-Python timing baseline).
6. Write Day 2 entry in this file. Keep each slice to its own small commit.

### Update 2026-11-18 (Day 2 run)
- Day 1 is now fully on GitHub: batch 7 (`docs/ROADMAP.md`, `PROGRESS.md`)
  was pushed twice with identical content — first by this Day 2 run
  (`61535dab`), then by the user's approved retry of the timed-out Day 1
  approval (`91fd0f17`). Same tree, harmless duplicate; head moved on
  normally from `91fd0f17`.

## Day 2 — 2026-11-18 — v0.1.0-day2

### Built
- `Tensor.sigmoid()` (numerically stable branch form), `Tensor.pow()` /
  elementwise `**`, `Tensor.max()` (grad flows to first argmax).
- `Sigmoid` + `GELU` modules in `nn.py` (exact tanh GELU, built from Tensor
  ops so autograd flows); both exported from `darklearn`.
- LR schedulers in `optim.py`: `StepLR` (multiply by gamma every step_size)
  and `CosineAnnealingLR` (cosine decay to eta_min over T_max, holds at
  eta_min past T_max).
- `clip_grad_norm_()` in `optim.py`: scales grads to a total-L2 budget,
  returns the pre-clip norm; no-op with no grads or norm already in budget.
- `examples/bench_matmul.py`: pure-Python matmul timing baseline.
- `examples/trainer_lite.py`: minimal `Trainer` (epochs over DataLoader,
  loss, optimiser, per-epoch scheduler, optional grad clipping) + a demo
  that trains the MLP through it.
- `tests/test_day2.py`: 7 new tests. `__version__` bumped to `0.1.0-day2`.

### Test results (2026-11-18, Python 3.12.3, local VM)
- `python3 tests/test_day2.py` — **7/7 PASS**:
  `test_tensor_sigmoid`, `test_tensor_pow`, `test_tensor_max`,
  `test_sigmoid_gelu_modules`, `test_lr_schedulers`,
  `test_clip_grad_norm`, `test_trainer_lite`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008 in 30 epochs**
- `python3 examples/bench_matmul.py` — **~7 MFLOPS** pure Python
  (16: 0.0010s, 32: 0.0099s, 64: 0.0778s, 128: 0.5866s per matmul; ~2.5s total)
- Slices 3+4 from the Day 2 plan (schedulers + grad clipping) landed as one
  combined `optim.py` commit — both are small utilities in the same file.

### Pushed to GitHub (main, 6 commits)
- `61535dab` Day 1 (7/7): roadmap + progress log
- `04d27137` Day 2 (1/6): tensor — sigmoid, pow (**), max with grads
- `353d3921` Day 2 (2/6): nn — Sigmoid + GELU activations
- `af8ea6cf` Day 2 (3/6): optim — StepLR + CosineAnnealingLR, clip_grad_norm_
- `00246bad` Day 2 (4/6): examples — bench_matmul.py baseline (~7 MFLOPS)
- `f9f67cee` Day 2 (5/6): examples — Trainer-lite loop
- (this log + version bump = Day 2 (6/6), pushed after this entry)

### Day 3 — next tasks (Month 2)
1. Add `BCELoss` (+ test with sigmoid outputs).
2. Add `AdamW` optimiser (decoupled weight decay) (+ test).
3. Add Xavier/Glorot init option to `Linear` (+ test).
4. Add `Dropout` module with train/eval mode flag on `Module` (+ test).
5. Add `Tensor.argmax()` + `Tensor.softmax()` helpers (+ gradchecks).
6. Write Day 3 entry in this file. Keep each slice to its own small commit.
