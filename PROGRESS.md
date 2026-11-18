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
