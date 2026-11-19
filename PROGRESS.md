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

## Day 3 — 2026-11-18 — v0.1.0-day3 (DarkCore — our own engine)

### Why (user direction, 2026-11-18)
User said the code felt like "other logics, not own" — wants darklearn's
own, better, optimized logic, and explicitly allows multiple languages,
not just Python. No code was copied from PyTorch/sklearn/micrograd and no
external lib is used, but Day 1–2 was generic tutorial-style (naive i-j-k
matmul, one closure per op). The maths is universal; the ENGINE is where
originality lives. Day 3 therefore pivots to DarkCore. The old
`tensor.py` engine is untouched and kept working — it is the correctness
reference the new engine is tested against.

### Built
- `darklearn/darkcore.py` (~600 lines, stdlib only): our own engine.
  - Tiled matmul: block 32, i-k-j order, locals cached, no attribute
    lookups in the inner loop (`matmul_tiled`, `matmul_tiled_into`).
  - Rows kernel: B transposed once into contiguous rows, each dot in C
    via `sum(map(mul, ...))` — the fast pure-Python path (`matmul_rows`).
  - Fused linear+ReLU in one pass, no intermediate tensors
    (`fused_linear_relu_raw`, tape-aware `fused_linear_relu`).
  - Tape autograd: flat `values` + `records` of
    `(op_code:int, out_idx, in_idx_tuple, saved)`, op codes 1=ADD,
    2=MUL, 3=MATMUL, 4=RELU, 5=SUM, 6=FUSED_LINEAR_RELU. Backward is a
    single iterative reverse loop — no closures, no recursive DFS.
  - `BufferPool`: scratch `array('f')` buffers recycled in backward
    (transposes, matmul temporaries); live outputs never pooled.
  - `fused_sgd_step`: grad + weight decay + momentum + update fused in
    one in-place pass over raw arrays; returns the velocity state.
  - `matmul()` dispatcher: C kernel when built, else rows kernel.
- `darklearn/_darkcore.c` (C stdlib + CPython API only): tiled matmul
  (block 32, i-k-j) + fused linear+ReLU, float32, bytes in/out.
- `scripts/build_c.py` (stdlib only): finds cc/gcc/clang, compiles with
  `-O3 -fPIC -shared` to `darklearn/_darkcore*.so`. No compiler ->
  message + exit 0, Python fallback keeps working. The `.so` is
  gitignored (`*.so` added) — build artifact, not source.
- `darklearn/__init__.py`: exports Tape, CoreTensor, BufferPool,
  matmul_tiled/rows, fused_linear_relu, fused_sgd_step, C_AVAILABLE;
  `__version__ = "0.1.0-day3"`.
- `tests/test_darkcore.py`: 8 tests. `examples/bench_darkcore.py`:
  naive vs tiled-16/32 vs rows vs C, with a correctness check first.
- `docs/ROADMAP.md`: new "DarkCore — our own engine" section.

### Measured results (2026-11-18, Python 3.12.3, gcc 13.3, local VM)
- `python3 tests/test_darkcore.py` — **8/8 PASS**; Day 1 **7/7**,
  Day 2 **7/7** — 22/22 total, no regressions.
- `examples/bench_darkcore.py` (MFLOPS, speedup vs naive Tensor @):

  | n   | naive | tiled-16 | tiled-32 | rows  | C       |
  |-----|-------|----------|----------|-------|---------|
  | 64  | 8.27  | 9.09 (1.10x) | 9.40 (1.14x) | 39.60 (4.79x) | 7838.95 (947.83x) |
  | 128 | 8.57  | 9.14 (1.07x) | 9.57 (1.12x) | 43.65 (5.10x) | 7617.77 (889.30x) |

- Honest miss: tiling alone did **not** reach the 2x target (~1.1x).
  Prototyped first and measured it: in CPython the bottleneck is
  bytecode, not cache, so tiling barely helps. The 2x+ comes from the
  rows kernel (~4–5x) and the C core (~900x). Tiling stays as the
  reference kernel and as the C kernel's layout — it is not presented
  as the speedup.
- C core built cleanly on this VM (`cc 13.3.0`, `-O3`); `C_AVAILABLE`
  is True and C results match Python bit-for-bit on the smoke check
  and within 1e-3 on the benchmark correctness check.

### Bugs fixed during Day 3
1. Test typos: `(64, 64)` written where `(64, 64, 64)` was meant, in
   three places in `test_darkcore.py`.
2. Test bug: Tensor-side `x` in the fused-backward test had no
   `requires_grad`, so `x.grad` was None — set `requires_grad=True`.
3. Test bug: exact `==` on float32 values (`0.9`) fails on rounding —
   switched to a tolerance comparison.

### Still naive / next to optimize
- Tape backward still builds small temporaries with list comprehensions
  (add/mul grads); a next step is pooling those too.
- No C fused-SGD or C backward yet — forward matmul + fused linear
  only. Rust is possible later; C first (no cargo needed).
- Old Day 3 plan (BCELoss, AdamW, Dropout, Xavier, argmax/softmax)
  moves to Day 4.

### Day 4 — next tasks
1. BCELoss (+ test with sigmoid outputs).
2. AdamW optimiser (decoupled weight decay) (+ test).
3. Xavier/Glorot init option to `Linear` (+ test).
4. Dropout module with train/eval mode flag on `Module` (+ test).
5. `Tensor.argmax()` + `Tensor.softmax()` helpers (+ gradchecks).
6. Write Day 4 entry in this file. Keep each slice to its own small commit.

## Day 4 — 2026-11-19 — v0.1.0-day4 (Month 2 — losses, optim, init, dropout)

### Built
- `Tensor.argmax(dim=None)` — flat int, or per-row / per-column list for
  2-d (first max wins on ties). No grad (an index is not differentiable).
- `Tensor.softmax(dim=-1)` — stable max-subtraction per slice, 1-d and
  2-d (row- or column-wise); custom backward via the softmax Jacobian
  `grad_i = s_i * (g_i - sum_j g_j * s_j)`, gradchecked 1-d and 2-d.
- `Tensor.clamp(min, max)` — elementwise clamp; grad passes through only
  where the input was inside the range.
- `BCELoss` in `nn.py`: mean binary cross-entropy, probs clamped to
  [1e-7, 1-1e-7] so log(0) can never give -inf; built from Tensor ops.
  End-to-end test trains a tiny sigmoid classifier with BCE + Adam.
- `AdamW` in `optim.py`: decoupled weight decay (`p -= lr*wd*p` before
  the Adam update). Test proves it differs from `Adam(weight_decay=...)`
  (0.89 vs 0.9 on the hand-computed one-step case) and matches Adam
  exactly when wd=0.
- Xavier/Glorot init for `Linear`: `init="xavier"` / `"glorot"`, std
  sqrt(2/(fan_in+fan_out)); `"he"` stays the default. Unknown init
  raises ValueError. Test checks measured std ≈ 0.1 (xavier) vs 0.141 (he).
- `Dropout` in `nn.py` + train/eval mode on `Module`: `Module.training`
  flag (default True), `train()` / `eval()` recurse into children.
  Inverted dropout (survivors scaled 1/(1-p)); identity in eval mode;
  grad flows scaled through survivors, zero where dropped. Keep rate
  verified ≈ 1-p over 2000 elements.
- `tests/test_day4.py`: 7 new tests. `__version__` bumped to `0.1.0-day4`.

### Test results (2026-11-19, Python 3.12.3, local VM)
- `python3 tests/test_day4.py` — **7/7 PASS**:
  `test_tensor_argmax`, `test_tensor_softmax`, `test_tensor_clamp`,
  `test_bce_loss`, `test_adamw`, `test_xavier_init`, `test_dropout`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)

### Bugs fixed during Day 4
1. Test bug: exact `==` on softmax of `[0,0,0]` fails on float32 rounding
   (0.33333334 vs 1/3) — switched to a tolerance comparison.
2. Missing export: `BCELoss` was in `nn.py` but not in
   `darklearn/__init__.py` — added to imports and `__all__`.

### Day 5 — next tasks (Month 3 — data)
1. `split_dataset` train/val split helper in `data.py` (+ test).
2. `Normalize` transform (mean/std) in `data.py` (+ test).
3. `TextDataset` — character-level text dataset for the tiny LM later (+ test).
4. `Dataset.map` — apply a transform over a dataset (+ test).
5. Example: train/val split demo on the MLP example.
6. Write Day 5 entry in this file. Keep each slice to its own small commit.
