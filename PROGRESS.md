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

## Day 5 — 2026-11-20 — v0.1.0-day5 (Month 3 — data)

### Built
- `Subset` + `split_dataset` in `data.py`: seeded shuffle, split by
  `lengths=(train, val)` or `val_fraction` in (0, 1) (rounded, clamped
  to 1..n-1). Same seed reproduces the split; subsets are disjoint and
  cover the dataset. Bad inputs raise ValueError.
- `Dataset.map(fn)` + `MappedDataset`: lazy view applying a transform
  to each item; original dataset untouched, maps compose.
- `Normalize` in `data.py`: `(x - mean) / std`, scalar or per-feature
  mean/std; works on feature-row lists and on Tensors (per-row for
  2-d, same shape back). `Normalize.from_dataset` computes per-feature
  mean/population-std; zero-variance features get std 1.0 (via eps)
  instead of dividing by zero.
- `TextDataset` in `data.py`: character-level dataset for the tiny LM
  (Month 4). Vocab = sorted unique chars; item `i` is
  `(input_ids, next_id)` — `seq_len` chars plus the one that follows —
  so it fits the existing DataLoader ((batch, seq_len) x, (batch, 1) y).
  `encode`/`decode` helpers, `stride` option.
- `examples/train_val_split.py`: MLP on 80% train / 20% val, reports
  both losses every 50 steps.
- `tests/test_day5.py`: 4 new tests. `__version__` bumped to `0.1.0-day5`.
- All new names exported from `darklearn`: `Subset`, `MappedDataset`,
  `split_dataset`, `Normalize`, `TextDataset`.

### Test results (2026-11-20, Python 3.12.3, local VM)
- `python3 tests/test_day5.py` — **4/4 PASS**:
  `test_text_dataset`, `test_normalize`, `test_split_dataset`,
  `test_dataset_map`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day4.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- Total: **33/33 PASS**
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (200 steps)

### Bugs fixed during Day 5
1. Real bug: `Normalize.__call__` on a 2-d Tensor flattened the whole
   tensor into one row before normalizing (length mismatch error).
   Fixed: normalize per row for 2-d Tensors, flat for 1-d.

### Day 7 — next tasks (Month 4 — transformer)
1. `Embedding` module in `nn.py` (token-id lookup, grad flows to rows used) (+ test).
2. `LayerNorm` module in `nn.py` (per-row mean/var normalize, gamma/beta) (+ test).
3. Positional encoding helper (sinusoidal, no parameters) (+ test).
4. Example: embed a BPE-encoded sentence, print shapes.
5. Write Day 7 entry in this file. Keep each slice to its own small commit.

## Day 6 — 2026-11-21 — v0.1.0-day6 (Month 3 — tokenizer)

### Built
- `darklearn/tokenizer.py`: byte-level BPE tokenizer from scratch
  (stdlib only: `json`, `re`). Base vocab = the 256 bytes, so any text
  encodes with no unknown tokens. `train(corpus, vocab_size=512,
  min_freq=2)` learns merges word by word (whitespace split, so merges
  never cross a word boundary): repeatedly merges the most frequent
  adjacent pair until the vocab cap or no pair meets `min_freq`.
  Deterministic (ties break by first-seen order). `encode` applies
  merges in rank order inside each word, whitespace runs stay raw
  bytes; `decode` is the exact inverse. `save`/`load` JSON
  (`darklearn-bpe-v1` format, vocab as id -> byte list, merges list).
- `TextDataset.from_token_ids(ids, seq_len, stride=1)` in `data.py`:
  same `(input_ids, next_id)` items as the character mode, but the ids
  come from a tokenizer (e.g. BPE). `vocab_size = max(ids) + 1` for the
  embedding layer. `encode`/`decode` raise a clear error in this mode
  (decoding is the tokenizer's job); character mode is unchanged.
- `tests/test_day6.py`: 7 new tests. `__version__` bumped to `0.1.0-day6`.
- `examples/train_bpe.py` + `examples/tiny_corpus.txt` (original
  sentences about darklearn, 873 chars): trains BPE, prints vocab +
  first merges + a sample encode/decode, builds a token-id
  `TextDataset` + `DataLoader` batch, saves `bpe_vocab.json` next to
  the example (generated file, not committed).
- `BPETokenizer` exported from `darklearn`.

### Test results (2026-11-21, Python 3.12.3, local VM)
- `python3 tests/test_day6.py` — **7/7 PASS**:
  `test_bpe_learns_merges`, `test_bpe_roundtrip`, `test_bpe_save_load`,
  `test_bpe_vocab_size_limit`, `test_bpe_deterministic`,
  `test_bpe_errors`, `test_text_dataset_from_token_ids`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day4.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day5.py` — **4/4 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- Total: **40/40 PASS**
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (unchanged)
- `python3 examples/train_bpe.py` — **vocab 320 (256 bytes + 64 merges)**,
  first merges `he, in, en, ar, or, er, to, an, the, ra`; sample
  "the model learns to predict the next token" -> 21 tokens for
  42 chars, round-trip exact; DataLoader batch x (4, 16), y (4, 1)

### Notes
- First push of the day (tokenizer.py) was interrupted by a service
  restart mid-call; the branch head was still Day 5's `34f6131d`,
  so the push had not landed and was safely retried — it then
  succeeded as `ba1551a9`. No duplicate commit resulted.
- Design choice: merges are learned within words only, so the space
  byte is never part of a merged token and whitespace round-trips
  exactly (tabs, newlines, double spaces all verified in tests).

## Day 7 — 2026-11-22 — v0.1.0-day7 (Month 4 — transformer)

### Built
- `Tensor.gather_rows(indices)` in `tensor.py`: row gather for 2-d
  tensors, `out[i] = self[indices[i]]`. Backward scatter-adds grads,
  so a row used twice gets the sum of both grads. Out-of-range ids
  raise IndexError.
- `Tensor.layer_norm(eps=1e-5)` in `tensor.py`: per-row
  `(x - mean) / sqrt(var + eps)` for 2-d, whole-tensor for 1-d.
  Backward is the standard formula using saved normalised values and
  per-row inverse std; gradchecked on a 1-d input.
- `Embedding` in `nn.py`: lookup table `(num_embeddings, dim)`,
  N(0, 1/sqrt(dim)) init. `forward(ids)` takes one int or a flat list
  of ids and returns `(len(ids), dim)`. Note: darklearn Tensors are
  1-d/2-d only, so a (batch, seq) batch is passed in flattened and
  comes back as (batch*seq, dim).
- `LayerNorm` in `nn.py`: per-row normalisation with learnable gamma
  (init 1) and beta (init 0), built on `Tensor.layer_norm` + the
  existing (m, n) + (n,) broadcast.
- `sinusoidal_positional_encoding(seq_len, d_model)` in `nn.py`:
  `sin(pos / 10000^(2i/d))` on even dims, `cos(...)` on odd dims, no
  parameters. Returns a (seq_len, d_model) Tensor to add to embeddings.
- `tests/test_day7.py`: 7 new tests. `__version__` bumped to `0.1.0-day7`.
- `examples/embed_sentence.py`: BPE-encodes "the model learns to
  predict the next token" (21 tokens), embeds with d_model=16, adds
  sinusoidal positions, prints shapes.
- `Embedding`, `LayerNorm`, `sinusoidal_positional_encoding` exported
  from `darklearn`.

### Test results (2026-11-22, Python 3.12.3, local VM)
- `python3 tests/test_day7.py` — **7/7 PASS**:
  `test_embedding_forward`, `test_embedding_grad`,
  `test_layernorm_forward`, `test_layernorm_gradcheck`,
  `test_layernorm_gamma_beta`, `test_positional_encoding`,
  `test_embedding_trains`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day4.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day5.py` — **4/4 PASS** (no regressions)
- `python3 tests/test_day6.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- Total: **47/47 PASS**
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (unchanged)
- `python3 examples/train_bpe.py` — **vocab 320**, batch x (4, 16) (unchanged)
- `python3 examples/embed_sentence.py` — **21 tokens -> (21, 16)
  embeddings + (21, 16) positions**, decode round-trip exact
- `test_embedding_trains`: embedding + Linear head learns token parity
  (even -> 1, odd -> 0) in 150 Adam steps; same-parity embeddings end
  up closer than cross-parity.

### Bugs fixed during Day 7
1. Test bug: `test_layernorm_forward` asserted unit variance for the
   constant row too — a constant row normalises to all zeros, so its
   variance is 0, not 1. Fixed the test to check variance only on the
   non-constant row.

### Day 8 — next tasks (Month 4 — transformer)
1. `Tensor.softmax` already exists — add scaled dot-product attention
   helper (scores = Q @ K.T / sqrt(d), causal mask option) (+ test).
2. `MultiHeadAttention` module in `nn.py` (Q/K/V projections, split
   heads, concat, output projection) (+ test, gradcheck on 1 head).
3. `TransformerBlock` in `nn.py` (attention + MLP + 2x LayerNorm,
   residual connections) (+ test).
4. Example: tiny GPT-style forward pass on a BPE-encoded sentence,
   print shapes per block.
5. Write Day 8 entry in this file. Keep each slice to its own small commit.

## Day 8 — 2026-11-23 — v0.1.0-day8 (Month 4 — transformer)

### Built
- `Tensor.narrow(dim, start, length)` in `tensor.py`: contiguous
  slice along dim 0 (rows) or 1 (columns) for 2-d, dim 0 for 1-d.
  Backward scatters the grad back into the parent (zeros outside
  the block). What splits a Q/K/V projection into attention heads
  and a weight matrix into per-head blocks.
- `scaled_dot_product_attention(q, k, v, causal=False)` in `nn.py`:
  `softmax(Q @ K.T / sqrt(d_k)) @ V` for (seq, d) inputs. Causal
  mask is a constant tensor (0 / -1e9) added to the scores, so no
  grad flows through it. Built only from Tensor ops.
- `MultiHeadAttention` in `nn.py`: one Linear(d_model, d_model)
  projection each for Q/K/V plus an output Linear. Heads split the
  projections by columns via `narrow`; the concatenated heads are
  never materialised — `out = sum_h head_out_h @ W_out[h-block, :] +
  b_out`, exactly `concat(heads) @ W_out.T + b_out`. Seeded
  projections use seed, seed+1, seed+2, seed+3 (same seed for all
  four would give identical Q/K/V weights).
- `TransformerBlock` in `nn.py`: pre-norm block, `x + attn(ln1(x))`
  then `x + mlp(ln2(x))`, MLP is Linear(d, 4d) -> GELU ->
  Linear(4d, d), optional Dropout (default off), causal flag passed
  through to attention.
- `tests/test_day8.py`: 11 new tests. `__version__` bumped to
  `0.1.0-day8`.
- `examples/tiny_gpt.py`: BPE-encodes "the model learns to predict
  the next token" (15 tokens), Embedding(281, 16) + positions, 2
  causal TransformerBlocks, Linear head; prints shapes per block,
  then trains next-token prediction for 80 Adam steps.
- `Embedding`, `LayerNorm`, `scaled_dot_product_attention`,
  `MultiHeadAttention`, `TransformerBlock` exported from `darklearn`.

### Test results (2026-11-23, Python 3.12.3, local VM)
- `python3 tests/test_day8.py` — **11/11 PASS**:
  `test_narrow_rows_cols`, `test_narrow_grad`,
  `test_sdpa_forward`, `test_sdpa_causal`, `test_sdpa_grad`,
  `test_mha_forward`, `test_mha_causal`, `test_mha_grad`,
  `test_block_forward`, `test_block_causal_and_grad`,
  `test_tiny_gpt_trains`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day4.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day5.py` — **4/4 PASS** (no regressions)
- `python3 tests/test_day6.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day7.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- Total: **58/58 PASS**
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (unchanged)
- `python3 examples/train_bpe.py` — **vocab 320**, batch x (4, 16) (unchanged)
- `python3 examples/embed_sentence.py` — **21 tokens -> (21, 16)
  embeddings + (21, 16) positions** (unchanged)
- `python3 examples/tiny_gpt.py` — **15 tokens -> (15, 16) per
  block, logits (15, 281); next-token loss 11.2851 -> 0.0001** in
  80 Adam steps (~15s)
- Strongest checks: causal tests prove changing rows 1.. never
  moves output row 0, through SDPA, MHA and the full block; the
  SDPA grad matches finite differences on one q element; the
  tiny-GPT training test drops loss below 1.0 in 60 steps.

### Day 9 — next tasks (Month 4 — transformer)
1. Greedy + sampling text generation from the tiny GPT
   (`generate()` helper: greedy argmax, temperature + top-k sampling).
2. Overfit-a-tiny-corpus test: train the tiny GPT until it
   reproduces a short sentence from its own generation.
3. Attention gradcheck on a 1-head MHA (numeric vs analytic on a
   small case).
4. Example: generate text from the trained tiny GPT.
5. Write Day 9 entry in this file. Keep each slice to its own small commit.

## Day 9 — 2026-11-24 — v0.1.0-day9 (Month 4 — generation)

### Built
- `darklearn/generate.py`: `generate(model, prompt_ids, ...)` —
  any callable `model(ids) -> Tensor` of shape `(seq, vocab)` can
  generate text. Greedy argmax (first max on ties); sampling
  divides logits by `temperature`, optionally keeps only the
  `top_k` largest logits, softmaxes, and draws with
  `random.Random(seed)` so a fixed seed reproduces the run.
  `eos_id` stops generation early; `max_new_tokens=0` returns the
  prompt unchanged; the prompt list is never mutated. Input
  validation (method, temperature > 0, top_k >= 1, non-empty
  prompt, non-negative `max_new_tokens`, model must return
  (len(ids), vocab) logits) all raise ValueError. No gradients are
  needed during generation — only the last logits row is read.
- `tests/test_day9.py`: 8 new tests. `__version__` bumped to
  `0.1.0-day9`.
- `examples/generate_text.py`: trains the Day 8 tiny GPT (80 Adam
  steps), greedy-generates the sentence back from "the model",
  and samples a continuation (t=0.8, top-k=5, seed=7).
- `generate` exported from `darklearn`.

### Test results (2026-11-24, Python 3.12.3, local VM)
- `python3 tests/test_day9.py` — **8/8 PASS**:
  `test_generate_greedy`, `test_generate_eos`,
  `test_generate_sample_reproducible`,
  `test_generate_sample_low_temperature`, `test_generate_top_k`,
  `test_generate_errors`, `test_mha_one_head_gradcheck`,
  `test_overfit_tiny_corpus`
- `python3 tests/test_day1.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day2.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day4.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day5.py` — **4/4 PASS** (no regressions)
- `python3 tests/test_day6.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day7.py` — **7/7 PASS** (no regressions)
- `python3 tests/test_day8.py` — **11/11 PASS** (no regressions)
- `python3 tests/test_darkcore.py` — **8/8 PASS** (no regressions)
- Total: **66/66 PASS**
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (unchanged)
- `python3 examples/train_bpe.py` — **vocab 320**, batch x (4, 16) (unchanged)
- `python3 examples/embed_sentence.py` — **21 tokens -> (21, 16)
  embeddings + (21, 16) positions** (unchanged)
- `python3 examples/tiny_gpt.py` — **15 tokens -> (15, 16) per
  block, logits (15, 281); next-token loss 11.2851 -> 0.0001** (unchanged)
- `python3 examples/generate_text.py` — **greedy reproduces
  "the model learns to predict the next token" exactly** from the
  prompt "the model"; sampled (t=0.8, top-k=5, seed=7):
  "the model learns to predict the next" (a different final token,
  as expected from sampling)
- Strongest checks: the 1-head MHA gradcheck passes numeric vs
  analytic grads through the whole module (Q/K/V projections,
  per-head attention, output projection); the overfit test trains
  "the cat sat" (5 tokens) to loss 0.0002 in 120 steps and greedy
  generation from its first 2 tokens reproduces it exactly.

### Bugs fixed during Day 9
1. Import bug (twice): `from darklearn import generate` returned
   the `darklearn.generate` *module*, not the function, because the
   module and the function share a name. Fixed by importing the
   function from the submodule (`from darklearn.generate import
   generate`) in the test and example; after the `__init__.py`
   export, `from darklearn import generate` also resolves to the
   function (verified).

### Day 10 — next tasks (Month 5 — LLM training loop)
1. `Trainer` upgrade: eval loop + perplexity metric in `examples/`
   or a `darklearn/trainer.py` (epochs, batches, logging, eval).
2. Checkpoint save/load for weights (`pickle`/`json`, stdlib only)
   (+ test: save, reload, identical outputs).
3. Gradient accumulation helper (+ test).
4. Early stopping on val loss (+ test).
5. Example: train the tiny GPT on the tiny corpus with the Trainer
   + checkpoint, reload, and generate from it.
6. Write Day 10 entry in this file. Keep each slice to its own small commit.

## Day 10 — 2026-11-25 — v0.1.0-day10 (Month 5 — training loop + checkpoints)

### Built
- `Tensor.cat(tensors, dim=0)` in `tensor.py`: concat for 1-d (flat)
  and 2-d (dim 0 stacks rows, dim 1 joins columns); backward slices
  grads back into each input. Needed to stack per-row logits into
  one batch tensor for the LM example. Gradchecked + explicit grad
  test; shape mismatches raise ValueError.
- `Module.named_parameters()` in `nn.py`: dotted names
  (`"fc1.weight"`, `"layers.0.bias"`, `"attn.q_proj.weight"`),
  same order as `parameters()` (verified on MLP + TransformerBlock).
  Checkpoint keys use these names.
- `darklearn/trainer.py`: library `Trainer` — epochs over a
  DataLoader, eval loop, `evaluate()`, `perplexity()` (exp of mean
  loss), gradient accumulation (`accum_steps`, loss divided before
  backward, optimizer steps every N micro-batches), per-epoch LR
  scheduler, `max_grad_norm` clipping before each step, `batch_fn`
  hook to reshape batches (what the LM example uses). `EarlyStopping`
  (patience, min_delta, min/max mode) snapshots the best weights in
  memory and restores them on stop. Supersedes
  `examples/trainer_lite.py` (kept, docstring now points here).
- Optimizer state in `optim.py`: `state_dict()` / `load_state_dict()`
  on `Optimizer` (lr), `SGD` (velocity), `Adam`/`AdamW` (step count
  `t`, first/second moments), keyed by param index so a round-trip
  restores exactly.
- `darklearn/checkpoint.py`: `state_dict(model)` /
  `load_state_dict(model, state)` (strict — unknown/missing keys and
  shape mismatches raise ValueError, weights copied in place so the
  optimizer keeps working), `save_checkpoint` / `load_checkpoint`
  with stdlib `pickle` (`darklearn-checkpoint-v1` format: weights +
  optimizer state + epoch + extra metadata). No tensors, no numpy.
- `examples/train_tiny_lm.py`: `TinyGPT` (a `Module` whose forward
  takes a token list or a (batch, seq) id Tensor), BPE corpus (113
  tokens), token-id `TextDataset` (seq_len 8, stride 1), 80/20 split,
  Trainer + early stopping, checkpoint to
  `examples/checkpoints/tiny_lm.pt` (gitignored), reload into a fresh
  model, generate from both.
- `Trainer`, `EarlyStopping`, `state_dict`, `load_state_dict`,
  `save_checkpoint`, `load_checkpoint` exported from `darklearn`;
  `__version__` bumped to `0.1.0-day10`.
- `tests/test_day10.py`: 8 new tests.

### Test results (2026-11-25, Python 3.12.3, local VM)
- `python3 tests/test_day10.py` — **8/8 PASS**:
  `test_tensor_cat`, `test_tensor_cat_grad`,
  `test_named_parameters`, `test_trainer_fit_and_eval`,
  `test_trainer_grad_accumulation`, `test_trainer_early_stopping`,
  `test_checkpoint_save_load`, `test_checkpoint_resume_matches`
- Full suite: **74/74 PASS** (Day 1: 7, Day 2: 7, Day 4: 7, Day 5: 4,
  Day 6: 7, Day 7: 7, Day 8: 11, Day 9: 8, DarkCore: 8, Day 10: 8)
- `python3 examples/train_mlp.py` — **loss 0.3624 -> 0.0000** (unchanged)
- `python3 examples/trainer_lite.py` — **loss 0.2364 -> 0.0008** (unchanged)
- `python3 examples/train_val_split.py` — **train 0.3934 -> 0.0000,
  val 0.2692 -> 0.0000** (unchanged)
- `python3 examples/train_bpe.py` — **vocab 320** (unchanged)
- `python3 examples/embed_sentence.py` — **21 tokens -> (21, 16)** (unchanged)
- `python3 examples/tiny_gpt.py` — **loss 11.2851 -> 0.0001** (unchanged)
- `python3 examples/generate_text.py` — **greedy reproduces the
  sentence** (unchanged)
- `python3 examples/train_tiny_lm.py` — **train loss 10.1170 ->
  0.8683 in 11 epochs (~72s), val perplexity 1.00 after restore**;
  checkpoint saved + reloaded (epoch 11, extra val_perplexity
  1.0002); reloaded model generates byte-identically to the trained
  one.
- Strongest check: `test_checkpoint_resume_matches` — 6 epochs
  straight vs 3 epochs + save + reload + 3 more gives the same final
  loss (0.0180) and byte-identical weights; `test_trainer_grad_accumulation`
  counts optimizer steps (4 steps for 8 micro-batches at accum=2).

### Honest notes
- Early stopping earned its keep in the example: val loss hit
  ~0.0000 at epoch 10, then one more Adam epoch at lr=0.05 bounced
  it to 3.78 (Adam keeps taking ~lr-sized steps near the minimum).
  The restore put the epoch-10 weights back, so the reported val
  perplexity (1.00) is measured on the restored model, not the last
  history entry — the example re-evaluates after `fit()` for exactly
  this reason.
- Free-running greedy generation still drifts: "the model learns
  predict predict predict" instead of the full sentence. The model
  was trained only on 8-token windows, so contexts longer than that
  (and the compounding of its own errors) are out of distribution.
  The example asserts what is actually verified — loss down, val
  perplexity 1.00, identical generation after reload — not that
  greedy reproduces the sentence (Day 9's 5-token overfit test
  already proves exact reproduction on a short sequence).
- Val perplexity 1.00 is optimistic: the corpus repeats the same
  sentence 8x and the split is random over overlapping sliding
  windows, so val windows are near-duplicates of train windows.

### Day 11 — next tasks (Month 5 — training loop, continued)
1. `Trainer` checkpoint callback: save the best model to disk
   automatically when val loss improves (`checkpoint_path`).
2. LR scheduler test with the Trainer (CosineAnnealingLR stepped
   per epoch, lr actually decays).
3. Longer training corpus for the tiny LM (multi-paragraph original
   text file) + train with the Trainer, report generation quality.
4. README quickstart for Trainer + checkpoints.
5. Write Day 11 entry in this file. Keep each slice to its own small commit.
