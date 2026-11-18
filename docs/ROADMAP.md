# darklearn Roadmap — 6–7 months, daily small steps

Rule: **5–8 small commits a day, each one real and tested.** No fake
commits, no big-bang rewrites. A day's batch is a slice: one op, one
layer, one test, one doc fix — never an untested dump.

## DarkCore — our own engine (Day 3 pivot)

The user asked for darklearn's *own* logic, not other people's. No code
was copied from PyTorch / sklearn / micrograd and no external library is
used — but the Day 1–2 engine was generic tutorial-style (naive i-j-k
matmul, one closure per op for backward). The maths is universal and
cannot change; the *engine design* is where originality lives, so from
Day 3 the fast path is DarkCore (`darklearn/darkcore.py`), running in
parallel to the old `tensor.py` (kept working, used as the correctness
reference in tests):

1. **Tiled matmul** — block 32, loop order i-k-j, locals cached, no
   attribute lookups in the inner loop. Measured honestly: ~1.15x in
   CPython, because bytecode — not cache — is the bottleneck.
2. **Rows kernel** — the fast pure-Python path: B transposed once into
   contiguous rows, each dot product in C via `sum(map(mul, ...))`
   (~4–5x). Same k-order summation, so bit-identical to naive.
3. **Fused linear+ReLU** — one pass over x/W, no intermediate tensors.
4. **Tape autograd** — flat lists, int op codes (1=ADD, 2=MUL,
   3=MATMUL, 4=RELU, 5=SUM, 6=FUSED_LINEAR_RELU), one iterative reverse
   loop. No closures, no recursive DFS.
5. **Buffer pool** — scratch `array('f')` buffers recycled in backward;
   live outputs are never pooled.
6. **Fused SGD step** — grad + weight decay + momentum + update in one
   in-place pass over raw arrays.
7. **Optional C core** (`darklearn/_darkcore.c`, built by
   `scripts/build_c.py`, stdlib only, `cc -O3`) — tiled matmul +
   fused linear+ReLU, ~900x the naive Python loop on this VM.
   Install never requires C: missing/failed build falls back to the
   Python kernels silently. Rust is possible later; C first, because it
   needs no toolchain beyond cc.

Rule for every DarkCore change: benchmark before/after, report the real
numbers, and never claim a speedup that was not measured.

## Month 1 — Tensor + Autograd (foundation)
- Week 1: Tensor storage (`array('f')`), add/sub/mul/matmul, sum/mean, reshape
- Week 2: relu/tanh/exp/log, transpose, broadcasting, gradcheck
- Week 3: autograd graph, backward topo-sort, grad accumulation, `Value`
- Week 4: benchmarks vs naive loops, profiling, fix hot loops, docs

## Month 2 — nn / optim / loss
- `Module`, `Linear`, `ReLU`/`Tanh`/`Sigmoid`/`GELU`, `Sequential`
- Losses: MSE, CrossEntropy (stable log-softmax), BCE
- Optimisers: SGD + momentum, Adam, AdamW, LR schedulers (step, cosine)
- Init: Xavier / He, weight decay, gradient clipping

## Month 3 — Data + Tokenizer
- `Dataset`, `TensorDataset`, `DataLoader` (shuffle, batching, drop_last)
- Text datasets, train/val split, normalisation
- BPE tokenizer from scratch (stdlib only), vocab save/load
- Data pipeline benchmarks

## Month 4 — Transformer / Attention
- `Embedding`, `LayerNorm`, `Dropout`, multi-head self-attention
- Positional encoding, causal mask, `TransformerBlock`
- Small GPT-style model, text generation (greedy + sampling)
- Attention gradcheck + overfit-a-tiny-corpus test

## Month 5 — LLM Training Loop + Checkpointing
- Trainer: epochs, batches, logging, eval loop, early stopping
- Checkpoint save/load (stdlib `pickle`/`json` for weights), resume
- Mixed-precision notes, gradient accumulation, perplexity metric
- Train a tiny LM on a small public-domain corpus, end-to-end

## Month 6 — Optimisation + Optional C Core
- Profile matmul/autograd; optimise hot paths in pure Python first
- Optional C extension (`_core.c`) for matmul/elementwise, with
  pure-Python fallback — never a hard requirement, never breaks stdlib-only install
- Benchmarks before/after, keep correctness tests identical for both paths

## Month 7 — Polish, Docs, Release 1.0
- Full test suite, examples (MLP, classifier, tiny LM), API docs
- README quickstarts, contributing guide, changelog
- Tag `v1.0.0`, publish to GitHub (PyPI only if user approves)

## Daily rhythm (each day)
1. Pick the next smallest slice from the current month.
2. Implement it + a test in 1–2 commits.
3. Run `python3 tests/test_day1.py` (later: full suite) + the example.
4. Log the day in `PROGRESS.md` — what changed, test result, next slice.
5. Stop at 5–8 commits; leave the next slice for tomorrow.
