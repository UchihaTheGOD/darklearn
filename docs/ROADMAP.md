# darklearn Roadmap — 6–7 months, daily small steps

Rule: **5–8 small commits a day, each one real and tested.** No fake
commits, no big-bang rewrites. A day's batch is a slice: one op, one
layer, one test, one doc fix — never an untested dump.

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
