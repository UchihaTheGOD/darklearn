"""Benchmark the data pipeline (Day 13, Month 6 baseline).

Measures DataLoader throughput on the story corpus (the same data
the tiny LM trains on) as windows/second, iterating only — no model,
no loss — so the number is purely the pipeline's cost: dataset
indexing, batching, and the ``batch_fn`` reshape. This is the
baseline Month 6 optimisation work compares against.

    python3 examples/bench_dataloader.py

Prints real measured throughput. The profile (Day 13, slice 1)
already showed the loader costs ~0.00s of a training epoch, so
expect a high number here — the point is to have it written down
before anyone tries to optimise it.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import BPETokenizer, DataLoader, TextDataset
from train_tiny_lm_long import lm_batch

HERE = Path(__file__).resolve().parent


def _time_loader(loader, batch_fn, reps):
    n_windows = 0
    t0 = time.perf_counter()
    for _ in range(reps):
        for xb, yb in loader:
            if batch_fn is not None:
                batch_fn((xb, yb))
            n_windows += xb.shape[0]
    secs = time.perf_counter() - t0
    return n_windows / secs


def main():
    text = (HERE / "darklearn_story.txt").read_text()
    tok = BPETokenizer()
    tok.train(text, vocab_size=384)
    ids = tok.encode(text)
    print(f"corpus: {len(text)} chars -> {len(ids)} tokens, "
          f"vocab {tok.vocab_size}")

    print(f"\n{'setup':>28} {'windows/s':>12} {'batches/s':>10}")
    for seq_len, stride in [(8, 4), (16, 4)]:
        ds = TextDataset.from_token_ids(ids, seq_len=seq_len,
                                        stride=stride)
        for batch_size, shuffle, use_fn in [
                (8, False, False), (8, True, True), (32, False, False)]:
            loader = DataLoader(ds, batch_size=batch_size,
                                shuffle=shuffle, seed=0)
            wps = _time_loader(loader, lm_batch if use_fn else None,
                               reps=5)
            label = (f"seq {seq_len}, batch {batch_size}, "
                     f"{'shuffled+batch_fn' if shuffle else 'plain'}")
            print(f"{label:>28} {wps:>12.0f} "
                  f"{wps / batch_size:>10.0f}")
    print("OK — dataloader benchmark complete. "
          "(Baseline for Month 6; see PROGRESS.md Day 13.)")


if __name__ == "__main__":
    main()
