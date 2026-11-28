"""Profile one tiny-LM training epoch (Day 13, Month 6 — profile first).

This is the *measurement* step, not a fix: it answers "where does the
time go?" for the Day 11 setup (story corpus, seq_len 8, d_model 16,
2 TransformerBlocks, batch 8) so the optimisation work that follows
is aimed at the real hot spots instead of guesses.

Two views, both printed:

1. ``cProfile`` on one training epoch — top functions by cumulative
   time (and their own time), so a single expensive helper shows up.
2. A phase breakdown with plain wall-clock timers, run as separate
   passes over the same loader: data-only (iterate the loader +
   ``batch_fn``), forward-only (model + loss, no backward), and the
   full step (forward + backward + optimizer step). The backward +
   optimizer share is the full step minus the forward-only pass, so
   no phase is guessed.

    python3 examples/profile_train_step.py

Takes ~2-3 minutes (one epoch is ~35s on this VM). Nothing here
changes the model or the training code.
"""

import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from darklearn import (
    Adam, BPETokenizer, CosineAnnealingLR, CrossEntropyLoss, DataLoader,
    TextDataset, split_dataset,
)
from darklearn.trainer import Trainer
from train_tiny_lm_long import TinyGPT, lm_batch

HERE = Path(__file__).resolve().parent
SEQ_LEN = 8
STRIDE = 4


def build():
    """Same setup as Day 11: story corpus, BPE vocab 384, 80/20 split."""
    text = (HERE / "darklearn_story.txt").read_text()
    tok = BPETokenizer()
    tok.train(text, vocab_size=384)
    ids = tok.encode(text)
    ds = TextDataset.from_token_ids(ids, seq_len=SEQ_LEN, stride=STRIDE)
    train_ds, val_ds = split_dataset(ds, val_fraction=0.2, seed=0)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, seed=0)
    val_loader = DataLoader(val_ds, batch_size=8)
    print(f"corpus: {len(text)} chars -> {len(ids)} tokens, "
          f"vocab {tok.vocab_size}")
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  "
          f"batches/epoch: {len(train_loader)}")
    return tok, train_loader, val_loader


def fresh_trainer(tok):
    model = TinyGPT(tok)
    opt = Adam(model.parameters(), lr=0.05)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.005)
    trainer = Trainer(model, CrossEntropyLoss(), opt, scheduler=sched,
                      max_grad_norm=1.0, batch_fn=lm_batch)
    return trainer


def phase_timings(trainer, train_loader):
    """Time the phases as separate passes (seconds per epoch)."""
    # data only: loader + batch_fn, no model at all
    t0 = time.perf_counter()
    for xb, yb in train_loader:
        lm_batch((xb, yb))
    data_s = time.perf_counter() - t0

    # forward only: model + loss, no backward, no optimizer step
    model, loss_fn = trainer.model, trainer.loss_fn
    model.train()
    t0 = time.perf_counter()
    for xb, yb in train_loader:
        x, y = lm_batch((xb, yb))
        loss_fn(model(x), y)
    fwd_s = time.perf_counter() - t0

    # full step: forward + backward + clip + optimizer step
    t0 = time.perf_counter()
    trainer._run_batches(train_loader, training=True)
    full_s = time.perf_counter() - t0
    return data_s, fwd_s, full_s


def profile_epoch(trainer, train_loader):
    pr = cProfile.Profile()
    pr.enable()
    trainer._run_batches(train_loader, training=True)
    pr.disable()
    out = StringIO()
    stats = pstats.Stats(pr, stream=out).sort_stats("cumulative")
    stats.print_stats(18)
    return out.getvalue()


def main():
    tok, train_loader, val_loader = build()
    trainer = fresh_trainer(tok)

    print("\n== phase breakdown (seconds per training epoch) ==")
    data_s, fwd_s, full_s = phase_timings(trainer, train_loader)
    bwd_opt_s = full_s - fwd_s
    print(f"  data (loader + batch_fn): {data_s:8.2f}s  "
          f"({100 * data_s / full_s:5.1f}%)")
    print(f"  forward (model + loss):   {fwd_s:8.2f}s  "
          f"({100 * fwd_s / full_s:5.1f}%)")
    print(f"  backward + clip + step:   {bwd_opt_s:8.2f}s  "
          f"({100 * bwd_opt_s / full_s:5.1f}%)")
    print(f"  full epoch:               {full_s:8.2f}s")

    print("\n== cProfile: one training epoch, top 18 by cumulative time ==")
    print(profile_epoch(fresh_trainer(tok), train_loader))
    print("OK — profile complete. No code was changed by this script.")


if __name__ == "__main__":
    main()
