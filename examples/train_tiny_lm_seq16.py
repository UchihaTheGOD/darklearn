"""Train the tiny GPT with seq_len 16 and compare with seq_len 8 (Day 12).

Same corpus (``examples/darklearn_story.txt``), same model shape
(d_model 16, 2 blocks), same seeds / lr / cosine schedule as Day 11 —
the only change is the context window: 16 tokens instead of 8. The
Day 11 seq-8 numbers are printed next to the measured seq-16 ones
so you can see whether the longer window helps or hurts on this
small corpus. This run also exercises two Day 12 features together:
the Trainer's checkpoint callback now saves the tokenizer into the
same file (one artifact = model + tokenizer), and beam search is
compared with greedy generation on the trained model.

    python3 examples/train_tiny_lm_seq16.py
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, BPETokenizer, CosineAnnealingLR, CrossEntropyLoss, DataLoader,
    Embedding, Linear, Module, Tensor, TextDataset, TransformerBlock,
    split_dataset, sinusoidal_positional_encoding,
)
from darklearn.checkpoint import load_checkpoint
from darklearn.generate import beam_search, generate
from darklearn.trainer import EarlyStopping, Trainer

HERE = Path(__file__).resolve().parent
SEQ_LEN = 16
STRIDE = 4
D_MODEL = 16

# Day 11, seq_len 8, measured 2026-11-26 on the same corpus/model/seeds:
# train 7.0625 -> 3.0393 in 4 epochs, best epoch 1, val loss 5.0573,
# val perplexity 157.16 (random guessing on the 384-way vocab is 384).
SEQ8 = {"train_first": 7.0625, "train_last": 3.0393, "epochs": 4,
        "best_epoch": 1, "val_loss": 5.0573, "val_ppl": 157.16}


class TinyGPT(Module):
    """Embedding + 2 causal blocks + Linear head (same shape as Day 10/11)."""

    def __init__(self, tok, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = tok.vocab_size
        self.emb = Embedding(tok.vocab_size, d_model, seed=1)
        self.blocks = [TransformerBlock(d_model, 2, seed=10 + i, causal=True)
                       for i in range(2)]
        self.head = Linear(d_model, tok.vocab_size, seed=20)

    def _single(self, seq_ids):
        x = self.emb(seq_ids) + sinusoidal_positional_encoding(
            len(seq_ids), self.d_model)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def forward(self, seq_ids):
        if isinstance(seq_ids, Tensor):
            rows = seq_ids.tolist()
            outs = [self._single([int(t) for t in row])[-1]
                    .reshape((1, self.vocab_size)) for row in rows]
            return Tensor.cat(outs, dim=0)
        return self._single(seq_ids)


def lm_batch(batch):
    x, y = batch
    return x, [int(row[0]) for row in y.tolist()]


def seq_logp(model, ids):
    """Total log-probability the model assigns to ``ids`` (sum over
    each next-token log-softmax). Used to compare greedy vs beam
    continuations on the same footing beam search optimises."""
    total = 0.0
    for i in range(1, len(ids)):
        row = [float(x) for x in model(ids[:i])[-1].tolist()]
        m = max(row)
        total += row[ids[i]] - m - math.log(
            sum(math.exp(x - m) for x in row))
    return total


def main():
    text = (HERE / "darklearn_story.txt").read_text()
    tok = BPETokenizer()
    tok.train(text, vocab_size=384)
    ids = tok.encode(text)
    print(f"corpus: {len(text)} chars -> {len(ids)} tokens  "
          f"vocab: {tok.vocab_size}")

    ds = TextDataset.from_token_ids(ids, seq_len=SEQ_LEN, stride=STRIDE)
    train_ds, val_ds = split_dataset(ds, val_fraction=0.2, seed=0)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, seed=0)
    val_loader = DataLoader(val_ds, batch_size=8)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    model = TinyGPT(tok)
    opt = Adam(model.parameters(), lr=0.05)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.005)
    stopper = EarlyStopping(patience=2, min_delta=0.001)
    ckpt = HERE / "checkpoints" / "tiny_lm_seq16.pt"
    trainer = Trainer(model, CrossEntropyLoss(), opt, scheduler=sched,
                      max_grad_norm=1.0, early_stopping=stopper,
                      batch_fn=lm_batch, checkpoint_path=str(ckpt),
                      tokenizer=tok)
    t0 = time.time()
    history = trainer.fit(train_loader, epochs=6, val_loader=val_loader,
                          log_every=1)
    print(f"trained in {time.time() - t0:.1f}s, "
          f"{len(history['train_loss'])} epochs, "
          f"best epoch {trainer.best_epoch} "
          f"(val loss {trainer.best_loss:.4f})")
    first, last = history["train_loss"][0], history["train_loss"][-1]
    assert last < first * 0.8, "train loss must go down"
    final_ppl = trainer.perplexity(val_loader)
    print(f"train loss {first:.4f} -> {last:.4f}, "
          f"val perplexity after restore {final_ppl:.2f}")
    # Better than uniform guessing over the vocab (384) is the only
    # hard bound: with 16-token windows the model still memorises
    # train windows faster than it generalises (same as Day 11).
    assert final_ppl < 384.0, f"val perplexity worse than guessing: {final_ppl}"
    assert ckpt.is_file(), "checkpoint callback must have written a file"
    print(f"best checkpoint on disk: {ckpt}")

    print("\nseq_len comparison (same corpus, model, seeds, lr):")
    print(f"  {'':10} {'seq 8 (Day 11)':>16} {'seq 16 (today)':>16}")
    print(f"  {'train loss':10} "
          f"{SEQ8['train_first']:.4f} -> {SEQ8['train_last']:.4f}  "
          f"{first:.4f} -> {last:.4f}")
    print(f"  {'epochs':10} {SEQ8['epochs']:>16} "
          f"{len(history['train_loss']):>16}")
    print(f"  {'best epoch':10} {SEQ8['best_epoch']:>16} "
          f"{trainer.best_epoch:>16}")
    print(f"  {'val loss':10} {SEQ8['val_loss']:>16.4f} "
          f"{trainer.best_loss:>16.4f}")
    print(f"  {'val ppl':10} {SEQ8['val_ppl']:>16.2f} {final_ppl:>16.2f}")

    # Reload the best checkpoint — model AND tokenizer — into fresh
    # objects: generation must match the trained model token for token.
    # (model2 is built from the original tokenizer only so its shapes
    # match; the tokenizer itself is restored from the file into tok2.)
    model2 = TinyGPT(tok)
    opt2 = Adam(model2.parameters(), lr=0.05)
    tok2 = BPETokenizer()
    meta = load_checkpoint(ckpt, model=model2, optimizer=opt2,
                           tokenizer=tok2)
    print(f"\ncheckpoint loaded: epoch={meta['epoch']}, "
          f"extra={meta['extra']}, tokenizer vocab={tok2.vocab_size}")
    assert tok2.merges == tok.merges, "tokenizer must round-trip"
    for prompt_text in ("The first piece", "Training is a loop"):
        prompt = tok.encode(prompt_text)
        gen_greedy = generate(model, prompt, max_new_tokens=12)
        gen_beam = beam_search(model, prompt, max_new_tokens=12,
                               beam_width=4)
        gen2_greedy = generate(model2, prompt, max_new_tokens=12)
        gen2_beam = beam_search(model2, prompt, max_new_tokens=12,
                                beam_width=4)
        assert gen_greedy == gen2_greedy, "reloaded model must match (greedy)"
        assert gen_beam == gen2_beam, "reloaded model must match (beam)"
        print(f"prompt {prompt_text!r}:")
        print(f"  greedy: {tok.decode(gen_greedy)!r} "
              f"(logp {seq_logp(model, gen_greedy):.2f})")
        print(f"  beam-4: {tok.decode(gen_beam)!r} "
              f"(logp {seq_logp(model, gen_beam):.2f})")
    print("OK — seq_len 16 trained, best checkpoint (model + tokenizer) "
          "saved + reloaded, greedy vs beam compared.")


if __name__ == "__main__":
    main()
