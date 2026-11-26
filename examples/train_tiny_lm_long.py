"""Train the tiny GPT on a longer, multi-paragraph corpus (Day 11).

Same Month 5 loop as Day 10, but the corpus is
``examples/darklearn_story.txt`` — ten original paragraphs about
darklearn itself (~3.3k chars) instead of one sentence repeated
8 times. This run also exercises the Day 11 features together:
the Trainer checkpoint callback (best val loss -> disk) and a
cosine LR schedule stepped per epoch.

Honest limits (same as Day 10, more visible here): the model is
tiny (d_model 16, 2 blocks) and the corpus is small and varied,
so free-running greedy generation repeats and drifts. The example
asserts what is actually verified — loss down, best checkpoint on
disk, identical generation after reload — and prints the greedy
output so you can judge the quality for yourself.

    python3 examples/train_tiny_lm_long.py
"""

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
from darklearn.generate import generate
from darklearn.trainer import EarlyStopping, Trainer

HERE = Path(__file__).resolve().parent
SEQ_LEN = 8
STRIDE = 4
D_MODEL = 16


class TinyGPT(Module):
    """Embedding + 2 causal blocks + Linear head (same shape as Day 10)."""

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
    stopper = EarlyStopping(patience=3, min_delta=0.001)
    ckpt = HERE / "checkpoints" / "tiny_lm_long.pt"
    trainer = Trainer(model, CrossEntropyLoss(), opt, scheduler=sched,
                      max_grad_norm=1.0, early_stopping=stopper,
                      batch_fn=lm_batch, checkpoint_path=str(ckpt))
    t0 = time.time()
    history = trainer.fit(train_loader, epochs=10, val_loader=val_loader,
                          log_every=2)
    print(f"trained in {time.time() - t0:.1f}s, "
          f"{len(history['train_loss'])} epochs, "
          f"best epoch {trainer.best_epoch} "
          f"(val loss {trainer.best_loss:.4f})")
    first, last = history["train_loss"][0], history["train_loss"][-1]
    assert last < first * 0.5, "train loss must go down"
    final_ppl = trainer.perplexity(val_loader)
    print(f"train loss {first:.4f} -> {last:.4f}, "
          f"val perplexity after restore {final_ppl:.2f}")
    # Measured 2026-11-26: ~157. The corpus is varied (384-way vocab,
    # random guessing would be 384) and the model is tiny, so it
    # memorizes train windows faster than it generalizes — val loss
    # is worst after epoch 1 and early stopping restores exactly
    # that best model. The bound below only guards against a
    # broken run (worse than guessing), not against a good one.
    assert final_ppl < 200.0, f"val perplexity too high: {final_ppl}"
    assert ckpt.is_file(), "checkpoint callback must have written a file"
    print(f"best checkpoint on disk: {ckpt}")

    # Reload the best checkpoint into a fresh model: generation must
    # match the trained model token for token.
    model2 = TinyGPT(tok)
    opt2 = Adam(model2.parameters(), lr=0.05)
    meta = load_checkpoint(ckpt, model=model2, optimizer=opt2)
    print(f"checkpoint loaded: epoch={meta['epoch']}, "
          f"extra={meta['extra']}")
    for prompt_text in ("The first piece", "Training is a loop"):
        prompt = tok.encode(prompt_text)
        gen1 = generate(model, prompt, max_new_tokens=12)
        gen2 = generate(model2, prompt, max_new_tokens=12)
        assert gen1 == gen2, "reloaded model must generate identically"
        print(f"prompt {prompt_text!r}:")
        print(f"  greedy: {tok.decode(gen1)!r}")
    print("OK — longer corpus trained, best checkpoint saved + reloaded.")
    print("Note: greedy output repeats and drifts — the model is tiny "
          "(d_model 16, 2 blocks) and the corpus is small and varied. "
          "It learns local word patterns, not real language yet.")


if __name__ == "__main__":
    main()
