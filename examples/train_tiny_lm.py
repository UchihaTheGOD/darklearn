"""Train the tiny GPT on a small corpus with the library Trainer (Day 10).

End-to-end Month 5 loop: BPE tokenizer -> token-id TextDataset ->
train/val split -> Trainer (eval loop, perplexity, early stopping) ->
checkpoint save -> reload into a fresh model -> generate from it.

The checkpoint is written to ``examples/checkpoints/tiny_lm.pt``
(gitignored — ``checkpoints/`` and ``*.pt`` are in ``.gitignore``).

    python3 examples/train_tiny_lm.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, BPETokenizer, CrossEntropyLoss, DataLoader, Embedding, Linear,
    Module, Tensor, TextDataset, TransformerBlock, split_dataset,
    sinusoidal_positional_encoding,
)
from darklearn.checkpoint import load_checkpoint, save_checkpoint
from darklearn.generate import generate
from darklearn.trainer import EarlyStopping, Trainer

TEXT = "the model learns to predict the next token"
REPEATS = 8
SEQ_LEN = 8
D_MODEL = 16


class TinyGPT(Module):
    """Embedding + 2 causal blocks + Linear head, as one Module.

    ``forward`` takes either a list of token ids (one sequence, the
    shape ``generate`` uses) or a (batch, seq) id Tensor (the shape
    the Trainer feeds it): each row is run through the single-sequence
    path, the last position's logits are kept, and the rows are
    stacked with ``Tensor.cat`` into (batch, vocab) next-token logits.
    """

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
    """DataLoader (x, y) -> (x, target ids): x stays a (batch, seq)
    id Tensor for TinyGPT; y (batch, 1) becomes a list of ints for
    CrossEntropyLoss."""
    x, y = batch
    return x, [int(row[0]) for row in y.tolist()]


def main():
    tok = BPETokenizer()
    tok.train(TEXT * REPEATS, vocab_size=320)
    ids = tok.encode(TEXT * REPEATS)
    print(f"corpus: {len(ids)} tokens  vocab: {tok.vocab_size}")

    ds = TextDataset.from_token_ids(ids, seq_len=SEQ_LEN, stride=1)
    train_ds, val_ds = split_dataset(ds, val_fraction=0.2, seed=0)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, seed=0)
    val_loader = DataLoader(val_ds, batch_size=4)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    model = TinyGPT(tok)
    opt = Adam(model.parameters(), lr=0.05)
    stopper = EarlyStopping(patience=5, min_delta=0.001)
    trainer = Trainer(model, CrossEntropyLoss(), opt,
                      max_grad_norm=1.0, early_stopping=stopper,
                      batch_fn=lm_batch)
    t0 = time.time()
    history = trainer.fit(train_loader, epochs=30, val_loader=val_loader,
                          log_every=5)
    print(f"trained in {time.time() - t0:.1f}s, "
          f"{len(history['train_loss'])} epochs")
    first, last = history["train_loss"][0], history["train_loss"][-1]
    assert last < first * 0.5, "train loss must go down"
    # Re-evaluate after early stopping: the history's last val entry
    # is from before the restore, and the restored model is the best
    # one (on this run the val loss hit ~0.0000 at epoch 10, then one
    # more Adam epoch at lr=0.05 bounced it up — the restore is what
    # saves the good weights).
    final_ppl = trainer.perplexity(val_loader)
    print(f"train loss {first:.4f} -> {last:.4f}, "
          f"val perplexity after restore {final_ppl:.2f}")
    assert final_ppl < 5.0, f"val perplexity too high: {final_ppl}"

    ckpt = Path(__file__).resolve().parent / "checkpoints" / "tiny_lm.pt"
    ckpt.parent.mkdir(exist_ok=True)
    save_checkpoint(ckpt, model, optimizer=opt,
                    epoch=len(history["train_loss"]),
                    extra={"val_perplexity": final_ppl})
    print(f"checkpoint saved: {ckpt}")

    # Reload into a fresh model and check the weights — and the
    # generation — are identical to the trained one.
    model2 = TinyGPT(tok)
    opt2 = Adam(model2.parameters(), lr=0.05)
    meta = load_checkpoint(ckpt, model=model2, optimizer=opt2)
    print(f"checkpoint loaded: epoch={meta['epoch']}, "
          f"extra={meta['extra']}")
    prompt = tok.encode("the model")
    gen1 = generate(model, prompt, max_new_tokens=9)
    gen2 = generate(model2, prompt, max_new_tokens=9)
    assert gen1 == gen2, "reloaded model must generate identically"
    print(f"prompt: {tok.decode(prompt)!r}")
    print(f"greedy (trained):  {tok.decode(gen1)!r}")
    print(f"greedy (reloaded): {tok.decode(gen2)!r}")
    print("OK — tiny LM trained, checkpointed, reloaded, generating.")


if __name__ == "__main__":
    main()
