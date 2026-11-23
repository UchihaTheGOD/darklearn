"""Tiny GPT-style forward pass (Day 8 example).

BPE-encodes a sentence, embeds it, adds sinusoidal positions, runs
it through two TransformerBlocks (causal), and trains a Linear head
to predict the next token for a few steps. Prints the shape after
each block and the loss before/after.

    python3 examples/tiny_gpt.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, BPETokenizer, CrossEntropyLoss, Embedding, Linear,
    TransformerBlock, sinusoidal_positional_encoding,
)

TEXT = "the model learns to predict the next token"


def main():
    tok = BPETokenizer()
    tok.train(TEXT * 4, vocab_size=320)
    ids = tok.encode(TEXT)
    print(f"tokens: {len(ids)}  vocab: {tok.vocab_size}")
    print(f"decoded back: {tok.decode(ids)!r}")
    assert tok.decode(ids) == TEXT

    d_model = 16
    emb = Embedding(tok.vocab_size, d_model, seed=1)
    blocks = [TransformerBlock(d_model, 2, seed=10 + i, causal=True)
              for i in range(2)]
    head = Linear(d_model, tok.vocab_size, seed=20)
    loss_fn = CrossEntropyLoss()

    x = emb(ids) + sinusoidal_positional_encoding(len(ids), d_model)
    print(f"embeddings: {x.shape}")
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"after block {i + 1}: {x.shape}")
    logits = head(x)
    print(f"logits: {logits.shape}")
    assert logits.shape == (len(ids), tok.vocab_size)

    # Next-token training: predict ids[1:] from positions [:-1].
    params = (emb.parameters() + blocks[0].parameters()
              + blocks[1].parameters() + head.parameters())
    opt = Adam(params, lr=0.05)
    first = None
    for step in range(80):
        x = emb(ids[:-1]) + sinusoidal_positional_encoding(
            len(ids) - 1, d_model)
        for block in blocks:
            x = block(x)
        loss = loss_fn(head(x), ids[1:])
        if step == 0:
            first = loss.item()
        for p in params:
            p.zero_grad()
        loss.backward()
        opt.step()
    print(f"next-token loss: {first:.4f} -> {loss.item():.4f}")
    assert loss.item() < first * 0.5, "loss must go down"
    print("OK — tiny GPT forward + training run works.")


if __name__ == "__main__":
    main()
