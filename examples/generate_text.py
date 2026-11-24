"""Generate text from the trained tiny GPT (Day 9 example).

Trains the Day 8 tiny GPT on "the model learns to predict the next
token" (80 Adam steps, next-token loss ~11.3 -> ~0.0001), then:
- greedy-generate the rest of the sentence from "the model" and
  check it reproduces the training sentence exactly;
- sample a continuation (temperature 0.8, top-k 5, fixed seed) and
  print the decoded text.

    python3 examples/generate_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, BPETokenizer, CrossEntropyLoss, Embedding, Linear,
    TransformerBlock, sinusoidal_positional_encoding,
)
from darklearn.generate import generate

TEXT = "the model learns to predict the next token"


def build_model(tok, d_model=16):
    """Embedding + 2 causal blocks + head; returns (model_fn, params)."""
    emb = Embedding(tok.vocab_size, d_model, seed=1)
    blocks = [TransformerBlock(d_model, 2, seed=10 + i, causal=True)
              for i in range(2)]
    head = Linear(d_model, tok.vocab_size, seed=20)
    params = (emb.parameters() + blocks[0].parameters()
              + blocks[1].parameters() + head.parameters())

    def model_fn(seq_ids):
        x = emb(seq_ids) + sinusoidal_positional_encoding(
            len(seq_ids), d_model)
        for block in blocks:
            x = block(x)
        return head(x)

    return model_fn, params


def main():
    tok = BPETokenizer()
    tok.train(TEXT * 4, vocab_size=320)
    ids = tok.encode(TEXT)
    print(f"tokens: {len(ids)}  vocab: {tok.vocab_size}")

    model_fn, params = build_model(tok)
    loss_fn = CrossEntropyLoss()
    opt = Adam(params, lr=0.05)
    first = None
    for step in range(80):
        logits = model_fn(ids[:-1])
        loss = loss_fn(logits, ids[1:])
        if step == 0:
            first = loss.item()
        for p in params:
            p.zero_grad()
        loss.backward()
        opt.step()
    print(f"next-token loss: {first:.4f} -> {loss.item():.4f}")
    assert loss.item() < 0.01, "tiny GPT must overfit the sentence"

    # Greedy: from the first 3 tokens, the model must reproduce the
    # rest of the sentence exactly.
    prompt = ids[:3]
    print(f"prompt: {tok.decode(prompt)!r}")
    greedy = generate(model_fn, prompt,
                      max_new_tokens=len(ids) - len(prompt))
    print(f"greedy: {tok.decode(greedy)!r}")
    assert greedy == ids, "greedy generation must reproduce the sentence"

    # Sampling: same prompt, temperature 0.8, top-k 5, fixed seed.
    sampled = generate(model_fn, prompt, max_new_tokens=10,
                       method="sample", temperature=0.8, top_k=5, seed=7)
    print(f"sampled (t=0.8, top-k=5, seed=7): {tok.decode(sampled)!r}")
    assert len(sampled) == len(prompt) + 10
    print("OK — tiny GPT generation works.")


if __name__ == "__main__":
    main()
