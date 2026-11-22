"""Embed a BPE-encoded sentence (Day 7 example).

Trains the Day 6 BPE tokenizer on the tiny corpus, encodes a sentence,
looks up token embeddings, adds sinusoidal positional encoding, and
prints shapes — the first step of the Month 4 transformer.

    python3 examples/embed_sentence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    BPETokenizer, Embedding, sinusoidal_positional_encoding,
)

HERE = Path(__file__).resolve().parent
SENTENCE = "the model learns to predict the next token"


def main():
    corpus = (HERE / "tiny_corpus.txt").read_text()
    tok = BPETokenizer().train(corpus, vocab_size=320, min_freq=2)
    ids = tok.encode(SENTENCE)
    print(f"sentence: {SENTENCE!r}")
    print(f"token ids ({len(ids)}): {ids}")
    print(f"decoded back: {tok.decode(ids)!r}")
    assert tok.decode(ids) == SENTENCE

    d_model = 16
    emb = Embedding(len(tok), d_model, seed=11)
    token_emb = emb(ids)                      # (seq_len, d_model)
    print(f"token embeddings: shape {token_emb.shape}")
    pe = sinusoidal_positional_encoding(len(ids), d_model)
    print(f"positional encoding: shape {pe.shape}")
    embedded = token_emb + pe                 # (seq_len, d_model)
    print(f"embedded + positions: shape {embedded.shape}")
    print("first token vector (first 6 values):",
          [round(v, 4) for v in embedded.tolist()[0][:6]])
    print("position 0 of the encoding:",
          [round(v, 4) for v in pe.tolist()[0][:6]])
    print("OK — ready to feed into attention (Day 8).")


if __name__ == "__main__":
    main()
