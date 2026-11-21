"""Train a byte-level BPE tokenizer on a small text file and show it working.

Usage:
    python3 examples/train_bpe.py [vocab_size]

Trains on examples/tiny_corpus.txt, prints the vocabulary size, the
first merges learned, a sample encode/decode, and a token-id
TextDataset batch through the DataLoader (the input for the tiny LM).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import BPETokenizer, DataLoader, TextDataset

CORPUS_PATH = Path(__file__).resolve().parent / "tiny_corpus.txt"


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    text = CORPUS_PATH.read_text(encoding="utf-8")
    tok = BPETokenizer().train(text, vocab_size=vocab_size, min_freq=2)
    print(f"corpus: {len(text)} chars, {len(text.split())} words")
    print(f"vocab size: {len(tok)} (256 bytes + {tok.num_merges} merges)")
    print("first merges:",
          [tok.vocab[256 + i] for i in range(min(10, tok.num_merges))])

    sample = "the model learns to predict the next token"
    ids = tok.encode(sample)
    print(f"\nencode: {sample!r}")
    print(f"  -> {ids} ({len(ids)} tokens for {len(sample)} chars)")
    print(f"decode back: {tok.decode(ids)!r}")
    assert tok.decode(ids) == sample

    # Token ids straight into the LM's dataset + loader.
    ds = TextDataset.from_token_ids(tok.encode(text), seq_len=16)
    loader = DataLoader(ds, batch_size=4, shuffle=True, seed=0)
    xb, yb = next(iter(loader))
    print(f"\nTextDataset: {len(ds)} windows, vocab_size {ds.vocab_size}")
    print(f"DataLoader batch: x {xb.shape}, y {yb.shape}")
    print("first x row decoded:",
          repr(tok.decode([int(v) for v in xb.tolist()[0]])))

    saved = Path(__file__).resolve().parent / "bpe_vocab.json"
    tok.save(saved)
    print(f"\nsaved vocab to {saved.name} "
          f"({BPETokenizer.load(saved).vocab_size} tokens on reload)")


if __name__ == "__main__":
    main()
