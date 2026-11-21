"""darklearn.tokenizer — byte-level BPE tokenizer, from scratch (stdlib only).

Byte-level BPE (the GPT-2 style): the base vocabulary is the 256 bytes,
so any text — any language, any punctuation — can be encoded with no
unknown tokens. ``train`` learns merges from a corpus; ``encode`` /
``decode`` round-trip exactly.

Training works word by word (whitespace split), so merges never cross a
word boundary: the space byte stays its own token. ``encode`` applies the
learned merges in rank order (lowest rank first) inside each word and
emits whitespace runs as raw byte tokens, so any text round-trips.
"""

from __future__ import annotations

import json
import re

__all__ = ["BPETokenizer"]


class BPETokenizer:
    """Byte-level byte-pair-encoding tokenizer.

    ``vocab`` maps token id -> bytes. Ids 0..255 are the single bytes;
    ids >= 256 are merges learned by :meth:`train`, in learned order.
    ``merges`` is the ordered list of ``(a, b)`` id pairs; a pair's
    position in that list is its rank (lower = merged earlier).
    """

    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_merges(self) -> int:
        return len(self.merges)

    def train(self, corpus, vocab_size: int = 512, min_freq: int = 2):
        """Learn merges from ``corpus`` (a string or an iterable of strings).

        Repeatedly finds the most frequent adjacent token pair across
        all words (weighted by word count) and merges it into a new
        token, until the vocabulary reaches ``vocab_size`` or no pair
        occurs at least ``min_freq`` times. Deterministic: ties are
        broken by first-seen order, so the same corpus always gives
        the same merges. Returns ``self`` so calls can chain.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte tokens)")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        texts = [corpus] if isinstance(corpus, str) else list(corpus)

        # Word counts: {word-bytes-tuple: count}. Words are whitespace
        # split, so merges never cross a word boundary.
        word_counts: dict[tuple[int, ...], int] = {}
        for text in texts:
            for word in re.findall(r"\S+", text):
                key = tuple(word.encode("utf-8"))
                word_counts[key] = word_counts.get(key, 0) + 1
        words = {word: list(word) for word in word_counts}

        while len(self.vocab) < vocab_size:
            pair_counts: dict[tuple[int, int], int] = {}
            for word, tokens in words.items():
                count = word_counts[word]
                for a, b in zip(tokens, tokens[1:]):
                    pair = (a, b)
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
            if not pair_counts:
                break
            # max() keeps the first max on ties (dicts preserve
            # insertion order), so training is deterministic.
            best_pair, best_freq = max(pair_counts.items(),
                                       key=lambda kv: kv[1])
            if best_freq < min_freq:
                break
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            for tokens in words.values():
                self._merge_in_place(tokens, best_pair, new_id)
        return self

    @staticmethod
    def _merge_in_place(tokens, pair, new_id):
        """Replace every occurrence of ``pair`` in ``tokens`` with ``new_id``."""
        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens[:] = out

    def _apply_merges(self, tokens):
        """Apply the learned merges to one word's token list, rank order."""
        if len(tokens) < 2 or not self.merges:
            return list(tokens)
        tokens = list(tokens)
        while len(tokens) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            tokens[best_idx:best_idx + 2] = [256 + best_rank]
        return tokens

    def encode(self, text: str):
        """Encode ``text`` to a list of token ids."""
        if not isinstance(text, str):
            raise TypeError("BPETokenizer.encode needs a str")
        ids = []
        for chunk in re.split(r"(\s+)", text):
            if not chunk:
                continue
            if chunk.isspace():
                # Whitespace runs stay raw bytes (never merged).
                ids.extend(chunk.encode("utf-8"))
            else:
                ids.extend(self._apply_merges(chunk.encode("utf-8")))
        return ids

    def decode(self, ids):
        """Decode a list of token ids back to a string (exact inverse)."""
        parts = []
        for token_id in ids:
            token = self.vocab.get(int(token_id))
            if token is None:
                raise ValueError(f"unknown token id {token_id}")
            parts.append(token)
        return b"".join(parts).decode("utf-8")

    def save(self, path):
        """Save the vocabulary + merges as JSON (``path`` str or Path)."""
        data = {
            "format": "darklearn-bpe-v1",
            "vocab": {str(token_id): list(token_bytes)
                      for token_id, token_bytes in self.vocab.items()},
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a tokenizer saved with :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != "darklearn-bpe-v1":
            raise ValueError(f"not a darklearn BPE file: {path}")
        tok = cls()
        tok.vocab = {int(token_id): bytes(token_bytes)
                     for token_id, token_bytes in data["vocab"].items()}
        tok.merges = [(int(a), int(b)) for a, b in data["merges"]]
        tok.merge_ranks = {pair: rank
                           for rank, pair in enumerate(tok.merges)}
        return tok
