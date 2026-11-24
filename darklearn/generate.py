"""darklearn.generate — greedy + sampling text generation (pure stdlib).

``generate()`` turns any next-token model into text. A model is any
callable ``model(ids) -> Tensor`` of shape ``(len(ids), vocab)`` whose
last row holds the logits for the next token — e.g. a tiny GPT built
from ``Embedding`` + ``TransformerBlock`` + a ``Linear`` head (see
``examples/generate_text.py``). Generation itself needs no gradients:
the model is called once per new token and only the last row of
logits is read.

Methods:
- ``"greedy"`` — always take the argmax (first max on ties).
- ``"sample"`` — divide logits by ``temperature``, optionally keep
  only the ``top_k`` largest logits, softmax, and draw one token
  with ``random.Random(seed)`` so a fixed seed reproduces the run.
"""

from __future__ import annotations

import math
import random

from .tensor import Tensor


def _softmax(logits: list[float]) -> list[float]:
    """Softmax over a plain list of floats (max-subtracted, stable)."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _sample(probs: list[float], rng: random.Random) -> int:
    """Draw one index from a categorical distribution (cumulative)."""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1  # rounding: land on the last token


def generate(model, prompt_ids, max_new_tokens: int = 20,
             method: str = "greedy", temperature: float = 1.0,
             top_k: int | None = None, seed: int | None = None,
             eos_id: int | None = None) -> list[int]:
    """Generate token ids from a next-token model.

    ``model`` is called as ``model(ids)`` and must return a 2-d
    ``Tensor`` of shape ``(len(ids), vocab)``. ``prompt_ids`` is a
    non-empty list of token ids; the return value is the prompt plus
    the generated ids. ``max_new_tokens`` caps how many are added.
    ``temperature`` must be > 0 (only used when sampling); ``top_k``
    keeps only the ``top_k`` largest logits before sampling (``None``
    keeps the whole vocab). Generation stops early after emitting
    ``eos_id`` when one is given.
    """
    if method not in ("greedy", "sample"):
        raise ValueError(f"unknown generate method {method!r}: "
                         f"use 'greedy' or 'sample'")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative int")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be a positive int or None, got {top_k}")
    ids = [int(i) for i in prompt_ids]
    if not ids:
        raise ValueError("generate needs a non-empty prompt_ids")
    rng = random.Random(seed)

    for _ in range(max_new_tokens):
        logits = model(ids)
        if len(logits.shape) != 2 or logits.shape[0] != len(ids):
            raise ValueError(
                f"model must return (len(ids), vocab) logits, got "
                f"shape {logits.shape} for {len(ids)} ids")
        row = [float(x) for x in logits[-1].tolist()]
        if method == "greedy":
            nxt = max(range(len(row)), key=lambda i: row[i])
        else:
            scaled = [x / temperature for x in row]
            if top_k is not None and top_k < len(scaled):
                # Keep the k largest logits; mask the rest to -inf so
                # the softmax gives them zero probability.
                threshold = sorted(scaled, reverse=True)[top_k - 1]
                scaled = [x if x >= threshold else float("-inf")
                          for x in scaled]
            nxt = _sample(_softmax(scaled), rng)
        ids.append(nxt)
        if eos_id is not None and nxt == int(eos_id):
            break
    return ids


__all__ = ["generate"]
