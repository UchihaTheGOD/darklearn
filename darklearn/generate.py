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

``beam_search()`` keeps the ``beam_width`` best partial sequences
by cumulative log-probability instead of committing to one token
at a time — it can find a better-scoring continuation than greedy
when the greedy first step leads nowhere good.
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


def _log_softmax(logits: list[float]) -> list[float]:
    """Log-softmax over a plain list of floats (max-subtracted, stable)."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    log_total = math.log(sum(exps))
    return [x - m - log_total for x in logits]


def beam_search(model, prompt_ids, max_new_tokens: int = 20,
                beam_width: int = 4, eos_id: int | None = None,
                length_penalty: float = 0.0) -> list[int]:
    """Beam-search generation from a next-token model.

    Same ``model`` contract as :func:`generate` (``model(ids)`` ->
    ``(len(ids), vocab)`` logits). Keeps the ``beam_width`` best
    partial sequences by cumulative log-probability, expanding each
    beam by its ``beam_width`` most likely next tokens each step.
    A beam that emits ``eos_id`` is finished and kept aside; the
    search stops early when every beam has finished. Returns the
    prompt plus the best-scoring continuation. With
    ``beam_width=1`` this is exactly greedy decoding.
    ``length_penalty`` divides a beam's score by
    ``(generated tokens) ** length_penalty`` before the final
    pick — 0.0 (the default) compares raw cumulative log-probs,
    which mildly favours short sequences; a penalty of 1.0
    compares mean log-prob per token instead.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError(f"beam_width must be a positive int, "
                         f"got {beam_width}")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative int")
    if length_penalty < 0.0:
        raise ValueError(f"length_penalty must be >= 0, "
                         f"got {length_penalty}")
    ids = [int(i) for i in prompt_ids]
    if not ids:
        raise ValueError("beam_search needs a non-empty prompt_ids")
    # (cumulative log-prob, token ids); the prompt starts at 0.0.
    beams: list[tuple[float, list[int]]] = [(0.0, list(ids))]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(max_new_tokens):
        candidates: list[tuple[float, list[int]]] = []
        for score, seq in beams:
            logits = model(seq)
            if len(logits.shape) != 2 or logits.shape[0] != len(seq):
                raise ValueError(
                    f"model must return (len(ids), vocab) logits, got "
                    f"shape {logits.shape} for {len(seq)} ids")
            logp = _log_softmax([float(x) for x in logits[-1].tolist()])
            order = sorted(range(len(logp)), key=lambda i: logp[i],
                           reverse=True)
            for nxt in order[:beam_width]:
                candidates.append((score + logp[nxt], seq + [nxt]))
        # Stable sort: ties keep the order the beams were made in,
        # so the result is deterministic.
        candidates.sort(key=lambda c: c[0], reverse=True)
        beams = []
        for score, seq in candidates[:beam_width]:
            if eos_id is not None and seq[-1] == int(eos_id):
                finished.append((score, seq))
            else:
                beams.append((score, seq))
        if not beams:
            break
    pool = finished + beams
    if length_penalty:
        def _key(candidate):
            score, seq = candidate
            generated = max(1, len(seq) - len(ids))
            return score / (generated ** length_penalty)
    else:
        def _key(candidate):
            return candidate[0]
    return max(pool, key=_key)[1]


__all__ = ["generate", "beam_search"]
