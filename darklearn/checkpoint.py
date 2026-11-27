"""darklearn.checkpoint — save / load model + optimizer state (stdlib).

Weights are stored as plain Python data (``pickle``): a dict of
``{dotted_name: {"shape": tuple, "data": [floats]}}`` from
``Module.named_parameters()``, plus the optimizer's ``state_dict()``
when one is given, the epoch, and any extra metadata the caller
wants (e.g. loss, config). No tensors, no numpy, no torch — a
checkpoint is just numbers, so it loads on any machine with
darklearn.

    save_checkpoint("model.pt", model, optimizer=opt, epoch=3)
    meta = load_checkpoint("model.pt", model=model, optimizer=opt)
    # meta == {"epoch": 3, "extra": {...}}; weights + optimizer state
    # are now back in the (freshly built) model and optimizer.

A tokenizer can travel in the same file, so one artifact is the
whole runnable model — pass ``tokenizer=tok`` to
``save_checkpoint`` and again to ``load_checkpoint`` with a fresh
``BPETokenizer``. The tokenizer state is plain data in the same
``darklearn-bpe-v1`` format ``BPETokenizer.save`` writes.
"""

from __future__ import annotations

import pickle

FORMAT = "darklearn-checkpoint-v1"


def state_dict(model) -> dict:
    """``{name: {"shape": ..., "data": [floats]}}`` for every
    parameter in ``model`` (see ``Module.named_parameters``)."""
    state = {}
    for name, p in model.named_parameters():
        state[name] = {"shape": tuple(p.shape),
                       "data": [float(x) for x in p._data]}
    return state


def load_state_dict(model, state: dict) -> None:
    """Copy ``state`` (from ``state_dict``) into ``model`` in place —
    the parameters keep their identity and ``requires_grad`` flag, so
    an optimizer built over them keeps working. Strict: unknown keys,
    missing keys and shape mismatches all raise ``ValueError``."""
    named = dict(model.named_parameters())
    for name in state:
        if name not in named:
            raise ValueError(f"checkpoint has unknown parameter {name!r}")
    for name, p in named.items():
        if name not in state:
            raise ValueError(f"checkpoint is missing parameter {name!r}")
        entry = state[name]
        shape = tuple(entry["shape"])
        data = entry["data"]
        if shape != tuple(p.shape):
            raise ValueError(f"checkpoint shape {shape} for {name!r} "
                             f"does not match model shape {tuple(p.shape)}")
        if len(data) != p.numel:
            raise ValueError(f"checkpoint data for {name!r} has "
                             f"{len(data)} values, expected {p.numel}")
        for i, value in enumerate(data):
            p._data[i] = float(value)


def tokenizer_state(tokenizer) -> dict:
    """Plain-data snapshot of a ``BPETokenizer`` (vocab + merges),
    in the same ``darklearn-bpe-v1`` format ``BPETokenizer.save``
    writes. Duck-typed — any object with ``vocab`` (id -> bytes)
    and ``merges`` (list of ``(a, b)`` pairs) works."""
    return {
        "format": "darklearn-bpe-v1",
        "vocab": {str(token_id): list(token_bytes)
                  for token_id, token_bytes in tokenizer.vocab.items()},
        "merges": [[a, b] for a, b in tokenizer.merges],
    }


def load_tokenizer_state(tokenizer, state: dict) -> None:
    """Restore a snapshot from :func:`tokenizer_state` into
    ``tokenizer`` in place (vocab, merges and merge ranks). A state
    that is not a ``darklearn-bpe-v1`` tokenizer snapshot raises
    ``ValueError``."""
    if not isinstance(state, dict) or state.get("format") != "darklearn-bpe-v1":
        raise ValueError("not a darklearn BPE tokenizer state")
    tokenizer.vocab = {int(token_id): bytes(token_bytes)
                       for token_id, token_bytes in state["vocab"].items()}
    tokenizer.merges = [(int(a), int(b)) for a, b in state["merges"]]
    tokenizer.merge_ranks = {pair: rank
                             for rank, pair in enumerate(tokenizer.merges)}


def save_checkpoint(path, model, optimizer=None, epoch: int | None = None,
                    extra: dict | None = None, tokenizer=None) -> dict:
    """Write a checkpoint file; returns the dict that was written.
    ``tokenizer`` (a ``BPETokenizer``) is snapshotted into the same
    file when given, so one artifact holds model + tokenizer."""
    payload = {
        "format": FORMAT,
        "model": state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "extra": dict(extra) if extra is not None else {},
        "tokenizer": (tokenizer_state(tokenizer)
                      if tokenizer is not None else None),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.DEFAULT_PROTOCOL)
    return payload


def load_checkpoint(path, model=None, optimizer=None, tokenizer=None) -> dict:
    """Read a checkpoint file. When ``model`` is given its weights
    are restored (strict — see ``load_state_dict``); when
    ``optimizer`` is given its state is restored too; when
    ``tokenizer`` is given its vocab + merges are restored. Returns
    ``{"epoch": ..., "extra": ...}``. A missing file raises
    ``FileNotFoundError``; a file that is not a darklearn checkpoint
    raises ``ValueError``; asking for optimizer or tokenizer state
    the file does not hold raises ``ValueError`` too."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path!r} is not a {FORMAT} checkpoint")
    if model is not None:
        load_state_dict(model, payload["model"])
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state to load")
        optimizer.load_state_dict(payload["optimizer"])
    if tokenizer is not None:
        state = payload.get("tokenizer")
        if state is None:
            raise ValueError("checkpoint has no tokenizer to load")
        load_tokenizer_state(tokenizer, state)
    return {"epoch": payload.get("epoch"), "extra": payload.get("extra", {})}


__all__ = ["FORMAT", "state_dict", "load_state_dict",
           "tokenizer_state", "load_tokenizer_state",
           "save_checkpoint", "load_checkpoint"]
