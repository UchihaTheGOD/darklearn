"""Day 12 tests — Month 5: Trainer resume, beam search, tokenizer in
checkpoints (stdlib only, plain `python3` runner)."""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darklearn import (
    Adam, BPETokenizer, DataLoader, Linear, MSELoss, ReLU, Sequential,
    Tensor, TensorDataset,
)
from darklearn.checkpoint import (
    load_checkpoint, load_tokenizer_state, save_checkpoint, state_dict,
    tokenizer_state,
)
from darklearn.generate import beam_search, generate
from darklearn.trainer import EarlyStopping, Trainer


def _mlp():
    return Sequential(Linear(2, 8, seed=1), ReLU(), Linear(8, 1, seed=2))


def _mlp_data(n=64, seed=0):
    import random
    rng = random.Random(seed)
    xs = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    ys = [x[0] + 2.0 * x[1] for x in xs]
    return TensorDataset(xs, ys)


def _split(data, seed=0):
    from darklearn import split_dataset
    return split_dataset(data, val_fraction=0.25, seed=seed)


def test_trainer_resume_matches():
    # 6 epochs straight == 3 epochs, then fit(resume_from=...) 3 more:
    # same final loss and byte-identical weights (Adam moments resume
    # too, and the seeded loader repeats the same batches).
    def run(epochs, resume_from=None):
        model = _mlp()
        loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
        opt = Adam(model.parameters(), lr=0.05)
        trainer = Trainer(model, MSELoss(), opt)
        h = trainer.fit(loader, epochs=epochs, log_every=99,
                        resume_from=resume_from)
        return h["train_loss"][-1], state_dict(model)

    loss_full, sd_full = run(6)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        model = _mlp()
        loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
        opt = Adam(model.parameters(), lr=0.05)
        Trainer(model, MSELoss(), opt).fit(loader, epochs=3, log_every=99)
        save_checkpoint(path, model, optimizer=opt, epoch=3)
        loss_resumed, sd_resumed = run(3, resume_from=path)
    assert abs(loss_full - loss_resumed) < 1e-6, \
        f"resumed loss {loss_resumed} != full {loss_full}"
    for k in sd_full:
        assert sd_full[k]["data"] == sd_resumed[k]["data"], k


def test_trainer_resume_restores_best():
    # The checkpoint callback's best file must survive a resumed run
    # whose epochs are all worse: best_loss/best_epoch are restored
    # from the checkpoint, and early stopping restarts from the
    # restored best instead of from scratch.
    model = _mlp()
    train_ds, val_ds = _split(_mlp_data())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, seed=0)
    val_loader = DataLoader(val_ds, batch_size=16)
    opt = Adam(model.parameters(), lr=0.05)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pt")
        trainer = Trainer(model, MSELoss(), opt, checkpoint_path=path)
        trainer.fit(train_loader, epochs=4, val_loader=val_loader,
                    log_every=99)
        saved_epoch = trainer.best_epoch
        saved_loss = trainer.best_loss
        meta = load_checkpoint(path, model=None)
        assert meta["epoch"] == saved_epoch

        # A zero-epoch resumed fit still loads the checkpoint and
        # restores the best tracking — the direct check.
        model0 = _mlp()
        stopper0 = EarlyStopping(patience=2, min_delta=0.0)
        trainer0 = Trainer(model0, MSELoss(),
                           Adam(model0.parameters(), lr=0.05),
                           early_stopping=stopper0, checkpoint_path=path)
        trainer0.fit(train_loader, epochs=0, val_loader=val_loader,
                     log_every=99, resume_from=path)
        assert trainer0.best_loss == saved_loss
        assert trainer0.best_epoch == saved_epoch
        assert stopper0.best_value == saved_loss
        assert stopper0.best_state is not None

        # Resume with a huge weight decay (not part of the saved
        # optimizer state, so it survives the restore): every
        # resumed epoch is worse, so nothing may overwrite the
        # best file, and early stopping (patience 2) must fire
        # after 2 epochs and put the checkpoint weights back.
        # Note: lr=0.0 would NOT freeze the run — resume restores
        # the saved lr (0.05) from the checkpoint, as it should.
        model2 = _mlp()
        opt2 = Adam(model2.parameters(), lr=0.05, weight_decay=10.0)
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        trainer2 = Trainer(model2, MSELoss(), opt2,
                           early_stopping=stopper, checkpoint_path=path)
        history = trainer2.fit(train_loader, epochs=10,
                               val_loader=val_loader, log_every=99,
                               resume_from=path)
        assert trainer2.best_loss == saved_loss, \
            f"best_loss moved: {trainer2.best_loss} != {saved_loss}"
        assert trainer2.best_epoch == saved_epoch
        assert len(history["train_loss"]) == 2, \
            f"early stopping should fire after 2 bad epochs, ran " \
            f"{len(history['train_loss'])}"
        assert stopper.stopped is True
        meta2 = load_checkpoint(path, model=None)
        assert meta2["epoch"] == saved_epoch, \
            "worse resumed epochs must not overwrite the best file"
        # The restored weights are the checkpoint's weights.
        assert abs(trainer2.evaluate(val_loader) - saved_loss) < 1e-6


def test_trainer_resume_no_optimizer_state():
    # A checkpoint saved without an optimizer has no moments to
    # resume from: fit(resume_from=...) must fail loudly, not
    # silently restart Adam from scratch.
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "plain.pt")
        save_checkpoint(path, model, epoch=2)  # no optimizer
        trainer = Trainer(model, MSELoss(),
                          Adam(model.parameters(), lr=0.05))
        try:
            trainer.fit(loader, epochs=1, log_every=99, resume_from=path)
        except ValueError as e:
            assert "no optimizer state" in str(e), e
        else:
            raise AssertionError("resume without optimizer state "
                                 "should raise ValueError")


def _myopic_model():
    # Vocab 3. Greedy's first pick (token 1) is only slightly better
    # than token 2, but its continuation is flat (best logp -1.099)
    # while token 2's continuation is peaked (best logp -0.013), so
    # the beam path scores better overall: -0.919 vs -1.904.
    table = {
        (0,): [-1.0, 0.1, 0.0],
        (0, 1): [0.0, 0.0, 0.0],
        (0, 2): [0.0, 5.0, 0.0],
    }

    def model(ids):
        row = table.get(tuple(ids), [0.0, 0.0, 0.0])
        return Tensor([row[:] for _ in ids])

    return model


def test_beam_search_beats_greedy():
    model = _myopic_model()
    greedy = generate(model, [0], max_new_tokens=2)
    beam = beam_search(model, [0], max_new_tokens=2, beam_width=2)
    assert greedy == [0, 1, 0], greedy
    assert beam == [0, 2, 1], beam
    # The beam result really does score higher under the model.
    def seq_logp(ids):
        total = 0.0
        for i in range(1, len(ids)):
            row = [float(x) for x in model(ids[:i])[-1].tolist()]
            m = max(row)
            total += row[ids[i]] - m - math.log(
                sum(math.exp(x - m) for x in row))
        return total
    assert seq_logp(beam) > seq_logp(greedy), \
        f"beam {seq_logp(beam):.3f} should beat greedy {seq_logp(greedy):.3f}"


def test_beam_search_width1_is_greedy():
    # Beam width 1 is exactly greedy decoding, on the myopic model
    # and on a peaked one where every step is forced.
    model = _myopic_model()
    for prompt in ([0], [0, 2]):
        assert beam_search(model, prompt, max_new_tokens=3,
                           beam_width=1) == generate(
                               model, prompt, max_new_tokens=3)


def test_beam_search_eos_and_errors():
    model = _myopic_model()
    # eos: the finished beam is returned once every beam ends in eos.
    out = beam_search(model, [0], max_new_tokens=5, beam_width=2, eos_id=0)
    assert out[-1] == 0 and out[0] == 0
    # max_new_tokens=0 returns the prompt unchanged.
    assert beam_search(model, [0, 2], max_new_tokens=0) == [0, 2]
    # length_penalty still returns a valid continuation.
    out = beam_search(model, [0], max_new_tokens=2, beam_width=2,
                      length_penalty=1.0)
    assert out[:1] == [0] and len(out) == 3
    # errors
    for bad in (lambda: beam_search(model, [], max_new_tokens=2),
                lambda: beam_search(model, [0], beam_width=0),
                lambda: beam_search(model, [0], max_new_tokens=-1),
                lambda: beam_search(model, [0], length_penalty=-1.0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("beam_search should have raised ValueError")


def test_checkpoint_with_tokenizer():
    tok = BPETokenizer()
    tok.train("the model learns to predict the next token " * 5,
              vocab_size=300)
    assert tok.num_merges > 0
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    opt = Adam(model.parameters(), lr=0.05)
    Trainer(model, MSELoss(), opt).fit(loader, epochs=2, log_every=99)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "full.pt")
        save_checkpoint(path, model, optimizer=opt, epoch=2, tokenizer=tok)
        # A fresh tokenizer + fresh model reload to identical state.
        tok2 = BPETokenizer()
        model2 = _mlp()
        opt2 = Adam(model2.parameters(), lr=0.05)
        meta = load_checkpoint(path, model=model2, optimizer=opt2,
                               tokenizer=tok2)
        assert meta["epoch"] == 2
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.merges == tok.merges
        assert tok2.merge_ranks == tok.merge_ranks
        text = "the model learns to predict the next token"
        assert tok2.encode(text) == tok.encode(text)
        assert tok2.decode(tok.encode(text)) == text
        # The state helpers work standalone, without a file.
        state = tokenizer_state(tok)
        assert state["format"] == "darklearn-bpe-v1"
        tok3 = BPETokenizer()
        load_tokenizer_state(tok3, state)
        assert tok3.merges == tok.merges


def test_checkpoint_tokenizer_missing_raises():
    model = _mlp()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "plain.pt")
        save_checkpoint(path, model, epoch=1)  # no tokenizer saved
        try:
            load_checkpoint(path, tokenizer=BPETokenizer())
        except ValueError as e:
            assert "no tokenizer" in str(e), e
        else:
            raise AssertionError("loading a missing tokenizer should raise")
    # A state dict that is not a BPE snapshot is rejected too.
    try:
        load_tokenizer_state(BPETokenizer(), {"format": "other"})
    except ValueError:
        pass
    else:
        raise AssertionError("bad tokenizer state should raise")


def test_trainer_checkpoint_with_tokenizer():
    # The Trainer's checkpoint callback saves the tokenizer into
    # the same file when one is given, so one artifact is the whole
    # runnable model.
    tok = BPETokenizer()
    tok.train("the model learns to predict the next token " * 5,
              vocab_size=300)
    model = _mlp()
    loader = DataLoader(_mlp_data(), batch_size=16, shuffle=True, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "best.pt")
        trainer = Trainer(model, MSELoss(),
                          Adam(model.parameters(), lr=0.05),
                          checkpoint_path=path, tokenizer=tok)
        trainer.fit(loader, epochs=3, log_every=99)
        tok2 = BPETokenizer()
        load_checkpoint(path, tokenizer=tok2)
        assert tok2.merges == tok.merges
        assert tok2.encode("the model") == tok.encode("the model")


TESTS = [
    test_trainer_resume_matches,
    test_trainer_resume_restores_best,
    test_trainer_resume_no_optimizer_state,
    test_beam_search_beats_greedy,
    test_beam_search_width1_is_greedy,
    test_beam_search_eos_and_errors,
    test_checkpoint_with_tokenizer,
    test_checkpoint_tokenizer_missing_raises,
    test_trainer_checkpoint_with_tokenizer,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
        else:
            print(f"PASS {t.__name__}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
