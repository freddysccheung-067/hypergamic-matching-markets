import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.python.ncf import NCFModel, NCFTrainer  # noqa: E402

N_AGENTS = 40


def make_interactions(n_pos=100, n_agents=N_AGENTS, seed=0):
    """Skewed log: positives pile onto a few popular items, 4 negatives each."""
    rng = np.random.default_rng(seed)
    popular = n_agents // 5

    def draw(n, high, taken):
        out = set()
        while len(out) < n:
            u, i = int(rng.integers(n_agents)), int(rng.integers(high))
            if u != i and (u, i) not in taken and (u, i) not in out:
                out.add((u, i))
        return out

    pos = draw(n_pos, popular, set())
    neg = draw(4 * n_pos, n_agents, pos)
    rows = [(u, i, 1) for u, i in sorted(pos)] + [(u, i, 0) for u, i in sorted(neg)]
    return np.array(rows, dtype=np.int32)


def test_predict_proba_strictly_in_open_unit_interval():
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16))
    u = torch.arange(N_AGENTS).repeat_interleave(N_AGENTS)
    i = torch.arange(N_AGENTS).repeat(N_AGENTS)
    p = model.predict_proba(u, i)
    assert p.shape == (N_AGENTS * N_AGENTS,)
    assert bool(((p > 0) & (p < 1)).all())


def test_forward_returns_logits_not_probabilities():
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16))
    u, i = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
    logits = model(u, i)
    assert logits.shape == (3,)
    assert torch.allclose(torch.sigmoid(logits), model.predict_proba(u, i))


def test_loss_decreases_over_20_epochs():
    torch.manual_seed(0)
    data = make_interactions(n_pos=100)
    assert data.shape == (500, 3)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16))
    trainer = NCFTrainer(model, lr=1e-2, device="cpu", batch_size=128, seed=0)
    losses = [trainer.train_epoch(data) for _ in range(20)]
    assert losses[-1] < losses[0]


def test_negative_top_up_only_when_short_and_avoids_log():
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=8, mlp_layers=(16,))
    trainer = NCFTrainer(model, device="cpu", seed=0)

    full = make_interactions(n_pos=20)
    assert len(trainer._top_up_negatives(trainer._validate(full))) == len(full)

    short = full[full[:, 2] == 1]  # positives only: every negative must be sampled
    topped = trainer._top_up_negatives(trainer._validate(short))
    added = topped[len(short):]
    assert len(added) == 4 * len(short)
    assert (added[:, 2] == 0).all()
    assert (added[:, 0] != added[:, 1]).all()
    logged = {(int(u), int(i)) for u, i, _ in short}
    assert not any((int(u), int(i)) in logged for u, i, _ in added)
    assert len({(int(u), int(i)) for u, i, _ in added}) == len(added)


def _recommend(seed, k=5, candidates=None):
    torch.manual_seed(seed)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16))
    trainer = NCFTrainer(model, lr=1e-2, device="cpu", batch_size=128, seed=seed)
    data = make_interactions(seed=seed)
    for _ in range(3):
        trainer.train_epoch(data)
    cands = np.arange(N_AGENTS) if candidates is None else candidates
    return trainer.top_k_recommendations(np.arange(N_AGENTS), cands, k)


@pytest.mark.parametrize("candidates", [None, np.arange(5, 35)])
def test_top_k_shape_uniqueness_and_no_self(candidates):
    k = 7
    recs = _recommend(0, k=k, candidates=candidates)
    assert set(recs) == set(range(N_AGENTS))
    allowed = set(range(N_AGENTS)) if candidates is None else set(candidates.tolist())
    for uid, items in recs.items():
        assert len(items) == k
        assert len(set(items)) == k
        assert uid not in items
        assert set(items) <= allowed
        assert all(isinstance(x, int) for x in items)


def test_top_k_rejects_k_larger_than_available_candidates():
    torch.manual_seed(0)
    model = NCFModel(10, 10, embed_dim=8, mlp_layers=(16,))
    trainer = NCFTrainer(model, device="cpu")
    ids = np.arange(10)
    assert len(trainer.top_k_recommendations(ids, ids, 9)[0]) == 9
    with pytest.raises(ValueError):
        trainer.top_k_recommendations(ids, ids, 10)


def test_top_k_chunking_matches_single_pass():
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16))
    trainer = NCFTrainer(model, device="cpu")
    ids = np.arange(N_AGENTS)
    whole = trainer.top_k_recommendations(ids, ids, 5)
    trainer.SCORE_CHUNK_ROWS = 3  # three users per forward pass
    assert trainer.top_k_recommendations(ids, ids, 5) == whole
    # The pair budget still caps the block when it is the tighter of the two;
    # one row per pass is the degenerate case the scale test runs into, so it
    # has to agree with the wide pass as well.
    trainer.SCORE_CHUNK_ROWS = 40
    trainer.SCORE_CHUNK_PAIRS = N_AGENTS  # one row per pass
    assert trainer.top_k_recommendations(ids, ids, 5) == whole


def test_same_seed_gives_identical_recommendations():
    assert _recommend(123) == _recommend(123)


@pytest.mark.parametrize("layers", [(32, 16), (128, 64, 32), (256,), (64, 32, 16, 8)])
def test_score_block_agrees_with_forward(layers):
    """The scoring fast path is the same function as ``forward``.

    It reassociates two sums (the first layer's bilinear form, and the GMF
    branch folded through the output weight), so it is exact in real arithmetic
    and differs only in the last bits in float. That is close enough to reorder a
    shortlist, which is the thing the run actually reads, so this pins the top-k
    rather than only the values — and pins it across layer shapes, because the
    loop that applies the later layers walks ``mlp_layers`` and a hardcoded
    3-layer assumption would pass at the base config and break on a sweep.
    """
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=layers).eval()
    users = torch.arange(N_AGENTS)
    cands = torch.arange(N_AGENTS)

    reference = model(
        users.repeat_interleave(N_AGENTS), cands.repeat(N_AGENTS)
    ).view(N_AGENTS, N_AGENTS)
    fast = model.score_block(users, cands, model.score_projections())

    assert fast.shape == reference.shape
    assert torch.allclose(fast, reference, atol=1e-6, rtol=1e-5)
    k = 5
    assert torch.equal(
        torch.topk(fast, k, dim=1).indices, torch.topk(reference, k, dim=1).indices
    )


def test_score_block_does_not_alias_the_projection_tensors():
    """Scoring writes into its own buffer, so masking one block cannot corrupt the next.

    ``score_block`` starts from a fresh matmul and adds in place; if it ever
    returned a view of a projection, the masked_fill the caller applies would
    write through to shared state and every block after the first would be
    scored against a corrupted model.
    """
    torch.manual_seed(0)
    model = NCFModel(N_AGENTS, N_AGENTS, embed_dim=16, mlp_layers=(32, 16)).eval()
    proj = model.score_projections()
    before = [t.clone() for t in proj]
    users, cands = torch.arange(8), torch.arange(N_AGENTS)

    reference = model.score_block(users, cands, proj).clone()
    # Mask and perturb the returned block the way the caller does; if it aliased
    # a projection, the next call would come back changed.
    block = model.score_block(users, cands, proj)
    block.masked_fill_(torch.eye(8, N_AGENTS, dtype=torch.bool), float("-inf"))
    again = model.score_block(users, cands, proj)

    assert torch.equal(reference, again), "masking one block changed the next"
    assert torch.isneginf(block[0, 0]) and torch.isfinite(reference[0, 0])
    for original, current in zip(before, proj):
        assert torch.equal(original, current), "score_block mutated the projections"
