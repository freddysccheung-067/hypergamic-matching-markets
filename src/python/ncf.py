"""Neural Collaborative Filtering (GMF + MLP) for the swipe-log recommender.

The model consumes the ``[user_id, item_id, label]`` log produced by
``generate_swipes`` (INTERFACES.md §2) and returns, for every user, a ranked
top-k candidate list that replaces the full preference ordering in matching.

Users and items live in one id space (an agent is both a user and a candidate),
so a user's own id is excluded from their recommendations and never drawn as a
sampled negative.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

NEGATIVES_PER_POSITIVE = 4
POS_WEIGHT = 4.0
_TOP_UP_ROUNDS = 20


class NCFModel(nn.Module):
    """GMF and MLP branches fused by a single linear output layer.

    ``forward`` returns raw logits. The sigmoid is applied only in
    ``predict_proba`` so training can use ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        mlp_layers: tuple[int, ...] = (128, 64, 32),
    ) -> None:
        super().__init__()
        if len(mlp_layers) == 0:
            raise ValueError("mlp_layers must contain at least one layer width")
        self.n_users = n_users
        self.n_items = n_items

        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_item = nn.Embedding(n_items, embed_dim)
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_item = nn.Embedding(n_items, embed_dim)

        layers: list[nn.Module] = []
        in_dim = 2 * embed_dim
        for width in mlp_layers:
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(mlp_layers[-1] + embed_dim, 1)

        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp = self.mlp(torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    def predict_proba(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(user_ids, item_ids))


class NCFTrainer:
    """Mini-batch trainer and batched top-k inference for an ``NCFModel``.

    ``seed`` fixes the shuffling and negative top-up draws. Model weights are
    initialised before the trainer exists, so seed torch (``torch.manual_seed``)
    before building the model to make a whole run reproducible.
    """

    # Sizing of the block of users scored in one forward pass. Two bounds,
    # because they fail in opposite directions and the old single pair budget
    # failed in one of them without saying so.
    #
    # SCORE_CHUNK_ROWS is the throughput floor. Per-pass Python and dispatch
    # overhead is fixed, so a pass that scores very few rows pays it per row
    # rather than per block: at N=10,000 a 8,192-pair budget resolves to ONE row
    # per pass — 5,000 passes — and a pair budget alone therefore degrades as N
    # grows, which is exactly the direction the scale test goes. The measured
    # cure is not to add passes but to raise the row floor, which is why the
    # constant is in rows.
    #
    # SCORE_CHUNK_PAIRS stays as the memory ceiling: the (rows, candidates, 128)
    # MLP activations are the big tensor, and at N=100,000 even a handful of rows
    # is hundreds of MB.
    #
    # Measured in the real run loop, one side scored against the other, this
    # machine, 24 threads, torch 2.14.1+cpu:
    #
    #     rows   N=5,000 top_k/call   N=10,000 top_k/call
    #        1           1.42 s              5.74 s
    #        2           1.00                4.83
    #        3           0.98                4.46
    #        4           0.85                4.32
    #        6           0.80                4.27   <- chosen
    #        8           0.77                5.90
    #       13           1.07                  -
    #       26           1.73                  -
    #      104           1.72-1.87             -     <- the old value, 1<<18 pairs
    #
    # The curve is flat-to-shallow between 2 and 8 and climbs sharply after, and
    # the plateau is in the same place at both N — which is why the knob is in
    # rows and not in pairs. Six sits in the trough at both sizes instead of on a
    # peak that would not survive a different cache.
    #
    # End to end this is worth NOTHING measurable, and the comment says so
    # because a later reader will otherwise assume otherwise. Median of
    # alternating runs, whole loop: 1.03x at N=5,000 (216.7 -> 209.7 s) and
    # 0.98x at N=10,000 (393.3 -> 400.6 s) — i.e. noise. Per-call timings do
    # improve ~2x, and an 8-epoch A/B did show that saving; across a full run it
    # does not survive, because per-call block timing and per-pass wall clock
    # are not the same quantity once 24 threads are synchronised on a small
    # block. What this constant buys is the degenerate case: without a row
    # floor, a pair budget resolves to one row per pass at N>=100,000, which is
    # thousands of synchronised passes per epoch. That is a robustness argument,
    # not a speed one.
    SCORE_CHUNK_ROWS = 6
    SCORE_CHUNK_PAIRS = 1 << 18

    def __init__(
        self,
        model: NCFModel,
        lr: float = 1e-3,
        l2_reg: float = 1e-5,
        device: str | torch.device | None = None,
        batch_size: int = 4096,
        seed: int | None = None,
        pos_weight: float = POS_WEIGHT,
        n_negatives: int = NEGATIVES_PER_POSITIVE,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ training

    def train_epoch(self, interactions: np.ndarray) -> float:
        """One pass over ``interactions`` (B, 3) int32; returns the mean loss."""
        data = self._validate(interactions)
        data = self._top_up_negatives(data)
        data = data[self._rng.permutation(len(data))]

        users = torch.from_numpy(data[:, 0]).to(self.device)
        items = torch.from_numpy(data[:, 1]).to(self.device)
        labels = torch.from_numpy(data[:, 2].astype(np.float32)).to(self.device)

        self.model.train()
        total = 0.0
        for start in range(0, len(data), self.batch_size):
            sl = slice(start, start + self.batch_size)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(users[sl], items[sl]), labels[sl])
            loss.backward()
            self.optimizer.step()
            total += loss.item() * (users[sl].shape[0])
        return total / len(data)

    def _validate(self, interactions: np.ndarray) -> np.ndarray:
        arr = np.asarray(interactions)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise ValueError(f"interactions must have shape (B, 3), B > 0; got {arr.shape}")
        arr = arr.astype(np.int64, copy=False)
        if arr[:, 0].min() < 0 or arr[:, 0].max() >= self.model.n_users:
            raise ValueError("user_id out of range for the model")
        if arr[:, 1].min() < 0 or arr[:, 1].max() >= self.model.n_items:
            raise ValueError("item_id out of range for the model")
        if not np.isin(arr[:, 2], (0, 1)).all():
            raise ValueError("labels must be 0 or 1")
        return arr

    def _top_up_negatives(self, data: np.ndarray) -> np.ndarray:
        """Bring the log up to ``n_negatives`` per positive, if it is short.

        ``generate_swipes`` already emits that ratio, so on a healthy epoch this
        is a no-op. It matters when the candidate pool is small enough that an
        agent could not supply the full quota (small test populations), and it
        must use the configured ratio rather than a constant — otherwise a sweep
        over ``ncf.n_negatives`` would be silently undone here.

        Extra pairs reuse the positives' users, avoid every pair already in the
        log and avoid user == item. If the pair space is nearly exhausted the
        top-up stops short rather than looping.
        """
        positives = data[data[:, 2] == 1]
        need = self.n_negatives * len(positives) - int((data[:, 2] == 0).sum())
        if need <= 0:
            return data

        n_items = self.model.n_items
        seen = data[:, 0] * n_items + data[:, 1]
        extra: list[np.ndarray] = []
        for _ in range(_TOP_UP_ROUNDS):
            u = self._rng.choice(positives[:, 0], size=need)
            i = self._rng.integers(0, n_items, size=need)
            keys = u * n_items + i
            ok = (u != i) & ~np.isin(keys, seen)
            _, first = np.unique(keys[ok], return_index=True)
            u, i, keys = u[ok][first], i[ok][first], keys[ok][first]
            if len(keys):
                extra.append(np.stack([u, i, np.zeros_like(u)], axis=1))
                seen = np.concatenate([seen, keys])
                need -= len(keys)
            if need <= 0:
                break
        return np.concatenate([data, *extra]) if extra else data

    # ------------------------------------------------------------------ turnover

    @torch.no_grad()
    def reinitialize(self, ids: np.ndarray, seed: int) -> None:
        """Re-draw the four embedding rows of each id, as at construction.

        Generational turnover puts a new agent in an old slot; without this the
        newcomer would be scored with the embedding the model learned for whoever
        held the slot before. Same init as ``NCFModel`` (normal, std 0.01), from a
        generator seeded by the caller so the multi-generation run stays
        reproducible and torch's global stream is left alone.

        The shared MLP and output layers are kept — they encode the market, not an
        agent — and so is Adam's state: its moment estimates for these rows are
        stale, but zeroing them would make the next step for exactly these rows the
        largest in the model, which is a bias of its own.
        """
        idx = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=self.device)
        if idx.numel() == 0:
            return
        gen = torch.Generator().manual_seed(int(seed))
        for emb in (self.model.gmf_user, self.model.gmf_item,
                    self.model.mlp_user, self.model.mlp_item):
            fresh = torch.normal(0.0, 0.01, size=(idx.numel(), emb.embedding_dim),
                                 generator=gen)
            emb.weight[idx] = fresh.to(emb.weight.device, emb.weight.dtype)

    # ----------------------------------------------------------------- inference

    @torch.no_grad()
    def top_k_recommendations(
        self,
        user_ids: np.ndarray,
        candidate_ids: np.ndarray,
        k: int,
    ) -> dict[int, list[int]]:
        """Top-``k`` candidates per user by model score, excluding the user.

        Users are scored against all candidates in batched forward passes (a
        block of users x every candidate per pass), never one pass per user.
        """
        users = np.asarray(user_ids, dtype=np.int64).ravel()
        cands = np.asarray(candidate_ids, dtype=np.int64).ravel()
        if k < 1:
            raise ValueError("k must be >= 1")
        if len(np.unique(cands)) != len(cands):
            raise ValueError("candidate_ids must be unique")
        n_cands = len(cands)
        min_valid = n_cands - (1 if np.isin(users, cands).any() else 0)
        if k > min_valid:
            raise ValueError(f"k={k} exceeds the {min_valid} candidates available per user")

        users_t = torch.from_numpy(users).to(self.device)
        cands_t = torch.from_numpy(cands).to(self.device)
        # Start from the throughput floor, then cut it down if that many rows
        # would exceed the memory ceiling. max(1, ...) is the last resort and
        # only matters above N ~ 32,000, where one row is already over budget.
        rows = min(self.SCORE_CHUNK_ROWS, max(1, self.SCORE_CHUNK_PAIRS // n_cands))

        was_training = self.model.training
        self.model.eval()
        try:
            recs: dict[int, list[int]] = {}
            for start in range(0, len(users), rows):
                u = users_t[start : start + rows]
                scores = self.model(
                    u.repeat_interleave(n_cands), cands_t.repeat(len(u))
                ).view(len(u), n_cands)
                scores.masked_fill_(u[:, None] == cands_t[None, :], float("-inf"))
                top = torch.topk(scores, k, dim=1).indices
                ids = cands_t[top].cpu().tolist()
                for uid, row in zip(users[start : start + rows].tolist(), ids):
                    recs[uid] = row
        finally:
            self.model.train(was_training)
        return recs
