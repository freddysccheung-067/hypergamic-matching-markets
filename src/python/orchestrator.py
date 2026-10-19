"""One-epoch pipeline for the hypergamic matching market (INTERFACES.md §6).

The loop each epoch:

    1. generate_swipes            -> the implicit-feedback log
    2. NCFTrainer.train_epoch     -> train the recommender on it
    3. top_k_recommendations      -> truncated preference lists
    4. deferred_acceptance        -> matching under those lists
    5. compute_welfare + metrics  -> one row of the metrics frame
    6. update_thresholds          -> θ moves, next epoch's swipe log changes

Stage 6 is what makes this a simulation rather than a repeated experiment: the
matching result feeds back into the swipe gate, so the market's history changes
the market's behaviour.

The full-information benchmark (THEORY.md §6.1) is O(N²) in memory and is run
*once per population*, not once per epoch — it depends only on traits, which
never change, whereas only θ evolves. INTERFACES.md §4 puts the cap at N=5000.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.python.config import SimConfig
from src.python.metrics import (
    algorithmic_loss,
    gini,
    market_clearing_rate,
    pareto_alpha,
    rank_satisfaction_index,
)
from src.python.ncf import NCFModel, NCFTrainer

try:  # the extension lands in build/Release (README "Installation")
    from _hypergamic import (
        build_full_info_pref_lists,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        sample_population,
        update_thresholds,
    )
except ImportError:  # pragma: no cover - convenience for a fresh checkout
    _BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "Release"
    if _BUILD.is_dir():
        sys.path.insert(0, str(_BUILD))
    from _hypergamic import (  # noqa: F401
        build_full_info_pref_lists,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        sample_population,
        update_thresholds,
    )

# Exact column order of INTERFACES.md §6. visualization.py reads these by name,
# but a stable order also makes a saved CSV diffable between runs.
COLUMNS = [
    "epoch",
    "generation",
    "delta",
    "seed",
    "gini",
    "mcr",
    "rsi",
    "bpr",
    "welfare_alg",
    "welfare_full",
    "algorithmic_loss",
    "n_matched",
    "n_blocking",
    "pareto_alpha",
    "mean_swipes_per_agent",
]

# INTERFACES.md §4: the full-information benchmark is O(N²) in memory, about
# 100 MB per side at N=5000.
FULL_INFO_MAX_N = 5000


class Simulation:
    """A complete run: population, recommender, matching and the metrics frame.

    ``config`` may be a :class:`SimConfig`, a plain dict in the YAML's shape, or
    ``None`` to read ``config_path``. The dict form exists so a sweep can mutate
    one block (``scripts/run_sweep.py`` changes ``hypergamy.delta``) without
    rebuilding the object.
    """

    def __init__(self, config: SimConfig | dict[str, Any] | None = None,
                 config_path: str = "experiments/base.yaml") -> None:
        if isinstance(config, SimConfig):
            self.config = config
        elif config is None:
            self.config = SimConfig.from_yaml(config_path)
        else:
            self.config = SimConfig.coerce(config)

        self.config_dict = self.config.to_dict()
        self.rng_seed = int(self.config.population.seed)
        self.n = int(self.config.population.n)

        # Torch seeding happens before the model is built: weight init draws
        # from torch's global generator, so seeding afterwards would leave the
        # initial weights outside the run's reproducibility guarantee.
        torch.manual_seed(self.rng_seed)

        self.states = sample_population(self.n, self.config_dict, self.rng_seed)
        self._split_sides()

        model = NCFModel(
            n_users=self.n,
            n_items=self.n,
            embed_dim=self.config.ncf.embed_dim,
            mlp_layers=tuple(self.config.ncf.mlp_layers),
        )
        self.trainer = NCFTrainer(
            model,
            lr=self.config.ncf.lr,
            l2_reg=self.config.ncf.l2_reg,
            seed=self.rng_seed,
            # Passed explicitly: NCFTrainer's own defaults would otherwise shadow
            # the config values and a sweep over either knob would measure
            # nothing.
            pos_weight=self.config.ncf.pos_weight,
            n_negatives=self.config.ncf.n_negatives,
        )

        # Cumulative match counts for the Pareto fit (THEORY.md §8).
        self._cumulative_matches = np.zeros(self.n, dtype=np.int64)

        # The benchmark is built lazily: it is only needed when it will be
        # reported, and at N just under the cap it costs seconds and 100 MB.
        self._full_prefs: tuple[list[list[int]], list[list[int]]] | None = None
        self._welfare_full: float = float("nan")
        self._benchmark_ready = False

    # ------------------------------------------------------------------ setup

    def _split_sides(self) -> None:
        """Dense per-side indexes, per INTERFACES.md §2.3.

        The preference lists, matches and blocking-pair counts all live in this
        dense space. ``_prop_slot`` and ``_acc_slot`` map a *row* to its position
        on its side, which is the only direction the orchestrator needs; the
        inverse is the row arrays themselves.
        """
        types = self.states[:, 1]
        self.prop_rows = np.flatnonzero(types == 0.0)
        self.acc_rows = np.flatnonzero(types == 1.0)

        self._prop_slot = np.full(self.n, -1, dtype=np.int64)
        self._acc_slot = np.full(self.n, -1, dtype=np.int64)
        self._prop_slot[self.prop_rows] = np.arange(self.prop_rows.size)
        self._acc_slot[self.acc_rows] = np.arange(self.acc_rows.size)

    @property
    def n_prop(self) -> int:
        return int(self.prop_rows.size)

    @property
    def n_acc(self) -> int:
        return int(self.acc_rows.size)

    def _ensure_benchmark(self) -> bool:
        """Build the full-information lists and matching once. False if skipped."""
        if self._benchmark_ready:
            return True
        if self.n > FULL_INFO_MAX_N:
            return False

        prop_prefs, acc_prefs = build_full_info_pref_lists(self.states, self.config_dict)
        self._full_prefs = (prop_prefs, acc_prefs)
        full = deferred_acceptance(prop_prefs, acc_prefs, False)
        # A complete-list run is stable by theorem, so n_blocking here is 0 and
        # is not reported; it is the check that the implementation is correct.
        self._welfare_full = float(compute_welfare(self.states, full["matches"],
                                                   self.config_dict)["welfare"])
        self._benchmark_ready = True
        return True

    # -------------------------------------------------------------- one epoch

    def _truncated_prefs(
        self,
        prop_recs: dict[int, list[int]],
        acc_recs: dict[int, list[int]],
    ) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
        """Turn per-side NCF recommendations into dense-index preference lists.

        The model scores a *global* id space — an agent is both a user and a
        candidate — while the matching uses dense per-side indexes (§2.3). Both
        conversion directions live here so the rest of the loop never has to
        remember which space it is in.

        Each recommendation list is already restricted to the other side by the
        caller's candidate set, but any id without a slot on the other side is
        dropped rather than mapped, so a list can come back shorter than
        ``top_k``.

        Exposure is counted per agent from the lists it appears on: that is what
        "visibility" means in THEORY.md §7.1, and it is now the right denominator
        because no same-side appearance is possible.
        """
        prop_prefs = [
            [int(self._acc_slot[i]) for i in prop_recs[int(row)] if self._acc_slot[i] >= 0]
            for row in self.prop_rows
        ]
        acc_prefs = [
            [int(self._prop_slot[i]) for i in acc_recs[int(row)] if self._prop_slot[i] >= 0]
            for row in self.acc_rows
        ]

        exposure = np.zeros(self.n, dtype=np.int64)
        for ids in prop_recs.values():
            exposure[np.asarray(ids, dtype=np.int64)] += 1
        for ids in acc_recs.values():
            exposure[np.asarray(ids, dtype=np.int64)] += 1
        return prop_prefs, acc_prefs, exposure

    def _proposals_by_row(self, proposals: np.ndarray) -> np.ndarray:
        """Map DA's dense proposal counts to row-indexed counts.

        The DA return is proposers first, then acceptors, both dense (§2.1, §2.3).
        ``update_thresholds`` writes ``states[:, 6]`` and so needs rows.
        """
        by_row = np.zeros(self.n, dtype=np.int32)
        by_row[self.prop_rows] = proposals[: self.n_prop]
        by_row[self.acc_rows] = proposals[self.n_prop:]
        return by_row

    def _matches_by_row(self, matches: np.ndarray) -> np.ndarray:
        by_row = np.zeros(self.n, dtype=np.int32)
        for pi, ai in matches.reshape(-1, 2):
            by_row[self.prop_rows[int(pi)]] += 1
            by_row[self.acc_rows[int(ai)]] += 1
        return by_row

    def run(self, n_epochs: int) -> pd.DataFrame:
        """Run ``n_epochs`` epochs and return the §6 metrics frame."""
        if n_epochs < 0:
            raise ValueError(f"n_epochs must be >= 0; got {n_epochs}")

        ok_benchmark = self._ensure_benchmark()
        top_k = int(self.config.ncf.top_k)
        eta = float(self.config.simulation.bayesian_eta)

        # The config's n_epochs is the full generation schedule, so a shorter run
        # still maps epochs onto the same generations a full run would.
        epochs_per_gen = max(1, int(self.config.simulation.n_epochs) //
                             int(self.config.simulation.n_generations))

        rows: list[dict[str, Any]] = []
        for epoch in range(n_epochs):
            log = generate_swipes(self.states, self.config_dict, epoch)
            self.trainer.train_epoch(log)

            # Score each side against the *other* side only. The model has no
            # notion of sides — an agent is a user and a candidate in one id
            # space — so asking for a global top-k would spend half of every list
            # on same-side agents and inflate the exposure counts with visibility
            # nobody could act on. Splitting here is what makes the lists mean
            # what the matching expects.
            prop_recs = self.trainer.top_k_recommendations(
                user_ids=self.prop_rows.astype(np.int64),
                candidate_ids=self.acc_rows.astype(np.int64),
                k=top_k,
            )
            acc_recs = self.trainer.top_k_recommendations(
                user_ids=self.acc_rows.astype(np.int64),
                candidate_ids=self.prop_rows.astype(np.int64),
                k=top_k,
            )
            prop_prefs, acc_prefs, exposure = self._truncated_prefs(prop_recs, acc_recs)

            da = deferred_acceptance(prop_prefs, acc_prefs, True)
            matches = da["matches"]
            n_matched = int(matches.shape[0])

            welfare_alg = float(compute_welfare(self.states, matches,
                                                self.config_dict)["welfare"])

            # BPR must be judged against the *true* ordering. The DA return's own
            # n_blocking is measured against the truncated lists it was given, and
            # a truncated run is stable with respect to those by construction —
            # reporting it would make bpr a column of zeros.
            #
            # Above the §4 cap there are no full lists, so there is no honest
            # count and bpr is NaN. The alternative — falling back to the DA's own
            # zero — would report perfect stability for a run that never measured
            # any.
            if self._full_prefs is not None:
                n_blocking = int(count_blocking_pairs(self._full_prefs[0],
                                                      self._full_prefs[1], matches))
                bpr = n_blocking / max(n_matched, 1)
            else:
                n_blocking = int(da["n_blocking"])
                bpr = float("nan")

            for pi, ai in matches.reshape(-1, 2):
                self._cumulative_matches[self.prop_rows[int(pi)]] += 1
                self._cumulative_matches[self.acc_rows[int(ai)]] += 1

            generation = epoch // epochs_per_gen
            gen_final = (epoch + 1) % epochs_per_gen == 0
            alpha = pareto_alpha(self._cumulative_matches) if gen_final else float("nan")

            rows.append({
                "epoch": epoch,
                "generation": generation,
                "delta": float(self.config.hypergamy.delta),
                "seed": self.rng_seed,
                "gini": gini(exposure),
                "mcr": market_clearing_rate(n_matched, self.n),
                "rsi": rank_satisfaction_index(matches, prop_prefs),
                "bpr": bpr,
                "welfare_alg": welfare_alg,
                "welfare_full": self._welfare_full if ok_benchmark else float("nan"),
                "algorithmic_loss": algorithmic_loss(welfare_alg, self._welfare_full)
                if ok_benchmark else float("nan"),
                "n_matched": n_matched,
                "n_blocking": n_blocking,
                "pareto_alpha": alpha,
                "mean_swipes_per_agent": float(len(log)) / float(self.n),
            })

            # θ ← θ + η·(matches/proposals − θ), in place on states[:, 6].
            update_thresholds(
                self.states,
                self._proposals_by_row(da["proposals"]),
                self._matches_by_row(matches),
                eta,
            )

        frame = pd.DataFrame(rows, columns=COLUMNS)
        if not frame.empty:
            # §6 fixes these dtypes; pandas infers int64/float64 and the counts
            # would otherwise drift to float on an empty or all-NaN column.
            frame = frame.astype({
                "epoch": "int64", "generation": "int64", "seed": "int64",
                "n_matched": "int64", "n_blocking": "int64",
            })
        return frame
