"""One-epoch pipeline for the hypergamic matching market (INTERFACES.md §6).

The loop each epoch:

    1. generate_swipes            -> the implicit-feedback log          (ncf arm)
    2. NCFTrainer.train_epoch     -> train the recommender on it         (ncf arm)
    3. top_k_recommendations      -> truncated preference lists          (all arms)
    4. deferred_acceptance        -> matching under those lists
    5. compute_welfare + metrics  -> one row of the metrics frame
    6. update_thresholds          -> θ moves, next epoch's swipe log changes

Stage 6 is what makes this a simulation rather than a repeated experiment: the
matching result feeds back into the swipe gate, so the market's history changes
the market's behaviour.

Steps 1-2 belong to the ``"ncf"`` arm only. The ablation arms of
docs/SPRINT_PROMPTS.md Session 3-C replace *only* the rule that produces the
preference lists in step 3; the population, seed, θ, utility parameters and the
deferred-acceptance call are held fixed across arms, because an arm-to-arm
comparison in which any of those moved would be measuring the wrong thing.

The full-information benchmark (THEORY.md §6.1) is O(N²) in memory and is run
*once per population*, not once per epoch — it depends only on traits, which
never change, whereas only θ evolves. INTERFACES.md §4 puts the cap at N=5000.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from src.python.config import SimConfig
from src.python.metrics import (
    algorithmic_loss,
    blocking_pair_ratio,
    gini,
    market_clearing_rate,
    pareto_alpha,
    rank_satisfaction_index,
    reach,
    top1_share,
)
from src.python.ncf import NCFModel, NCFTrainer

try:  # the extension lands in build/Release (README "Installation")
    from _hypergamic import (
        admissible_prefs,
        build_full_info_pref_lists,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        sample_population,
        update_thresholds,
        utility_sd,
    )
except ImportError:  # pragma: no cover - convenience for a fresh checkout
    _BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "Release"
    if _BUILD.is_dir():
        sys.path.insert(0, str(_BUILD))
    from _hypergamic import (  # noqa: F401
        admissible_prefs,
        build_full_info_pref_lists,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        sample_population,
        update_thresholds,
        utility_sd,
    )

# sd of a standard logistic. Same constant as kLogisticSd in include/utilities.h;
# repeated here so the orchestrator can log the ratio without a second binding.
_LOGISTIC_SD = 1.8138

log = logging.getLogger(__name__)

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
    # Session 3-C. Appended rather than inserted: §6 says adding a column is
    # free and renaming one breaks the plots, and appending keeps every saved
    # CSV readable by column name.
    "arm",
    "reach",
    "top1_share",
]

# The four ablation arms of docs/SPRINT_PROMPTS.md Session 3-C. Only the rule
# that produces the preference lists differs; everything upstream is shared.
RecommendArm = Literal["ncf", "random", "popularity", "oracle"]
ARMS: tuple[str, ...] = ("ncf", "random", "popularity", "oracle")

# INTERFACES.md §4: the full-information benchmark is O(N²) in memory, about
# 100 MB per side at N=5000.
FULL_INFO_MAX_N = 5000


class Simulation:
    """A complete run: population, recommender, matching and the metrics frame.

    ``config`` may be a :class:`SimConfig`, a plain dict in the YAML's shape, or
    ``None`` to read ``config_path``. The dict form exists so a sweep can mutate
    one block (``scripts/run_sweep.py`` changes ``hypergamy.delta``) without
    rebuilding the object.

    ``recommender`` selects the ablation arm (docs/SPRINT_PROMPTS.md 3-C):

    ``"ncf"``         swipe log -> NCF training -> top-k. The historical path; the
                      only arm in which anything is learned, and the only one that
                      calls ``generate_swipes``.
    ``"random"``      k drawn uniformly *without replacement* from each chooser's
                      θ-admissible set, reseeded per (seed, epoch).
    ``"popularity"``  every chooser on a side gets the same list: the top-k of the
                      other side by in-degree. One rule per side, no history.
    ``"oracle"``      the first k θ-admissible candidates by true utility.

    All four run on the same population, seed, θ, utility params and
    ``deferred_acceptance`` call. Only the rule producing the preference lists
    differs — that is what makes the comparison an ablation rather than four
    experiments.

    ``freeze_theta`` stops stage 6 from mutating θ. It exists for tests that need
    all arms to see identical states at *every* epoch rather than only at epoch 0:
    with the loop live the arms legitimately diverge, because each arm's matching
    outcome moves θ differently, and that divergence is the feedback the model is
    supposed to have. Freezing it is a diagnostic, never a headline setting.
    """

    def __init__(self, config: SimConfig | dict[str, Any] | None = None,
                 config_path: str = "experiments/base.yaml",
                 recommender: RecommendArm = "ncf",
                 freeze_theta: bool = False) -> None:
        if recommender not in ARMS:
            raise ValueError(
                f"recommender must be one of {ARMS}; got {recommender!r}"
            )
        self.recommender: str = recommender
        self.freeze_theta = bool(freeze_theta)

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
        self._log_noise_ratio()

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

        # (T, N) exposure series, set by run(). The Lorenz plot needs the
        # trajectory, not only the final state, and a caller that wants it should
        # not have to re-read the frame column by column.
        self.exposure_series: np.ndarray = np.zeros((0, self.n), dtype=np.int64)

        # The state matrix each epoch was handed, recorded only when asked for.
        # The ablation test asserts these are identical across arms, and an
        # assertion that reads a live attribute from four objects cannot be
        # faked by an arm that quietly resampled: it would have to equal the
        # others' matrices to pass.
        self.record_states = False
        self.states_seen: list[np.ndarray] = []

    # ------------------------------------------------------------------ setup

    def _log_noise_ratio(self) -> None:
        """Log the implied chemistry noise-to-signal ratio once at construction.

        ``shock_scale`` is in absolute utility units, so its meaning moves with
        whatever alpha and beta happen to be: 1.5 is a large shock at one
        weighting and a whisper at another. The ratio,

            shock_scale * 1.8138 / utility_sd,

        is the scale-free statement of the same thing, and logging it once per run
        is what makes a sweep's ``shock_scale`` axis interpretable after a retune.
        ``utility_signal_sd`` is exposed on the instance so a caller can record it
        without parsing the log.
        """
        signal_sd = float(utility_sd(self.states, self.config_dict))
        self.utility_signal_sd = signal_sd
        scale = float(self.config.hypergamy.shock_scale)
        self.noise_ratio = (scale * _LOGISTIC_SD / signal_sd) if signal_sd > 0.0 else float("nan")
        log.info(
            "utility noise-to-signal: shock_scale=%.4g, utility_sd=%.4f, ratio=%.3f",
            scale, signal_sd, self.noise_ratio,
        )

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

    # ------------------------------------------------------- ablation arms

    def _exposure_from_dense(
        self,
        prop_prefs: list[list[int]],
        acc_prefs: list[list[int]],
    ) -> np.ndarray:
        """Row-indexed exposure counts from dense per-side preference lists.

        The same quantity ``_truncated_prefs`` computes, but from the lists
        themselves rather than from the global-id recommendations they were built
        from. The three non-NCF arms never hold global ids, so this is the only
        conversion path they can share — and sharing it is what guarantees the
        exposure column means the same thing in all four arms.

        A proposer appearing in many acceptors' lists is exposed many times, which
        is exactly THEORY.md §7.1's "profile exposures received across an epoch".

        The mapping matters and is easy to get backwards: ``prop_prefs[p]`` holds
        *acceptor* slots, so its entries are exposures *received* by acceptors. The
        two loops below therefore index ``prop_prefs`` through ``acc_rows`` and
        ``acc_prefs`` through ``prop_rows``. Swapping them would still give the
        right Gini — a permutation within a side leaves the distribution alone —
        while attributing every agent's exposure to the wrong agent, which is the
        silent version of this bug.
        """
        exposure = np.zeros(self.n, dtype=np.int64)
        for lst in prop_prefs:
            if lst:
                exposure[self.acc_rows[np.asarray(lst, dtype=np.int64)]] += 1
        for lst in acc_prefs:
            if lst:
                exposure[self.prop_rows[np.asarray(lst, dtype=np.int64)]] += 1
        return exposure

    def _oracle_prefs(self, top_k: int) -> tuple[list[list[int]], list[list[int]]]:
        """Top-k θ-admissible candidates by true utility, per side.

        Reads the ranked admissible sets straight from C++, so the first ``k``
        entries are exactly the positives ``generate_swipes`` would report — the
        same utilities, the same θ gate, the same tie-break. The arm is therefore
        a genuine upper bound on *ranking*, differing from the NCF arm only in
        that nothing was estimated.
        """
        prop_adm, acc_adm = admissible_prefs(self.states, self.config_dict)
        return (
            [list(lst[:top_k]) for lst in prop_adm],
            [list(lst[:top_k]) for lst in acc_adm],
        )

    def _random_prefs(
        self, top_k: int, epoch: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """k drawn uniformly WITHOUT replacement from each chooser's admissible set.

        Two specification points, both load-bearing:

        * From the *admissible* set, not the whole market. Sampling the market
          would let this arm reach candidates the other arms' θ gate excludes,
          which makes it differ in two ways at once (the rule and the gate) and
          turns the control into a comparison against a different market.
        * Without replacement. With replacement an agent could appear twice in one
          list, which the matching rejects (a repeated id makes a rank lookup
          ambiguous) and which would also distort the exposure count.

        ``min(k, len(set))`` so a chooser with fewer than k admissible candidates
        gets a shorter list rather than an error, matching how ``generate_swipes``
        truncates at the admissible bound.

        Seeded from (run seed, epoch) so the same (arm, seed) reproduces the
        exposure matrix exactly; the generator is rebuilt each epoch so epoch t
        does not depend on how many draws epoch t-1 happened to make.
        """
        prop_adm, acc_adm = admissible_prefs(self.states, self.config_dict)
        rng = np.random.default_rng([self.rng_seed, epoch])

        def draw(lists: list[list[int]]) -> list[list[int]]:
            out: list[list[int]] = []
            for lst in lists:
                if not lst:
                    out.append([])
                    continue
                take = min(top_k, len(lst))
                # choice(size=take, replace=False) is a uniform draw of a
                # take-subset, which is what "uniformly without replacement" means
                # and is not the same as taking the first k of a shuffle.
                picked = rng.choice(len(lst), size=take, replace=False)
                out.append([lst[int(i)] for i in picked])
            return out

        return draw(prop_adm), draw(acc_adm)

    def _popularity_prefs(
        self,
        top_k: int,
        prev_prop: list[list[int]] | None,
        prev_acc: list[list[int]] | None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """One shared list per side: the top-k of the other side by in-degree.

        Once every chooser on a side holds the same list, exactly ``2k`` agents
        carry all the exposure in equal amounts — which is where the closed forms
        of THEORY.md §7.1/§7.2 come from:

            Gini = 1 - 2k/N,   MCR = 2k/N.

        In-degree is counted over the *previous* epoch's exposure (`prev_*`), so
        the first epoch has no history.

        **Epoch 0 falls back to the oracle ranking.** Starting from all-zero
        in-degrees would rank the market by id, which is a different and accidental
        baseline: it would hand exposure to ids 0..k-1, i.e. the first k agents of
        the population, with no claim attached to them. Using the oracle ranking
        instead gives the arm the best possible information on its first step and
        makes the arm's subsequent concentration attributable to the popularity
        rule rather than to sampling order. It also means epoch 0 is identical to
        the oracle arm by construction, which is asserted in the tests rather than
        left implicit.

        Ties are broken by ascending dense index, matching the tie-break used
        everywhere else in the matching layer.
        """
        if prev_prop is None or prev_acc is None:
            # No history: rank by true utility, and say so (see docstring).
            return self._oracle_prefs(top_k)

        # In-degree of each *proposer* = how many acceptor lists named him;
        # in-degree of each acceptor = how many proposer lists named her.
        prop_indeg = np.zeros(self.n_prop, dtype=np.int64)
        acc_indeg = np.zeros(self.n_acc, dtype=np.int64)
        for lst in prev_acc:
            if lst:
                prop_indeg[np.asarray(lst, dtype=np.int64)] += 1
        for lst in prev_prop:
            if lst:
                acc_indeg[np.asarray(lst, dtype=np.int64)] += 1

        # lexsort's last key is primary: descending degree, then ascending index.
        prop_order = np.lexsort((np.arange(self.n_prop), -prop_indeg))
        acc_order = np.lexsort((np.arange(self.n_acc), -acc_indeg))

        shared_prop = [int(i) for i in acc_order[:top_k]]
        shared_acc = [int(i) for i in prop_order[:top_k]]

        return ([list(shared_acc)] * self.n_prop, [list(shared_prop)] * self.n_acc)

    def _arm_prefs(
        self, epoch: int, top_k: int, prev: tuple[list[list[int]], list[list[int]]] | None
    ) -> tuple[list[list[int]], list[list[int]], np.ndarray]:
        """Preference lists for this (arm, epoch), plus exposure.

        The NCF arm is handled by the caller, which needs the swipe log as well;
        this covers the three learning-free arms.
        """
        if self.recommender == "oracle":
            prop_prefs, acc_prefs = self._oracle_prefs(top_k)
        elif self.recommender == "random":
            prop_prefs, acc_prefs = self._random_prefs(top_k, epoch)
        elif self.recommender == "popularity":
            prev_prop, prev_acc = prev if prev is not None else (None, None)
            prop_prefs, acc_prefs = self._popularity_prefs(top_k, prev_prop, prev_acc)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unhandled arm {self.recommender!r}")

        return prop_prefs, acc_prefs, self._exposure_from_dense(prop_prefs, acc_prefs)

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
        series: list[np.ndarray] = []
        self.states_seen = []
        prev_prefs: tuple[list[list[int]], list[list[int]]] | None = None

        for epoch in range(n_epochs):
            # Recorded before anything touches the states, so this is the matrix
            # the epoch's preference rule actually saw. All four arms must agree
            # at epoch 0; with freeze_theta they must agree at every epoch.
            if self.record_states:
                self.states_seen.append(self.states.copy())

            n_swipes = 0
            if self.recommender == "ncf":
                log = generate_swipes(self.states, self.config_dict, epoch)
                self.trainer.train_epoch(log)
                n_swipes = len(log)

                # Score each side against the *other* side only. The model has no
                # notion of sides — an agent is a user and a candidate in one id
                # space — so asking for a global top-k would spend half of every
                # list on same-side agents and inflate the exposure counts with
                # visibility nobody could act on. Splitting here is what makes the
                # lists mean what the matching expects.
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
            else:
                # No swipe log and no training: these arms never learn, which is
                # the point of them. Everything downstream of the preference
                # lists — the DA call, welfare, the metrics, the θ step — is the
                # same code the NCF arm runs.
                prop_prefs, acc_prefs, exposure = self._arm_prefs(epoch, top_k, prev_prefs)
            prev_prefs = (prop_prefs, acc_prefs)
            series.append(exposure)


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
                bpr = blocking_pair_ratio(n_blocking, n_matched)
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
                "algorithmic_loss": algorithmic_loss(
                    welfare_alg=welfare_alg, welfare_full=self._welfare_full
                ) if ok_benchmark else float("nan"),
                "n_matched": n_matched,
                "n_blocking": n_blocking,
                "pareto_alpha": alpha,
                # Zero for every arm that never builds a swipe log, which is the
                # honest value: those arms do not train, so they have no swipes.
                "mean_swipes_per_agent": float(n_swipes) / float(self.n),
                "arm": self.recommender,
                "reach": reach(exposure),
                "top1_share": top1_share(exposure),
            })

            # θ ← θ + η·(matches/proposals − θ), in place on states[:, 6].
            # Skipped under freeze_theta, which is a diagnostic that holds the
            # market still so the arms can be compared at identical states rather
            # than only at identical epoch-0 states.
            if not self.freeze_theta:
                update_thresholds(
                    self.states,
                    self._proposals_by_row(da["proposals"]),
                    self._matches_by_row(matches),
                    eta,
                )

        self.exposure_series = (
            np.stack(series) if series else np.zeros((0, self.n), dtype=np.int64)
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
