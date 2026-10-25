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
THEORY.md §7.7 replace *only* the rule that produces the
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

from src.python.comparison import EpochOutcome, agent_payoffs
from src.python.config import EnvironmentBConfig, SimConfig
from src.python.metrics import (
    algorithmic_loss,
    blocking_pair_ratio,
    gini,
    market_clearing_rate,
    mean_proposal_intent,
    pareto_alpha,
    proposal_cost_total,
    rank_satisfaction_index,
    reach,
    separating_threshold,
    top1_share,
)
from src.python.environment_b import (
    Subgraph,
    build_subgraph,
    complete_prefs,
    utility_table,
)
from src.python.ncf import NCFModel, NCFTrainer

try:  # the extension lands in build/Release (README "Installation")
    from _hypergamic import (
        admissible_prefs,
        build_full_info_pref_lists,
        calibrate_thresholds,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        mean_admissible_utility,
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
        calibrate_thresholds,
        compute_welfare,
        count_blocking_pairs,
        deferred_acceptance,
        generate_swipes,
        mean_admissible_utility,
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
    # Session 5 (THEORY.md §10.2). Appended for the same reason as the ablation
    # columns: adding is free, renaming breaks the plots. The first two are the
    # quantities H1 is stated in terms of; proposal_cost_total is the signal-cost
    # term of the welfare sum.
    "mean_proposal_intent",
    "separating_threshold",
    "proposal_cost_total",
    # Session 5-E (THEORY.md §10.4). W's value term, its defection term and its
    # total; the signal-cost term is proposal_cost_total above, not repeated. All
    # three come from compute_welfare and are cross-checked against the per-agent
    # payoff vector the cross-environment accounting reads. The paired columns are
    # NaN on a row from a lone run: welfare_common needs an A outcome for every
    # agent B did not match, and only src/python/comparison.py's paired_run has one.
    "value_created",
    "defection_loss",
    "welfare_total",
    "welfare_common",
    "welfare_selected",
    "welfare_selection_artifact",
]

# The four ablation arms of THEORY.md §7.7. Only the rule
# that produces the preference lists differs; everything upstream is shared.
RecommendArm = Literal["ncf", "random", "popularity", "oracle", "complete"]
ARMS: tuple[str, ...] = ("ncf", "random", "popularity", "oracle", "complete")
# The four arms Environment A is ablated over. "complete" is Environment B's list
# rule (THEORY.md §10.3) and is not one of them. scripts/run_ablation.py keeps its own
# four-arm tuple, so the published A-side ablation does not grow a fifth arm.
ABLATION_ARMS: tuple[str, ...] = ("ncf", "random", "popularity", "oracle")

# INTERFACES.md §4: the full-information benchmark is O(N²) in memory, about
# 100 MB per side at N=5000.
FULL_INFO_MAX_N = 5000

# Survival parameters for a config with no environment_b block: sigma = 1 for every
# pair, so nothing dissolves and the defection term is exactly zero. The same
# agent_payoffs call then serves both cases rather than a second code path.
_NO_DEFECTION = EnvironmentBConfig(size_fraction=1.0, values_min=0.0, signal_cost=0.0,
                                   defection_penalty=0.0, sigma_0=1.0, sigma_alpha=0.0,
                                   sigma_beta=0.0, partner_loss=0.0)


def received_proposals(states: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row count and mean proposer intent of the proposals each agent received.

    ``pairs`` is (Q, 2) row ids, [proposer, receiver] — the label-1 rows of a swipe
    log, columns 0 and 1, or the complete arm's proposal set. The receiver is
    column 1: column 0 is the agent who swiped. Intent is read from ``states``
    column 11 at the *proposer's* row. An agent who received nothing has a count of
    0 and a mean intent of NaN, never 0.0, because no proposal is not the same
    thing as proposals from zero-intent agents.
    """
    n = states.shape[0]
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    volume = np.bincount(pairs[:, 1], minlength=n).astype(np.float64)
    intent_sum = np.bincount(pairs[:, 1], weights=states[pairs[:, 0], 11], minlength=n)
    mean_intent = np.full(n, np.nan)
    got = volume > 0
    mean_intent[got] = intent_sum[got] / volume[got]
    return volume, mean_intent


def flooding_delta(volume: np.ndarray, mean_intent: np.ndarray,
                   sides: tuple[np.ndarray, ...], weight: float,
                   fallback_intent: float) -> np.ndarray:
    """The flooding term of THESIS.md H2, per row: w * (V_i / V̄_side - 1) * (1 - τ̄_i).

    V̄ is the mean over the rows of i's own side, so w is scale-free in volume: a
    market where every agent receives twice as much has the same term as before,
    and only *dispersion* in received volume moves θ. An agent who received
    nothing has no τ̄; ``fallback_intent`` (the population mean intent, the
    posterior a pooling acceptor would hold) stands in for it, which makes the term
    -w * (1 - fallback) — an unswamped agent lowers her bar. A side that received
    nothing at all has no V̄ and gets no term.
    """
    delta = np.zeros(volume.shape[0])
    for rows in sides:
        v_bar = float(volume[rows].mean()) if rows.size else 0.0
        if v_bar <= 0.0:
            continue
        tau = np.where(np.isnan(mean_intent[rows]), fallback_intent, mean_intent[rows])
        delta[rows] = weight * (volume[rows] / v_bar - 1.0) * (1.0 - tau)
    return delta


class Simulation:
    """A complete run: population, recommender, matching and the metrics frame.

    ``config`` may be a :class:`SimConfig`, a plain dict in the YAML's shape, or
    ``None`` to read ``config_path``. The dict form exists so a sweep can mutate
    one block (``scripts/run_sweep.py`` changes ``hypergamy.delta``) without
    rebuilding the object.

    ``recommender`` selects the ablation arm (THEORY.md §7.7):

    ``"ncf"``         swipe log -> NCF training -> top-k. The historical path; the
                      only arm in which anything is learned, and the only one that
                      calls ``generate_swipes``.
    ``"random"``      k drawn uniformly *without replacement* from each chooser's
                      θ-admissible set, reseeded per (seed, epoch).
    ``"popularity"``  every chooser on a side gets the same list: the top-k of the
                      other side by in-degree. One rule per side, no history.
    ``"oracle"``      the first k θ-admissible candidates by true utility.
    ``"complete"``    Environment B (THEORY.md §10.3): the oracle rule restricted to
                      the ``environment_b`` subgraph, with that block's signal cost
                      c_B applied to the proposers. Agents outside the subgraph get
                      empty lists, so ``mcr`` is over the whole population and an
                      excluded agent counts as unmatched — the A outcome they would
                      have had is added back by the welfare layer, not here.

    All run on the same population, seed, θ, utility params and
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

        # The thread count is part of the reproducibility contract, not a
        # performance knob to leave to the environment. torch's threaded
        # reductions are not bitwise associative, so the same seed at 1 and 8
        # threads diverges at ~2e-7 per training step; across a hundred epochs
        # of this feedback loop that reaches a Gini spread of 0.007-0.015,
        # which is the size of the effects the project reports. `ncf.num_threads`
        # defaults to 1 and docs/INTERFACES.md §3 states the contract.
        torch.set_num_threads(int(self.config.ncf.num_threads))

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

        # Environment B state, built only for the "complete" arm. The subgraph is a
        # property of the population and the filter, both fixed, so it is built once;
        # the utility table is O(N²) and reused by every epoch.
        self.subgraph: Subgraph | None = None
        self._utility_table: tuple[np.ndarray, np.ndarray] | None = None
        self._complete_proposals = np.zeros((0, 2), dtype=np.int64)
        if recommender == "complete":
            try:
                env = self.config.require_environment_b()
            except ValueError as exc:
                raise ValueError(f'recommender="complete": {exc}') from exc
            if self.n > FULL_INFO_MAX_N:
                raise ValueError(
                    f"recommender=\"complete\" is O(N²) and capped at N={FULL_INFO_MAX_N}; "
                    f"got N={self.n}"
                )
            self.subgraph = build_subgraph(self.states, env, self.rng_seed)
            self._utility_table = utility_table(self.states, self.config_dict)

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

        # Per-epoch EpochOutcome (src/python/comparison.py), recorded only when
        # asked for. A paired A/B comparison reads these rather than the frame,
        # because the comparability rule is per agent and the frame is per epoch.
        self.record_outcomes = False
        self.outcomes: list[EpochOutcome] = []

        # Per-epoch market record, kept only when asked for: the swipe log, the
        # proposal set the matching was restricted to, DA's per-row proposal and
        # match counts — the exact inputs update_thresholds read — and θ after the
        # step, so Δθ exists for the last epoch too. An attribute, not a config key,
        # for the same reason as record_states: it changes what is kept, never what
        # is computed.
        self.record_market = False
        self.market_log: list[dict[str, Any]] = []

        # One entry per turnover run_multigen performed: {"generation", "rows"}.
        self.turnover_log: list[dict[str, Any]] = []
        # The noise-to-signal ratio each generation of run_multigen started with.
        self.noise_ratio_by_generation: list[float] = []

        # Exogenous status drift (THEORY.md §10.6), in raw status units per epoch:
        # the configured gamma is in epoch-0 population sd, converted once here so
        # the step does not change as the drift itself moves the sd.
        self._status_step = (float(self.config.simulation.status_drift)
                             * float(np.std(self.states[:, 2], ddof=1)))

        # Survival parameters for defection. Both environments run the same
        # formula (THEORY.md §10.3); A differs only in d = 0. Without an
        # environment_b block there are no survival parameters at all, so no
        # dissolution is drawn and the defection term is zero — stated, not
        # silently defaulted.
        self._env = self.config.environment_b

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

    def _refresh_after_drift(self) -> None:
        """Rebuild everything computed from traits once, now that a trait moved.

        Three caches assume traits are fixed: the full-information benchmark, B's
        utility table, and nothing else. Both are rebuilt rather than kept, because
        a stale benchmark would report algorithmic_loss against a market that no
        longer exists, and a stale table would rank B's candidates on epoch-0
        status. Each costs well under a second at N = 1000.
        """
        self._benchmark_ready = False
        self._full_prefs = None
        self._ensure_benchmark()
        if self._utility_table is not None:
            self._utility_table = utility_table(self.states, self.config_dict)

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
        elif self.recommender == "complete":
            env = self.config.environment_b
            assert env is not None and self.subgraph is not None
            assert self._utility_table is not None
            built = complete_prefs(self.states, self._utility_table, self.subgraph,
                                   top_k, env.signal_cost)
            prop_prefs, acc_prefs = built.prop_prefs, built.acc_prefs
            self._complete_proposals = built.proposals
        elif self.recommender == "random":
            prop_prefs, acc_prefs = self._random_prefs(top_k, epoch)
        elif self.recommender == "popularity":
            prev_prop, prev_acc = prev if prev is not None else (None, None)
            prop_prefs, acc_prefs = self._popularity_prefs(top_k, prev_prop, prev_acc)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unhandled arm {self.recommender!r}")

        return prop_prefs, acc_prefs, self._exposure_from_dense(prop_prefs, acc_prefs)

    def _restrict_to_proposals(
        self,
        prop_prefs: list[list[int]],
        acc_prefs: list[list[int]],
        proposals: np.ndarray,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Keep only pairs the *proposer* proposed, in the given order.

        Below c = 0 this is not called and the lists are untouched, so every
        pre-signalling number is reproduced exactly.

        Above c = 0 the reachable set is the proposal set, and the reason is not a
        convention: a proposal costs the proposer ``c``, so it is the proposer's own
        act. A recommender can rank the candidates in front of an agent; it cannot
        spend on their behalf. Letting the NCF shortlist pairs the agent declined to
        pay for would make the cost decorative — the matching would fill up with
        pairs the model says nobody would have proposed, and a prohibitive cost
        would still produce a clearing market.

        **Only the proposer's side is restricted.** The acceptor's list is a ranking,
        and her willingness is already what "accept" means in deferred acceptance;
        intersecting both lists instead would require a *mutual* top-k positive,
        which at k = 20 of 100 candidates thins the market by a factor of five long
        before the cost bites — measured as mcr falling from 0.21 to 0.0 at c = 0.5,
        a fifth of the way to the separating cutoff. That is a different experiment
        (a mutual-consent market), not this one.

        Which side is "the proposer" is M by construction (§5 of this file's
        contract: M proposes, W accepts). A W-proposing arm would restrict the other
        list; that is the same code with the arguments swapped.

        Args:
            proposals: (P, 2) array of *row* ids, [proposer_row, acceptor_row].
        """
        if proposals.shape[0] == 0:
            return [[] for _ in prop_prefs], [[] for _ in acc_prefs]

        # proposer row -> the set of acceptor slots it proposed to. A dict of sets
        # rather than a row-indexed array: a proposer appears once per proposal, so
        # an array indexed by row would keep only the last one — which is exactly
        # the bug this shape avoids.
        proposed: dict[int, set[int]] = {}
        for u, v in proposals:
            proposed.setdefault(int(u), set()).add(int(self._acc_slot[int(v)]))

        out_prop: list[list[int]] = [
            [a for a in lst if a in proposed.get(int(row), ())]
            for row, lst in zip(self.prop_rows, prop_prefs)
        ]

        # The acceptor's list is kept whole and in order: it is her ranking, and
        # the restriction is about who spent the cost, not about who may say yes.
        return out_prop, [list(lst) for lst in acc_prefs]

    def _accept_prob_from_log(
        self, prev_log: np.ndarray | None, prev: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Per-side acceptance probability for the *next* epoch's proposal decision.

        p is the previous epoch's empirical accept rate, measured on the swipe log
        rather than assumed: among the positive swipes a side made, the share that
        the other agent reciprocated. That is the closest thing the model has to
        "a proposal was accepted", it is observable without a matching step, and it
        is available for the epoch about to run — the C++ decision happens before
        deferred acceptance, so a rate taken from this epoch's matches could not be
        used without a second pass.

        Epoch 0 has no history. It uses 0.5 for both sides and says so; the
        constant is not a claim about the market, it is the neutral prior that
        makes the first epoch's decision non-degenerate.

        A side with no positive swipes this epoch has no *new* measurement, and
        keeps the p it already had (``prev``). It does not fall back to 0.5: that
        reset is what made the market cycle. At c = 1 the measured rate is about
        0.015, so τ* = c / (V̄ p) is above every agent's intent and the next log is
        empty; a reset to 0.5 then reopened the market, the reopened market
        measured 0.015 again, and every odd epoch was empty. "Nobody proposed" is
        not evidence that proposals have become likelier. Only a side that has
        never been measured — ``prev`` is None — uses the 0.5 prior.
        """
        prior = {"accept_prob_m": 0.5, "accept_prob_w": 0.5}
        prev = prior if prev is None else prev
        if prev_log is None or prev_log.shape[0] == 0:
            return dict(prev)

        users = prev_log[:, 0].astype(np.int64)
        items = prev_log[:, 1].astype(np.int64)
        positive = prev_log[:, 2] == 1
        users, items = users[positive], items[positive]

        # Reciprocity: a pair (u, v) is a mutual positive when the reverse row
        # (v, u) is also positive. Encoded as a single int64 key so the lookup is a
        # set membership rather than a Python dict over tuples.
        keys = users * np.int64(self.n) + items
        mutual = np.isin(items * np.int64(self.n) + users, keys)

        out: dict[str, float] = {}
        for side, rows in (("m", self.prop_rows), ("w", self.acc_rows)):
            mask = np.isin(users, rows)
            n_sent = int(np.count_nonzero(mask))
            # A side that made no positive swipe has no new measurement and keeps
            # the p it had (see docstring).
            key = f"accept_prob_{side}"
            out[key] = (
                float(np.count_nonzero(mutual & mask)) / float(n_sent) if n_sent else prev[key]
            )
        return out

    def _mean_admissible_utility_by_side(self) -> tuple[float, float]:
        """V̄ per proposing side: the pooled mean over θ-admissible candidates.

        Pooled by admissible-set size, not as a mean of per-agent means: an agent
        with three admissible candidates and one with two hundred should not carry
        equal weight in the number the cost is compared against.

        A side with no admissible candidate at all yields NaN here, which
        ``separating_threshold`` turns into +inf — the right reading, since no
        proposal is possible at any finite cost.
        """
        means, counts = mean_admissible_utility(self.states, self.config_dict)
        out = []
        for rows in (self.prop_rows, self.acc_rows):
            c = counts[rows]
            total = float(c.sum())
            if total <= 0.0:
                out.append(float("nan"))
            else:
                ok = np.isfinite(means[rows])
                out.append(float(np.sum(means[rows][ok] * c[ok]) / total))
        return out[0], out[1]

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

    def run(self, n_epochs: int, *, start_epoch: int = 0) -> pd.DataFrame:
        """Run ``n_epochs`` epochs and return the §6 metrics frame.

        ``start_epoch`` is the absolute index of the first epoch, so a caller that
        runs one generation at a time (``run_multigen``) gets an ``epoch`` column
        that continues across calls rather than restarting at 0. Everything keyed
        on the epoch — the swipe stream, the random arm's draw, the defection draw,
        status drift, and the ``generation`` label — reads the absolute index, so a
        generation run at ``start_epoch = 40`` sees exactly the streams epochs 40..
        of one long run would. At the default 0 this is the single-population path,
        unchanged.

        Per-call history — the popularity arm's in-degrees and the measured accept
        rate — starts fresh on every call. Across a generation boundary that is
        the right reading rather than a shortcut: a fifth of the slots now hold
        different agents, and an in-degree carried over would credit a newcomer
        with the popularity of the agent it replaced.
        """
        if n_epochs < 0:
            raise ValueError(f"n_epochs must be >= 0; got {n_epochs}")
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0; got {start_epoch}")

        ok_benchmark = self._ensure_benchmark()
        top_k = int(self.config.ncf.top_k)
        eta = float(self.config.simulation.bayesian_eta)
        flooding_w = float(self.config.hypergamy.flooding_weight)

        # The config's n_epochs is the full generation schedule, so a shorter run
        # still maps epochs onto the same generations a full run would.
        epochs_per_gen = max(1, int(self.config.simulation.n_epochs) //
                             int(self.config.simulation.n_generations))

        rows: list[dict[str, Any]] = []
        series: list[np.ndarray] = []
        self.states_seen = []
        self.outcomes = []
        self.market_log = []
        prev_prefs: tuple[list[list[int]], list[list[int]]] | None = None

        # p for each proposing side, from the previous epoch's log. None means "no
        # history", which _accept_prob_from_log maps to 0.5 for both sides.
        accept_prob: dict[str, float] = {"accept_prob_m": 0.5, "accept_prob_w": 0.5}
        # V̄ per side, recomputed when θ moves: θ is the gate that defines the
        # admissible pool, so a stale V̄ would report a threshold against a pool the
        # agent no longer has. Recomputed only on the epochs that actually need it.
        v_bar_m, v_bar_w = self._mean_admissible_utility_by_side()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            # Exogenous drift (THEORY.md §10.6), applied before anything reads the
            # states so the epoch sees s_w(t) = s_w(0) + gamma*t. Epoch 0 is never
            # shifted, which is what lets the four factorial cells assert one shared
            # population at their first epoch. Traits are re-standardized from the
            # states on every C++ call, so moving W's raw status moves her
            # standardized status *and* the population mean the cutoff is read
            # against — the compositional channel, with no algorithm involved.
            if self._status_step != 0.0 and epoch > 0:
                self.states[self.acc_rows, 2] += self._status_step
                self._refresh_after_drift()
                v_bar_m, v_bar_w = self._mean_admissible_utility_by_side()

            # Recorded before anything touches the states, so this is the matrix
            # the epoch's preference rule actually saw. All four arms must agree
            # at epoch 0; with freeze_theta they must agree at every epoch.
            if self.record_states:
                self.states_seen.append(self.states.copy())

            # The config the C++ decision reads. A copy per epoch: signal_cost is
            # fixed but accept_prob is this epoch's measurement, and the state
            # matrix must not be mutated by a config write.
            epoch_config = dict(self.config_dict)
            epoch_config["signal"] = dict(accept_prob)

            n_swipes = 0
            swipe_log = None
            if self.recommender == "ncf":
                swipe_log = generate_swipes(self.states, epoch_config, epoch)
                self.trainer.train_epoch(swipe_log)
                n_swipes = len(swipe_log)

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

            # ── The proposal set constrains the matching (THEORY.md §10.2) ────
            # At c > 0 only proposed pairs are reachable, for every arm: the cost
            # is the proposer's own, so a recommender ranks but cannot spend. The
            # NCF arm also builds the swipe log here when c > 0, because the log is
            # what says which pairs were proposed — at c = 0 it is not built for
            # the non-NCF arms and nothing is restricted, so those arms are
            # bit-identical to before.
            is_complete = self.recommender == "complete"
            # The complete arm pays c_B, not A's cost, and has already applied it
            # inside complete_prefs: the cost is the proposer's act, so the lists it
            # built are the proposal set. Running A's restriction on top would apply
            # a second cost, from a different parameter, to the same pairs.
            cost = (float(self.config.environment_b.signal_cost) if is_complete
                    else float(self.config.hypergamy.signal_cost))
            proposal_log = swipe_log
            if cost > 0.0 and proposal_log is None and not is_complete:
                # Every arm needs the same proposal set, so it comes from the same
                # function the NCF arm uses, on the same states and epoch. This is
                # the only extra O(N²) call the non-NCF arms pay, and only when a
                # cost is switched on.
                proposal_log = generate_swipes(self.states, epoch_config, epoch)

            n_proposals = 0
            if is_complete:
                proposals = self._complete_proposals
                n_proposals = int(proposals.shape[0])
            elif proposal_log is not None and proposal_log.shape[0]:
                positive = proposal_log[:, 2] == 1
                prop_rows = proposal_log[positive, 0]
                acc_rows = proposal_log[positive, 1]
                n_proposals = int(prop_rows.size)
                proposals = np.stack([prop_rows, acc_rows], axis=1).astype(np.int64) \
                    if n_proposals else np.zeros((0, 2), dtype=np.int64)
            else:
                proposals = np.zeros((0, 2), dtype=np.int64)

            if cost > 0.0 and not is_complete:
                prop_prefs, acc_prefs = self._restrict_to_proposals(
                    prop_prefs, acc_prefs, proposals
                )
                # Exposure is a property of the lists the matching actually saw, so
                # it is recomputed after the restriction rather than before. The
                # alternative — keeping the pre-restriction counts — would report a
                # Gini over a market that no longer exists.
                exposure = self._exposure_from_dense(prop_prefs, acc_prefs)

            prev_prefs = (prop_prefs, acc_prefs)
            series.append(exposure)

            da = deferred_acceptance(prop_prefs, acc_prefs, True)
            matches = da["matches"]
            n_matched = int(matches.shape[0])

            welfare_alg = float(compute_welfare(self.states, matches,
                                                self.config_dict)["welfare"])

            # ── §10.4 welfare, per agent and in total ─────────────────────────
            # The matching is in dense index space; the payoff vector and the
            # defection draw are in rows. Defection uses B's d only in B: A runs
            # the same survival formula with d = 0 (THEORY.md §10.3).
            pair_rows = np.stack([self.prop_rows[matches[:, 0]],
                                  self.acc_rows[matches[:, 1]]], axis=1) \
                if n_matched else np.zeros((0, 2), dtype=np.int64)
            if self._env is not None:
                penalty = float(self._env.defection_penalty) if is_complete else 0.0
                env = self._env
            else:
                penalty, env = 0.0, _NO_DEFECTION
            payoff, defection = agent_payoffs(
                self.states, self.config_dict, pair_rows, proposals, cost,
                env, penalty, self.rng_seed, epoch,
            )
            loss = defection.loss
            w10 = compute_welfare(self.states, matches, self.config_dict,
                                  cost, n_proposals, loss)
            # The vector and the C++ sum are two routes to one number; they may
            # differ in summation order and nothing else.
            if not np.isclose(payoff.sum(), w10["welfare_total"], rtol=1e-12, atol=1e-9):
                raise RuntimeError(
                    f"epoch {epoch}: per-agent payoff sums to {payoff.sum()!r} but "
                    f"compute_welfare reports {w10['welfare_total']!r}"
                )
            if self.record_outcomes:
                matched_mask = np.zeros(self.n, dtype=bool)
                matched_mask[pair_rows.ravel()] = True
                entrant = np.ones(self.n, dtype=bool)
                if is_complete:
                    assert self.subgraph is not None
                    entrant[:] = False
                    entrant[self.subgraph.rows_m] = True
                    entrant[self.subgraph.rows_w] = True
                self.outcomes.append(EpochOutcome(
                    states=self.states.copy(), payoff=payoff, matched=matched_mask,
                    entrant=entrant, seed=self.rng_seed, epoch=epoch,
                    prop_prefs=prop_prefs, acc_prefs=acc_prefs,
                    proposals=np.asarray(proposals, dtype=np.int64).reshape(-1, 2),
                    cost=cost, pairs=pair_rows.astype(np.int64),
                ))

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

            # ── Signalling columns (THEORY.md §10.2) ──────────────────────────
            # The proposals are the label-1 rows of the epoch's swipe log, computed
            # above. An arm with no log and c = 0 made no proposal, which is why it
            # reports 0 cost and NaN intent rather than a plausible number.
            if n_proposals:
                proposer_rows = proposals[:, 0]
                mpi = mean_proposal_intent(self.states[:, 11], proposer_rows)
            else:
                n_proposals = 0
                mpi = float("nan")

            # τ* per side. One column in the §6 frame, so the two sides are pooled
            # by proposal count: the number a reader wants next to E[τ | proposal]
            # is the cut that produced those proposals, weighted by who proposed.
            # Both sides are also logged at INFO so the paired numbers survive a
            # run without a second column.
            n_m = int(np.count_nonzero(np.isin(proposer_rows, self.prop_rows))) if n_proposals else 0
            n_w = n_proposals - int(n_m)

            def _tau_star(v_bar: float, p: float) -> float:
                return separating_threshold(cost, v_bar, p)

            tau_m = _tau_star(v_bar_m, accept_prob["accept_prob_m"])
            tau_w = _tau_star(v_bar_w, accept_prob["accept_prob_w"])
            weights = np.array([n_m, n_w], dtype=np.float64)
            if n_proposals > 0 and weights.sum() > 0:
                taus = np.array([tau_m, tau_w], dtype=np.float64)
                finite = np.isfinite(taus)
                tau_star = (
                    float(np.average(taus[finite], weights=weights[finite]))
                    if finite.any() else float("inf")
                )
            else:
                # No proposals to weight by. τ* is still defined — it is c/(V̄·p),
                # a property of the market and the cost, not of what anyone did —
                # so it is reported as the unweighted mean of the finite sides
                # rather than as NaN. NaN here would hide the very case the
                # no-proposal regime is about: a threshold high enough to empty the
                # market should read as a large number, not as "not measured".
                taus = np.array([tau_m, tau_w], dtype=np.float64)
                finite = np.isfinite(taus)
                tau_star = float(np.mean(taus[finite])) if finite.any() else float("inf")
            log.info(
                "epoch %d: c=%.4g p_m=%.4f p_w=%.4f Vbar_m=%.4f Vbar_w=%.4f "
                "tau*_m=%.4f tau*_w=%.4f proposals=%d",
                epoch, cost, accept_prob["accept_prob_m"], accept_prob["accept_prob_w"],
                v_bar_m, v_bar_w, tau_m, tau_w, n_proposals,
            )

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
                "mean_proposal_intent": mpi,
                "separating_threshold": tau_star,
                "proposal_cost_total": proposal_cost_total(cost, n_proposals),
                "value_created": float(w10["value_created"]),
                "defection_loss": float(w10["defection_loss"]),
                "welfare_total": float(w10["welfare_total"]),
                # Filled in by comparison.paired_run, which is the only caller that
                # holds the A outcome every unmatched B agent is scored at.
                "welfare_common": float("nan"),
                "welfare_selected": float("nan"),
                "welfare_selection_artifact": float("nan"),
            })

            # θ ← θ + η·(matches/proposals − θ), in place on states[:, 6].
            # Skipped under freeze_theta, which is a diagnostic that holds the
            # market still so the arms can be compared at identical states rather
            # than only at identical epoch-0 states.
            prop_by_row = self._proposals_by_row(da["proposals"])
            match_by_row = self._matches_by_row(matches)

            # The flooding term reads the proposals each agent *received* this
            # epoch, so it is computed from the states the epoch saw, before the
            # yield step moves θ. An arm that built no swipe log at c = 0 made no
            # proposals, so at w > 0 its log is built here by the same call the
            # c > 0 path uses; at w = 0 nothing below runs and no extra draw is made.
            flood = None
            if flooding_w != 0.0 and not self.freeze_theta:
                if is_complete or proposal_log is not None:
                    flood_pairs = proposals
                else:
                    log_w = generate_swipes(self.states, epoch_config, epoch)
                    flood_pairs = log_w[log_w[:, 2] == 1, :2]
                volume, recv_intent = received_proposals(self.states, flood_pairs)
                flood = flooding_delta(
                    volume, recv_intent, (self.prop_rows, self.acc_rows), flooding_w,
                    float(self.states[:, 11].mean()),
                )

            if not self.freeze_theta:
                update_thresholds(self.states, prop_by_row, match_by_row, eta)
                if flood is not None:
                    self.states[:, 6] += flood
                # θ moved, so the admissible pool moved with it. V̄ is a mean over
                # that pool and would otherwise be a number about a market that no
                # longer exists.
                v_bar_m, v_bar_w = self._mean_admissible_utility_by_side()

            if self.record_market:
                self.market_log.append({
                    "epoch": epoch,
                    "swipe_log": None if swipe_log is None else swipe_log.copy(),
                    "proposals": np.asarray(proposals, dtype=np.int64).reshape(-1, 2).copy(),
                    "da_proposals": prop_by_row.copy(),
                    "matches": match_by_row.copy(),
                    "theta_after": self.states[:, 6].copy(),
                    "flood": None if flood is None else flood.copy(),
                })

            # Next epoch's p, measured on this epoch's log. Read after the matching
            # step on purpose: the log is this epoch's behavior, and nothing after
            # the swipe call mutates it.
            if swipe_log is not None:
                accept_prob = self._accept_prob_from_log(swipe_log, accept_prob)

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

    # ------------------------------------------------------ generational turnover

    def run_multigen(self) -> pd.DataFrame:
        """``simulation.n_generations`` generations with turnover between them.

        Each generation is one ``run()`` call of ``n_epochs // n_generations``
        epochs, started at the absolute epoch where the previous one ended. There
        is no second epoch loop: ``run()`` derives ``generation`` from the absolute
        epoch and fits ``pareto_alpha`` on each generation's final row, so both
        come out of the one place they were already computed.

        Between generations, ``_turnover`` replaces the bottom
        ``turnover_fraction`` of agents by cumulative match count (THEORY.md §8).
        There is no turnover after the last generation: nothing would observe it.

        ``exposure_series``, ``states_seen``, ``outcomes`` and ``market_log`` hold the
        whole run, all generations in epoch order, as they would after one long
        ``run()``.
        """
        sim_cfg = self.config.simulation
        n_epochs, n_gen = int(sim_cfg.n_epochs), int(sim_cfg.n_generations)
        # run() maps epochs to generations as epoch // (n_epochs // n_generations).
        # A remainder would either be dropped or spill into a generation n_gen that
        # was never configured, so it is refused rather than resolved silently.
        if n_epochs < n_gen or n_epochs % n_gen != 0:
            raise ValueError(
                f"run_multigen needs simulation.n_epochs ({n_epochs}) to be a positive "
                f"multiple of simulation.n_generations ({n_gen})"
            )
        epochs_per_gen = n_epochs // n_gen

        frames: list[pd.DataFrame] = []
        series: list[np.ndarray] = []
        states_seen: list[np.ndarray] = []
        outcomes: list[EpochOutcome] = []
        market_log: list[dict[str, Any]] = []
        self.turnover_log = []
        # Newcomers change the population's utility spread, so the noise-to-signal
        # ratio is a per-generation number, recorded for each one as it starts.
        self.noise_ratio_by_generation = []
        for g in range(n_gen):
            if g > 0:
                self._turnover(g)
            self.noise_ratio_by_generation.append(self.noise_ratio)
            frames.append(self.run(epochs_per_gen, start_epoch=g * epochs_per_gen))
            series.append(self.exposure_series)
            states_seen.extend(self.states_seen)
            outcomes.extend(self.outcomes)
            market_log.extend(self.market_log)

        self.exposure_series = np.concatenate(series)
        self.states_seen = states_seen
        self.outcomes = outcomes
        self.market_log = market_log

        frame = pd.concat(frames, ignore_index=True)
        if not np.array_equal(frame["epoch"].to_numpy(), np.arange(len(frame))):
            raise RuntimeError("run_multigen: epoch is not contiguous across generations")
        return frame

    def _turnover(self, generation: int) -> np.ndarray:
        """Replace the bottom ``turnover_fraction`` of agents; return their rows.

        Called before ``generation`` starts. The rules, each tested in
        tests/python/test_multigen.py:

        * **Who leaves.** The ``round(turnover_fraction * N)`` agents with the fewest
          cumulative matches. Ties — and at a clearing rate near a quarter most
          agents are tied at zero or one — are broken by a permutation seeded on
          (seed, generation), not by row: rows below N/2 are all M, so a row
          tie-break would retire one side first.
        * **Slot, not traits.** A newcomer takes the leaver's row, so id, side and
          every dense index stay valid. Everything else in the row is new: traits,
          taste multipliers, key and the institutional columns all come from a
          fresh population drawn from the ORIGINAL distributions at a seed derived
          from (seed, generation) — reproducible, and not a repeat of generation 0.
        * **θ is recalibrated, not inherited.** A newcomer carrying a rejected
          agent's θ would start life already pessimistic. It is calibrated against
          the market it actually enters (survivors plus newcomers), by the same
          quantile rule the sampler uses; survivors keep the θ they have earned.
        * **A new key** (THEORY.md §2.3), so the newcomer's chemistry is fresh with
          everyone and no other pair's chemistry moves — every other pair is a
          function of two keys, neither of which changed.
        * **Cumulative matches reset to 0**, or the next Pareto fit would measure
          the dead agent's history plus the newcomer's.
        * **The recommender forgets the slot.** Its embedding rows are re-drawn, for
          the same reason θ is not inherited: a newcomer holding a rejected agent's
          embedding would start out ranked as that agent was.

        Survivors' rows are not written at all, so they are bit-identical.
        """
        n_replace = int(round(float(self.config.simulation.turnover_fraction) * self.n))
        if n_replace <= 0:
            self.turnover_log.append({"generation": generation,
                                      "rows": np.zeros(0, dtype=np.int64)})
            return np.zeros(0, dtype=np.int64)

        tiebreak = np.random.default_rng([self.rng_seed, generation, _TURNOVER_TAG])
        order = np.lexsort((tiebreak.permutation(self.n), self._cumulative_matches))
        rows = np.sort(order[:n_replace]).astype(np.int64)

        fresh_seed = _generation_seed(self.rng_seed, generation)
        fresh = sample_population(self.n, self.config_dict, fresh_seed)
        # Same rows in both matrices are the same slot: the sampler lays out M then
        # W by row, so a row's side does not depend on the seed.
        if not np.array_equal(fresh[:, :2], self.states[:, :2]):
            raise RuntimeError("turnover: the fresh population's id/side layout differs")
        self.states[rows] = fresh[rows]

        # θ: the configured cutoff verbatim if one is set, otherwise the quantile
        # rule, computed on a copy so that only the newcomers' column is taken.
        if self.config.population.init_threshold is not None:
            self.states[rows, 6] = float(self.config.population.init_threshold)
        else:
            calibrated = self.states.copy()
            calibrate_thresholds(calibrated, self.config_dict,
                                 float(self.config.population.admissible_fraction))
            self.states[rows, 6] = calibrated[rows, 6]

        keys = self.states[:, 10]
        if np.unique(keys).size != keys.size:
            # 52-bit keys over N agents: a collision is ~N²/2^53, but a shared key
            # would give two agents identical chemistry with everyone, so it is
            # checked rather than assumed.
            raise RuntimeError(f"turnover: duplicate key after generation {generation}")

        self._cumulative_matches[rows] = 0
        self.trainer.reinitialize(rows, seed=fresh_seed)

        # Caches built from traits. The benchmark and B's utility table are the two
        # _refresh_after_drift already knows about; B's membership also depends on
        # the newcomers' values, so the subgraph is rebuilt with them.
        self._refresh_after_drift()
        if self.recommender == "complete":
            self.subgraph = build_subgraph(self.states, self.config.require_environment_b(),
                                           self.rng_seed)
        self._log_noise_ratio()

        self.turnover_log.append({"generation": generation, "rows": rows})
        log.info("generation %d: replaced %d agents (seed %d)", generation, rows.size,
                 fresh_seed)
        return rows


# Third word of the tie-break seed, so it never coincides with the random arm's
# (seed, epoch) generator.
_TURNOVER_TAG = 0x7E4E0FE5


def _generation_seed(seed: int, generation: int) -> int:
    """The population seed newcomers of ``generation`` are drawn at.

    Derived through a SeedSequence rather than as ``seed + generation``: seed 42's
    generation 1 would otherwise be seed 43's generation 0, and a seed sweep would
    reuse one population as another run's newcomers.
    """
    state = np.random.SeedSequence([int(seed), int(generation), _TURNOVER_TAG])
    return int(state.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
