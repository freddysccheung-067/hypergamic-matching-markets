"""Environment B: the constrained subgraph (THEORY.md §10.3, §10.5).

Four pieces, in the order the theory introduces them:

  * ``build_subgraph``   who is eligible, side-balanced, with the balancing reported;
  * ``complete_prefs``   complete-information ranking inside the subgraph, truncated
                         at k — the list rule behind ``recommender="complete"``;
  * ``resolve_defection`` the survival formula, shared by both environments;
  * ``thin_market_curve`` clearance, acceptability and critical mass against size.

Nothing here owns a second copy of the market. The orchestrator's ``"complete"`` arm
and the thin-market sweep both call ``complete_prefs`` and ``deferred_acceptance``,
so the number the sweep reports is the number a ``Simulation`` produces.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from src.python.config import EnvironmentBConfig

try:  # the extension lands in build/Release (README "Installation")
    from _hypergamic import deferred_acceptance, utility_for_pairs
except ImportError:  # pragma: no cover - convenience for a fresh checkout
    _BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "Release"
    if _BUILD.is_dir():
        sys.path.insert(0, str(_BUILD))
    from _hypergamic import deferred_acceptance, utility_for_pairs  # noqa: F401

log = logging.getLogger(__name__)

# Column indexes of the state matrix (docs/INTERFACES.md §1).
_TYPE, _THETA, _INTENT, _VALUES = 1, 6, 11, 12

# p in the proposal decision tau * V * p > c. Epoch-0 neutral prior of THEORY.md
# §10.2. Inside B there is no previous-epoch swipe log to measure it on, so it stays
# at the prior rather than at a number nobody observed.
NEUTRAL_ACCEPT_PROB = 0.5


# ── 1. Subgraph construction ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SubgraphReport:
    """What the construction did, so no clearing number is read against a market
    whose composition was never stated."""

    n_pool_m: int          # drawn into the pool, before the filter
    n_pool_w: int
    n_pass_m: int          # passed g >= values_min
    n_pass_w: int
    dropped_side: str | None   # "M" / "W" when balancing removed anyone, else None
    dropped_ids: np.ndarray    # row ids removed to balance, ascending
    n_eligible: int            # |B| after balancing

    @property
    def balanced_by_dropping(self) -> bool:
        return self.dropped_ids.size > 0


@dataclass(frozen=True)
class Subgraph:
    rows_m: np.ndarray   # row ids, ascending
    rows_w: np.ndarray
    report: SubgraphReport

    @property
    def n_eligible(self) -> int:
        return int(self.rows_m.size + self.rows_w.size)


def build_subgraph(
    states: np.ndarray,
    env: EnvironmentBConfig,
    seed: int,
    replicate: int = 0,
    warn: bool = True,
) -> Subgraph:
    """The agents eligible for Environment B.

    An agent is eligible when it is in the size_fraction pool AND ``g >= values_min``
    AND the two sides end up the same size. Each side's pool is a seeded permutation
    prefix of that side (keyed on ``(seed, side, replicate)``), so the pool is a
    random share of the population rather than the first agents in id order;
    ``replicate`` exists so the thin-market sweep can average over pools.

    **Balancing is reported, never silent.** If the filter leaves one side larger,
    the excess is dropped from the larger side, highest ids first. That rule is a
    function of the ids alone, so the same population gives the same subgraph. The
    drop is returned in ``report`` and logged at WARNING: a silently unbalanced
    subgraph would make every clearing number in B uninterpretable, because the
    unmatched surplus side would depress clearance for a reason that has nothing to
    do with information or size.

    ``warn=False`` is for the thin-market sweep, which builds hundreds of pools and
    reports the drops as a column (``n_dropped``) instead of one log line each. The
    report is returned either way; ``warn`` only controls the log.
    """
    types = states[:, _TYPE]
    sides = (np.flatnonzero(types == 0.0), np.flatnonzero(types == 1.0))
    values = states[:, _VALUES]

    pooled: list[np.ndarray] = []
    passed: list[np.ndarray] = []
    for side_idx, rows in enumerate(sides):
        take = max(1, int(round(env.size_fraction * rows.size)))
        if take >= rows.size:
            pool = rows
        else:
            rng = np.random.default_rng([int(seed), side_idx, int(replicate)])
            pool = np.sort(rng.permutation(rows)[:take])
        pooled.append(pool)
        passed.append(pool[values[pool] >= env.values_min])

    rows_m, rows_w = passed
    dropped_side: str | None = None
    dropped = np.zeros(0, dtype=np.int64)
    if rows_m.size != rows_w.size:
        keep = min(rows_m.size, rows_w.size)
        if rows_m.size > rows_w.size:
            dropped_side, dropped, rows_m = "M", rows_m[keep:], rows_m[:keep]
        else:
            dropped_side, dropped, rows_w = "W", rows_w[keep:], rows_w[:keep]
        if warn:
            log.warning(
                "environment_b: filter left the subgraph unbalanced (%d M, %d W passed "
                "g >= %.3g); dropped %d %s agents by highest id to balance at %d per side",
                passed[0].size, passed[1].size, env.values_min,
                dropped.size, dropped_side, keep,
            )

    if rows_m.size == 0:
        raise ValueError(
            f"environment_b subgraph is empty (size_fraction={env.size_fraction}, "
            f"values_min={env.values_min}); loosen the filter or enlarge the pool"
        )

    report = SubgraphReport(
        n_pool_m=int(pooled[0].size), n_pool_w=int(pooled[1].size),
        n_pass_m=int(passed[0].size), n_pass_w=int(passed[1].size),
        dropped_side=dropped_side, dropped_ids=dropped.astype(np.int64),
        n_eligible=int(rows_m.size + rows_w.size),
    )
    return Subgraph(rows_m=rows_m.astype(np.int64), rows_w=rows_w.astype(np.int64),
                    report=report)


# ── 2. Complete-information visibility ───────────────────────────────────────


def utility_table(states: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Full pair utilities, as dense-slot matrices ``(U_mw, U_wm)``.

    ``U_mw[p, a]`` is proposer p's utility toward acceptor a, ``U_wm[a, p]`` the
    reverse. Computed on the *whole* population's standardization, which is what
    ``generate_swipes`` and ``compute_welfare`` use — a subgraph that restandardized
    on its own members would be on a different utility scale from A, and every
    cross-environment comparison would be off by that rescaling. O(N²) memory, the
    same cap as the full-information benchmark (INTERFACES.md §4).
    """
    types = states[:, _TYPE]
    m_rows = np.flatnonzero(types == 0.0).astype(np.int32)
    w_rows = np.flatnonzero(types == 1.0).astype(np.int32)
    mm, ww = np.meshgrid(m_rows, w_rows, indexing="ij")
    pairs = np.stack([mm.ravel(), ww.ravel()], axis=1)
    u_mw = np.asarray(utility_for_pairs(states, pairs, config)).reshape(m_rows.size, w_rows.size)
    u_wm = np.asarray(
        utility_for_pairs(states, pairs[:, ::-1].copy(), config)
    ).reshape(m_rows.size, w_rows.size).T.copy()
    return u_mw, u_wm


@dataclass(frozen=True)
class CompletePrefs:
    prop_prefs: list[list[int]]    # dense acceptor slots, one list per proposer
    acc_prefs: list[list[int]]     # dense proposer slots, one list per acceptor
    proposals: np.ndarray          # (P, 2) row ids [proposer, acceptor] that survived the cost


def complete_prefs(
    states: np.ndarray,
    table: tuple[np.ndarray, np.ndarray],
    subgraph: Subgraph,
    top_k: int,
    signal_cost: float,
    accept_prob: float = NEUTRAL_ACCEPT_PROB,
) -> CompletePrefs:
    """Every eligible agent ranks every eligible agent of the other side, top k.

    Complete information: the list is built from the *true* utility, with no
    recommender in between. Same rule as the oracle arm — θ-admissible candidates,
    descending utility, ties by ascending dense index, truncated at k — but over the
    subgraph's members only, which is the whole difference between the environments:
    B sees its pool perfectly, A sees a market it cannot enumerate.

    Lists are returned in the *full* dense index space with empty lists for agents
    outside B, so ``deferred_acceptance`` and the exposure count run unchanged.

    At ``signal_cost > 0`` the proposer side is further cut to the candidates
    satisfying ``tau * V * p > c`` after truncation, exactly the order
    ``generate_swipes`` applies it in. The acceptor's list stays whole: the cost is
    paid by whoever proposes (THEORY.md §10.2).
    """
    u_mw, u_wm = table
    types = states[:, _TYPE]
    n_prop = int((types == 0.0).sum())
    n_acc = int((types == 1.0).sum())
    prop_slot = np.full(states.shape[0], -1, dtype=np.int64)
    acc_slot = np.full(states.shape[0], -1, dtype=np.int64)
    prop_slot[np.flatnonzero(types == 0.0)] = np.arange(n_prop)
    acc_slot[np.flatnonzero(types == 1.0)] = np.arange(n_acc)

    m_slots = prop_slot[subgraph.rows_m]
    w_slots = acc_slot[subgraph.rows_w]
    theta = states[:, _THETA]
    intent = states[:, _INTENT]

    def rank(u_row: np.ndarray, bar: float) -> np.ndarray:
        """Indexes into the candidate slice, best first, ties to the lower index."""
        ok = np.flatnonzero(u_row >= bar)
        # A stable sort on -u keeps ascending index among equal utilities.
        return ok[np.argsort(-u_row[ok], kind="stable")][:top_k]

    prop_prefs: list[list[int]] = [[] for _ in range(n_prop)]
    acc_prefs: list[list[int]] = [[] for _ in range(n_acc)]
    proposed: list[tuple[int, int]] = []

    for row, slot in zip(subgraph.rows_m, m_slots):
        u_row = u_mw[slot, w_slots]
        order = rank(u_row, theta[row])
        if signal_cost > 0.0 and order.size:
            keep = intent[row] * u_row[order] * accept_prob > signal_cost
            order = order[keep]
        prop_prefs[int(slot)] = [int(w_slots[i]) for i in order]
        proposed.extend((int(row), int(subgraph.rows_w[i])) for i in order)

    for row, slot in zip(subgraph.rows_w, w_slots):
        order = rank(u_wm[slot, m_slots], theta[row])
        acc_prefs[int(slot)] = [int(m_slots[i]) for i in order]

    proposals = (np.asarray(proposed, dtype=np.int64) if proposed
                 else np.zeros((0, 2), dtype=np.int64))
    return CompletePrefs(prop_prefs, acc_prefs, proposals)


# ── 3. Defection ─────────────────────────────────────────────────────────────

_U64 = np.uint64


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        return x ^ (x >> _U64(31))


def survival_uniform(seed: int, epoch: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """U(0,1) draw per pair, a pure function of ``(seed, epoch, i, j)``.

    Hash-based rather than a stateful generator, so the draw for a pair does not
    depend on how many pairs were drawn before it or in what order: reordering the
    match list, or resolving a subset, gives each pair the same number. The pair is
    canonicalised to ``(min, max)`` so (i, j) and (j, i) are the same event.
    """
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    h = _splitmix64(np.full(lo.shape, int(seed) & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64))
    for part in (np.full(lo.shape, int(epoch), dtype=np.uint64), lo, hi):
        h = _splitmix64(h ^ _splitmix64(part))
    return (h >> _U64(11)).astype(np.float64) * (1.0 / (1 << 53))


def pairwise_alignment(g_i: np.ndarray, g_j: np.ndarray) -> np.ndarray:
    """g_ij in [0, 1]: 1 at identical values, 0 at the extremes of the range.

    Pairwise, from ``|g_i - g_j|``, not from either agent's own g. Values live on
    [0, 1], so the absolute gap already lies in [0, 1] and the map is 1 - gap.
    """
    return 1.0 - np.abs(np.asarray(g_i, dtype=np.float64) - np.asarray(g_j, dtype=np.float64))


def survival_probability(
    tau_i: np.ndarray, tau_j: np.ndarray, g_i: np.ndarray, g_j: np.ndarray,
    env: EnvironmentBConfig,
) -> np.ndarray:
    """sigma_ij = sigma_0 * (tau_i tau_j)^alpha * g_ij^beta  (THEORY.md §10.3).

    ``d`` is not an argument: it prices a dissolution, it does not change how likely
    one is. That is what lets the same formula run in A with d = 0.
    """
    g_ij = pairwise_alignment(g_i, g_j)
    tau = np.asarray(tau_i, dtype=np.float64) * np.asarray(tau_j, dtype=np.float64)
    return env.sigma_0 * tau ** env.sigma_alpha * g_ij ** env.sigma_beta


@dataclass(frozen=True)
class DefectionOutcome:
    survived: np.ndarray        # bool per input pair
    sigma: np.ndarray           # survival probability per pair
    defector_rows: np.ndarray   # row id of the defector of each dissolved pair
    partner_loss: float         # sum over dissolved of L * (1 - tau_defector)
    defector_penalty: float     # sum over dissolved of d

    @property
    def n_dissolved(self) -> int:
        return int(self.defector_rows.size)

    @property
    def loss(self) -> float:
        return self.partner_loss + self.defector_penalty


def resolve_defection(
    states: np.ndarray,
    pairs: np.ndarray,
    env: EnvironmentBConfig,
    penalty: float,
    seed: int,
    epoch: int,
) -> DefectionOutcome:
    """Draw survival for matched pairs and price the dissolutions.

    One code path for both environments: Environment A passes ``penalty = 0`` and
    gets the same survival draws and the same partner loss, with no penalty term.

    The defector of a dissolved pair is the lower-intent member (ties to the lower
    row id): the less committed agent is the one who leaves. That is a modelling
    choice, stated because §10.3 does not fix who defects, only what it costs.

    Args:
        pairs: (P, 2) row ids of matched pairs.
        penalty: d, charged to the defector of each dissolved pair.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return DefectionOutcome(np.zeros(0, bool), np.zeros(0), empty, 0.0, 0.0)

    a, b = pairs[:, 0], pairs[:, 1]
    sigma = survival_probability(states[a, _INTENT], states[b, _INTENT],
                                 states[a, _VALUES], states[b, _VALUES], env)
    survived = survival_uniform(seed, epoch, a, b) < sigma

    gone = ~survived
    ta, tb = states[a[gone], _INTENT], states[b[gone], _INTENT]
    a_defects = (ta < tb) | ((ta == tb) & (a[gone] < b[gone]))
    defector = np.where(a_defects, a[gone], b[gone])
    tau_def = np.where(a_defects, ta, tb)
    return DefectionOutcome(
        survived=survived,
        sigma=sigma,
        defector_rows=defector.astype(np.int64),
        partner_loss=float(np.sum(env.partner_loss * (1.0 - tau_def))),
        defector_penalty=float(penalty) * int(gone.sum()),
    )


# ── 4. The thin-market curve ─────────────────────────────────────────────────


def thin_market_curve(
    states: np.ndarray,
    config: dict[str, Any],
    env: EnvironmentBConfig,
    fractions: list[float] | np.ndarray,
    *,
    rho: float = 0.5,
    n_replicates: int = 100,
    top_k: int | None = None,
    seed: int | None = None,
    table: tuple[np.ndarray, np.ndarray] | None = None,
    bar_quantile: float | None = None,
) -> pd.DataFrame:
    """Clearance and acceptability as the subgraph shrinks (THEORY.md §10.5).

    One row per ``size_fraction``, at epoch 0 (θ as calibrated, no feedback):

    ``eligible_n``      mean |B| over replicates.
    ``empty_share``     share of replicates in which the filter left nobody; the
                        rates below are over the non-empty pools.
    ``n_dropped``       mean agents removed to side-balance the pool (reported, §10.3).
    ``clearance_prob``  matched agents / |B|, from the same lists and the same
                        ``deferred_acceptance`` call the ``"complete"`` arm uses.
    ``p_acceptable``    measured: share of eligible agents with at least one
                        acceptable candidate (utility >= own bar) in the pool.
    ``p_acceptable_analytic``  mean over those agents of 1 - F_i(bar_i)^n, n = the
                        opposite pool size: the formula of §10.5, binomial.
    ``p_acceptable_exact``  the same with the without-replacement correction,
                        1 - C(B_i, n) / C(K, n). The pool is a subset of a finite
                        reference set, so this is the exact law; the binomial is its
                        large-K limit and drifts from it as n/K grows.
    ``critical_mass``   N* = ln(1 - rho) / ln F, F the mean F_i(bar_i).

    **The bar u.** By default it is the agent's own θ, the cut the model already
    uses to decide who is acceptable. θ is calibrated at the ``1 - admissible_fraction``
    quantile, so F = 0.8 at the shipped 0.2 and N* is about 3: the bar is so low that
    a pool of a few dozen almost surely clears it, and p_acceptable is 1.0 at every
    size worth sweeping. That is a finding about the calibration, not about B, and it
    makes the default curve uninformative about thin markets. ``bar_quantile=q``
    replaces θ with each agent's q-quantile of utility over its reference set, which
    puts F = q and moves N* to ln(1-rho)/ln q: the stricter bar is the one at which
    pool size starts to matter. Both are reported in THEORY.md §10.5.

    **Which distribution F is evaluated on.** F_i is agent i's own CDF of utility
    toward the *reference set*: every opposite-side agent that passes the entry
    filter, not only the sampled pool. Evaluating it on the pool being tested would
    be circular — a pool in which nobody is acceptable has F = 1 by construction —
    and evaluating it on the unfiltered population would be the wrong conditioning,
    since §10.5 defines F conditional on the entry filter. Per-agent rather than
    pooled because agents differ in their own utility distributions, and the mean of
    1 - F_i^n is not 1 - (mean F_i)^n.

    ``p_acceptable`` is a one-sided existence condition. ``clearance_prob`` also
    needs the acceptability to be mutual, the list to survive truncation at k and the
    proposer to pay c_B, so it lies at or below it; the gap between the two is the
    cost of everything except raw pool size.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1); got {rho}")
    seed = int(config["population"]["seed"] if seed is None else seed)
    top_k = int(config["ncf"]["top_k"] if top_k is None else top_k)
    if table is None:
        table = utility_table(states, config)
    u_mw, u_wm = table

    types = states[:, _TYPE]
    prop_rows = np.flatnonzero(types == 0.0)
    acc_rows = np.flatnonzero(types == 1.0)
    prop_slot = np.full(states.shape[0], -1, dtype=np.int64)
    acc_slot = np.full(states.shape[0], -1, dtype=np.int64)
    prop_slot[prop_rows] = np.arange(prop_rows.size)
    acc_slot[acc_rows] = np.arange(acc_rows.size)
    theta = states[:, _THETA]
    passes = states[:, _VALUES] >= env.values_min

    # F_i(bar_i) over each agent's reference set, once: it does not depend on the
    # pool, only on the filter.
    ref_acc = np.flatnonzero(passes[acc_rows])
    ref_prop = np.flatnonzero(passes[prop_rows])
    if ref_acc.size == 0 or ref_prop.size == 0:
        raise ValueError("entry filter leaves an empty reference set")
    if bar_quantile is None:
        bar = theta.copy()
    else:
        if not 0.0 < bar_quantile < 1.0:
            raise ValueError(f"bar_quantile must be in (0, 1); got {bar_quantile}")
        bar = np.zeros_like(theta)
        bar[prop_rows] = np.quantile(u_mw[:, ref_acc], bar_quantile, axis=1)
        bar[acc_rows] = np.quantile(u_wm[:, ref_prop], bar_quantile, axis=1)
    below_m = np.sum(u_mw[:, ref_acc] < bar[prop_rows][:, None], axis=1)
    below_w = np.sum(u_wm[:, ref_prop] < bar[acc_rows][:, None], axis=1)
    f_m = below_m / ref_acc.size
    f_w = below_w / ref_prop.size

    def p_none_exact(below: np.ndarray, k_ref: int, n: int) -> np.ndarray:
        """P(no acceptable candidate in an n-subset drawn WITHOUT replacement).

        C(below, n) / C(k_ref, n). The binomial F^n is its with-replacement limit and
        overstates the failure probability once n is a sizeable share of k_ref; at
        n = k_ref the exact value is 0 (the whole reference set is in the pool) while
        F^n is still positive.
        """
        if n > k_ref:
            raise ValueError("pool larger than its reference set")
        out = np.zeros(below.shape, dtype=np.float64)
        can = below >= n
        out[can] = np.exp(
            gammaln(below[can] + 1) - gammaln(below[can] - n + 1)
            - gammaln(k_ref + 1) + gammaln(k_ref - n + 1)
        )
        return out

    out: list[dict[str, float]] = []
    for frac in fractions:
        cfg_env = EnvironmentBConfig(**{**env.to_dict(), "size_fraction": float(frac)})
        reps = 1 if frac >= 1.0 else int(n_replicates)

        n_elig, n_drop, matched, hit, ana, exact, n_agents = [], [], 0, 0, 0.0, 0.0, 0
        n_empty = 0
        for r in range(reps):
            try:
                sub = build_subgraph(states, cfg_env, seed, replicate=r, warn=False)
            except ValueError:
                # A pool the filter empties is a real outcome at small sizes, not an
                # error: nobody is eligible, so nothing clears. It is counted and
                # reported as empty_share; the rates below are over the non-empty pools.
                n_empty += 1
                n_elig.append(0)
                n_drop.append(0)
                continue
            n_side = sub.rows_m.size
            m_slots, w_slots = prop_slot[sub.rows_m], acc_slot[sub.rows_w]

            prefs = complete_prefs(states, table, sub, top_k, env.signal_cost)
            da = deferred_acceptance(prefs.prop_prefs, prefs.acc_prefs, True)
            matched += 2 * int(da["matches"].shape[0])
            n_elig.append(sub.n_eligible)
            n_drop.append(sub.report.dropped_ids.size)

            ok_m = (u_mw[np.ix_(m_slots, w_slots)] >= bar[sub.rows_m][:, None]).any(axis=1)
            ok_w = (u_wm[np.ix_(w_slots, m_slots)] >= bar[sub.rows_w][:, None]).any(axis=1)
            hit += int(ok_m.sum() + ok_w.sum())
            ana += float((1.0 - f_m[m_slots] ** n_side).sum()
                         + (1.0 - f_w[w_slots] ** n_side).sum())
            exact += float((1.0 - p_none_exact(below_m[m_slots], ref_acc.size, n_side)).sum()
                           + (1.0 - p_none_exact(below_w[w_slots], ref_prop.size, n_side)).sum())
            n_agents += 2 * n_side

        f_bar = float(np.concatenate([f_m, f_w]).mean())
        nan = float("nan")
        out.append({
            "size_fraction": float(frac),
            "empty_share": n_empty / reps,
            "eligible_n": float(np.mean(n_elig)),
            "n_dropped": float(np.mean(n_drop)),
            "clearance_prob": matched / n_agents if n_agents else nan,
            "p_acceptable": hit / n_agents if n_agents else nan,
            "p_acceptable_analytic": ana / n_agents if n_agents else nan,
            "p_acceptable_exact": exact / n_agents if n_agents else nan,
            "critical_mass": (float(np.log(1.0 - rho) / np.log(f_bar))
                              if 0.0 < f_bar < 1.0 else float("nan")),
        })
    return pd.DataFrame(out)
