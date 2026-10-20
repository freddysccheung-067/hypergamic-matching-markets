"""Attribution: which term drives selection, and what the market cannot tell us.

This module exists to answer one question and to refuse another.

It *answers* "how much does each term of the utility drive who reaches the top of
the market", by ablation: zero one weight, rebuild every agent's ranking, and
measure how much of the baseline shortlist survives.

It *refuses* "does status reflect attractiveness". Status and attractiveness are
drawn independently in `sample_population`, so in this model there is no causal
path between them to estimate. The correlation an observer sees among matched or
shortlisted pairs is induced by the market sorting on both traits at once; it
carries no information about individuals. THEORY.md §9 states this, and
`selection_correlation` is the function that measures the artifact rather than
asserting it away.

Precise definitions, because the loose ones differ:

**top-k** is per agent. Agent i's top-k is the first k entries of i's ranking of
the opposite side by `utility_toward` — the same ordering `generate_swipes` and
the oracle ablation arm use, ties broken by ascending dense slot. It is a shortlist,
not a set of individuals: the same candidate appears in many agents' top-k.

**overlap** is share-in-common, ``|A ∩ B| / k``, averaged over agents. Not Jaccard.
Jaccard divides by the union, so the same ablation reports ~0.94 where share-in-common
reports 0.967, and a threshold written for one reads as a different claim under the
other.

**The θ gate is switched off.** Every function here ranks an *unthresholded* copy of
the state matrix, with ``states[:, 6]`` set to -inf. θ was calibrated against the
base weights, so leaving it in place while zeroing a weight would change the
admissible set as a side effect and the ablation would measure two things at once.
Attribution is about the ranking, so the gate is removed to isolate it.

The terminal caveat: leave-one-term-out is a one-at-a-time ablation *at one set of
weights and one pair of trait distributions*. It is not an intrinsic importance
ranking and it does not transfer. Status dominates here partly because alpha
multiplies a heavy-tailed lognormal; at a Pareto tail, or at beta/alpha >= 2, the
ranking of terms moves. Report the configuration alongside the number or the number
means nothing.
"""

from __future__ import annotations

import copy
import itertools
import warnings
from typing import Any

import numpy as np

# Map the human-readable term names used by the attribution API onto the config
# keys that carry them. The utility's five weights are (alpha, beta, gamma, delta,
# lambda): delta is the cutoff the indicator compares against, and lambda is the
# size of the bonus. "indicator" therefore means lambda — zeroing delta does not
# switch the indicator off, it moves the cutoff to her own status, which is a
# different experiment. Both are reachable; this dict names the important one.
TERM_WEIGHTS: dict[str, str] = {
    "status": "alpha",
    "attract": "beta",
    "distance": "gamma",
    "indicator": "lambda",
}

# Trait columns in the flat state matrix (docs/INTERFACES.md §1).
TRAIT_COLUMNS: dict[str, int] = {
    "status": 2,
    "attractiveness": 3,
    "sociability": 4,
    "age": 5,
}


def _extension():
    """Import the built extension, with a message that says how to build it.

    Imported lazily rather than at module scope so that ``import
    src.python.attribution`` works on a checkout where the extension has not been
    built, and so a test module can skip cleanly on ImportError instead of dying
    during collection.
    """
    try:
        import _hypergamic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "the _hypergamic extension is required; build it with "
            "`cmake --build build --config Release` and put build/Release on sys.path"
        ) from exc
    return _hypergamic


def _with_weights(config: dict[str, Any], **weights: float) -> dict[str, Any]:
    """A deep copy of ``config`` with the given hypergamy weights overridden."""
    out = copy.deepcopy(config)
    block = dict(out.get("hypergamy", {}))
    block.update(weights)
    out["hypergamy"] = block
    return out


def _unthresholded(states: np.ndarray) -> np.ndarray:
    """A float64 copy of ``states`` with the θ column set to -inf.

    See the module docstring: θ is calibrated against the base weights, so
    re-ranking under a modified weight while leaving θ in place would confound the
    ablation with a change in the admissible set.
    """
    out = np.array(states, dtype=np.float64, order="C", copy=True)
    out[:, 6] = -np.inf
    return out


def _top_k_by_agent(
    states: np.ndarray,
    config: dict[str, Any],
    k: int,
    side: str = "both",
) -> list[list[int]]:
    """Each agent's first ``k`` candidates, in ranked order, as row indexes.

    Concatenates both sides, so the returned list has one entry per state row:
    entry ``i`` is the top-k of whichever side agent ``i`` is on, holding rows of
    the opposite side. Both sides are included by default because the market is
    two-sided and an ablation that moved only the women's shortlists would
    misreport the effect of a symmetric weight.

    ``side="W"`` restricts the result to women's shortlists and leaves every other
    entry empty. That is required for δ, and only for δ: the indicator term exists
    only in the acceptor's utility (see ``Agent::utility_toward``), so a man's
    ranking is *invariant to δ by construction*. Pooling both sides halves the
    apparent effect and makes an inert indicator look like a live one — measured
    0.923 pooled against 0.846 on the women alone at the shipped weights.

    ``side`` also accepts ``"M"`` for the symmetric case; nothing in this module
    needs it, and it exists so the restriction is expressible in either direction
    rather than hard-coded to one.
    """
    ext = _extension()
    states = _unthresholded(states)
    proposers, acceptors = ext.build_full_info_pref_lists(states, config)

    n = states.shape[0]
    prop_rows = np.flatnonzero(states[:, 1] == 0.0)
    acc_rows = np.flatnonzero(states[:, 1] == 1.0)

    out: list[list[int]] = [[] for _ in range(n)]
    # build_full_info_pref_lists indexes dense per-side slots (INTERFACES §2.3), so
    # a slot has to be mapped back to a row before the traits can be looked up.
    if side in ("both", "M"):
        for slot, row in enumerate(prop_rows):
            out[int(row)] = [int(acc_rows[j]) for j in proposers[slot][:k]]
    if side in ("both", "W"):
        for slot, row in enumerate(acc_rows):
            out[int(row)] = [int(prop_rows[j]) for j in acceptors[slot][:k]]
    return out


def _mean_share_in_common(a: list[list[int]], b: list[list[int]], k: int) -> float:
    """Mean over non-empty agents of ``|top-k_a(i) ∩ top-k_b(i)| / k``.

    Averaged per agent rather than pooled: pooling would let agents with more
    candidates (none here, both sides are equal) and agents whose shortlist
    happened to overlap more cast more weight. Per-agent keeps it a statement about
    the typical shortlist, which is the object the guardrail tests also use.

    Empty entries are skipped rather than counted as zero overlap. ``side="W"``
    leaves the men's entries empty, and counting them would average in a row of
    zeros that describes no agent at all.
    """
    if not a or not b:
        return float("nan")
    total = 0.0
    counted = 0
    for left, right in zip(a, b):
        if not left and not right:
            continue
        total += len(set(left) & set(right)) / k
        counted += 1
    return total / counted if counted else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN rather than 0.0 when either side is constant.

    A constant trait has no correlation with anything; returning 0.0 would read as
    "no relationship", which is a different and false statement.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    denom = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def leave_one_term_out_effect(
    states: np.ndarray,
    config: dict[str, Any],
    terms: tuple[str, ...] = ("status", "attract", "distance", "indicator"),
    k: int = 1000,
) -> dict[str, float]:
    """Mean shortlist overlap between the baseline top-k and each ablations'.

    For each term, the term's weight is set to zero, every agent's ranking is
    rebuilt, and the mean share of the baseline shortlist that survives is
    returned. Low overlap means the term was doing the selecting; an overlap of
    exactly 1.0 means the term had no effect on the ranking at all.

    Read the absolute value, not just the ordering. A term at 0.97 is present but
    nearly inert at these weights: it reorders about one shortlist entry in thirty.
    The indicator is the one to watch, because ``lambda`` interacts with ``delta``
    and can be fully inert if no candidate straddles the cutoff (THEORY.md §2.2).

    ``k`` larger than one side's candidate count is clamped, and the clamp is
    reported through the returned values' being computed at the clamped k — callers
    that care can compare against ``states.shape[0] // 2``.

    Measured at the shipped config (N=1000, seed 42, chemistry on), these are the
    values this project cites; re-measure with scripts/run_attribution.py after any
    retune rather than trusting them.
    """
    ext = _extension()  # noqa: F841  (fail fast, and before the deep copies)
    n_side = int(np.count_nonzero(states[:, 1] == 0.0))
    k = max(1, min(k, n_side))

    baseline = _top_k_by_agent(states, config, k)
    out: dict[str, float] = {}
    for term in terms:
        if term not in TERM_WEIGHTS:
            raise ValueError(
                f"unknown term {term!r}; expected one of {sorted(TERM_WEIGHTS)}"
            )
        ablated = _top_k_by_agent(states, _with_weights(config, **{TERM_WEIGHTS[term]: 0.0}), k)
        out[term] = _mean_share_in_common(baseline, ablated, k)
    return out


def selection_correlation(
    states: np.ndarray,
    config: dict[str, Any],
    top_n: int,
) -> dict[str, dict[str, float]]:
    """Trait-pair correlation among shortlisted pairs, against the population.

    For every unordered pair of traits this returns three numbers per pair:

        population  the correlation across all agents — the generative one
        selected    the correlation across pooled (user, candidate) pairs where the
                    candidate is in the user's top-``top_n``, counting a candidate
                    once per shortlist it appears on
        n_pairs     how many such pairs there were

    The distinction is the whole point. ``population`` is a property of the draw;
    ``selected`` is a property of the search, and it is what an analyst computing a
    correlation from a swipe log or a matched-pairs table would see. When the
    selected value differs from the population value, the difference is a selection
    artifact and must not be reported as a relationship between traits.

    Which direction to expect: with the market ranking on both traits and a
    threshold-like shortlist, the induced correlation at small top_n is
    *negative* (selecting on two traits makes the selected set trade one against
    the other). At large top_n it decays toward zero as the shortlist approaches
    the whole population. Do not assume the sign: measure it, and note that at
    ``top_n`` around 10 the sampling error of a Pearson coefficient is ~0.3, so a
    single market's top-10 value is not an estimate of anything.

    To get an interval, call this once per seed and compare the spread — that is
    what scripts/run_attribution.py does. A bootstrap inside one market would
    resample pairs, not markets, and would understate the uncertainty that matters.
    """
    _extension()
    top_n = max(1, int(top_n))
    traits = sorted(TRAIT_COLUMNS)
    columns = {name: states[:, TRAIT_COLUMNS[name]] for name in traits}

    top_k = _top_k_by_agent(states, config, top_n)
    users = np.repeat(np.arange(states.shape[0]), [len(t) for t in top_k])
    candidates = np.fromiter(
        itertools.chain.from_iterable(top_k), dtype=np.int64, count=int(users.size)
    )

    out: dict[str, dict[str, float]] = {}
    for a, b in itertools.combinations(traits, 2):
        out[f"{a}~{b}"] = {
            "population": _pearson(columns[a], columns[b]),
            "selected": _pearson(columns[a][candidates], columns[b][candidates]),
            "n_pairs": float(candidates.size),
        }
    return out


def delta_power(
    states: np.ndarray,
    config: dict[str, Any],
    deltas: tuple[float, ...] = (0.0, 0.5, 1.0),
    k: int = 1000,
) -> float:
    """Share of the shortlist held in common between the smallest and largest δ.

    δ only reorders anything when the indicator bisects the competitive set: some
    candidate above a woman's cutoff and some below must both be plausible. When
    status dominates the ranking, every candidate in her shortlist sits on the same
    side of the cutoff, the indicator is constant there, and δ does nothing at all
    — overlap exactly 1.0, for every δ.

    That failure is silent, which is why this function warns rather than returning
    1.0 quietly: a sweep over δ at inert settings produces a flat line that reads as
    "δ does not matter" when it means "this configuration cannot test δ". Tests
    treat the warning as a signal; experiments must not ignore it.

    The shipped guardrail is THEORY.md §2.2 and beta/alpha >= 2; this function is
    the same measurement on the Python side, and it exists so the condition can be
    checked from a script without rebuilding the C++ tests.

    **Women only.** δ enters the acceptor's utility alone, so a man's ranking does
    not depend on it at all and pooling both sides would dilute every figure by
    roughly half — turning an inert indicator into one that appears to move a
    quarter of the market. `tests/cpp/test_delta_power.cpp` measures the same
    women-only quantity; the two are meant to agree.
    """
    _extension()
    if len(deltas) < 2:
        raise ValueError("delta_power needs at least two deltas to compare")
    smallest, largest = min(deltas), max(deltas)

    n_side = int(np.count_nonzero(states[:, 1] == 0.0))
    k = max(1, min(k, n_side))

    low = _top_k_by_agent(states, _with_weights(config, delta=smallest), k, side="W")
    high = _top_k_by_agent(states, _with_weights(config, delta=largest), k, side="W")
    overlap = _mean_share_in_common(low, high, k)

    if overlap == 1.0:
        warnings.warn(
            f"delta is inert at this configuration: delta={smallest} and "
            f"delta={largest} produce identical top-{k} shortlists (overlap 1.000) "
            "for every woman in the market. Any sweep over delta measures nothing "
            "here. See THEORY.md §2.2: the indicator needs beta/alpha >= 2 to "
            "bisect the competitive set.",
            RuntimeWarning,
            stacklevel=2,
        )
    return overlap
