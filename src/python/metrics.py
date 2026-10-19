"""Metrics for the matching market (docs/THEORY.md §7).

Pure functions over numpy arrays, with no dependency on the C++ extension, so
they are testable on hand-built fixtures and reusable by the notebook and the
plots.

Every function is written to return a finite float for the cases a run actually
produces. Where a metric is genuinely undefined — a Pareto fit on a constant
distribution, a rank index with nothing matched — it returns NaN rather than a
plausible-looking default, because a fabricated number is worse than a visible
hole: the orchestrator writes NaN into the DataFrame and pandas keeps it
visible, whereas a 0.0 would be read as a result.
"""

from __future__ import annotations

import numpy as np


def gini(counts: np.ndarray) -> float:
    """Visibility Gini over per-agent exposure counts (THEORY.md §7.1).

    ``counts[i]`` is the number of exposures agent ``i`` received in an epoch.
    The formula is the standard one,

        G = Σ_i Σ_j |x_i − x_j| / (2 n² x̄),

    evaluated through its sorted form, which is the same quantity in O(n log n):

        G = (2 Σ_i i·x_(i)) / (n Σ_i x_i) − (n + 1)/n,   x_(i) sorted ascending.

    A population with no exposure at all returns 0.0: every agent is equally
    invisible, which is perfect equality, not a missing measurement.
    """
    x = np.asarray(counts, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    total = x.sum()
    if total <= 0.0:
        return 0.0
    x = np.sort(x)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, x)) / (n * total) - (n + 1.0) / n)


def market_clearing_rate(n_matched: int, n_agents: int) -> float:
    """MCR = 2·|μ|/N (INTERFACES.md §6), the share of agents who end matched.

    Written with the 2 so it is comparable to 1.0 rather than 0.5. THEORY.md §7.2
    states the same quantity as |μ|/(N/2).
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    return 2.0 * float(n_matched) / float(n_agents)


def rank_satisfaction_index(
    matches: np.ndarray,
    prop_prefs: list[list[int]],
) -> float:
    """Mean normalized rank of the partner in each matched proposer's list.

    ``matches`` holds dense ``[proposer_index, acceptor_index]`` pairs and
    ``prop_prefs[p]`` is proposer ``p``'s preference list of acceptor indexes, so
    the two share an index space and no id mapping is needed here.

    Normalized as ``(rank + 1) / len(list)`` so the result lies in ``(0, 1]``:
    rank 0 (the top choice) gives ``1/len`` and the last entry gives 1.0. Lower is
    better, and a value of 1.0 means every proposer settled for the bottom of
    their own list.

    A partner missing from the list contributes nothing rather than being
    scored at 1.0. That should not happen in the truncated regime, because an
    acceptor only accepts proposers she lists, but the full-information benchmark
    and hand-built fixtures both reach this function and a silent 1.0 there would
    blend a bug into the average.
    """
    if len(matches) == 0:
        return float("nan")
    ranks: list[float] = []
    for pi, ai in np.asarray(matches).reshape(-1, 2):
        prefs = prop_prefs[int(pi)]
        if len(prefs) == 0:
            continue
        try:
            rank = prefs.index(int(ai))
        except ValueError:
            continue
        ranks.append((rank + 1.0) / len(prefs))
    if not ranks:
        return float("nan")
    return float(np.mean(ranks))


def algorithmic_loss(welfare_alg: float, welfare_full: float) -> float:
    """Δ = (SW(μ*) − SW(μ_alg)) / SW(μ*) (THEORY.md §7.4).

    Undefined — and NaN — when the benchmark was not run, and when the benchmark's
    own welfare is zero, since the ratio has no scale to divide by. Negative
    values are possible when the truncated matching happens to beat the
    full-information one on this utility, which is a real property of the model
    and not an error worth hiding.
    """
    if not np.isfinite(welfare_full) or welfare_full == 0.0:
        return float("nan")
    return float((welfare_full - welfare_alg) / welfare_full)


def pareto_alpha(cumulative_counts: np.ndarray) -> float:
    """Pareto tail index fitted to cumulative match counts (THEORY.md §8).

    MLE with ``x_min`` fixed at the smallest positive count:

        α̂ = n / Σ_i ln(x_i / x_min).

    Only positive counts enter the fit: a zero has no logarithm and, more to the
    point, an agent with no cumulative matches is outside the support rather than
    an observation at the boundary.

    NaN when the positive part is degenerate — fewer than two distinct values
    means there is no tail to describe, and the MLE is infinite.
    """
    x = np.asarray(cumulative_counts, dtype=np.float64).ravel()
    x = x[x > 0.0]
    if x.size < 2:
        return float("nan")
    x_min = x.min()
    if np.unique(x).size < 2:
        return float("nan")
    return float(x.size / np.log(x / x_min).sum())
