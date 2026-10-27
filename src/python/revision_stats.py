"""Shared statistics for the second revision's analyses.

Every interval here says what it is: a Student-t interval over seeds at a stated
level, a BCa bootstrap interval over seeds, or a two-one-sided-test (TOST)
interval at 90%, which is the 90% t interval by construction. The replicate is
always the seed: rounds within a run are one trajectory, not independent draws.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

MARGIN = 0.002  # docs/PREREGISTRATION.md E5 and R2-0


def t_interval(values, level: float = 0.95) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "se": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "lo": mean - half, "hi": mean + half}


def t_test(values) -> dict[str, float]:
    """One-sample two-sided t test against zero."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    res = stats.ttest_1samp(x, 0.0)
    return {"t": float(res.statistic), "p": float(res.pvalue)}


def bca_interval(values, level: float = 0.95, seed: int = 20261027,
                 resamples: int = 10_000) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.ptp(x) == 0.0:
        return float("nan"), float("nan")
    res = stats.bootstrap((x,), np.mean, confidence_level=level, method="BCa",
                          n_resamples=resamples, random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def tost(values, margin: float = MARGIN) -> dict[str, float]:
    """Two one-sided t tests of |mean| < margin, and the 90% interval they imply."""
    ti = t_interval(values, level=0.90)
    n, mean, se = ti["n"], ti["mean"], ti["se"]
    t_lower = (mean + margin) / se
    t_upper = (margin - mean) / se
    p_lower = float(stats.t.sf(t_lower, n - 1))
    p_upper = float(stats.t.sf(t_upper, n - 1))
    return {"n": n, "mean": mean, "se": se, "t_lower": t_lower, "t_upper": t_upper,
            "p": max(p_lower, p_upper), "lo90": ti["lo"], "hi90": ti["hi"],
            "passes": bool(ti["lo"] > -margin and ti["hi"] < margin)}


def entry_round(series: dict[int, float] | np.ndarray, margin: float = MARGIN) -> int | None:
    """First round from which |gap| < margin to the end of the run; None if never.

    ``series`` is indexed by round 0..T-1. A run whose last round is outside the
    margin is censored and returns None.
    """
    gap = np.asarray(series if isinstance(series, np.ndarray)
                     else [series[r] for r in sorted(series)], dtype=np.float64)
    inside = np.abs(gap) < margin
    if not inside[-1]:
        return None
    outside = np.flatnonzero(~inside)
    return int(outside[-1] + 1) if outside.size else 0


def first_stays_below(series: np.ndarray, threshold: float) -> int | None:
    """First round from which the gap stays below ``threshold``; None if never."""
    gap = np.asarray(series, dtype=np.float64)
    below = gap < threshold
    if not below[-1]:
        return None
    above = np.flatnonzero(~below)
    return int(above[-1] + 1) if above.size else 0


def bh_reject(pvalues, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up at level q; returns a boolean mask."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    m = p.size
    passed = ranked <= q * (np.arange(1, m + 1) / m)
    out = np.zeros(m, dtype=bool)
    if passed.any():
        last = int(np.flatnonzero(passed)[-1])
        out[order[: last + 1]] = True
    return out


def bh_qvalues(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out
