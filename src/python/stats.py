"""Interval estimation over seeds (docs/STATS.md).

Every headline in this project is a mean over five seeds. Five seeds give one
number and a standard deviation, and a standard deviation is a *spread*, not an
interval: it says how far apart the seeds fell, not where the population mean
lies. This module turns a vector of per-seed values into an interval estimate of
the quantity those seeds are draws from.

Two intervals are produced for every quantity, side by side:

- **BCa** (bias-corrected and accelerated) — the one to quote. It corrects for
  the bootstrap distribution being shifted or skewed relative to the statistic,
  which matters here because several headline quantities (Gini, algorithmic
  loss) are bounded and their seed distribution is visibly asymmetric at n = 5.
- **percentile** — the plain 2.5/97.5 quantiles of the same resampling. It makes
  no correction at all. It is reported beside BCa so a reader can see whether
  the correction did anything: when the two agree the correction is not load
  bearing, and when they disagree the BCa interval is the one that has been
  checked against a nominal coverage target.

The resampling unit is the **seed**, never the epoch. Epochs within a run are a
trajectory through one population, not independent draws, and resampling them
would shrink every interval by roughly sqrt(n_epochs) for no reason. See
``scripts/run_ablation.py`` for the same convention in the point estimates.

A gap between two arms is bootstrapped from the **per-seed paired differences**,
not from the two arms' separate means: both arms at one seed share a population,
so the difference at that seed is the observation, and forming it before
resampling keeps the pairing. This is the same reason the ablation script forms
gaps per seed.

Nothing here assumes normality. Nothing here is a confidence interval for a
prediction: it is an interval for the *seed-averaged* value of a quantity in a
model with no empirical calibration, and docs/STATS.md §1 says what may and may
not be concluded from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy import stats as _st


DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """One quantity's estimate, with both intervals and the seed count behind it.

    ``method`` records which interval a consumer should read: every row written
    by this project says ``bca``, and carries ``lo_percentile``/``hi_percentile``
    beside it as the cross-check. Keeping both on one object is what stops a
    downstream table from quoting the percentile interval by accident and
    calling it a confidence interval.
    """

    point: float
    lo: float
    hi: float
    lo_percentile: float
    hi_percentile: float
    n_seeds: int
    n_resamples: int
    method: str = "bca"

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def sd(self) -> float:
        """Half-width over 1.96 — the normal-theory sd the interval implies.

        Provided so a reader can compare an interval against the mean ± sd the
        rest of the repository quotes. It is a convenience, not an estimate to
        publish: at n = 5 it is itself noisy, which is the whole point.
        """
        return self.width / (2.0 * _st.norm.ppf(0.975))

    def half_width_sd_units(self, sd: float) -> float:
        """Half-width divided by a given sd — the multiplier to state beside ±.

        The repository's existing tables read ``mean ± sd``. Multiplying that sd
        by this number gives the 95% interval's half-width, so a reader can see
        how much wider the honest interval is without reading another table. For
        a normal at n = 5 the value is about 2.78; anything near 2 means the
        quantity is being reported as if it were better determined than it is.
        """
        if not np.isfinite(sd) or sd <= 0.0:
            return float("nan")
        return (self.width / 2.0) / sd

    def as_row(self, **labels: object) -> dict:
        """Flatten to a CSV row, labels first then the interval's own fields."""
        row: dict = dict(labels)
        row.update({
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "lo_percentile": self.lo_percentile,
            "hi_percentile": self.hi_percentile,
            "n_seeds": self.n_seeds,
            "n_resamples": self.n_resamples,
            "method": self.method,
        })
        return row


def _as_vector(values: Iterable[float]) -> np.ndarray:
    """Drop non-finite entries; a NaN seed is a missing run, not a zero.

    Dropping is preferred to raising because the repository's CSVs carry NaN for
    quantities an arm does not produce (``algorithmic_loss`` on a run without a
    benchmark). Silently treating that as 0 would move the point estimate.
    """
    x = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                   dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _point(x: np.ndarray, statistic: Callable[[np.ndarray], float]) -> float:
    return float(statistic(x))


def _bootstrap(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample ``x`` with replacement ``n_resamples`` times.

    Means are drawn as one (B, n) index matrix, which is both fast and exactly
    the same distribution a per-resample loop would give. Any other statistic
    falls back to a loop; only the mean is used for the headlines, so the loop
    exists for the tests rather than for the pipeline.
    """
    n = x.size
    if statistic is np.mean or statistic is np.nanmean:
        idx = rng.integers(0, n, size=(n_resamples, n))
        return x[idx].mean(axis=1)
    out = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        out[b] = statistic(x[rng.integers(0, n, size=n)])
    return out


def _acceleration(x: np.ndarray, statistic: Callable[[np.ndarray], float]) -> float:
    """Jackknife acceleration, the BCa skewness correction.

    ``a`` enters the BCa quantile through ``1 - a(z0 + z)`` and is undefined for
    a degenerate sample (zero spread), where it is set to 0.0 — which reduces
    that term to the identity and leaves the bias correction alone doing the
    work.
    """
    n = x.size
    if n < 2:
        return 0.0
    theta_dot = np.empty(n, dtype=np.float64)
    for i in range(n):
        theta_dot[i] = statistic(np.delete(x, i))
    mean_dot = theta_dot.mean()
    d = mean_dot - theta_dot
    denom = 6.0 * float(np.sum(d * d)) ** 1.5
    if denom == 0.0:
        return 0.0
    return float(np.sum(d ** 3) / denom)


def bca_interval(
    values: Iterable[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    rng: np.random.Generator | int | None = None,
) -> Interval:
    """BCa and percentile intervals for ``statistic`` over the seed vector.

    ``rng`` is seeded explicitly by the caller (see ``scripts/run_intervals.py``)
    so that re-running the pipeline reproduces ``intervals.csv`` byte for byte.
    Passing ``None`` draws fresh entropy, which is fine interactively and not
    fine for a committed table.

    Degenerate inputs are handled rather than raised on, because a table that
    drops a row is harder to audit than one that shows a zero-width interval:

    - ``n == 0``: every field NaN. Nothing was measured.
    - ``n == 1``: the point estimate with a NaN interval. One seed cannot bound
      anything, and widening a single value into a range would invent precision.
    - zero spread: point estimate as both bounds. The seeds agreed exactly, and
      the interval says so.
    """
    x = _as_vector(values)
    n = x.size
    if n == 0:
        nan = float("nan")
        return Interval(nan, nan, nan, nan, nan, 0, n_resamples)
    point = _point(x, statistic)
    if n == 1:
        nan = float("nan")
        return Interval(point, nan, nan, nan, nan, 1, n_resamples)
    if np.all(x == x[0]):
        return Interval(point, point, point, point, point, n, n_resamples)

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)
    boot = _bootstrap(x, statistic, n_resamples, gen)

    # Percentile interval: no correction, so it is the honest "what did the
    # resampling distribution say" reading to compare BCa against.
    lo_p, hi_p = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])

    # Bias correction: how far the bootstrap centre sits from the estimate, in
    # normal quantiles. Ties are not split — with continuous statistics a tie is
    # measure zero, and splitting would only matter on the discrete ones.
    prop = float(np.mean(boot < point))
    prop = min(max(prop, 1.0 / (n_resamples + 1.0)), 1.0 - 1.0 / (n_resamples + 1.0))
    z0 = float(_st.norm.ppf(prop))
    a = _acceleration(x, statistic)

    def _adjust(z_alpha: float) -> float:
        denom = 1.0 - a * (z0 + z_alpha)
        if denom <= 0.0:  # acceleration too large for this tail; clip rather than explode
            return alpha / 2.0 if z_alpha < 0 else 1.0 - alpha / 2.0
        return float(_st.norm.cdf(z0 + (z0 + z_alpha) / denom))

    lo_q = _adjust(_st.norm.ppf(alpha / 2.0))
    hi_q = _adjust(_st.norm.ppf(1.0 - alpha / 2.0))
    lo_q = min(max(lo_q, 0.0), 1.0)
    hi_q = min(max(hi_q, 0.0), 1.0)
    lo, hi = np.percentile(boot, [100.0 * lo_q, 100.0 * hi_q])

    return Interval(float(point), float(lo), float(hi), float(lo_p), float(hi_p),
                    n, n_resamples)


def paired_gap(
    left: Iterable[float],
    right: Iterable[float],
    **kwargs: object,
) -> Interval:
    """Interval for a per-seed difference, formed before resampling.

    ``left[i]`` and ``right[i]`` must be the two arms at the same seed. The
    difference is taken elementwise and the bootstrap runs over those differences,
    so the population shared by the pair cancels instead of entering the width
    twice. Lengths must match: a ragged pair means the arms did not run on the
    same seeds, and averaging over whatever survived would silently compare
    different populations.
    """
    lhs = _as_vector(left)
    rhs = _as_vector(right)
    if lhs.size != rhs.size:
        raise ValueError(
            f"paired_gap needs one value per seed on both sides; got "
            f"{lhs.size} and {rhs.size}"
        )
    return bca_interval(lhs - rhs, **kwargs)  # type: ignore[arg-type]


def t_interval(
    values: Iterable[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Student-t interval: ``(mean, lo, hi)`` on the seed vector.

    Reported beside BCa because at five seeds the bootstrap cannot see the tails
    it has not observed. Measured on N(0, 1) with 4,000 replications: BCa covers
    84.3% at n = 5 and 92.6% at n = 20, where the t interval covers 94.9% and
    93.5%. So the bootstrap is not a licence to shrink an interval at small n —
    it is optimistic, and the t interval is the honest floor until n is large
    enough for the resampling distribution to have tails of its own.

    Nothing here is used at large n, where the two agree to within a percent.
    """
    x = _as_vector(values)
    n = x.size
    if n < 2:
        return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"))
    m = float(x.mean())
    se = float(x.std(ddof=1)) / np.sqrt(n)
    q = float(_st.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return (m, m - q * se, m + q * se)
