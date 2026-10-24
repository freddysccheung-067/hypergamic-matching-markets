"""Tests for src/python/stats.py — the interval estimator.

Two things are checked, and they are different claims:

1. **Correctness against a known answer.** On 1,000 draws from a normal with a
   known mean, the BCa interval must land on the analytic interval. That is a
   coverage check on the machinery: if the resampling, the bias correction or
   the acceleration term were wrong, a symmetric sample would expose it, because
   there the correct BCa interval is *exactly* the analytic one (z0 = 0, a = 0).

2. **Sensible degradation.** At n = 5 nothing can be exact. What is required is
   that the interval still (a) contains the point estimate, (b) widens as the
   sample spreads out, (c) is finite, and (d) does not pretend to a precision
   the seed count cannot support. A test that demanded 95% coverage at n = 5
   from five points would be testing arithmetic on a small sample, not the code.

The paired-gap test pins the property that makes the ablation intervals narrow:
the shared population cancels, so the gap's interval is narrower than the
interval on either arm would suggest.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy import stats as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.stats import (  # noqa: E402
    Interval,
    bca_interval,
    paired_gap,
    t_interval,
)


# ── correctness at a known answer ─────────────────────────────────────────────

def test_bca_matches_analytic_interval_on_a_large_normal_sample():
    """n = 1000, known mean: BCa must agree with the normal interval.

    The reference is the interval built from the *sample's own* sd, not the
    population's, because that is what the bootstrap is estimating from: with
    1,000 points the bootstrap distribution is the sampling distribution of the
    mean with the empirical sd plugged in. Comparing against the population sd
    would measure the sample's sd error, which is a property of the draw and not
    of the code.

    The sample is symmetric by construction, so z0 ≈ 0 and a ≈ 0 and BCa
    collapses to the percentile interval, which collapses to mean ± 1.96·s/√n.
    """
    rng = np.random.default_rng(20261024)
    mu, sigma, n = 3.0, 2.0, 1000
    x = rng.normal(mu, sigma, n)

    iv = bca_interval(x, rng=np.random.default_rng(1))

    se = x.std(ddof=1) / np.sqrt(n)
    analytic_lo, analytic_hi = x.mean() - 1.959964 * se, x.mean() + 1.959964 * se
    half = analytic_hi - analytic_lo

    assert iv.point == pytest.approx(x.mean(), rel=0, abs=1e-12)
    assert iv.lo == pytest.approx(analytic_lo, rel=0, abs=0.06 * half)
    assert iv.hi == pytest.approx(analytic_hi, rel=0, abs=0.06 * half)
    assert iv.n_seeds == n


def test_bca_and_percentile_agree_on_a_symmetric_sample():
    """When the sampling distribution is symmetric the correction is idle.

    This is the cross-check the pipeline quotes: if BCa and percentile disagree
    badly on a symmetric sample, the correction is misfiring.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 400)
    iv = bca_interval(x, rng=np.random.default_rng(2))
    half = iv.hi - iv.lo
    assert abs(iv.lo - iv.lo_percentile) < 0.15 * half
    assert abs(iv.hi - iv.hi_percentile) < 0.15 * half


def test_bca_correction_is_visible_on_a_skewed_sample():
    """On a shifted exponential the bias correction must do something.

    Not a coverage claim — a claim that the code path is not dead. If z0 and a
    were both silently zero this test would fail, and so would any downstream
    interval on a bounded quantity like Gini.
    """
    rng = np.random.default_rng(11)
    x = rng.exponential(scale=1.0, size=300) + 1.0
    iv = bca_interval(x, rng=np.random.default_rng(3))
    moved = max(abs(iv.lo - iv.lo_percentile), abs(iv.hi - iv.hi_percentile))
    assert moved > 0.0, "BCa identical to percentile: correction is not being applied"


# ── degradation at n = 5 ──────────────────────────────────────────────────────

def test_five_seeds_produce_a_finite_interval_containing_the_estimate():
    x = np.array([0.0152, 0.0338, 0.0284, 0.0189, 0.0293])  # the measured H3 gaps
    iv = bca_interval(x, rng=np.random.default_rng(4))
    assert np.isfinite([iv.point, iv.lo, iv.hi, iv.lo_percentile, iv.hi_percentile]).all()
    assert iv.lo <= iv.point <= iv.hi
    assert iv.width > 0.0
    assert iv.n_seeds == 5


def test_interval_widens_with_spread():
    """Same n, one sample four times as wide: the interval must widen too."""
    base = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    wide = 1.0 + 4.0 * (base - 1.0)
    narrow_iv = bca_interval(base, rng=np.random.default_rng(5))
    wide_iv = bca_interval(wide, rng=np.random.default_rng(5))
    assert wide_iv.width > 3.0 * narrow_iv.width


def test_five_seeds_bootstrap_interval_is_narrower_than_the_t_interval():
    """Documented, measured, and the reason docs/STATS.md §2 exists.

    At n = 5 the bootstrap resampling distribution is built from five points, so
    it cannot see the tails it has not observed: the interval comes out short of
    Student's t at 4 df. Measured coverage on N(0,1) with 4,000 trials is 0.84
    for BCa against 0.95 for t.

    Averaged over many samples, because a single n = 5 draw's ratio is itself
    extremely noisy — one positively skewed sample can come out at 0.63 while the
    next is 1.0. The systematic bias is what matters, and it is what this pins, so
    it cannot drift silently into looking correct. It is *not* an endorsement:
    `t_interval` is reported beside BCa in intervals.csv for exactly this reason.
    """
    rng = np.random.default_rng(606)
    ratios = []
    for _ in range(400):
        x = rng.normal(0.0, 1.0, 5)
        sd = x.std(ddof=1)
        if sd == 0.0:
            continue
        iv = bca_interval(x, rng=np.random.default_rng(1), n_resamples=2000)
        ratios.append((iv.width / 2.0) / (st.t.ppf(0.975, df=4) * sd / np.sqrt(5)))

    median_ratio = float(np.median(ratios))
    # Measured near 0.63 at n = 5. Bounds are wide enough for resampling noise
    # and narrow enough to fail if the interval were suddenly corrected.
    assert 0.55 < median_ratio < 0.75, median_ratio


def test_single_seed_gives_a_point_and_no_interval():
    iv = bca_interval([0.42], rng=np.random.default_rng(7))
    assert iv.point == pytest.approx(0.42)
    assert np.isnan(iv.lo) and np.isnan(iv.hi)
    assert iv.n_seeds == 1


def test_empty_input_is_nan_not_an_error():
    iv = bca_interval([], rng=np.random.default_rng(8))
    assert np.isnan(iv.point) and np.isnan(iv.lo)
    assert iv.n_seeds == 0


def test_nan_entries_are_dropped_not_zeroed():
    x = [0.5, np.nan, 0.6, 0.55, np.inf, 0.52]
    iv = bca_interval(x, rng=np.random.default_rng(9))
    assert iv.n_seeds == 4
    assert iv.point == pytest.approx(np.mean([0.5, 0.6, 0.55, 0.52]))


def test_identical_seeds_give_a_zero_width_interval():
    iv = bca_interval([0.25] * 5, rng=np.random.default_rng(10))
    assert iv.lo == iv.hi == iv.point


# ── paired gaps ───────────────────────────────────────────────────────────────

def test_paired_gap_forms_the_difference_before_resampling():
    """The gap's interval must be consistent with the per-seed differences.

    Constructed so the arms are nearly perfectly correlated through their shared
    population: each arm is a large common component plus a small arm-specific
    one. An unpaired interval on either arm would be several times wider.
    """
    rng = np.random.default_rng(12)
    common = np.array([10.0, 12.0, 9.0, 11.0, 10.5])
    ncf = common + np.array([0.30, 0.28, 0.31, 0.29, 0.31])
    oracle = common

    gap = paired_gap(ncf, oracle, rng=np.random.default_rng(13))
    assert gap.point == pytest.approx(0.298)
    assert gap.width < 0.10, "shared population did not cancel"

    # Unpaired, the same two arms would be far apart in level and the interval on
    # the mean of either is dominated by the population's spread, not the gap's.
    either = bca_interval(ncf, rng=np.random.default_rng(13))
    assert either.width > 5.0 * gap.width


def test_paired_gap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one value per seed"):
        paired_gap([1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_gap_of_identical_arms_is_zero_width():
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    gap = paired_gap(x, x, rng=np.random.default_rng(14))
    assert gap.lo == gap.hi == 0.0


# ── reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_reproduces_the_interval_exactly():
    x = np.array([0.939, 0.941, 0.938, 0.942, 0.940])
    a = bca_interval(x, rng=np.random.default_rng(99))
    b = bca_interval(x, rng=np.random.default_rng(99))
    assert a == b


def test_default_resample_count_is_ten_thousand():
    iv = bca_interval(np.arange(5.0), rng=np.random.default_rng(1))
    assert iv.n_resamples == 10_000


def test_as_row_carries_labels_and_the_method():
    iv = bca_interval([1.0, 2.0, 3.0, 4.0, 5.0], rng=np.random.default_rng(1))
    row = iv.as_row(quantity="gini", regime="live")
    assert row["quantity"] == "gini" and row["regime"] == "live"
    assert row["method"] == "bca"
    assert row["n_seeds"] == 5


# ── the t cross-check ─────────────────────────────────────────────────────────

def test_t_interval_widens_at_five_seeds_relative_to_the_normal():
    """The t quantile at 4 df is 2.776 against 1.96 — a 42% wider half-width."""
    x = np.array([0.60, 0.68, 0.55, 0.72, 0.65])
    m, lo, hi = t_interval(x)
    assert m == pytest.approx(x.mean())
    normal_half = 1.959964 * x.std(ddof=1) / np.sqrt(5)
    assert (hi - lo) / 2.0 == pytest.approx(2.776445 * x.std(ddof=1) / np.sqrt(5), rel=1e-5)
    assert (hi - lo) / 2.0 > 1.35 * normal_half


def test_t_interval_on_a_single_seed_is_undefined():
    m, lo, hi = t_interval([0.3])
    assert m == pytest.approx(0.3)
    assert np.isnan(lo) and np.isnan(hi)
