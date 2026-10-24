"""Validation of the simulator against Bilderbeek (2026), "Hypergamy Is Solved".

Three checks, one test each. The paper's results are closed-form, so a
disagreement here is a bug in this project, not in the paper.

  1. Theorem 2.2 (deficit D) -- a max-cardinality bipartite matching problem under
     a compatibility constraint s_m >= s_w. It is NOT a stability result and
     ``deferred_acceptance`` does not compute it; scipy's exact matching does.
  2. Theorem 3.3 (height deficit) -- the Gaussian closed form, against the same
     exact matching at N = 5,000.
  3. Theorem 5.1 (assortative uniqueness) -- this one is the Gale-Shapley engine.

Agreement on D says nothing about the recommender's visibility constraint; see
docs/validation.md.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import norm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import _hypergamic  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from _hypergamic import deferred_acceptance  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- helpers


def unmatched_w_mass(s_m: np.ndarray, s_w: np.ndarray, delta: float = 0.0) -> float:
    """Fraction of W left out by the exact maximum matching on s_m >= s_w + delta.

    The compatibility graph is nested (each woman's neighbours are an upper set of
    men), so it is built row by row from one sort of M instead of as an N x N dense
    matrix. Rows are women, columns are men.
    """
    n_w, n_m = len(s_w), len(s_m)
    order = np.argsort(s_m, kind="stable").astype(np.int32)
    start = np.searchsorted(s_m[order], s_w + delta, side="left")
    indptr = np.zeros(n_w + 1, dtype=np.int64)
    np.cumsum(n_m - start, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for i in range(n_w):
        indices[indptr[i] : indptr[i + 1]] = order[start[i] :]
    graph = csr_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(n_w, n_m)
    )
    matched_to = maximum_bipartite_matching(graph, perm_type="column")
    return float((matched_to < 0).sum()) / n_w


def empirical_deficit(s_m: np.ndarray, s_w: np.ndarray, delta: float = 0.0) -> float:
    """sup_t (G_W(t) - G_M(t + delta))+ from the empirical survival functions.

    G(t) = P(s >= t). Both step functions only change at sample points, so the
    supremum is attained on the grid of W's own statuses (G_W steps down just
    after each) and M's statuses shifted by delta; evaluating on their union is
    exact.
    """
    sm, sw = np.sort(s_m), np.sort(s_w)
    grid = np.concatenate([sw, sm - delta])
    g_w = 1.0 - np.searchsorted(sw, grid, side="left") / len(sw)
    g_m = 1.0 - np.searchsorted(sm, grid + delta, side="left") / len(sm)
    return max(float((g_w - g_m).max()), 0.0)


def gaussian_deficit(delta: float, gap: float, sigma: float) -> float:
    """Theorem 3.3: D(delta) = (2 Phi((delta - gap) / 2 sigma) - 1)+."""
    return max(2.0 * norm.cdf((delta - gap) / (2.0 * sigma)) - 1.0, 0.0)


def unequal_variance_deficit(delta, m_m, m_w, sd_m, sd_w) -> float:
    """sup_t (G_W(t) - G_M(t + delta))+ for N(m_m, sd_m^2) vs N(m_w, sd_w^2), exact
    to grid resolution. Theorem 3.3 does not cover this case; Theorem 2.2 does."""
    t = np.linspace(m_w - 12 * sd_w, m_w + 12 * sd_w, 400_001)
    gap = norm.sf(t, m_w, sd_w) - norm.sf(t + delta, m_m, sd_m)
    return max(float(gap.max()), 0.0)


# ------------------------------------------------------------- 1. Theorem 2.2, deficit


def test_theorem_2_2_deficit_matches_exact_matching():
    n = 300
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # Different means *and* variances, so the survival functions cross and D > 0.
        s_m = rng.normal(0.0, 1.0, n)
        s_w = rng.normal(0.6, 1.3, n)
        measured = unmatched_w_mass(s_m, s_w)
        predicted = empirical_deficit(s_m, s_w)
        rows.append((seed, measured, predicted))
        assert measured > 0.0, "setup must produce a nonzero deficit to be a test"
        assert abs(measured - predicted) <= 1.0 / n, (seed, measured, predicted)

    # Example 2.4: M ~ Unif[0,1], W ~ Beta(2,1)  =>  D = 1/4 at t = 1/2.
    # Beta(2,1) by inverse CDF: sqrt(U). Sampling error dominates at small N.
    big = 20_000
    rng = np.random.default_rng(7)
    s_m = rng.uniform(size=big)
    s_w = np.sqrt(rng.uniform(size=big))
    measured = unmatched_w_mass(s_m, s_w)
    assert abs(measured - 0.25) < 0.02, measured
    # The exact matching still equals the empirical sup to one agent at this N.
    assert abs(measured - empirical_deficit(s_m, s_w)) <= 1.0 / big

    print("\n[check 1] measured vs sup_t(G_W - G_M)+ at N=300:")
    for seed, meas, pred in rows:
        print(
            f"  seed {seed}: matching {meas:.4f}  sup {pred:.4f}  diff {abs(meas - pred):.4f}"
        )
    print(f"[check 1] Example 2.4, N={big}: measured {measured:.4f}  predicted 0.2500")


# --------------------------------------------------------- 2. Theorem 3.3, height deficit


def test_theorem_3_3_height_deficit():
    m_m, m_w, sigma = 175.3, 161.3, 7.6
    gap = m_m - m_w
    one_foot = 30.48
    n = 5_000

    # The closed form itself: D(0) = 0 and the free region ends at the mean gap.
    assert gaussian_deficit(0.0, gap, sigma) == 0.0
    assert gaussian_deficit(gap, gap, sigma) == 0.0
    d_foot = gaussian_deficit(one_foot, gap, sigma)

    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        s_m = rng.normal(m_m, sigma, n)
        s_w = rng.normal(m_w, sigma, n)  # equal variance: Theorem 3.3's hypothesis
        d0 = unmatched_w_mass(s_m, s_w, 0.0)
        d30 = unmatched_w_mass(s_m, s_w, one_foot)
        out.append((seed, d0, d30))
        # D(0) ~ 0: the paper bounds it by 1e-150; at N=5000 it is exactly zero.
        assert d0 == 0.0, (seed, d0)
        # Exact matching against the closed form. The sample sup is biased up by
        # O(N^-1/2); 0.03 is about three standard errors and still tight.
        assert abs(d30 - d_foot) < 0.03, (seed, d30, d_foot)
        # And against the sample's own sup, to one agent.
        assert abs(d30 - empirical_deficit(s_m, s_w, one_foot)) <= 1.0 / n

    mean30 = float(np.mean([o[2] for o in out]))
    assert abs(mean30 - d_foot) < 0.01, (mean30, d_foot)

    # The paper's headline "0.74 at one foot" is the *unequal-variance* calibration
    # (sigma_M = 7.6, sigma_W = 7.1), which Theorem 3.3 does not cover. Theorem 2.2
    # does: compare the exact sup with the matching on that population.
    sd_m, sd_w = 7.6, 7.1
    d_unequal = unequal_variance_deficit(one_foot, m_m, m_w, sd_m, sd_w)
    assert abs(d_unequal - 0.74) < 0.005, d_unequal
    meas_unequal = []
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        s_m = rng.normal(m_m, sd_m, n)
        s_w = rng.normal(m_w, sd_w, n)
        meas = unmatched_w_mass(s_m, s_w, one_foot)
        meas_unequal.append(meas)
        assert abs(meas - d_unequal) < 0.03, (seed, meas, d_unequal)

    print(
        f"\n[check 2] equal sigma=7.6, N={n}: closed form D(0)=0, D(30.48)={d_foot:.4f}"
    )
    for seed, d0, d30 in out:
        print(f"  seed {seed}: D(0) {d0:.4f}  D(30.48) {d30:.4f}")
    print(f"  mean D(30.48) {mean30:.4f}  sd {np.std([o[2] for o in out], ddof=1):.4f}")
    print(
        f"[check 2] unequal sigma (7.6, 7.1): exact sup {d_unequal:.4f} (paper 0.74); "
        f"matching mean {np.mean(meas_unequal):.4f} sd {np.std(meas_unequal, ddof=1):.4f}"
    )


# ------------------------------------------------- 3. Theorem 5.1, assortative uniqueness


def _ranked_lists(score: np.ndarray) -> list[list[int]]:
    """Row i's preference list: partners by descending ``score[i]``."""
    return [[int(j) for j in np.argsort(-row, kind="stable")] for row in score]


def _matching_pairs(result, proposer_is_m: bool) -> set[tuple[int, int]]:
    """Matches as (m, w) pairs regardless of which side proposed."""
    pairs = (tuple(map(int, p)) for p in result["matches"].tolist())
    return {(p, a) if proposer_is_m else (a, p) for p, a in pairs}


def test_theorem_5_1_assortative_uniqueness_and_its_precondition():
    n = 300
    diverged = []
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        s_m = rng.normal(size=n)
        s_w = rng.normal(size=n)

        # Status-ranked, complete: everyone ranks the other side by status alone.
        m_prefs = _ranked_lists(np.tile(s_w, (n, 1)))
        w_prefs = _ranked_lists(np.tile(s_m, (n, 1)))

        m_proposes = deferred_acceptance(m_prefs, w_prefs, False)
        w_proposes = deferred_acceptance(w_prefs, m_prefs, False)
        a = _matching_pairs(m_proposes, proposer_is_m=True)
        b = _matching_pairs(w_proposes, proposer_is_m=False)

        # k-th highest M with k-th highest W: argsort(argsort(s)) is each agent's
        # ascending rank, so equal rank means equal k.
        rank_m = np.argsort(np.argsort(s_m))
        rank_w = np.argsort(np.argsort(s_w))
        w_at_rank = np.empty(n, dtype=int)
        w_at_rank[rank_w] = np.arange(n)
        expected = {(i, int(w_at_rank[rank_m[i]])) for i in range(n)}

        assert len(a) == n and len(b) == n
        assert a == b, f"seed {seed}: proposing side changed the matching"
        assert a == expected, f"seed {seed}: not positive assortative"
        assert m_proposes["n_blocking"] == 0 and w_proposes["n_blocking"] == 0

        # Break the precondition: perturb each agent's own view of the other side.
        # Status is still the dominant term, but the lists are no longer common.
        eps = 0.01
        pm = _ranked_lists(np.tile(s_w, (n, 1)) + eps * rng.normal(size=(n, n)))
        pw = _ranked_lists(np.tile(s_m, (n, 1)) + eps * rng.normal(size=(n, n)))
        pa = deferred_acceptance(pm, pw, False)
        pb = deferred_acceptance(pw, pm, False)
        a2 = _matching_pairs(pa, proposer_is_m=True)
        b2 = _matching_pairs(pb, proposer_is_m=False)
        # Both are still stable (zero blocking pairs), yet they are different
        # matchings: stability survives, uniqueness does not.
        assert pa["n_blocking"] == 0 and pb["n_blocking"] == 0
        assert a2 != b2, f"seed {seed}: perturbed lists still gave one matching"
        diverged.append(len(a2 - b2))

    print(f"\n[check 3] n={n}, five seeds: M- and W-proposing matchings identical and")
    print("  equal to the rank-k pairing on every seed (0 blocking pairs).")
    print(
        f"[check 3] eps=0.01 perturbation: pairs that differ by direction = {diverged}"
    )
