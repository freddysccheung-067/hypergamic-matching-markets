"""Verification of the matching machinery against closed-form results.

Five checks. Each is a theorem or an exact identity, so a disagreement is a bug in
this project.

  1. The deficit identity. Under a nested compatibility constraint (a woman may
     match a man whose status is at least hers plus delta), the unmatched mass of
     a maximum matching equals sup_t (G_W(t) - G_M(t + delta))+, with G the
     survival functions. This is the deficiency form of Hall's marriage theorem
     (Hall, 1935): the worst set of women is an upper set when neighborhoods are
     nested. It is a maximum-cardinality matching problem, not a stability
     result, so scipy's exact matching computes it, not ``deferred_acceptance``.
  2. The Gaussian closed form of the deficit. For equal variances the supremum
     of the difference of two shifted normal survival functions is attained at
     the midpoint, so D(delta) = (2 Phi((delta - gap) / (2 sigma)) - 1)+.
  3. Uniqueness under common rankings. When every agent on a side ranks the other
     side identically, the proposer-optimal and acceptor-optimal stable matchings
     coincide (Gale and Shapley, 1962; the lattice of stable matchings collapses
     to a point, Knuth, 1996), and the matching is positive assortative.
  4. The popularity closed form. A ranker that hands every agent on a side the
     same k candidates exposes exactly 2k of N agents, equally, so the exposure
     Gini is 1 - 2k/N and reach is 2k/N.
  5. Truncated deferred acceptance is stable on the lists it is given. On the
     top-k shortlists of a run, no pair listed by both sides prefers each other
     to their partners; against true preferences, blocking pairs remain, which is
     what truncation costs.

Agreement here says the machinery is correct. It says nothing about whether the
model describes a population; see docs/validation.md.
"""

from __future__ import annotations

import copy
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

# The Gaussian setting of checks 2 and 2b: M ~ N(1, 1), W ~ N(0, 1) and, for the
# unequal-variance case, W ~ N(0, 0.9^2); margin delta = 2.
GAUSS_M, GAUSS_W, GAUSS_SD, GAUSS_SD_W, GAUSS_DELTA = 1.0, 0.0, 1.0, 0.9, 2.0

# --------------------------------------------------------------------------- helpers


def unmatched_w_mass(s_m: np.ndarray, s_w: np.ndarray, delta: float = 0.0) -> float:
    """Fraction of W left out by the exact maximum matching on s_m >= s_w + delta.

    The compatibility graph is nested (each woman's neighbors are an upper set of
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
    supremum is attained on the grid of W's own statuses and M's statuses shifted
    by delta; evaluating on their union is exact.
    """
    sm, sw = np.sort(s_m), np.sort(s_w)
    grid = np.concatenate([sw, sm - delta])
    g_w = 1.0 - np.searchsorted(sw, grid, side="left") / len(sw)
    g_m = 1.0 - np.searchsorted(sm, grid + delta, side="left") / len(sm)
    return max(float((g_w - g_m).max()), 0.0)


def gaussian_deficit(delta: float, gap: float, sigma: float) -> float:
    """Equal variances: D(delta) = (2 Phi((delta - gap) / (2 sigma)) - 1)+."""
    return max(2.0 * norm.cdf((delta - gap) / (2.0 * sigma)) - 1.0, 0.0)


def unequal_variance_deficit(delta, m_m, m_w, sd_m, sd_w) -> float:
    """sup_t (G_W(t) - G_M(t + delta))+ for N(m_m, sd_m^2) vs N(m_w, sd_w^2), exact
    to grid resolution. No closed form; the supremum is taken numerically."""
    t = np.linspace(m_w - 12 * sd_w, m_w + 12 * sd_w, 400_001)
    gap = norm.sf(t, m_w, sd_w) - norm.sf(t + delta, m_m, sd_m)
    return max(float(gap.max()), 0.0)


def _ranked_lists(score: np.ndarray) -> list[list[int]]:
    """Row i's preference list: partners by descending ``score[i]``."""
    return [[int(j) for j in np.argsort(-row, kind="stable")] for row in score]


def _matching_pairs(result, proposer_is_m: bool) -> set[tuple[int, int]]:
    """Matches as (m, w) pairs regardless of which side proposed."""
    pairs = (tuple(map(int, p)) for p in result["matches"].tolist())
    return {(p, a) if proposer_is_m else (a, p) for p, a in pairs}


def blocking_pairs_on_lists(prop_prefs: list[list[int]], acc_prefs: list[list[int]],
                            pairs: np.ndarray) -> int:
    """Pairs listed by both sides that prefer each other to their partners.

    Written here, independently of the C++ counter: a pair (p, a) blocks when a is
    on p's list, p is on a's list, and each ranks the other above its partner or
    has none. ``pairs`` holds (proposer slot, acceptor slot) per match.
    """
    partner_p = {int(p): int(a) for p, a in pairs}
    partner_a = {int(a): int(p) for p, a in pairs}
    rank_a = [{p: r for r, p in enumerate(lst)} for lst in acc_prefs]
    count = 0
    for p, lst in enumerate(prop_prefs):
        mine = partner_p.get(p)
        for a in lst:
            if a == mine:
                break  # every later entry is ranked below his partner
            if p not in rank_a[a]:
                continue
            hers = partner_a.get(a)
            if hers is None or rank_a[a][p] < rank_a[a][hers]:
                count += 1
    return count


def truncated_da_check(config: dict, seeds, rounds: int, arms=("ncf", "oracle")):
    """Blocking pairs on the lists and against true preferences, every round.

    Returns (total blocking pairs against the lists, total against true
    preferences, number of (arm, seed, round) matchings checked).
    """
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    on_lists = against_truth = checked = 0
    for arm in arms:
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg["population"]["seed"] = int(seed)
            sim = Simulation(config=cfg, recommender=arm)
            sim.record_outcomes = True
            frame = sim.run(rounds)
            for out, n_true in zip(sim.outcomes, frame["n_blocking"]):
                slots = np.stack([sim._prop_slot[out.pairs[:, 0]],
                                  sim._acc_slot[out.pairs[:, 1]]], axis=1)
                on_lists += blocking_pairs_on_lists(out.prop_prefs, out.acc_prefs, slots)
                against_truth += int(n_true)
                checked += 1
    return on_lists, against_truth, checked


def popularity_check(config: dict, sizes, seeds, rounds: int) -> float:
    """Largest |Gini - (1 - 2k/N)| and |reach - 2k/N| over the given runs.

    Round 0 is skipped: the popularity arm has no history then and ranks by true
    utility, which is a documented fallback and not the popularity rule.
    """
    from src.python.orchestrator import Simulation  # noqa: PLC0415

    worst = 0.0
    k = int(config["ncf"]["top_k"])
    for n in sizes:
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg["population"].update(n=int(n), seed=int(seed), admissible_fraction=20 / n)
            frame = Simulation(config=cfg, recommender="popularity").run(rounds)
            later = frame[frame["epoch"] >= 1]
            worst = max(worst, float((later["gini"] - (1 - 2 * k / n)).abs().max()),
                        float((later["reach"] - 2 * k / n).abs().max()))
    return worst


def shipped_config(n: int | None = None) -> dict:
    from src.python.config import SimConfig  # noqa: PLC0415

    cfg = SimConfig.from_yaml(PROJECT_ROOT / "experiments" / "base.yaml").to_dict()
    if n is not None:
        cfg["population"]["n"] = n
    return cfg

# ---------------------------------------------------------- 1. the deficit identity


def test_deficit_identity_matches_exact_matching():
    n = 300
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # Different means *and* variances, so the survival functions cross and D > 0.
        s_m = rng.normal(0.0, 1.0, n)
        s_w = rng.normal(0.6, 1.3, n)
        measured = unmatched_w_mass(s_m, s_w)
        predicted = empirical_deficit(s_m, s_w)
        assert measured > 0.0, "setup must produce a nonzero deficit to be a test"
        assert abs(measured - predicted) <= 1.0 / n, (seed, measured, predicted)

    # A case with a closed form: M ~ Unif[0, 1], W ~ Beta(2, 1), so G_W(t) = 1 - t^2,
    # G_M(t) = 1 - t, and sup_t (t - t^2) = 1/4 at t = 1/2. Beta(2, 1) is sqrt(U).
    big = 20_000
    rng = np.random.default_rng(7)
    s_m = rng.uniform(size=big)
    s_w = np.sqrt(rng.uniform(size=big))
    measured = unmatched_w_mass(s_m, s_w)
    assert abs(measured - 0.25) < 0.02, measured
    assert abs(measured - empirical_deficit(s_m, s_w)) <= 1.0 / big


# ------------------------------------------------- 2. the Gaussian closed form


def test_gaussian_closed_form_of_the_deficit():
    gap = GAUSS_M - GAUSS_W
    d_closed = gaussian_deficit(GAUSS_DELTA, gap, GAUSS_SD)
    assert gaussian_deficit(gap, gap, GAUSS_SD) == 0.0  # no deficit up to the mean gap
    n = 5_000
    measured = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        s_m = rng.normal(GAUSS_M, GAUSS_SD, n)
        s_w = rng.normal(GAUSS_W, GAUSS_SD, n)
        d = unmatched_w_mass(s_m, s_w, GAUSS_DELTA)
        measured.append(d)
        # The sample supremum is biased up by O(N^-1/2); 0.03 is about three
        # standard errors.
        assert abs(d - d_closed) < 0.03, (seed, d, d_closed)
        assert abs(d - empirical_deficit(s_m, s_w, GAUSS_DELTA)) <= 1.0 / n
    assert abs(float(np.mean(measured)) - d_closed) < 0.01

    # Unequal variances: no closed form, so the exact supremum is the comparison.
    d_exact = unequal_variance_deficit(GAUSS_DELTA, GAUSS_M, GAUSS_W, GAUSS_SD, GAUSS_SD_W)
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        s_m = rng.normal(GAUSS_M, GAUSS_SD, n)
        s_w = rng.normal(GAUSS_W, GAUSS_SD_W, n)
        assert abs(unmatched_w_mass(s_m, s_w, GAUSS_DELTA) - d_exact) < 0.03


# ------------------------------------------- 3. uniqueness under common rankings


def test_common_rankings_give_one_assortative_stable_matching():
    n = 300
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        s_m = rng.normal(size=n)
        s_w = rng.normal(size=n)
        m_prefs = _ranked_lists(np.tile(s_w, (n, 1)))
        w_prefs = _ranked_lists(np.tile(s_m, (n, 1)))
        m_proposes = deferred_acceptance(m_prefs, w_prefs, False)
        w_proposes = deferred_acceptance(w_prefs, m_prefs, False)
        a = _matching_pairs(m_proposes, proposer_is_m=True)
        b = _matching_pairs(w_proposes, proposer_is_m=False)

        # k-th highest M with k-th highest W.
        rank_m = np.argsort(np.argsort(s_m))
        rank_w = np.argsort(np.argsort(s_w))
        w_at_rank = np.empty(n, dtype=int)
        w_at_rank[rank_w] = np.arange(n)
        expected = {(i, int(w_at_rank[rank_m[i]])) for i in range(n)}
        assert len(a) == n and a == b == expected
        assert m_proposes["n_blocking"] == 0 and w_proposes["n_blocking"] == 0

        # Break the precondition: each agent's own view of the other side is
        # perturbed, so the lists are no longer common. Both directions stay
        # stable and the matchings differ.
        eps = 0.01
        pm = _ranked_lists(np.tile(s_w, (n, 1)) + eps * rng.normal(size=(n, n)))
        pw = _ranked_lists(np.tile(s_m, (n, 1)) + eps * rng.normal(size=(n, n)))
        pa = deferred_acceptance(pm, pw, False)
        pb = deferred_acceptance(pw, pm, False)
        assert pa["n_blocking"] == 0 and pb["n_blocking"] == 0
        assert _matching_pairs(pa, True) != _matching_pairs(pb, False)


# --------------------------------------------------- 4. the popularity closed form


def test_popularity_ranker_meets_its_closed_form():
    assert popularity_check(shipped_config(), (500, 1000), (0, 1), 3) < 1e-9


# --------------------------------------- 5. truncated deferred acceptance is stable


def test_truncated_deferred_acceptance_is_stable_on_its_lists():
    on_lists, against_truth, checked = truncated_da_check(shipped_config(300), (0, 1), 3)
    assert checked == 12
    assert on_lists == 0
    # Truncation is not free: against true preferences some pairs block.
    assert against_truth > 0


def test_the_blocking_counter_finds_a_planted_blocking_pair():
    # Proposer 0 lists acceptor 1 first, acceptor 1 lists proposer 0 first, and the
    # planted matching pairs them with others: (0, 1) blocks.
    prop = [[1, 0], [0, 1]]
    acc = [[1, 0], [0, 1]]
    assert blocking_pairs_on_lists(prop, acc, np.array([[0, 0], [1, 1]])) == 2
    assert blocking_pairs_on_lists(prop, acc, np.array([[0, 1], [1, 0]])) == 0


# ------------------------------------------------- 6. the committed table artifact


def test_the_committed_validation_csv_matches():
    """The committed CSV must equal a fresh computation, row for row."""
    import csv as _csv  # noqa: PLC0415

    path = PROJECT_ROOT / "data" / "processed" / "rigor" / "validation.csv"
    assert path.exists(), f"missing {path}; run: python scripts/run_validation.py"
    with path.open(newline="", encoding="utf-8") as handle:
        committed = list(_csv.DictReader(handle))

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_validation  # noqa: PLC0415

    fresh = run_validation.compute()
    assert len(committed) == len(fresh), (
        f"committed CSV has {len(committed)} rows, a fresh computation has "
        f"{len(fresh)}; rerun python scripts/run_validation.py"
    )
    for old, new in zip(committed, fresh):
        assert old == new, (
            "committed validation row differs from a fresh computation:\n"
            f"  committed: {old}\n  fresh:     {new}\n"
            "rerun: python scripts/run_validation.py"
        )
