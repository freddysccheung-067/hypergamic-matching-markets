"""Environment B: the constrained subgraph (THEORY.md §10.3, §10.5).

The properties the comparison depends on:

  * a subgraph the filter leaves unbalanced is balanced deterministically and the
    drop is reported, never silently used;
  * the defection survival is one formula for both environments, d does not enter
    it, and it falls as pairwise alignment falls;
  * as the subgraph grows to the whole population with no filter and no cost, B's
    market is A's complete-information arm;
  * the measured acceptability agrees with its analytic form across the sweep.

The last one has a subtlety worth stating where the assertion is. §10.5 writes
``1 - F(u)^n``, which is the with-replacement law. A pool is a subset of a finite
reference set, so the exact law is hypergeometric, ``1 - C(B,n)/C(K,n)``. The two
differ by up to ~0.06 once n is a sizeable share of K. The 0.03 tolerance is asserted
against the exact form, which is what the measurement is a sample of; the binomial's
gap is asserted to be explained by that correction rather than waved through.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build" / "Release"))

try:
    import _hypergamic  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python import environment_b as eb  # noqa: E402
from src.python.config import EnvironmentBConfig, SimConfig  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402


def _cfg(n: int = 400, seed: int = 11, **env_overrides) -> SimConfig:
    d = SimConfig.from_yaml().to_dict()
    d["population"]["n"] = n
    d["population"]["seed"] = seed
    d["ncf"]["top_k"] = 20
    d["environment_b"].update(env_overrides)
    return SimConfig.from_dict(d)


@pytest.fixture(scope="module")
def sim() -> Simulation:
    return Simulation(config=_cfg(), recommender="complete")


# ── 1. balancing is deterministic and reported ───────────────────────────────


def test_unbalanced_filter_is_dropped_deterministically_and_reported(caplog):
    """A filter that leaves the sides unequal is balanced by highest id and said so."""
    cfg = _cfg(n=400, size_fraction=1.0, values_min=0.4)
    states = Simulation(config=cfg, recommender="oracle").states
    env = cfg.environment_b

    with caplog.at_level(logging.WARNING, logger="src.python.environment_b"):
        first = eb.build_subgraph(states, env, seed=11)
    second = eb.build_subgraph(states, env, seed=11)

    rep = first.report
    # The premise: this population really is unbalanced by the filter. A test that
    # could not fail would prove nothing.
    assert rep.n_pass_m != rep.n_pass_w, "fixture no longer unbalances; pick another seed"
    assert rep.balanced_by_dropping
    assert first.rows_m.size == first.rows_w.size == min(rep.n_pass_m, rep.n_pass_w)

    # Reported: in the returned report AND in the log.
    assert rep.dropped_side in ("M", "W")
    assert rep.dropped_ids.size == abs(rep.n_pass_m - rep.n_pass_w)
    assert any("unbalanced" in r.getMessage() for r in caplog.records)

    # Deterministic: same inputs, same subgraph, same drop.
    assert np.array_equal(first.rows_m, second.rows_m)
    assert np.array_equal(first.rows_w, second.rows_w)
    assert np.array_equal(rep.dropped_ids, second.report.dropped_ids)

    # By id: everything dropped has a higher id than everything kept on that side.
    kept = first.rows_m if rep.dropped_side == "M" else first.rows_w
    assert rep.dropped_ids.min() > kept.max()

    # Nothing in the subgraph violates the filter.
    members = np.concatenate([first.rows_m, first.rows_w])
    assert (states[members, 12] >= env.values_min).all()


def test_balanced_filter_reports_no_drop(caplog):
    """The report is not a standing warning: a balanced pool reports nothing dropped."""
    cfg = _cfg(n=400, size_fraction=1.0, values_min=0.0)
    states = Simulation(config=cfg, recommender="oracle").states
    with caplog.at_level(logging.WARNING, logger="src.python.environment_b"):
        sub = eb.build_subgraph(states, cfg.environment_b, seed=11)
    assert not sub.report.balanced_by_dropping
    assert sub.report.dropped_side is None
    assert sub.n_eligible == 400
    assert not [r for r in caplog.records if "unbalanced" in r.getMessage()]


# ── 3. defection ─────────────────────────────────────────────────────────────


def test_survival_does_not_depend_on_d_and_d_zero_is_a_pure_loss_free_penalty(sim):
    """One code path: A is d = 0, and d changes the price of a dissolution, not its odds."""
    states = sim.states
    env = sim.config.environment_b
    rng = np.random.default_rng(0)
    m = rng.choice(np.flatnonzero(states[:, 1] == 0.0), 60, replace=False)
    w = rng.choice(np.flatnonzero(states[:, 1] == 1.0), 60, replace=False)
    pairs = np.stack([m, w], axis=1)

    env_d = EnvironmentBConfig(**{**env.to_dict(), "defection_penalty": 3.0})
    a = eb.resolve_defection(states, pairs, env, penalty=0.0, seed=5, epoch=2)
    b = eb.resolve_defection(states, pairs, env_d, penalty=env_d.defection_penalty,
                             seed=5, epoch=2)

    # Same survival probabilities and the same realised draws, d = 0 or d = 3.
    assert np.array_equal(a.sigma, b.sigma)
    assert np.array_equal(a.survived, b.survived)
    assert a.n_dissolved > 0, "no pair dissolved; the comparison below would be vacuous"

    # A has no penalty term; the partner loss is the same in both.
    assert a.defector_penalty == 0.0
    assert b.defector_penalty == pytest.approx(3.0 * b.n_dissolved)
    assert a.partner_loss == pytest.approx(b.partner_loss)
    assert a.loss == pytest.approx(a.partner_loss)


def test_survival_draw_is_independent_of_order_and_reproducible(sim):
    """sigma's draw is a function of (seed, epoch, i, j) alone."""
    states, env = sim.states, sim.config.environment_b
    rng = np.random.default_rng(1)
    m = rng.choice(np.flatnonzero(states[:, 1] == 0.0), 80, replace=False)
    w = rng.choice(np.flatnonzero(states[:, 1] == 1.0), 80, replace=False)
    pairs = np.stack([m, w], axis=1)
    perm = rng.permutation(len(pairs))

    base = eb.resolve_defection(states, pairs, env, 0.0, seed=9, epoch=4)
    shuffled = eb.resolve_defection(states, pairs[perm], env, 0.0, seed=9, epoch=4)
    flipped = eb.resolve_defection(states, pairs[:, ::-1].copy(), env, 0.0, seed=9, epoch=4)

    assert np.array_equal(base.survived[perm], shuffled.survived)
    assert np.array_equal(base.survived, flipped.survived)  # (i, j) is (j, i)
    assert np.array_equal(base.survived,
                          eb.resolve_defection(states, pairs, env, 0.0, 9, 4).survived)

    # Epoch and seed are in the key: changing either changes the draws.
    u0 = eb.survival_uniform(9, 4, pairs[:, 0], pairs[:, 1])
    assert not np.array_equal(u0, eb.survival_uniform(9, 5, pairs[:, 0], pairs[:, 1]))
    assert not np.array_equal(u0, eb.survival_uniform(10, 4, pairs[:, 0], pairs[:, 1]))
    assert ((u0 >= 0.0) & (u0 < 1.0)).all()
    # Roughly uniform: not a proof, a guard against a constant or a stuck hash.
    big = eb.survival_uniform(1, 0, np.arange(20000), np.arange(20000) + 20000)
    assert abs(big.mean() - 0.5) < 0.01


def test_survival_probability_is_monotone_decreasing_in_value_gap(sim):
    """Holding intent fixed, a wider |g_i - g_j| never raises survival, and lowers it."""
    env = sim.config.environment_b
    gap = np.linspace(0.0, 1.0, 101)
    tau = np.full_like(gap, 0.7)
    g_i = np.zeros_like(gap)
    sigma = eb.survival_probability(tau, tau, g_i, g_i + gap, env)

    assert np.all(np.diff(sigma) <= 0.0)
    assert sigma[0] > sigma[-1]
    assert sigma[0] == pytest.approx(env.sigma_0 * 0.7 ** (2 * env.sigma_alpha))
    assert sigma[-1] == pytest.approx(0.0)
    assert ((sigma >= 0.0) & (sigma <= 1.0)).all()

    # Pairwise, not individual: the same gap at a different level of g gives the same
    # survival, which a formula on g_i alone could not do.
    low = eb.survival_probability(tau[:1], tau[:1], np.array([0.1]), np.array([0.3]), env)
    high = eb.survival_probability(tau[:1], tau[:1], np.array([0.6]), np.array([0.8]), env)
    assert low == pytest.approx(high)


# ── 2 / 5. the complete arm ──────────────────────────────────────────────────


def test_complete_is_a_recommender_arm_and_runs_unchanged_through_the_harness(sim):
    """No parallel code path: it is a Simulation arm producing the §6 frame."""
    from src.python.orchestrator import ARMS, COLUMNS

    assert "complete" in ARMS
    frame = Simulation(config=_cfg(), recommender="complete").run(2)
    assert list(frame.columns) == COLUMNS
    assert (frame["arm"] == "complete").all()
    assert frame["n_matched"].iloc[0] > 0
    # Lists exist only inside B: an agent outside it is never exposed.
    sub = sim.subgraph
    outside = np.setdiff1d(np.arange(sim.n), np.concatenate([sub.rows_m, sub.rows_w]))
    run = Simulation(config=_cfg(), recommender="complete")
    run.run(1)
    assert (run.exposure_series[0][outside] == 0).all()


def test_complete_arm_needs_an_environment_b_block():
    d = SimConfig.from_yaml().to_dict()
    d["population"]["n"] = 200
    d.pop("environment_b")
    with pytest.raises(ValueError, match="environment_b"):
        Simulation(config=d, recommender="complete")


def test_b_signal_cost_must_exceed_as_only_when_b_is_run():
    """Checked where B runs, so A's cost sweeps stay unconstrained by B's presence."""
    # An A-side config with a cost above B's is valid on its own (the H1 grid).
    d = SimConfig.from_yaml().to_dict()
    d["population"]["n"] = 200
    d["hypergamy"]["signal_cost"] = 1.0
    cfg = SimConfig.from_dict(d)
    Simulation(config=cfg, recommender="oracle")

    # Running B against it is the error.
    with pytest.raises(ValueError, match="must exceed"):
        Simulation(config=cfg, recommender="complete")

    # Equal is also rejected: B must be strictly costlier.
    d["hypergamy"]["signal_cost"] = d["environment_b"]["signal_cost"]
    with pytest.raises(ValueError, match="must exceed"):
        SimConfig.from_dict(d).require_environment_b()


def test_full_size_b_clearance_approaches_the_complete_information_arm_of_a():
    """size_fraction -> 1, no filter, no cost: B's market is A's oracle market.

    This is the reduction that makes B a *restriction* of A rather than a different
    model. It must hold exactly, not approximately: the same lists, the same
    deferred-acceptance call. The cost is set to a value so small it binds on no pair.
    """
    cfg = _cfg(n=400, size_fraction=1.0, values_min=0.0, signal_cost=1e-9)
    a = Simulation(config=cfg, recommender="oracle").run(1)
    b = Simulation(config=cfg, recommender="complete").run(1)
    assert b["mcr"].iloc[0] == pytest.approx(a["mcr"].iloc[0], abs=1e-12)
    assert b["n_matched"].iloc[0] == a["n_matched"].iloc[0]

    # And approaches it: clearance moves toward A's as the pool grows. Compared at
    # a thin and a near-full pool, so a constant would fail.
    target = a["mcr"].iloc[0]
    gaps = []
    for frac in (0.1, 0.5, 0.9):
        c = _cfg(n=400, size_fraction=frac, values_min=0.0, signal_cost=1e-9)
        gaps.append(abs(Simulation(config=c, recommender="complete").run(1)["mcr"].iloc[0]
                        - target))
    assert gaps[-1] < gaps[0]


# ── 4 / 5. the thin-market curve ─────────────────────────────────────────────

SWEEP = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]


@pytest.fixture(scope="module")
def sweep_inputs():
    cfg = _cfg(n=600, seed=21)
    s = Simulation(config=cfg, recommender="complete")
    return s.states, s.config_dict, cfg.environment_b, eb.utility_table(s.states, s.config_dict)


@pytest.mark.parametrize("bar_quantile", [None, 0.95, 0.99])
def test_measured_and_analytic_acceptability_agree_across_the_sweep(sweep_inputs, bar_quantile):
    """Within 0.03 at every size, at the default bar and at two stricter ones.

    The default bar is θ, at which p_acceptable is ~1 everywhere past a tiny pool, so
    agreeing there is nearly free; the 0.95 and 0.99 bars are where pool size binds
    and the test has teeth. Asserted against the exact (without-replacement) law; see
    the module docstring for why the binomial is reported alongside rather than used.
    """
    states, config, env, table = sweep_inputs
    curve = eb.thin_market_curve(states, config, env, SWEEP, n_replicates=150,
                                 table=table, bar_quantile=bar_quantile)

    gap = (curve["p_acceptable"] - curve["p_acceptable_exact"]).abs()
    assert gap.max() < 0.03, curve.round(4).to_string()

    # The binomial is the same quantity to first order and is never *better* than the
    # exact law as a description of the measurement, by more than sampling error.
    binom_gap = (curve["p_acceptable"] - curve["p_acceptable_analytic"]).abs()
    assert (binom_gap + 0.01 >= gap).all()

    # The ordering the whole argument rests on: acceptability rises with pool size.
    assert (np.diff(curve["p_acceptable"].to_numpy()) >= -0.02).all()
    if bar_quantile is not None:
        assert curve["p_acceptable"].iloc[0] < 0.7 < curve["p_acceptable"].iloc[-1]


def test_critical_mass_matches_its_definition(sweep_inputs):
    """N* = ln(1 - rho) / ln F, with F recomputed here from the utility table.

    F is *not* assumed equal to the bar quantile q. "Strictly below the q-quantile"
    covers slightly less than q of a finite reference set (0.987 rather than 0.99 at
    ~230 candidates), and at q = 0.99 that moves N* from 229 to 181. The test
    therefore rebuilds F from the table independently of the function under test.
    """
    states, config, env, table = sweep_inputs
    u_mw, u_wm = table
    types = states[:, 1]
    prop_rows, acc_rows = np.flatnonzero(types == 0.0), np.flatnonzero(types == 1.0)
    passes = states[:, 12] >= env.values_min
    ref_a, ref_p = np.flatnonzero(passes[acc_rows]), np.flatnonzero(passes[prop_rows])

    for q, rho in ((0.95, 0.5), (0.99, 0.9)):
        bar_m = np.quantile(u_mw[:, ref_a], q, axis=1)
        bar_w = np.quantile(u_wm[:, ref_p], q, axis=1)
        f = np.concatenate([(u_mw[:, ref_a] < bar_m[:, None]).mean(axis=1),
                            (u_wm[:, ref_p] < bar_w[:, None]).mean(axis=1)]).mean()

        row = eb.thin_market_curve(states, config, env, [0.5], rho=rho, n_replicates=2,
                                   table=table, bar_quantile=q).iloc[0]
        n_star = row["critical_mass"]
        assert n_star == pytest.approx(np.log(1 - rho) / np.log(f), rel=1e-9)
        assert 1.0 - f ** n_star == pytest.approx(rho, abs=1e-9)
        # F sits at or just under q: the grain of the reference set, not an error.
        assert q - 0.02 < f <= q



def test_curve_reports_pool_size_and_drops(sweep_inputs):
    states, config, env, table = sweep_inputs
    curve = eb.thin_market_curve(states, config, env, SWEEP, n_replicates=10, table=table)
    assert list(curve["eligible_n"].round().astype(int)) == sorted(
        curve["eligible_n"].round().astype(int)
    )
    assert (curve["n_dropped"] >= 0).all()
    assert (curve["clearance_prob"] <= curve["p_acceptable"] + 1e-9).all(), (
        "clearance needs acceptability to be mutual and to survive truncation and the "
        "cost, so it cannot exceed the one-sided existence probability"
    )
