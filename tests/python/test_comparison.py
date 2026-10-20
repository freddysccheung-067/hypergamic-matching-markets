"""The institutional comparison and its comparability rule (THEORY.md §10.4, §10.6).

What these tests pin:

  * the §10.4 total reproduces the pre-session welfare exactly where it must (every
    intent 1, c = 0, no dissolution), checked against values recorded *before*
    compute_welfare was extended rather than against the new code;
  * welfare_common at size_fraction = 1 with no filter and no cost is A's welfare;
  * both accountings read one state matrix, A and B share population and seed, and
    a mismatched pair raises;
  * the factorial's four cells share a population and seed — and nothing about the
    sign of any term.

**Two properties the session prompt asked for do not hold, and are tested as
measured.** "welfare_selected >= welfare_common always" and "the gap grows as
values_min rises" are both false in this model. The artifact is N times the gap
between the mean entrant's payoff and the mean agent's payoff under the common rule;
nothing forces the entrants to be better off. It turns negative when B's proposers
spend more on c_B than they gain, and when the filter is strict enough that B is too
thin to match its own entrants. The tests below assert the identity the artifact
*does* satisfy, the region where it is non-negative, and the counterexamples, so that
a change that silently flips any of them fails. THEORY.md §10.4 states the
correction.
"""

from __future__ import annotations

import copy
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

from _hypergamic import compute_welfare, deferred_acceptance  # noqa: E402

from src.python import comparison as cmp  # noqa: E402
from src.python.config import SimConfig  # noqa: E402
from src.python.factorial import CELLS, run_factorial  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

logging.getLogger("src.python.environment_b").setLevel(logging.ERROR)

# Oracle arm, epoch 0, shipped config, recorded before Session 5-E touched
# compute_welfare (and identical to the welfare_alg column of that run).
PRE_SESSION_WELFARE = {42: 2193.2503578352744, 43: 2048.172665594763}


def _cfg(seed: int = 42, n: int | None = None, **env) -> dict:
    d = SimConfig.from_yaml().to_dict()
    d["population"]["seed"] = seed
    if n is not None:
        d["population"]["n"] = n
    d["environment_b"].update(env)
    return d


# ── 1. Welfare accounting ────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", sorted(PRE_SESSION_WELFARE))
def test_total_equals_pre_session_welfare_at_unit_intent(seed: int) -> None:
    sim = Simulation(config=_cfg(seed), recommender="oracle")
    m = deferred_acceptance(*sim._oracle_prefs(20), True)["matches"]

    w = compute_welfare(sim.states, m, sim.config_dict)
    assert w["welfare"] == PRE_SESSION_WELFARE[seed]

    unit = sim.states.copy()
    unit[:, 11] = 1.0
    w1 = compute_welfare(unit, m, sim.config_dict, 0.0, 0, 0.0)
    # Bit-for-bit: 1.0 * u is exact and the accumulation order is the same.
    assert w1["welfare_total"] == PRE_SESSION_WELFARE[seed]
    assert w1["value_created"] == w1["welfare"]


def test_shipped_intent_scales_value_and_is_not_the_old_welfare() -> None:
    """At Beta(2,2) intent value_created is about half of welfare, as documented."""
    sim = Simulation(config=_cfg(42), recommender="oracle")
    m = deferred_acceptance(*sim._oracle_prefs(20), True)["matches"]
    w = compute_welfare(sim.states, m, sim.config_dict)
    assert w["value_created"] / w["welfare"] == pytest.approx(0.5157, abs=5e-4)


def test_terms_are_subtracted_and_negative_terms_refused() -> None:
    sim = Simulation(config=_cfg(42), recommender="oracle")
    m = deferred_acceptance(*sim._oracle_prefs(20), True)["matches"]
    w = compute_welfare(sim.states, m, sim.config_dict, 0.25, 40, 3.5)
    assert w["signal_cost"] == 10.0
    assert w["welfare_total"] == w["value_created"] - 10.0 - 3.5
    for bad in ((-0.1, 1, 0.0), (0.1, -1, 0.0), (0.1, 1, -1.0), (float("nan"), 1, 0.0)):
        with pytest.raises(ValueError):
            compute_welfare(sim.states, m, sim.config_dict, *bad)


def test_frame_terms_add_up_and_match_the_payoff_vector() -> None:
    cfg = _cfg(42, n=400, size_fraction=0.5)
    for arm in ("oracle", "complete"):
        sim = Simulation(config=cfg, recommender=arm)
        sim.record_outcomes = True
        f = sim.run(3)
        np.testing.assert_allclose(
            f.welfare_total, f.value_created - f.proposal_cost_total - f.defection_loss,
            rtol=0, atol=1e-9)
        sums = [o.payoff.sum() for o in sim.outcomes]
        np.testing.assert_allclose(f.welfare_total, sums, rtol=1e-12, atol=1e-9)


# ── 2. The two accountings ───────────────────────────────────────────────────


def test_common_at_full_size_equals_a_side_welfare() -> None:
    """size_fraction = 1, no filter, d = 0, negligible c_B: B is A's oracle arm."""
    cfg = _cfg(42, size_fraction=1.0, values_min=0.0, signal_cost=1e-300,
               defection_penalty=0.0)
    r = cmp.paired_run(cfg, 2)
    assert np.array_equal(r.frame_a.n_matched, r.frame_b.n_matched)
    np.testing.assert_allclose(r.frame_b.welfare_common, r.frame_a.welfare_total,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(r.welfare_common_residual, r.frame_a.welfare_total,
                               rtol=0, atol=1e-9)


def test_selection_artifact_identity() -> None:
    """selected - common is N * (mean entrant payoff - mean agent payoff)."""
    r = cmp.paired_run(_cfg(42, size_fraction=0.35, signal_cost=0.01), 1)
    ob, oa = r.sim_b.outcomes[0], r.sim_a.outcomes[0]
    acc = cmp.welfare_accounts(ob, oa)
    n = ob.payoff.size
    per_agent = np.where(ob.matched, ob.payoff, oa.payoff + ob.payoff)
    assert per_agent.sum() == pytest.approx(acc.welfare_common, rel=1e-12)
    expected = n * (ob.payoff[ob.entrant].mean() - per_agent.mean())
    assert acc.welfare_selection_artifact == pytest.approx(expected, rel=1e-12, abs=1e-9)
    assert r.frame_b.welfare_selection_artifact[0] == acc.welfare_selection_artifact


def test_selection_artifact_is_nonnegative_where_b_is_cheap_and_not_thin() -> None:
    # Measured at N = 1000, seed 42, c_B = 0.01: 46 to 299 across these sizes.
    for sf in (0.10, 0.25, 0.35, 0.50):
        r = cmp.paired_run(_cfg(42, size_fraction=sf, signal_cost=0.01), 1)
        assert r.frame_b.welfare_selection_artifact[0] >= 0.0, sf


def test_selection_artifact_sign_is_not_guaranteed() -> None:
    """The prompt's ">= 0 always" is false; these are the measured counterexamples.

    Cost: at c_B = 0.25 every entrant pays for up to k proposals and is matched at
    most once, so entrants end below the population. Thinness: at values_min 0.8
    B is too small to match its own entrants. Either way the selected mean is the
    lower one.
    """
    costly = cmp.paired_run(_cfg(42, size_fraction=0.5, signal_cost=0.25), 1)
    assert costly.frame_b.welfare_selection_artifact[0] < 0.0
    strict = cmp.paired_run(_cfg(42, size_fraction=0.25, signal_cost=1e-9,
                                 values_min=0.8), 1)
    assert strict.frame_b.welfare_selection_artifact[0] < 0.0


def test_selection_artifact_is_not_monotone_in_values_min() -> None:
    """Rises while the filter removes low-alignment agents, falls once B is thin.

    Measured at N = 1000, seed 42, size_fraction 0.5, c_B ~ 0: 11.5, 105.7, 241.8,
    188.6, -106.5 at values_min 0.0, 0.2, 0.4, 0.6, 0.8. The prompt's "gap grows as
    values_min rises" holds on the first half only.
    """
    art = []
    for vm in (0.0, 0.2, 0.4, 0.6, 0.8):
        r = cmp.paired_run(_cfg(42, size_fraction=0.5, signal_cost=1e-9, values_min=vm), 1)
        art.append(float(r.frame_b.welfare_selection_artifact[0]))
    assert art[0] < art[1] < art[2], art
    assert art[2] > art[3] > art[4], art


def test_both_accountings_read_one_state_matrix() -> None:
    r = cmp.paired_run(_cfg(42, n=400, size_fraction=0.5), 3)
    for oa, ob in zip(r.sim_a.outcomes, r.sim_b.outcomes):
        cols = np.arange(oa.states.shape[1]) != 6
        assert np.array_equal(oa.states[:, cols], ob.states[:, cols])
    # Epoch 0: θ has not moved in either run, so the matrices are equal outright.
    assert np.array_equal(r.sim_a.outcomes[0].states, r.sim_b.outcomes[0].states)


def test_a_and_b_share_population_and_seed() -> None:
    r = cmp.paired_run(_cfg(43, n=400, size_fraction=0.5), 1)
    assert r.sim_a.rng_seed == r.sim_b.rng_seed == 43
    assert r.sim_a.outcomes[0].seed == r.sim_b.outcomes[0].seed == 43
    assert np.array_equal(Simulation(config=_cfg(43, n=400)).states,
                          r.sim_b.outcomes[0].states)


def test_mismatched_population_raises() -> None:
    a = cmp.paired_run(_cfg(42, n=400, size_fraction=0.5), 1)
    b = cmp.paired_run(_cfg(43, n=400, size_fraction=0.5), 1)
    oa, ob_other = a.sim_a.outcomes[0], b.sim_b.outcomes[0]
    with pytest.raises(ValueError, match="different seeds"):
        cmp.welfare_accounts(ob_other, oa)

    # Same seed label, different population: the label is not trusted.
    ob = a.sim_b.outcomes[0]
    tampered = copy.copy(ob)
    states = ob.states.copy()
    states[0, 2] += 1e-12
    object.__setattr__(tampered, "states", states)
    with pytest.raises(ValueError, match="do not share a population"):
        cmp.welfare_accounts(tampered, oa)

    with pytest.raises(ValueError, match="different epochs"):
        cmp.welfare_accounts(a.sim_b.outcomes[0],
                             cmp.EpochOutcome(**{**oa.__dict__, "epoch": 1}))


def test_residual_a_without_removal_is_a() -> None:
    r = cmp.paired_run(_cfg(42, n=400, size_fraction=0.5), 1)
    oa = r.sim_a.outcomes[0]
    res = cmp.residual_a_payoff(oa, np.zeros(oa.payoff.size, bool), r.sim_a.config_dict,
                                r.sim_b.config.environment_b)
    assert np.array_equal(res, oa.payoff)


# ── 5. Factorial ─────────────────────────────────────────────────────────────


def test_factorial_cells_share_population_and_seed() -> None:
    """The only assertion on the decomposition. No term has an asserted sign."""
    res = run_factorial(_cfg(44, n=300), gamma=0.02, n_epochs=4, window=2)
    assert res.seed == 44
    names = [c[0] for c in CELLS]
    assert set(res.epoch0_states) == set(names)
    for name in names[1:]:
        assert np.array_equal(res.epoch0_states[names[0]], res.epoch0_states[name])
    w = res.cells
    assert res.interaction == pytest.approx(
        w["both"] - w["ncf"] - w["drift"] + w["baseline"], abs=1e-12)


def test_drift_moves_only_w_status_and_only_after_epoch_0() -> None:
    cfg = _cfg(42, n=300)
    cfg["simulation"]["status_drift"] = 0.05
    sim = Simulation(config=cfg, recommender="oracle", freeze_theta=True)
    sim.record_states = True
    sim.run(3)
    s0, s1, s2 = sim.states_seen
    w = s0[:, 1] == 1.0
    step = 0.05 * np.std(s0[:, 2], ddof=1)
    np.testing.assert_allclose(s1[w, 2] - s0[w, 2], step, rtol=1e-12)
    np.testing.assert_allclose(s2[w, 2] - s0[w, 2], 2 * step, rtol=1e-12)
    others = np.ones(s0.shape[1], bool)
    others[2] = False
    assert np.array_equal(s0[:, others], s2[:, others])
    assert np.array_equal(s0[~w], s2[~w])


def test_zero_drift_is_bit_identical() -> None:
    cfg = _cfg(42, n=300)
    a = Simulation(config=cfg, recommender="oracle").run(3)
    cfg["simulation"]["status_drift"] = 0.0
    b = Simulation(config=cfg, recommender="oracle").run(3)
    assert a.equals(b)
