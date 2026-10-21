"""The signalling cost, the proposal decision and the separating threshold.

THEORY.md §10.2 states the mechanism; THESIS.md H1 is the hypothesis it is
supposed to support. This file is the set of properties the rest of the project
depends on, in the order of the contract:

  * c = 0 changes nothing, at several seeds and through the whole orchestrator;
  * E[tau | proposal] is non-decreasing in c, and strictly increasing overall;
  * above the cutoff the market closes: no proposals, and mcr = 0;
  * the proposal count falls monotonically in c;
  * intent does not enter utility_toward — the guard against the failure mode
    where the cost is implemented inside the utility and every welfare number
    silently moves.

The measurement behind the H1 test is worth stating here rather than only in the
assertion: at the shipped weights the utility V of an admissible pair is of order
4, so tau * V * p is of order 1 and a cost of 0.05 cannot bind on anything. The
c = 0 and c = 0.05 rows are therefore exactly equal at every seed, and that is a
property of the model, not a defect in the mechanism. The test asserts
non-decrease over the full grid and a strict increase from the first row to the
last, which is what H1 actually claims.
"""

import hashlib
import pathlib
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build" / "Release"))

try:
    import _hypergamic
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python.config import SimConfig  # noqa: E402
from src.python.metrics import (  # noqa: E402
    mean_proposal_intent,
    proposal_cost_total,
    separating_threshold,
)
from src.python.orchestrator import Simulation  # noqa: E402


def base_config(**overrides):
    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


def with_cost(cfg, cost):
    """A config with the cost set, as the C++ decision reads it.

    signal_cost lives in hypergamy; accept_prob is handed in per side by the
    orchestrator. Here it is pinned at the epoch-0 default so this file measures
    the c-axis alone and nothing else moves between rows.
    """
    out = yaml.safe_load(yaml.safe_dump(cfg))
    out["hypergamy"]["signal_cost"] = float(cost)
    out["signal"] = {"accept_prob": 0.5}
    return out


def positives(states, cfg):
    """(n_positives, mean proposer intent) for one config."""
    log = _hypergamic.generate_swipes(states, cfg, 0)
    pos = log[log[:, 2] == 1]
    if pos.shape[0] == 0:
        return 0, float("nan")
    return int(pos.shape[0]), float(states[pos[:, 0], 11].mean())


# ── 1. c = 0 reproduces the pre-session swipe log ────────────────────────────

PINNED = [
    "2adea38ae4533316a16e398c6163bb99616457038827cb01c0dc284af516bc4b",
    "823a8f7ffebd8221727be183adc1db668d5cf56dcfea21133bf27d7270ddb005",
    "e72d11404fb2063c92182ad617bc7db42d46be5a6b8880b7e1b793b4be50fc9e",
]


def test_cost_zero_reproduces_the_pinned_hash():
    """base.yaml, seed 42, N=1000, epochs 0-2 — the hash heterogeneity pins.

    The two heterogeneity knobs are forced off to match that recording; the cost
    is the default 0.0 and is not set at all, which is also the case that matters:
    a config that predates the key must run unchanged. admissible_fraction is
    pinned at 0.2, the value the hash was recorded at (see test_heterogeneity.py).
    """
    cfg = base_config(population={"taste_sigma": 0.0, "admissible_fraction": 0.2},
                      hypergamy={"shock_scale": 0.0})
    assert "signal_cost" not in cfg["hypergamy"] or cfg["hypergamy"]["signal_cost"] == 0.0
    states = _hypergamic.sample_population(1000, cfg, 42)

    digests = [
        hashlib.sha256(_hypergamic.generate_swipes(states, cfg, epoch).tobytes()).hexdigest()
        for epoch in (0, 1, 2)
    ]
    assert digests == PINNED


@pytest.mark.parametrize("seed", [7, 11, 23, 101])
def test_explicit_zero_cost_matches_an_absent_key(seed):
    """Setting the key to 0.0 is the same run as omitting it, at any seed."""
    cfg = base_config(population={"seed": seed})
    states = _hypergamic.sample_population(400, cfg, seed)
    implicit = _hypergamic.generate_swipes(states, cfg, 0)
    explicit = _hypergamic.generate_swipes(states, with_cost(cfg, 0.0), 0)
    assert np.array_equal(implicit, explicit)


def test_cost_zero_leaves_the_orchestrator_frame_identical():
    """The whole pipeline, not only the swipe log: every pre-existing column.

    A separate pinned comparison of the *frame* rather than the log, because the
    log does not see the matching, the welfare read or the θ step, and a change
    in any of those would be invisible to the hash above. Recorded at
    admissible_fraction 0.2 and pinned there.
    """
    cfg = base_config(population={"n": 200, "admissible_fraction": 0.2},
                      simulation={"n_epochs": 6})
    frame = Simulation(config=cfg).run(3)

    assert list(frame.columns) == _COLUMNS_EXPECTED
    # Rounded because the frame is float arithmetic through numpy; the claim is
    # that the two runs agree to a value, and a bit-for-bit float comparison
    # across a pandas round trip would be testing pandas.
    got = frame[["gini", "mcr", "rsi", "welfare_alg", "n_matched", "reach"]].to_numpy()
    expected = np.array([
        [0.7975775, 0.21, 0.3642857143, 143.4235516629, 21, 0.24],
        [0.7992350, 0.21, 0.4642857143, 131.0997282758, 21, 0.225],
        [0.7997825, 0.20, 0.4800000000, 143.9094424529, 20, 0.215],
    ])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


# The §6 column list as it stands after this session. Compared as a tuple so the
# order is part of the contract: the CSV is read back by name, but the plots index
# positionally in one place and a reorder would break them.
_COLUMNS_EXPECTED = [
    "epoch", "generation", "delta", "seed", "gini", "mcr", "rsi", "bpr",
    "welfare_alg", "welfare_full", "algorithmic_loss", "n_matched", "n_blocking",
    "pareto_alpha", "mean_swipes_per_agent", "arm", "reach", "top1_share",
    "mean_proposal_intent", "separating_threshold", "proposal_cost_total",
    # Session 5-E, appended (THEORY.md §10.4). They do not enter the values pinned
    # below, so the c = 0 identity this test exists for is untouched by them.
    "value_created", "defection_loss", "welfare_total",
    "welfare_common", "welfare_selected", "welfare_selection_artifact",
]


# ── 2. H1: E[tau | proposal] rises with c ────────────────────────────────────

COST_GRID = (0.0, 0.05, 0.25, 1.0)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_h1_mean_proposal_intent_is_non_decreasing_in_cost(seed):
    """THESIS H1, at the shipped weights and three seeds.

    Non-decreasing over the grid and strictly higher at the top than at the
    bottom. The weak form on the *steps* is deliberate: c = 0.05 is below
    min(tau * V * p) over the admissible set at these weights, so it selects on
    nothing and the first two rows are equal. Asserting strict increase at every
    step would be asserting a property of the utility scale, not of the
    mechanism, and it would fail for the right reason at the wrong test.

    The kill condition in THESIS.md is "E[tau | proposal] is flat in c over the
    grid". A flat curve fails the strict-increase assertion below.
    """
    cfg = base_config(population={"seed": seed})
    states = _hypergamic.sample_population(1000, cfg, seed)

    rows = []
    for cost in COST_GRID:
        n_pos, mean_tau = positives(states, with_cost(cfg, cost))
        rows.append((cost, n_pos, mean_tau))

    taus = [r[2] for r in rows]
    for (c_lo, _, t_lo), (c_hi, _, t_hi) in zip(rows, rows[1:]):
        assert t_hi >= t_lo - 1e-12, (
            f"E[tau | proposal] fell from {t_lo:.6f} at c={c_lo} to {t_hi:.6f} "
            f"at c={c_hi} (seed {seed}); H1 would be refuted, not the test"
        )
    assert taus[-1] > taus[0] + 1e-3, (
        f"E[tau | proposal] is effectively flat over {COST_GRID} "
        f"(seed {seed}): {taus}. H1 is refuted at these weights."
    )


def test_h1_holds_through_the_orchestrator():
    """The same claim measured on the §6 column the orchestrator reports.

    The C++-level test above measures the log; this one measures what a reader of
    the results frame sees. They can disagree — the orchestrator estimates p per
    side from history and restricts the matching to proposals — so both are
    asserted rather than one standing in for the other.
    """
    measured = []
    for cost in (0.0, 0.25, 1.0):
        cfg = base_config(population={"n": 200}, simulation={"n_epochs": 4},
                          hypergamy={"signal_cost": cost})
        frame = Simulation(config=cfg).run(1)
        measured.append(float(frame["mean_proposal_intent"].iloc[0]))
    assert measured[0] <= measured[1] <= measured[2]
    assert measured[2] > measured[0]


# ── 3. Above the cutoff, the market closes ───────────────────────────────────

def test_no_proposals_above_the_cutoff():
    """c above every tau * V * p: the log has no positive and mcr is 0.

    The cutoff is computed, not guessed: tau_max * V_max * p, with V_max taken
    from the utilities of the top-k candidates the log reports at c = 0. Using the
    pool mean V̄ instead would understate it — V̄ is below the maximum by
    construction, and a cost between the two still leaves the best pair proposing.
    """
    cfg = base_config()
    states = _hypergamic.sample_population(1000, cfg, 42)

    log0 = _hypergamic.generate_swipes(states, with_cost(cfg, 0.0), 0)
    pos0 = log0[log0[:, 2] == 1]
    pairs = np.stack([pos0[:, 0], pos0[:, 1]], axis=1).astype(np.int32)
    utilities = _hypergamic.utility_for_pairs(states, pairs, cfg)
    tau = states[pos0[:, 0], 11]
    cutoff = float(np.max(tau * utilities * 0.5))
    assert cutoff > 0.0

    just_above = cutoff * 1.01
    n_pos, mean_tau = positives(states, with_cost(cfg, just_above))
    assert n_pos == 0
    assert np.isnan(mean_tau)


def test_a_closed_market_clears_nothing():
    """mcr = 0 when the cost is prohibitive, through the orchestrator."""
    cfg = base_config(population={"n": 200}, simulation={"n_epochs": 4},
                      hypergamy={"signal_cost": 40.0})
    frame = Simulation(config=cfg).run(2)
    assert (frame["mcr"] == 0.0).all()
    assert (frame["n_matched"] == 0).all()
    assert (frame["proposal_cost_total"] == 0.0).all()
    # A threshold above every intent is still reported, not swallowed into NaN:
    # the market closing is a measurement.
    assert frame["separating_threshold"].notna().all()
    assert (frame["separating_threshold"] > 1.0).all()


def test_zero_accept_probability_closes_the_market_at_any_positive_cost():
    """p = 0 makes tau* infinite: nothing is worth a cost that never pays."""
    cfg = base_config()
    states = _hypergamic.sample_population(400, cfg, 5)
    closed = with_cost(cfg, 0.5)
    closed["signal"] = {"accept_prob": 0.0}
    assert positives(states, closed)[0] == 0
    # And the reported threshold is infinite, which is the analytic statement of
    # the same fact.
    assert separating_threshold(0.5, 4.0, 0.0) == float("inf")


# ── 4. The proposal count falls monotonically in c ───────────────────────────

@pytest.mark.parametrize("seed", [42, 44])
def test_proposal_count_falls_monotonically_in_cost(seed):
    cfg = base_config(population={"seed": seed})
    states = _hypergamic.sample_population(1000, cfg, seed)

    counts = [positives(states, with_cost(cfg, c))[0] for c in
              (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    for a, b in zip(counts, counts[1:]):
        assert b <= a, f"proposal count rose along the c grid: {counts}"
    assert counts[0] > 0
    assert counts[-1] < counts[0]


def test_proposal_count_and_mcr_both_fall_through_the_orchestrator():
    mcr = []
    for cost in (0.0, 0.25, 1.0, 2.0):
        cfg = base_config(population={"n": 200}, simulation={"n_epochs": 4},
                          hypergamy={"signal_cost": cost})
        frame = Simulation(config=cfg).run(1)
        mcr.append(float(frame["mcr"].iloc[0]))
    assert mcr == sorted(mcr, reverse=True), mcr


# ── 5. Intent does not enter utility_toward ──────────────────────────────────

def test_intent_does_not_enter_the_utility():
    """Perturb every intent value; utility_toward must not move by one bit.

    This is the guard for the failure mode THEORY.md §10.2 names: putting the
    cost — or the intent that multiplies it — inside the utility function would
    silently change every welfare number, because compute_welfare reads the same
    function. The proposal decision may read intent; the value of a pair may not.
    """
    cfg = base_config()
    states = _hypergamic.sample_population(300, cfg, 42)

    # A well-spread set of ordered pairs, both orientations.
    rng = np.random.default_rng(1)
    pairs = np.stack([rng.integers(0, 300, 4000), rng.integers(0, 300, 4000)], axis=1)
    pairs = pairs[(states[pairs[:, 0], 1] != states[pairs[:, 1], 1])].astype(np.int32)

    before = _hypergamic.utility_for_pairs(states, pairs, cfg)

    perturbed = states.copy()
    # Every intent to an extreme, then to its opposite extreme, so any linear or
    # multiplicative dependence shows up rather than cancelling.
    perturbed[:, 11] = 0.001
    low = _hypergamic.utility_for_pairs(perturbed, pairs, cfg)
    perturbed[:, 11] = 1.0
    high = _hypergamic.utility_for_pairs(perturbed, pairs, cfg)

    assert np.array_equal(before, low)
    assert np.array_equal(before, high)

    # The same guard for the other two new columns, which are equally not traits.
    perturbed[:, 11] = states[:, 11]
    perturbed[:, 12] = 0.987
    perturbed[:, 13] = 1.0
    assert np.array_equal(before, _hypergamic.utility_for_pairs(perturbed, pairs, cfg))


def test_intent_changes_the_proposal_decision_and_only_that():
    """Intent is live in the decision — otherwise the guard above proves nothing.

    A guard that passes because intent is ignored everywhere would pass on code
    that never implemented the mechanism. Set every intent below the cutoff and
    the proposals must vanish, with every trait column untouched.
    """
    cfg = base_config()
    states = _hypergamic.sample_population(400, cfg, 11)
    costed = with_cost(cfg, 1.0)

    log = _hypergamic.generate_swipes(states, costed, 0)
    assert (log[:, 2] == 1).any(), "the cost binds on nothing; pick a larger c"

    low_intent = states.copy()
    low_intent[:, 11] = 1e-12
    assert not (low_intent[:, :11] - states[:, :11]).any(), "the fixture moved a trait"

    closed = _hypergamic.generate_swipes(low_intent, costed, 0)
    assert not (closed[:, 2] == 1).any()
    # The rows are still there, as refusals: an omitted row would be
    # indistinguishable from a candidate the agent never saw.
    assert closed.shape[0] == log.shape[0]


# ── The new metrics, as pure functions ───────────────────────────────────────

def test_mean_proposal_intent_is_nan_without_proposals():
    assert np.isnan(mean_proposal_intent(np.array([0.5, 0.6]), np.array([], dtype=int)))


def test_mean_proposal_intent_reads_the_proposer_column():
    intent = np.array([0.1, 0.2, 0.3, 0.9])
    # Proposals 0->2 and 3->1: the mean is over the proposers, 0.1 and 0.9.
    assert mean_proposal_intent(intent, np.array([0, 3])) == pytest.approx(0.5)


def test_separating_threshold_is_the_analytic_cut():
    assert separating_threshold(1.0, 4.0, 0.5) == pytest.approx(0.5)
    assert separating_threshold(0.0, 4.0, 0.5) == 0.0
    assert separating_threshold(1.0, 0.0, 0.5) == float("inf")
    assert separating_threshold(1.0, 4.0, 0.0) == float("inf")


def test_proposal_cost_total_is_the_spent_cost():
    assert proposal_cost_total(0.25, 80) == pytest.approx(20.0)
    assert proposal_cost_total(0.25, 0) == 0.0


# ── Config plumbing ──────────────────────────────────────────────────────────

def test_signal_cost_defaults_to_zero_and_survives_the_dict():
    cfg = SimConfig.from_dict(base_config())
    assert cfg.hypergamy.signal_cost == 0.0
    assert cfg.to_dict()["hypergamy"]["signal_cost"] == 0.0


def test_base_yaml_ships_a_zero_cost():
    """The shipped config is the pooling regime: Environment A has no cost."""
    assert SimConfig.from_yaml().hypergamy.signal_cost == 0.0


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="signal_cost"):
        SimConfig.from_dict(base_config(hypergamy={"signal_cost": -0.1}))


def test_unknown_keys_in_the_population_block_are_still_rejected():
    """The strict check has to grow with the block, or a typo samples a default."""
    cfg = base_config()
    cfg["population"]["intent_beta_aa"] = 3.0
    with pytest.raises(ValueError, match="unknown key population.intent_beta_aa"):
        SimConfig.from_dict(cfg)


def test_intent_and_values_beta_shapes_are_validated():
    for key in ("intent_beta_a", "intent_beta_b", "values_beta_a", "values_beta_b"):
        with pytest.raises(ValueError, match=key):
            SimConfig.from_dict(base_config(population={key: 0.0}))
    with pytest.raises(ValueError, match="community_prob"):
        SimConfig.from_dict(base_config(population={"community_prob": 1.5}))


def test_new_sampling_knobs_are_drawn_on_their_own_stream():
    """The four trait columns and every key are bit-identical at any knob setting.

    The property the contract promises, asserted at the boundary a caller sees:
    changing the Beta shapes and the membership probability must not move a
    single pre-existing column.
    """
    cfg = base_config()
    states = _hypergamic.sample_population(500, cfg, 7)

    moved = base_config(population={
        "intent_beta_a": 5.0, "intent_beta_b": 0.7,
        "values_beta_a": 0.4, "values_beta_b": 3.0,
        "community_prob": 0.1,
    })
    other = _hypergamic.sample_population(500, moved, 7)

    assert np.array_equal(states[:, :11], other[:, :11])
    assert not np.array_equal(states[:, 11], other[:, 11])
    assert not np.array_equal(states[:, 13], other[:, 13])
    assert other[:, 13].mean() < states[:, 13].mean()


def test_intent_is_strictly_positive_and_values_stay_in_the_unit_interval():
    cfg = base_config()
    states = _hypergamic.sample_population(2000, cfg, 3)
    assert (states[:, 11] > 0.0).all()
    assert (states[:, 11] <= 1.0).all()
    assert (states[:, 12] >= 0.0).all()
    assert (states[:, 12] <= 1.0).all()
    assert set(np.unique(states[:, 13])) <= {0.0, 1.0}


def test_a_sharp_beta_puts_intent_near_its_mode():
    """The shapes are live, not merely validated: a=8, b=2 peaks near 0.9."""
    cfg = base_config(population={"intent_beta_a": 8.0, "intent_beta_b": 2.0})
    states = _hypergamic.sample_population(2000, cfg, 9)
    assert states[:, 11].mean() > 0.7
