"""The second revision's switches, and the guarantee that each default is a no-op.

Five switches were added for the experiments registered in
``docs/PREREGISTRATION.md`` R2: the training dose (``ncf.passes_per_round``),
exploration slots (``ncf.explore_share``), a floating-point probe
(``ncf.init_perturbation``), an open market (``simulation.arrival_rate``), and two
recorders that change what is kept but never what is computed (diagnostics and the
shadow ranker). Every committed number in the repository was produced without
them, so the contract pinned first is that a run at the defaults is the run from
before they existed, checked against a committed 1-thread file rather than only
against itself.
"""

from __future__ import annotations

import copy
import csv
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

CONFIG = "experiments/base.yaml"


def _small(seed: int = 42, ncf: dict | None = None, simulation: dict | None = None) -> dict:
    cfg = SimConfig.from_yaml(CONFIG).to_dict()
    cfg["population"]["seed"] = int(seed)
    cfg["population"]["n"] = 120
    cfg["population"]["admissible_fraction"] = 0.1
    cfg["ncf"]["top_k"] = 8
    cfg["ncf"].update(ncf or {})
    cfg["simulation"].update(simulation or {})
    return cfg


def _frame(cfg: dict, rounds: int = 4, arm: str = "ncf", **kw):
    sim = Simulation(config=copy.deepcopy(cfg), recommender=arm, **kw)
    return sim, sim.run(rounds)


# ── defaults are no-ops ─────────────────────────────────────────────────────────

def test_defaults_leave_the_config_dict_unchanged():
    """No new key appears in to_dict at its default, so config hashes do not move."""
    d = SimConfig.from_yaml(CONFIG).to_dict()
    for key in ("passes_per_round", "explore_share", "init_perturbation", "perturbation_seed"):
        assert key not in d["ncf"]
    for key in ("arrival_rate", "departure_rate"):
        assert key not in d["simulation"]


def test_explicit_defaults_are_bit_identical_to_omitted_ones():
    _, omitted = _frame(_small())
    _, explicit = _frame(_small(
        ncf={"passes_per_round": 1.0, "explore_share": 0.0, "init_perturbation": 0.0},
        simulation={"arrival_rate": 0.0, "departure_rate": 0.0}))
    assert omitted.equals(explicit)


def test_shipped_run_reproduces_the_committed_one_thread_series():
    """Seed 42 at N = 1000, 1 thread, rounds 0-4, against rigor/exploratory_1t.

    That directory was written by the previous revision before any of these
    switches existed. Matching it to the printed digits is the strict no-op check
    against committed output rather than against this code's own default.
    """
    cfg = SimConfig.from_yaml(CONFIG).to_dict()
    cfg["population"]["seed"] = 42
    frame = Simulation(config=cfg, recommender="ncf").run(5)
    path = PROJECT_ROOT / "data" / "processed" / "rigor" / "exploratory_1t" / "metrics_seed42.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        committed = {int(r["epoch"]): r for r in csv.DictReader(fh) if r["arm"] == "ncf"}
    for _, row in frame.iterrows():
        ref = committed[int(row["epoch"])]
        assert float(ref["gini"]) == pytest.approx(row["gini"], abs=1e-12)
        assert float(ref["reach"]) == pytest.approx(row["reach"], abs=1e-12)
        assert int(ref["n_matched"]) == int(row["n_matched"])


# ── recorders read and never write ──────────────────────────────────────────────

def test_diagnostics_do_not_change_the_run_and_stay_in_range():
    _, off = _frame(_small())
    sim = Simulation(config=_small(), recommender="ncf")
    sim.record_diagnostics = True
    on = sim.run(4)
    assert off.equals(on)
    for row in sim.round_log:
        assert -1.0 <= row["spearman_mean"] <= 1.0
        for name in ("item_share", "modal_share", "identical_share", "jaccard_oracle"):
            assert 0.0 <= row[name] <= 1.0 + 1e-12
        assert row["spearman_q10"] <= row["spearman_q50"] <= row["spearman_q90"]


def test_untrained_ranker_hands_every_agent_nearly_the_same_list():
    """At round 0 the scores are mostly a shared per-candidate term (R2-E12, C6)."""
    cfg = SimConfig.from_yaml(CONFIG).to_dict()
    cfg["population"]["seed"] = 43
    sim = Simulation(config=cfg, recommender="ncf")
    sim.record_diagnostics = True
    frame = sim.run(1)
    k, n = 20, 1000
    assert frame["gini"].iloc[0] == pytest.approx(1 - 2 * k / n, abs=2e-3)
    assert sim.round_log[0]["modal_share"] > 0.95
    assert sim.round_log[0]["item_share"] > 0.9


def test_shadow_ranker_does_not_change_its_arm():
    _, plain = _frame(_small(), arm="oracle")
    _, shadowed = _frame(_small(), arm="oracle", shadow_ranker=True)
    assert plain.equals(shadowed)


def test_shadow_ranker_starts_as_the_ncf_arm_starts():
    """Same seed, same init, same round-0 log: identical round-0 exposure."""
    ncf_sim, ncf = _frame(_small(), rounds=1, arm="ncf")
    sh_sim, _ = _frame(_small(), rounds=1, arm="oracle", shadow_ranker=True)
    assert sh_sim.round_log[0]["shadow_gini"] == ncf["gini"].iloc[0]
    assert sh_sim.round_log[0]["shadow_grad_steps"] == ncf_sim.round_log[0]["grad_steps"]


def test_shadow_ranker_is_refused_on_the_ncf_arm():
    with pytest.raises(ValueError, match="shadow_ranker"):
        Simulation(config=_small(), recommender="ncf", shadow_ranker=True)


def test_recorded_lists_reproduce_the_exposure_gini():
    from src.python.metrics import gini

    sim = Simulation(config=_small(), recommender="ncf")
    sim.record_lists = True
    frame = sim.run(3)
    for t, (prop, acc) in enumerate(sim.lists_log):
        exposure = np.zeros(sim.n, dtype=np.int64)
        for slots, rows in ((prop, sim.acc_rows), (acc, sim.prop_rows)):
            flat = slots[slots >= 0]
            np.add.at(exposure, rows[flat], 1)
        assert gini(exposure) == frame["gini"].iloc[t]


# ── the dose ────────────────────────────────────────────────────────────────────

def test_grad_steps_scale_with_the_dose():
    steps = {}
    for dose in (0.5, 1.0, 2.0):
        sim, _ = _frame(_small(ncf={"passes_per_round": dose}), rounds=3)
        steps[dose] = [r["grad_steps"] for r in sim.round_log]
    per_round = {d: np.diff([0] + s) for d, s in steps.items()}
    assert np.all(per_round[2.0] == 2 * per_round[1.0])
    assert np.all(per_round[0.5] <= per_round[1.0])
    assert np.all(per_round[0.5] >= 1)


def test_a_higher_dose_changes_the_run():
    _, one = _frame(_small())
    _, two = _frame(_small(ncf={"passes_per_round": 2.0}))
    assert not one["gini"].equals(two["gini"])


# ── exploration ─────────────────────────────────────────────────────────────────

def test_exploration_slots_are_distinct_and_at_the_end():
    sim = Simulation(config=_small(ncf={"explore_share": 0.25}), recommender="ncf")
    sim.record_lists = True
    sim.run(3)
    k, m = 8, 2
    for prop, acc in sim.lists_log:
        for lists in (prop, acc):
            assert np.all(lists >= 0)
            for row in lists:
                assert len(set(row.tolist())) == k
    # The exploiting head is the ranker's own top k - m: rerun without
    # exploration and the heads agree at round 0, where nothing has diverged.
    plain = Simulation(config=_small(), recommender="ncf")
    plain.record_lists = True
    plain.run(1)
    np.testing.assert_array_equal(sim.lists_log[0][0][:, :k - m], plain.lists_log[0][0][:, :k - m])


def test_exploration_that_leaves_no_exploiting_slot_is_refused():
    with pytest.raises(ValueError, match="explore_share"):
        Simulation(config=_small(ncf={"explore_share": 0.95}), recommender="ncf")


# ── perturbation ────────────────────────────────────────────────────────────────

def test_perturbation_moves_the_initial_weights_by_about_its_size():
    a = Simulation(config=_small(), recommender="ncf")
    b = Simulation(config=_small(ncf={"init_perturbation": 1e-7, "perturbation_seed": 3}),
                   recommender="ncf")
    wa = a.trainer.model.gmf_user.weight.detach().numpy()
    wb = b.trainer.model.gmf_user.weight.detach().numpy()
    diff = np.abs(wa - wb)
    assert 0 < diff.max() < 1e-6
    assert np.median(diff) > 0


# ── arrivals ────────────────────────────────────────────────────────────────────

def test_arrivals_replace_the_configured_share_every_round():
    cfg = _small(simulation={"arrival_rate": 0.05, "departure_rate": 0.05})
    sim = Simulation(config=cfg, recommender="ncf")
    sim.run(4)
    assert [e["epoch"] for e in sim.arrivals_log] == [1, 2, 3]
    for entry in sim.arrivals_log:
        assert entry["rows"].size == 6
        assert np.all(sim.arrival_round[entry["rows"]] >= entry["epoch"])
    assert sim.arrival_round.max() == 3


def test_every_arm_sees_the_same_arrivals():
    cfg = _small(simulation={"arrival_rate": 0.05, "departure_rate": 0.05})
    seen = {}
    for arm in ("ncf", "oracle"):
        sim = Simulation(config=copy.deepcopy(cfg), recommender=arm)
        sim.record_states = True
        sim.run(4)
        traits = [s[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]] for s in sim.states_seen]
        seen[arm] = (traits, [e["rows"].tolist() for e in sim.arrivals_log])
    assert seen["ncf"][1] == seen["oracle"][1]
    for a, b in zip(seen["ncf"][0], seen["oracle"][0]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("simulation, match", [
    ({"arrival_rate": 0.05, "departure_rate": 0.0}, "must equal"),
    ({"arrival_rate": 0.001, "departure_rate": 0.001}, "0 agents"),
])
def test_arrival_rates_that_cannot_be_honoured_are_refused(simulation, match):
    with pytest.raises(ValueError, match=match):
        SimConfig.coerce(_small(simulation=simulation))
