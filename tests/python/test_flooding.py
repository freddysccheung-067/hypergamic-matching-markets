"""The flooding term of the θ step (THESIS.md H2) and the per-epoch market record.

The shipped θ step is a yield rule: θ += η(matches/proposals - θ). Received volume
enters it only as the acceptor's yield denominator, so more proposals at fixed
matches push θ *down*, and received intent never enters at all. H2 needs the
opposite sign. hypergamy.flooding_weight adds it behind a flag; these tests pin
that the flag at 0 changes nothing, that it does what it says above 0, and that the
two accounts are distinguishable from data.
"""

import hashlib
import math
import pathlib
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import _hypergamic  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import (  # noqa: E402
    Simulation,
    flooding_delta,
    received_proposals,
    update_thresholds,
)


def small_config(**hypergamy):
    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["population"]["n"] = 200
    cfg["population"]["admissible_fraction"] = 0.05
    cfg["ncf"]["top_k"] = 10
    cfg["hypergamy"].update(hypergamy)
    return cfg


def trajectory_digest(sim, frame) -> str:
    h = hashlib.sha256()
    for s in sim.states_seen:
        h.update(s.tobytes())
    h.update(sim.states.tobytes())
    h.update(frame["n_matched"].to_numpy().tobytes())
    return h.hexdigest()


# ── w = 0 is the shipped model ──────────────────────────────────────────────────

# Eight epochs of the oracle arm at N=200, captured at commit e9cff10 before the
# flag existed. The oracle arm is used because it involves no torch: the digest is
# a function of the C++ core and this orchestrator only. The single-epoch swipe
# digests in test_heterogeneity.py and test_signal.py cannot catch a change to the
# θ step, which runs after generate_swipes; a θ trajectory can.
#
# The digest is bitwise, and the population sampler calls exp/log/pow, which MSVC's
# CRT and glibc round differently in the last bit. The algorithm is the same; the
# bytes are not. Each platform therefore has its own pin, captured from the same
# code at the same commit: Windows/MSVC locally, Linux/GCC on the CI runner (stable
# across the Intel and AMD runner CPUs it has landed on). Any other platform has no
# pin and the digest tests skip rather than guess.
_PINS = {
    "win32": {
        0.0: "8e8eb7d6bb6173d40974f6748b8dd0ef00c596e68819d6d4200d760c91584b0c",
        1.0: "15d3958eff72036854aec994e888beda70dec340db8a6c99514d5738f422c623",
    },
    "linux": {
        0.0: "62604ee9c4f627828d84c4414b7988fa0ca497f0de6402179000ac1fa1c1b3c3",
        1.0: "c416889632c4b5ea27bf08d983a78755d28ce41a60d079e2e6462fa4c189e680",
    },
}


def pinned_trajectory(cost):
    pins = _PINS.get(sys.platform)
    if pins is None:
        pytest.skip(f"no pinned trajectory digest for platform {sys.platform!r}")
    return pins[cost]


@pytest.mark.parametrize("cost", [0.0, 1.0])
def test_zero_weight_reproduces_the_pre_flag_trajectory(cost):
    sim = Simulation(small_config(signal_cost=cost, flooding_weight=0.0), recommender="oracle")
    sim.record_states = True
    frame = sim.run(8)
    assert trajectory_digest(sim, frame) == pinned_trajectory(cost)


@pytest.mark.parametrize("cost", [0.0, 1.0])
def test_record_market_changes_nothing(cost):
    cfg = small_config(signal_cost=cost)
    runs = []
    for record in (False, True):
        sim = Simulation(cfg, recommender="oracle")
        sim.record_states = True
        sim.record_market = record
        frame = sim.run(8)
        runs.append(trajectory_digest(sim, frame))
    assert runs[0] == runs[1]
    assert runs[0] == pinned_trajectory(cost)


def test_zero_weight_is_absent_from_the_config_dict():
    """The dict every C++ call receives must be the one it received before the knob."""
    assert "flooding_weight" not in SimConfig.coerce(small_config()).to_dict()["hypergamy"]
    out = SimConfig.coerce(small_config(flooding_weight=0.2)).to_dict()["hypergamy"]
    assert out["flooding_weight"] == 0.2


# ── Validation ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf])
def test_flooding_weight_rejects_negative_and_non_finite(bad):
    with pytest.raises(ValueError, match="flooding_weight"):
        SimConfig.coerce(small_config(flooding_weight=bad))


def test_flooding_weight_round_trips():
    cfg = SimConfig.coerce(small_config(flooding_weight=0.05))
    again = SimConfig.from_dict(cfg.to_dict())
    assert again.hypergamy.flooding_weight == 0.05
    assert again.to_dict() == cfg.to_dict()


# ── The market record ───────────────────────────────────────────────────────────

def test_market_log_holds_what_update_thresholds_read():
    """Replaying the yield step from the record reproduces θ_{t+1} exactly."""
    sim = Simulation(small_config(), recommender="oracle")
    sim.record_states = True
    sim.record_market = True
    sim.run(5)
    assert len(sim.market_log) == 5
    eta = float(sim.config.simulation.bayesian_eta)
    for seen, rec in zip(sim.states_seen, sim.market_log):
        replay = seen.copy()
        update_thresholds(replay, rec["da_proposals"], rec["matches"], eta)
        assert np.array_equal(replay[:, 6], rec["theta_after"])
    assert np.array_equal(sim.market_log[-1]["theta_after"], sim.states[:, 6])


# ── received_proposals ──────────────────────────────────────────────────────────

def test_received_counts_by_column_one_and_reads_proposer_intent():
    states = np.zeros((4, 14))
    states[:, 11] = [0.2, 0.8, 0.5, 0.5]
    # Rows 0 and 1 both swipe on 2; row 0 also swipes on 3. Column 0 is the
    # swiper, so 0 and 1 receive nothing.
    pairs = np.array([[0, 2], [1, 2], [0, 3]])
    volume, intent = received_proposals(states, pairs)
    assert volume.tolist() == [0.0, 0.0, 2.0, 1.0]
    assert np.isnan(intent[0]) and np.isnan(intent[1])
    assert intent[2] == pytest.approx(0.5)
    assert intent[3] == pytest.approx(0.2)


# ── flooding_delta ──────────────────────────────────────────────────────────────

def test_flooding_raises_above_average_volume_and_spares_full_intent():
    rows = np.arange(4)
    volume = np.array([1.0, 2.0, 5.0, 5.0])         # V̄ = 3.25
    intent = np.array([0.5, 0.5, 0.5, 1.0])
    d = flooding_delta(volume, intent, (rows,), 0.2, 0.5)
    assert d[0] < 0.0 and d[1] < 0.0                 # below V̄: bar comes down
    assert d[2] > 0.0                                # above V̄, pooled: bar goes up
    assert d[3] == 0.0                               # above V̄, τ̄ = 1: nothing to defend
    assert d[2] == pytest.approx(0.2 * (5.0 / 3.25 - 1.0) * 0.5)


def test_flooding_is_scale_free_in_volume():
    rows = np.arange(3)
    volume = np.array([1.0, 3.0, 8.0])
    intent = np.full(3, 0.4)
    a = flooding_delta(volume, intent, (rows,), 0.1, 0.5)
    b = flooding_delta(10.0 * volume, intent, (rows,), 0.1, 0.5)
    assert np.allclose(a, b)


def test_flooding_normalises_within_side():
    volume = np.array([10.0, 30.0, 1.0, 3.0])
    intent = np.full(4, 0.0)
    d = flooding_delta(volume, intent, (np.array([0, 1]), np.array([2, 3])), 1.0, 0.5)
    assert np.allclose(d, [-0.5, 0.5, -0.5, 0.5])


def test_zero_receiver_gets_a_finite_term():
    volume = np.array([0.0, 4.0])
    _, intent = received_proposals(np.zeros((2, 14)), np.array([[0, 1]] * 4))
    assert np.isnan(intent[0])
    d = flooding_delta(volume, intent, (np.arange(2),), 0.2, 0.5)
    assert np.all(np.isfinite(d))
    assert d[0] == pytest.approx(0.2 * (0.0 - 1.0) * 0.5)


def test_side_with_no_proposals_gets_no_term():
    d = flooding_delta(np.zeros(3), np.full(3, np.nan), (np.arange(3),), 0.2, 0.5)
    assert np.array_equal(d, np.zeros(3))


def test_positive_weight_raises_theta_for_above_average_receivers_at_fixed_matches():
    """Same matches, same DA counts: the only difference between the arms is w."""
    states = np.zeros((4, 14))
    states[:, 1] = 1.0
    states[:, 6] = 0.3
    states[:, 11] = 0.5
    da = np.array([3, 3, 3, 3], dtype=np.int32)
    matched = np.array([1, 1, 1, 1], dtype=np.int32)
    volume = np.array([2.0, 2.0, 6.0, 6.0])
    intent = np.full(4, 0.5)

    yield_only = states.copy()
    update_thresholds(yield_only, da, matched, 0.1)
    flooded = yield_only.copy()
    flooded[:, 6] += flooding_delta(volume, intent, (np.arange(4),), 0.2, 0.5)
    lift = flooded[:, 6] - yield_only[:, 6]
    assert np.all(lift[2:] > 0.0)
    assert np.all(lift[:2] < 0.0)


# ── The two accounts disagree in sign ───────────────────────────────────────────

def _fixture_market():
    """Acceptors on a balanced volume x intent grid, one match each, one shared θ.

    A crossed grid rather than random draws: the yield step is 1/V, which a linear
    fit in V cannot absorb, and with random intents the misfit picks up a chance
    correlation with τ̄. On the grid τ̄ is orthogonal to every function of V, so the
    yield rule's intent coefficient is zero by construction rather than by luck.
    """
    v_levels = np.arange(1.0, 30.0)
    t_levels = np.linspace(0.1, 0.9, 9)
    volume, intent = (a.ravel() for a in np.meshgrid(v_levels, t_levels))
    theta = np.full(volume.size, 0.3)
    matched = np.ones(volume.size, dtype=np.int32)
    return volume, intent, theta, matched


def _coefficients(dtheta, volume, intent):
    vrel = volume / volume.mean()
    X = np.column_stack([np.ones_like(vrel), vrel, intent])
    beta, *_ = np.linalg.lstsq(X, dtheta, rcond=None)
    return beta[1], beta[2]


def test_the_two_accounts_produce_opposite_signs():
    volume, intent, theta, matched = _fixture_market()
    states = np.zeros((theta.size, 14))
    states[:, 1] = 1.0
    states[:, 6] = theta
    states[:, 11] = intent
    da = volume.astype(np.int32)

    shipped = states.copy()
    update_thresholds(shipped, da, matched, 0.1)
    b_vol_y, b_tau_y = _coefficients(shipped[:, 6] - theta, volume, intent)

    flooded = shipped.copy()
    flooded[:, 6] += flooding_delta(volume, intent, (np.arange(theta.size),), 0.2, 0.5)
    b_vol_f, _ = _coefficients(flooded[:, 6] - theta, volume, intent)

    # Intent enters the flooding term only through (V/V̄ - 1) * (1 - τ̄), which is
    # orthogonal to τ̄ over the whole crossed grid, so the all-rows τ̄ coefficient is
    # zero up to roundoff and its sign is noise. Among above-average-volume rows the
    # factor (V/V̄ - 1) is positive, and there intent genuinely lowers the bar.
    busy = volume > volume.mean()
    _, b_tau_busy = _coefficients((flooded[:, 6] - theta)[busy], volume[busy], intent[busy])

    assert b_vol_y < 0.0                     # yield: more proposals, lower yield
    assert abs(b_tau_y) < 1e-12              # yield: intent is not read
    assert b_vol_f > 0.0                     # flooding: volume raises the bar
    assert b_tau_busy < -1e-3                # flooding: intent lowers it, where volume is high


# ── w > 0 end to end ────────────────────────────────────────────────────────────

def test_positive_weight_moves_theta_and_records_the_term():
    sim = Simulation(small_config(flooding_weight=0.2), recommender="oracle")
    sim.record_states = True
    sim.record_market = True
    sim.run(4)
    for seen, rec in zip(sim.states_seen, sim.market_log):
        assert rec["flood"] is not None and np.all(np.isfinite(rec["flood"]))
        replay = seen.copy()
        update_thresholds(replay, rec["da_proposals"], rec["matches"],
                          float(sim.config.simulation.bayesian_eta))
        assert np.allclose(replay[:, 6] + rec["flood"], rec["theta_after"], rtol=0, atol=1e-12)
    assert np.any(sim.market_log[0]["flood"] != 0.0)
    assert np.all(np.isfinite(sim.states[:, 6]))


def test_freeze_theta_freezes_the_flooding_term_too():
    sim = Simulation(small_config(flooding_weight=0.2), recommender="oracle", freeze_theta=True)
    theta0 = sim.states[:, 6].copy()
    sim.run(3)
    assert np.array_equal(sim.states[:, 6], theta0)
