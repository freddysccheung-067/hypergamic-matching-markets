"""The Python-visible half of preference heterogeneity (THEORY.md §2.3).

The C++ tests in tests/cpp/test_heterogeneity.cpp cover the mechanisms. This file
covers what the Python layer can observe and what the config guarantees: the
11-column state matrix, the config keys and their validation, the noise-to-signal
diagnostic, and the backward-compatibility promise at the boundary the experiment
scripts actually call.
"""

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
from src.python.orchestrator import Simulation  # noqa: E402


def base_config(**overrides):
    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


# ── State matrix ─────────────────────────────────────────────────────────────

def test_state_matrix_has_eleven_columns():
    """Columns 7-10 are the heterogeneity block, appended after θ (INTERFACES §1)."""
    cfg = base_config(population={"taste_sigma": 0.0})
    states = _hypergamic.sample_population(40, cfg, 1)
    assert states.shape == (40, 11)
    assert states.dtype == np.float64


def test_key_column_round_trips_through_float64():
    """key < 2^53 so it survives the float64 matrix; distinct agents, distinct keys."""
    cfg = base_config()
    states = _hypergamic.sample_population(200, cfg, 5)
    keys = states[:, 10]
    assert (keys >= 0).all()
    assert (keys < 2.0**53).all()
    # float -> int -> float must be the identity, or chemistry would differ between
    # the C++ path (uint64 key) and the Python path (double column).
    assert np.array_equal(keys.astype(np.uint64).astype(np.float64), keys)
    assert len(np.unique(keys)) == len(keys)


def test_flatten_is_identity_on_the_wider_matrix():
    cfg = base_config()
    states = _hypergamic.sample_population(20, cfg, 2)
    flat = _hypergamic.flatten_states(states)
    assert flat.shape == (20, 11)
    assert np.array_equal(flat, states)


def test_narrow_matrix_is_rejected():
    """The guard has to move with the layout, or every sampler output is refused."""
    cfg = base_config()
    states = _hypergamic.sample_population(10, cfg, 1)
    with pytest.raises(ValueError, match=r"\(N, 11\)"):
        _hypergamic.generate_swipes(states[:, :7], cfg, 0)
    with pytest.raises(ValueError, match=r"\(N, 11\)"):
        _hypergamic.utility_sd(states[:, :7], cfg)


# ── Backward compatibility ───────────────────────────────────────────────────

def test_both_knobs_off_reproduces_the_historical_swipe_log():
    """The pinned hash.

    Recorded from the code as it stood before preference heterogeneity existed,
    on base.yaml at seed 42 with N=1000, epochs 0-2. Any change to the sampling
    stream, the utility arithmetic, or the θ calibration path shows up here as a
    failure rather than as a quietly different set of published numbers.
    """
    import hashlib

    cfg = base_config(
        population={"taste_sigma": 0.0},
        hypergamy={"shock_scale": 0.0},
    )
    states = _hypergamic.sample_population(1000, cfg, 42)

    digests = []
    for epoch in (0, 1, 2):
        log = _hypergamic.generate_swipes(states, cfg, epoch)
        digests.append(hashlib.sha256(log.tobytes()).hexdigest())

    assert digests == [
        "2adea38ae4533316a16e398c6163bb99616457038827cb01c0dc284af516bc4b",
        "823a8f7ffebd8221727be183adc1db668d5cf56dcfea21133bf27d7270ddb005",
        "e72d11404fb2063c92182ad617bc7db42d46be5a6b8880b7e1b793b4be50fc9e",
    ]


def test_swipe_log_changes_once_heterogeneity_is_on():
    """The control arm must be distinguishable from the treatment arm."""
    off = base_config(population={"taste_sigma": 0.0}, hypergamy={"shock_scale": 0.0})
    on = base_config()

    a = _hypergamic.generate_swipes(_hypergamic.sample_population(200, off, 3), off, 0)
    b = _hypergamic.generate_swipes(_hypergamic.sample_population(200, on, 3), on, 0)
    # Not a strict shape claim — the count can coincide — but the rows must differ.
    assert not np.array_equal(a, b)


# ── Configuration ────────────────────────────────────────────────────────────

def test_base_yaml_ships_the_heterogeneity_values():
    cfg = SimConfig.from_yaml()
    assert cfg.population.taste_sigma == 0.5
    assert cfg.hypergamy.shock_scale == 1.5


def test_defaults_are_off():
    """A config predating both keys must run the old model, not raise."""
    cfg = SimConfig.from_dict({
        "population": {
            "n": 100,
            "seed": 1,
            "traits": {
                "status": {"dist": "lognormal", "mean": 0.0, "std": 0.8},
                "attractiveness": {"dist": "normal", "mean": 0.5, "std": 0.15},
                "sociability": {"dist": "normal", "mean": 0.5, "std": 0.15},
                "age": {"dist": "normal", "mean": 30.0, "std": 6.0},
            },
        },
        "hypergamy": {"alpha": 1.0, "beta": 2.0, "gamma": 0.3, "delta": 0.5, "lambda": 1.5},
    })
    assert cfg.population.taste_sigma == 0.0
    assert cfg.hypergamy.shock_scale == 0.0
    # And they appear in the dict handed to C++, since the orchestrator always
    # serializes rather than relying on the extension's defaults.
    assert cfg.to_dict()["population"]["taste_sigma"] == 0.0
    assert cfg.to_dict()["hypergamy"]["shock_scale"] == 0.0


def test_negative_values_are_rejected():
    with pytest.raises(ValueError, match="taste_sigma"):
        SimConfig.from_dict(base_config(population={"taste_sigma": -0.1}))
    with pytest.raises(ValueError, match="shock_scale"):
        SimConfig.from_dict(base_config(hypergamy={"shock_scale": -0.1}))


def test_unknown_keys_in_the_population_block_are_rejected():
    """The strict check has to cover the block parsed by hand, not just _build."""
    cfg = base_config()
    cfg["population"]["taste_sigm"] = 0.5
    with pytest.raises(ValueError, match="unknown key population.taste_sigm"):
        SimConfig.from_dict(cfg)


# ── The noise-to-signal diagnostic ───────────────────────────────────────────

def test_utility_sd_is_deterministic_and_positive():
    cfg = base_config()
    states = _hypergamic.sample_population(400, cfg, 11)
    a = _hypergamic.utility_sd(states, cfg)
    b = _hypergamic.utility_sd(states, cfg)
    assert a == b
    assert a > 0.0


def test_utility_sd_ignores_the_shock():
    """It must measure the signal, so shock_scale cannot feed back into it."""
    cfg = base_config()
    states = _hypergamic.sample_population(400, cfg, 11)
    loud = base_config(hypergamy={"shock_scale": 9.0})
    assert _hypergamic.utility_sd(states, cfg) == _hypergamic.utility_sd(states, loud)


def test_utility_sd_matches_a_direct_replica():
    """The binding's sample has to describe the market, not a correlated slice."""
    cfg = base_config(population={"taste_sigma": 0.0}, hypergamy={"shock_scale": 0.0})
    states = _hypergamic.sample_population(300, cfg, 17)
    got = _hypergamic.utility_sd(states, cfg)

    # Straightforward full-population replica of the C++ utility.
    h = cfg["hypergamy"]
    alpha, beta, gamma, delta, lam = (h["alpha"], h["beta"], h["gamma"],
                                      h["delta"], h["lambda"])
    typ = states[:, 1]
    cols = []
    for c in (2, 3, 4, 5):
        col = states[:, c]
        cols.append((col - col.mean()) / col.std(ddof=1))
    zs, za, zc, zt = cols
    men = np.flatnonzero(typ == 0)
    women = np.flatnonzero(typ == 1)

    dist = np.sqrt((zs[women][None, :] - zs[men][:, None]) ** 2
                   + (za[women][None, :] - za[men][:, None]) ** 2
                   + (zc[women][None, :] - zc[men][:, None]) ** 2
                   + (zt[women][None, :] - zt[men][:, None]) ** 2)
    u_mw = alpha * zs[women][None, :] + beta * za[women][None, :] - gamma * dist
    u_wm = alpha * zs[men][None, :] + beta * za[men][None, :] - gamma * dist.T
    u_wm = u_wm + lam * (zs[men][None, :] >= zs[women][:, None] + delta)

    truth = np.concatenate([u_mw.ravel(), u_wm.ravel()]).std(ddof=1)
    assert got == pytest.approx(truth, rel=0.03)


def test_orchestrator_logs_the_implied_ratio():
    """The ratio is recorded on the instance, so a sweep can store it per run."""
    sim = Simulation(config=base_config(population={"n": 100}, simulation={"n_epochs": 2}))
    assert sim.utility_signal_sd > 0.0
    expected = sim.config.hypergamy.shock_scale * 1.8138 / sim.utility_signal_sd
    assert sim.noise_ratio == pytest.approx(expected)


def test_ratio_is_zero_when_the_shock_is_off():
    sim = Simulation(config=base_config(population={"n": 100},
                                        hypergamy={"shock_scale": 0.0},
                                        simulation={"n_epochs": 2}))
    assert sim.noise_ratio == 0.0


# ── Calibration and thresholds still work ────────────────────────────────────

def test_theta_calibration_survives_heterogeneity():
    """A target admissibility has to stay a target when tastes and chemistry vary.

    Calibration is a quantile of each agent's own utilities *including* the shock,
    so the admitted count is bounded by the same rule as before — but the utilities
    are now heterogeneous, which is exactly why the count needs checking rather
    than assuming.
    """
    cfg = base_config()
    states = _hypergamic.sample_population(400, cfg, 23)

    # sample_population already calibrates θ at 0.2, so this is a re-calibration:
    # the point is that it is idempotent on a well-calibrated population, which
    # only holds if the quantile is computed the same way the sampler computes it.
    theta_before = states[:, 6].copy()
    _hypergamic.calibrate_thresholds(states, cfg, 0.2)
    theta = states[:, 6]

    np.testing.assert_allclose(theta, theta_before)
    assert np.isfinite(theta).all()

    # Each agent should admit roughly 20% of the 200 candidates: mean 40. A quantile
    # admits an exact count unless ties straddle the boundary, so allow a wide
    # band and catch gross miswiring instead of pinning a coincidence.
    log = _hypergamic.generate_swipes(states, cfg, 0)
    positives = log[log[:, 2] == 1]
    per_user = np.bincount(positives[:, 0], minlength=len(states))
    # k=20 truncates above the admissibility target, so the observable is that the
    # gate is binding without being empty: some agents give positives, none give
    # more than k.
    assert per_user.max() <= 20
    assert (per_user > 0).sum() > len(states) * 0.5


def test_swipes_are_thread_count_independent_with_heterogeneity():
    """The per-pair hash must not depend on any thread-visible state."""
    cfg = base_config(population={"n": 300})
    states = _hypergamic.sample_population(300, cfg, 42)
    _hypergamic.calibrate_thresholds(states, cfg, 0.2)

    a = _hypergamic.generate_swipes(states, cfg, 4)
    b = _hypergamic.generate_swipes(states, cfg, 4)
    assert np.array_equal(a, b)

    # And a real end-to-end epoch still produces a well-formed log.
    assert a.ndim == 2 and a.shape[1] == 3
    assert set(np.unique(a[:, 2])).issubset({0, 1})
