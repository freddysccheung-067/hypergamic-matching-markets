"""Contract tests for the orchestrator and the C++ binding surface (§1, §2, §6)."""

import pathlib
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build" / "Release"))

# The orchestrator imports the extension at module scope, so collection fails
# without it. CI's test-python job does not build one (test-cpp does), and a
# developer may simply not have run cmake; skip with the command that fixes it
# rather than reporting a collection error as a test failure.
try:
    import _hypergamic  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import COLUMNS, Simulation  # noqa: E402


def base_config(**overrides):
    """The shipped config as a dict, with nested overrides applied by section."""
    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


# ── §6 smoke test ────────────────────────────────────────────────────────────────

def test_smoke_n200_ten_epochs():
    """N=200, 10 epochs: 10 rows, exact §6 columns, no NaN where required."""
    sim = Simulation(config=base_config(population={"n": 200}, simulation={"n_epochs": 10}))
    df = sim.run(10)

    assert df.shape == (10, len(COLUMNS))
    assert list(df.columns) == COLUMNS
    assert not df[["gini", "mcr", "rsi"]].isna().any().any()
    assert df["gini"].between(0.0, 1.0).all()
    assert df["epoch"].tolist() == list(range(10))


def test_metrics_schema_dtypes():
    sim = Simulation(config=base_config(population={"n": 100}, simulation={"n_epochs": 4}))
    df = sim.run(2)
    for col in ("epoch", "generation", "seed", "n_matched", "n_blocking"):
        assert df[col].dtype == np.int64, col
    for col in ("delta", "gini", "mcr", "rsi", "bpr", "welfare_alg"):
        assert df[col].dtype == np.float64, col


def test_pareto_alpha_only_on_generation_final_rows():
    sim = Simulation(
        config=base_config(
            population={"n": 100}, simulation={"n_epochs": 10, "n_generations": 5}
        )
    )
    df = sim.run(6)
    # 10 epochs over 5 generations: every second epoch closes a generation.
    filled = df["pareto_alpha"].notna().tolist()
    assert filled == [False, True, False, True, False, True]


def test_empty_run_returns_empty_frame_with_schema():
    sim = Simulation(config=base_config(population={"n": 100}))
    df = sim.run(0)
    assert df.shape == (0, len(COLUMNS))
    assert list(df.columns) == COLUMNS


# ── §2.3 index space ────────────────────────────────────────────────────────────

def test_pref_lists_are_dense_per_side():
    import _hypergamic as h

    cfg = base_config(population={"n": 40})
    states = h.sample_population(40, cfg, 42)
    prop, acc = h.build_full_info_pref_lists(states, cfg)

    assert len(prop) == 20 and len(acc) == 20
    for lst in prop:
        assert len(lst) == 20
        assert max(lst) < 20 and min(lst) >= 0
    for lst in acc:
        assert len(lst) == 20
        assert max(lst) < 20 and min(lst) >= 0


def test_full_info_matching_is_stable():
    """A complete-list Gale-Shapley run must have no blocking pair (THEORY.md §6.1)."""
    import _hypergamic as h

    cfg = base_config(population={"n": 60})
    states = h.sample_population(60, cfg, 7)
    prop, acc = h.build_full_info_pref_lists(states, cfg)

    da = h.deferred_acceptance(prop, acc, False)
    assert da["n_blocking"] == 0
    assert h.count_blocking_pairs(prop, acc, da["matches"]) == 0
    # Balanced population, complete lists: everyone matches.
    assert len(da["matches"]) == 30
    assert len(da["unmatched_proposers"]) == 0


def test_truncated_run_measures_instability_against_true_prefs():
    """A truncated matching is stable against its own lists but not the true ones."""
    import _hypergamic as h

    cfg = base_config(population={"n": 200})
    states = h.sample_population(200, cfg, 42)
    prop, acc = h.build_full_info_pref_lists(states, cfg)
    truncated = h.deferred_acceptance([lst[:5] for lst in prop],
                                      [lst[:5] for lst in acc], True)

    assert truncated["n_blocking"] == 0, "stable w.r.t. the lists it was given"
    assert h.count_blocking_pairs(prop, acc, truncated["matches"]) > 0, (
        "unstable w.r.t. the true ordering — this is what BPR reports"
    )


def test_deferred_acceptance_return_shapes():
    import _hypergamic as h

    cfg = base_config(population={"n": 40})
    states = h.sample_population(40, cfg, 1)
    prop, acc = h.build_full_info_pref_lists(states, cfg)
    da = h.deferred_acceptance(prop, acc, False)

    assert set(da) == {
        "matches", "unmatched_proposers", "unmatched_acceptors", "n_blocking", "proposals"
    }
    assert da["matches"].shape == (20, 2) and da["matches"].dtype == np.int32
    assert da["proposals"].shape == (40,) and da["proposals"].dtype == np.int32
    assert isinstance(da["n_blocking"], int)
    assert da["proposals"].sum() > 0


def test_proposals_count_rejections():
    """Every acceptor's proposal count equals the number of attempts against her."""
    import _hypergamic as h

    cfg = base_config(population={"n": 100})
    states = h.sample_population(100, cfg, 3)
    prop, acc = h.build_full_info_pref_lists(states, cfg)
    da = h.deferred_acceptance(prop, acc, False)

    # Proposer-side attempts made == acceptor-side attempts received.
    assert da["proposals"][:50].sum() == da["proposals"][50:].sum()
    # Complete lists: a proposer proposes at least once, acceptors too.
    assert (da["proposals"][:50] >= 1).all()


# ── §2.2 welfare consistency ────────────────────────────────────────────────────

def test_welfare_matches_utility_arithmetic():
    """compute_welfare must equal U_prop(acc)+U_acc(prop) recomputed in numpy."""
    import _hypergamic as h

    cfg = base_config(population={"n": 40})
    states = h.sample_population(40, cfg, 11)
    prop, acc = h.build_full_info_pref_lists(states, cfg)
    da = h.deferred_acceptance(prop, acc, False)
    w = h.compute_welfare(states, da["matches"], cfg)

    # The four pre-§10.4 keys keep their names and meaning; Session 5-E appended
    # six more (INTERFACES.md §2.2), which test_comparison.py checks term by term.
    assert {"welfare", "welfare_proposer", "welfare_acceptor", "n_pairs"} <= set(w)
    assert set(w) - {"welfare", "welfare_proposer", "welfare_acceptor", "n_pairs"} == {
        "value_created", "value_proposer", "value_acceptor",
        "signal_cost", "defection_loss", "welfare_total",
    }
    assert w["n_pairs"] == len(da["matches"])
    assert w["welfare"] == pytest.approx(w["welfare_proposer"] + w["welfare_acceptor"])


def test_update_thresholds_mutates_in_place_and_skips_idle_agents():
    import _hypergamic as h

    cfg = base_config(population={"n": 40})
    states = h.sample_population(40, cfg, 5)
    before = states[:, 6].copy()

    proposals = np.ones(40, dtype=np.int32)
    proposals[:4] = 0
    matches = np.zeros(40, dtype=np.int32)
    matches[10] = 1

    assert h.update_thresholds(states, proposals, matches, 0.5) is None
    assert not np.allclose(before, states[:, 6]), "θ should have moved"
    assert np.array_equal(before[:4], states[:, 6][:4]), "no proposals -> unchanged"
    # η=0.5, yield 0 vs 1: θ + 0.5·(0 − θ) and θ + 0.5·(1 − θ).
    np.testing.assert_allclose(states[10, 6], before[10] + 0.5 * (1.0 - before[10]))
    np.testing.assert_allclose(states[20, 6], before[20] + 0.5 * (0.0 - before[20]))


def test_update_thresholds_rejects_float32_states():
    import _hypergamic as h

    cfg = base_config(population={"n": 40})
    states32 = h.sample_population(40, cfg, 5).astype(np.float32)
    with pytest.raises(Exception, match="float64"):
        h.update_thresholds(
            states32,
            np.ones(40, dtype=np.int32),
            np.zeros(40, dtype=np.int32),
            0.1,
        )


# ── config validation ───────────────────────────────────────────────────────────

def test_config_loads_shipped_yaml():
    cfg = SimConfig.from_yaml()
    assert cfg.population.n == 1000
    assert cfg.hypergamy.lam == 1.5, "lambda must map to lam"
    assert cfg.to_dict()["hypergamy"]["lambda"] == 1.5


def test_config_rejects_odd_n():
    raw = base_config(population={"n": 101})
    with pytest.raises(ValueError, match="even"):
        SimConfig.from_dict(raw)


def test_config_rejects_top_k_too_large():
    raw = base_config(population={"n": 100}, ncf={"top_k": 50})
    with pytest.raises(ValueError, match="top_k"):
        SimConfig.from_dict(raw)


def test_config_rejects_negative_delta():
    raw = base_config(hypergamy={"delta": -0.5})
    with pytest.raises(ValueError, match="delta"):
        SimConfig.from_dict(raw)


def test_config_rejects_unknown_key():
    raw = base_config(ncf={"topk": 20})
    with pytest.raises(ValueError, match="unknown key"):
        SimConfig.from_dict(raw)


# ── the θ gate's operating point (THEORY.md §3) ─────────────────────────────────

def test_shipped_config_is_in_the_live_gate_regime():
    """phi < 2k/N at the shipped N and k, so the θ gate truncates at epoch 0."""
    import warnings

    from src.python.config import InertGateWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", InertGateWarning)
        cfg = SimConfig.from_yaml()
    assert cfg.population.admissible_fraction < 2 * cfg.ncf.top_k / cfg.population.n


@pytest.mark.parametrize("phi", [0.04, 0.05, 0.2])
def test_inert_gate_warns_and_does_not_raise(phi):
    """At or above 2k/N the market is static. Legal — the inert ablation runs there
    on purpose — so it warns, and the config still builds."""
    from src.python.config import InertGateWarning

    raw = base_config(population={"admissible_fraction": phi})
    with pytest.warns(InertGateWarning, match="2 \* ncf.top_k / population.n"):
        cfg = SimConfig.from_dict(raw)
    assert cfg.population.admissible_fraction == phi


def test_the_inert_warning_follows_k_and_n_not_a_fixed_phi():
    """phi and k interact: 0.02 is live at k=20, N=1000 and inert at k=5."""
    import warnings

    from src.python.config import InertGateWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", InertGateWarning)
        SimConfig.from_dict(base_config(population={"admissible_fraction": 0.02}))
    with pytest.warns(InertGateWarning):
        SimConfig.from_dict(base_config(population={"admissible_fraction": 0.02},
                                        ncf={"top_k": 5}))


def test_explicit_init_threshold_does_not_warn():
    """An absolute θ bypasses calibration, so phi says nothing about the gate."""
    import warnings

    from src.python.config import InertGateWarning

    raw = base_config(population={"admissible_fraction": 0.2, "init_threshold": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertGateWarning)
        SimConfig.from_dict(raw)
