"""Generational turnover (docs/SPRINT_PROMPTS.md Session 4-A, THEORY.md §8).

``run_multigen`` drives ``run()`` one generation at a time and calls ``_turnover``
between generations. These tests pin the replacement rules — who leaves, what a
newcomer gets, what nobody else gets — and the frame contract: one contiguous
``epoch`` index, a non-decreasing ``generation`` and a Pareto fit on the
generation-final rows only.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import _hypergamic
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

N = 200
EPOCHS = 6
GENERATIONS = 3

# Every state column a newcomer draws fresh: the four traits, the three taste
# multipliers, the key and the three institutional columns. Columns 0-1 (id,
# side) are the slot and must not move; column 6 (θ) is tested on its own.
_TRAIT_COLS = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
_KEY = 10


def _config(turnover_fraction: float = 0.2, seed: int = 42) -> dict:
    cfg = SimConfig.from_yaml().to_dict()
    cfg["population"]["n"] = N
    cfg["population"]["seed"] = seed
    cfg["simulation"]["n_epochs"] = EPOCHS
    cfg["simulation"]["n_generations"] = GENERATIONS
    cfg["simulation"]["turnover_fraction"] = turnover_fraction
    return cfg


def _after_first_generation(**kw) -> Simulation:
    """A simulation that has run generation 0 and not yet turned over."""
    sim = Simulation(config=_config(**kw), recommender="oracle")
    sim.run(EPOCHS // GENERATIONS)
    return sim


# ── 1. exactly round(turnover_fraction * N) per generation ──────────────────


@pytest.mark.parametrize("fraction", [0.2, 0.15, 0.0])
def test_turnover_replaces_exactly_the_configured_count(fraction: float) -> None:
    sim = Simulation(config=_config(turnover_fraction=fraction), recommender="oracle")
    sim.run_multigen()

    expected = int(round(fraction * N))
    # One turnover per generation boundary, none after the last generation.
    assert [t["generation"] for t in sim.turnover_log] == list(range(1, GENERATIONS))
    for t in sim.turnover_log:
        assert t["rows"].size == expected
        assert np.unique(t["rows"]).size == expected


def test_the_replaced_agents_are_the_bottom_by_cumulative_matches() -> None:
    sim = _after_first_generation()
    counts = sim._cumulative_matches.copy()
    rows = sim._turnover(1)

    kept = np.setdiff1d(np.arange(N), rows)
    # Ties at the boundary may go either way; nobody kept may have fewer matches
    # than anybody replaced.
    assert counts[rows].max() <= counts[kept].min()


# ── 2. newcomers: new key, new traits, reset count, recalibrated θ ──────────


def test_replaced_slots_get_new_keys_new_traits_and_reset_counts() -> None:
    sim = _after_first_generation()
    before = sim.states.copy()
    # The leavers are the bottom by count, so after a short generation most of
    # them hold 0 already and a missing reset would pass unseen. Lifting every
    # count by one keeps the order — the same agents leave — and makes "reset to
    # 0" a claim that can fail.
    sim._cumulative_matches += 1
    rows = sim._turnover(1)
    after = sim.states

    # The slot: id and side are kept, so every dense index stays valid.
    np.testing.assert_array_equal(after[rows, :2], before[rows, :2])
    # A key not held by anyone before turnover — fresh with everyone, not swapped.
    assert not np.isin(after[rows, _KEY], before[:, _KEY]).any()
    # Every newcomer differs from its predecessor in every drawn column except
    # where a clamp can coincide (attractiveness/sociability at 0 or 1).
    changed = after[rows][:, _TRAIT_COLS] != before[rows][:, _TRAIT_COLS]
    assert changed[:, [0, 3, 7]].all()          # status, age, key: continuous, unclamped
    assert changed.mean() > 0.95
    assert (sim._cumulative_matches[rows] == 0).all()


def test_newcomer_theta_is_recalibrated_not_inherited() -> None:
    sim = _after_first_generation()
    before = sim.states.copy()
    rows = sim._turnover(1)

    # The quantile rule, applied to the market the newcomer enters.
    expected = sim.states.copy()
    _hypergamic.calibrate_thresholds(expected, sim.config_dict,
                                     sim.config.population.admissible_fraction)
    np.testing.assert_array_equal(sim.states[rows, 6], expected[rows, 6])
    # And not the leaver's θ, which the θ step has dragged down all generation.
    assert (sim.states[rows, 6] != before[rows, 6]).all()


def test_newcomers_are_reproducible_and_not_a_repeat_of_generation_zero() -> None:
    a, b = _after_first_generation(), _after_first_generation()
    initial = Simulation(config=_config(), recommender="oracle").states
    rows_a, rows_b = a._turnover(1), b._turnover(1)

    np.testing.assert_array_equal(rows_a, rows_b)
    np.testing.assert_array_equal(a.states, b.states)
    # The generation-1 draw is a different population from the generation-0 one.
    assert (a.states[rows_a, 2] != initial[rows_a, 2]).all()


# ── 3. everyone else is bit-identical ───────────────────────────────────────


def test_survivors_are_bit_identical_across_turnover() -> None:
    sim = _after_first_generation()
    before = sim.states.copy()
    counts = sim._cumulative_matches.copy()
    rows = sim._turnover(1)

    kept = np.setdiff1d(np.arange(N), rows)
    # Every column, θ included: traits, keys and the θ they have earned.
    assert before[kept].tobytes() == sim.states[kept].tobytes()
    np.testing.assert_array_equal(sim._cumulative_matches[kept], counts[kept])


def test_survivor_chemistry_with_each_other_is_unchanged() -> None:
    """Nobody else's chemistry changes — the half tests/cpp/test_heterogeneity.cpp
    asserts for a re-keyed agent, checked here end to end through the binding."""
    sim = _after_first_generation()
    cfg = sim.config_dict
    off_cfg = {**cfg, "hypergamy": {**cfg["hypergamy"], "shock_scale": 0.0}}
    before = sim.states.copy()
    rows = sim._turnover(1)
    kept = np.setdiff1d(np.arange(N), rows)
    kept_m = kept[np.isin(kept, sim.prop_rows)]
    kept_w = kept[np.isin(kept, sim.acc_rows)]
    # Both directions of every survivor pair.
    pairs = np.array([(m, w) for m in kept_m for w in kept_w]
                     + [(w, m) for m in kept_m for w in kept_w], dtype=np.int32)

    # Utilities are standardized against population trait statistics, which the
    # newcomers move, so raw utilities of survivor pairs legitimately shift. The
    # chemistry term is what must not: it is isolated as (shock on) - (shock off)
    # inside one population, and that difference must agree before and after.
    def chemistry(states: np.ndarray) -> np.ndarray:
        return (_hypergamic.utility_for_pairs(states, pairs, cfg)
                - _hypergamic.utility_for_pairs(states, pairs, off_cfg))

    np.testing.assert_allclose(chemistry(sim.states), chemistry(before), rtol=0, atol=1e-12)


# ── 4. epoch is contiguous, generation non-decreasing ───────────────────────


def test_epoch_is_contiguous_and_generation_non_decreasing() -> None:
    df = Simulation(config=_config(), recommender="oracle").run_multigen()

    assert len(df) == EPOCHS
    assert df["epoch"].tolist() == list(range(len(df)))
    gen = df["generation"].to_numpy()
    assert (np.diff(gen) >= 0).all()
    assert gen.tolist() == [g for g in range(GENERATIONS) for _ in range(EPOCHS // GENERATIONS)]


# ── 5. pareto_alpha on generation-final rows only ───────────────────────────


@pytest.mark.parametrize("arm", ["ncf", "oracle"])
def test_pareto_alpha_is_set_exactly_on_generation_final_rows(arm: str) -> None:
    df = Simulation(config=_config(), recommender=arm).run_multigen()

    final = df["generation"].ne(df["generation"].shift(-1)).to_numpy()
    assert final.sum() == GENERATIONS
    np.testing.assert_array_equal(df["pareto_alpha"].notna().to_numpy(), final)


# ── 6. reproducibility ──────────────────────────────────────────────────────


def test_same_seed_gives_the_same_multigen_frame() -> None:
    a = Simulation(config=_config(), recommender="ncf").run_multigen()
    b = Simulation(config=_config(), recommender="ncf").run_multigen()
    pd.testing.assert_frame_equal(a, b)

    c = Simulation(config=_config(seed=43), recommender="ncf").run_multigen()
    assert not a["gini"].equals(c["gini"])


# ── contract edges ──────────────────────────────────────────────────────────


def test_run_multigen_refuses_epochs_that_do_not_divide_into_generations() -> None:
    cfg = _config()
    cfg["simulation"]["n_epochs"] = 7
    with pytest.raises(ValueError, match="multiple"):
        Simulation(config=cfg, recommender="oracle").run_multigen()


def test_one_generation_is_exactly_run() -> None:
    """With one generation run_multigen adds nothing: same frame as run()."""
    cfg = _config()
    cfg["simulation"]["n_generations"] = 1
    a = Simulation(config=cfg, recommender="ncf").run_multigen()
    b = Simulation(config=cfg, recommender="ncf").run(EPOCHS)
    pd.testing.assert_frame_equal(a, b)
