"""End-to-end smoke test: config in, §6 frame out (INTERFACES.md §6).

Every other test module pins one component. This one pins the assembled pipeline
— ``Sample_population`` → swipe log → preference lists → deferred acceptance →
metrics → generational turnover — at a size small enough to run on every commit,
so a change that is locally correct but breaks the composition fails here.

The assertions are deliberately about *shape and range*, not about particular
values: a value assertion at one seed would encode the seed's noise rather than
the contract, and would have to be re-recorded after every legitimate model
change. Where a range would be vacuous on a broken run — ``len(matches) <= N//2``
is the clearest — the test asserts the corresponding non-vacuity alongside it.
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
    import _hypergamic  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "C++ extension not built — run: cmake --build build --config Release",
        allow_module_level=True,
    )

from src.python.config import SimConfig  # noqa: E402
from src.python.orchestrator import COLUMNS, Simulation  # noqa: E402

N = 100
# 5 epochs *per generation*, two generations, so 10 rows in total. The session
# prompt asked for "5 epochs, n_generations=2" and "a DataFrame with 10 rows",
# which cannot both hold one row to an epoch; it also cannot be run, because
# run_multigen refuses an n_epochs that is not a multiple of n_generations. The
# reading taken here is the one that satisfies the row count and the divisibility
# rule together: EPOCHS_PER_GEN = 5, GENERATIONS = 2, EPOCHS = 10.
EPOCHS_PER_GEN = 5
GENERATIONS = 2
EPOCHS = EPOCHS_PER_GEN * GENERATIONS
DELTA = 0.5
SEED = 42

# The §6 columns in the order the contract lists them. Spelled out here rather
# than imported so that a reordering in orchestrator.py fails this test instead
# of silently redefining what "the §6 order" means.
SECTION_6_COLUMNS = [
    "epoch", "generation", "delta", "seed",
    "gini", "mcr", "rsi", "bpr",
    "welfare_alg", "welfare_full", "algorithmic_loss",
    "n_matched", "n_blocking", "pareto_alpha",
    "mean_swipes_per_agent",
]

# Columns whose contract is "never NaN on any row". The benchmark and the
# signalling columns are excluded: welfare_full / algorithmic_loss are NaN
# without a benchmark and mean_proposal_intent is NaN without a swipe log.
NEVER_NAN = ["gini", "mcr", "rsi", "bpr", "n_matched"]


def _config(seed: int = SEED) -> dict:
    cfg = SimConfig.from_yaml().to_dict()
    cfg["population"]["n"] = N
    cfg["population"]["seed"] = seed
    cfg["simulation"]["n_epochs"] = EPOCHS
    cfg["simulation"]["n_generations"] = GENERATIONS
    cfg["hypergamy"]["delta"] = DELTA
    return cfg


@pytest.fixture(scope="module")
def run() -> tuple[Simulation, pd.DataFrame]:
    sim = Simulation(config=_config(), recommender="ncf")
    # Outcomes and the market log are opt-in (orchestrator.record_outcomes /
    # record_market, both False by default) because they retain one copy of the
    # state matrix per epoch. This module inspects the matching, so it turns them
    # on; the §6 frame is asserted from the same run and must not depend on them.
    sim.record_outcomes = True
    sim.record_market = True
    return sim, sim.run_multigen()


# ── shape and schema ────────────────────────────────────────────────────────


def test_frame_has_one_row_per_epoch(run) -> None:
    _, frame = run
    assert frame.shape[0] == EPOCHS
    assert frame["epoch"].tolist() == list(range(EPOCHS))


def test_frame_columns_match_contract(run) -> None:
    _, frame = run
    assert list(frame.columns) == COLUMNS
    # The first fifteen are §6's own columns, in §6's order. Anything after them
    # was appended by a later session, which §6 permits.
    assert list(frame.columns[: len(SECTION_6_COLUMNS)]) == SECTION_6_COLUMNS


def test_config_values_reach_the_frame(run) -> None:
    """delta and seed are copied through, not recomputed."""
    _, frame = run
    assert (frame["delta"] == DELTA).all()
    assert (frame["seed"] == SEED).all()


def test_generation_is_derived_from_the_absolute_epoch(run) -> None:
    _, frame = run
    assert frame["generation"].tolist() == [e // EPOCHS_PER_GEN for e in frame["epoch"]]
    assert frame["generation"].is_monotonic_increasing


# ── missing values ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("column", NEVER_NAN)
def test_core_metrics_are_never_nan(run, column: str) -> None:
    _, frame = run
    assert frame[column].notna().all(), f"{column} has NaN at epoch {frame.index[frame[column].isna()]}"


def test_pareto_alpha_is_fitted_only_on_generation_final_rows(run) -> None:
    """§6: non-NaN exactly on each generation's final row, NaN everywhere else.

    Asserted as a set equality against the expected rows rather than as a count,
    so an off-by-one that still fits on *some* single row fails.
    """
    _, frame = run
    expected = {e for e in frame["epoch"] if (e + 1) % EPOCHS_PER_GEN == 0}
    fitted = set(frame.loc[frame["pareto_alpha"].notna(), "epoch"])

    assert fitted == expected
    assert fitted, "run produced no generation-final row, so this test is vacuous"
    assert np.isfinite(frame.loc[frame["pareto_alpha"].notna(), "pareto_alpha"]).all()


# ── ranges ──────────────────────────────────────────────────────────────────


def test_gini_is_a_share_in_unit_interval(run) -> None:
    _, frame = run
    assert ((frame["gini"] >= 0.0) & (frame["gini"] <= 1.0)).all()


def test_clearing_rate_is_a_positive_share(run) -> None:
    """mcr ∈ (0, 1]: at least one pair clears, and 2·n_matched/N cannot exceed 1."""
    _, frame = run
    assert ((frame["mcr"] > 0.0) & (frame["mcr"] <= 1.0)).all()


def test_rank_satisfaction_is_a_normalized_rank(run) -> None:
    """rsi ∈ (0, 1] — a matched partner is never ranked below last."""
    _, frame = run
    assert ((frame["rsi"] > 0.0) & (frame["rsi"] <= 1.0)).all()


def test_mcr_agrees_with_n_matched(run) -> None:
    """mcr is defined as 2·n_matched/N, so the two columns cannot disagree."""
    _, frame = run
    expected = 2.0 * frame["n_matched"].to_numpy() / N
    assert np.allclose(frame["mcr"].to_numpy(), expected, rtol=0, atol=1e-12)


def test_n_matched_is_a_consistent_integer(run) -> None:
    _, frame = run
    assert pd.api.types.is_integer_dtype(frame["n_matched"])
    assert (frame["n_matched"] > 0).all()
    # Bipartite matching over N/2 a side: no agent is in two pairs.
    assert (frame["n_matched"] <= N // 2).all()


# ── the matching itself ─────────────────────────────────────────────────────


def test_no_agent_is_matched_twice(run) -> None:
    """``len(matches) <= N//2`` plus each row appearing once.

    The half-population bound only says something if the pairs are also
    disjoint; a bug that matched one agent to many would pass the bound.
    """
    sim, _ = run

    assert sim.outcomes, "no epoch outcomes were recorded"
    for outcome in sim.outcomes:
        pairs = np.asarray(outcome.pairs, dtype=np.int64).reshape(-1, 2)
        rows = pairs.ravel()
        assert pairs.shape[0] <= N // 2
        assert len(set(rows.tolist())) == rows.size, "an agent appears in two pairs"


def test_a_swipe_log_is_produced_for_the_ncf_arm(run) -> None:
    """Every NCF row generated swipes; mean_swipes_per_agent is a share of N."""
    sim, frame = run
    assert (frame["mean_swipes_per_agent"] > 0.0).all()
    assert sim.market_log, "NCF arm produced no market log"


def test_the_frame_is_not_vacuous(run) -> None:
    """A run that matched nothing would satisfy most bounds above trivially."""
    sim, frame = run
    assert frame["n_matched"].sum() > 0
    assert sim.outcomes, "no epoch outcomes were recorded"
