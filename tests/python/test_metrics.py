"""Pure-function tests for src/python/metrics.py, plus the §6 schema guard.

The metric functions are numpy-only and are exercised here without the C++
extension. The one test that needs it — the schema assertion, which runs a real
``Simulation`` — skips itself with the build command rather than failing
collection for the rest of the file.
"""

import pathlib
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "build" / "Release"))

from src.python.metrics import (  # noqa: E402
    algorithmic_loss,
    blocking_pair_ratio,
    gini,
    market_clearing_rate,
    pareto_alpha,
    rank_satisfaction_index,
)

TOL = 1e-12


# ── gini ────────────────────────────────────────────────────────────────────────

def test_gini_of_equal_counts_is_zero():
    assert gini(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0, abs=TOL)


def test_gini_of_one_holder_hand_derived():
    """Sorted form: 2·16/(4·4) − 5/4 = 0.75.

    Cross-checked against the pairwise definition, 3 pairs of difference 4 counted
    twice over 2·n²·x̄ = 32.
    """
    assert gini(np.array([0.0, 0.0, 0.0, 4.0])) == pytest.approx(0.75, abs=TOL)
    assert 2 * 3 * 4.0 / (2 * 16 * 1.0) == pytest.approx(0.75)


def test_gini_single_nonzero_entry_is_n_minus_one_over_n():
    for n in (2, 3, 7, 10, 50):
        counts = np.zeros(n)
        counts[0] = 13.0
        assert gini(counts) == pytest.approx((n - 1) / n, abs=TOL), n


def test_gini_is_invariant_to_order_and_scale():
    rng = np.random.default_rng(20261019)
    counts = rng.integers(0, 40, size=64).astype(np.float64)
    reference = gini(counts)

    assert gini(rng.permutation(counts)) == pytest.approx(reference, abs=TOL)
    assert gini(counts * 2.5) == pytest.approx(reference, abs=TOL)
    assert gini(counts * 1000.0) == pytest.approx(reference, abs=TOL)


# ── market clearing rate ───────────────────────────────────────────────────────

def test_mcr_counts_both_sides():
    assert market_clearing_rate(50, 100) == pytest.approx(1.0)
    assert market_clearing_rate(25, 100) == pytest.approx(0.5)
    assert market_clearing_rate(0, 100) == 0.0


# ── rank satisfaction index ────────────────────────────────────────────────────

def test_rsi_bounds_and_extremes():
    prefs = [[0, 1, 2], [1, 0, 2]]
    # Top choice in a 3-list: (0+1)/3.
    assert rank_satisfaction_index(np.array([[0, 0]]), prefs) == pytest.approx(1 / 3)
    # Bottom choice: 1.0.
    assert rank_satisfaction_index(np.array([[0, 2]]), prefs) == pytest.approx(1.0)
    assert np.isnan(rank_satisfaction_index(np.zeros((0, 2), dtype=np.int32), prefs))


def test_rsi_stays_in_the_open_closed_unit_interval():
    rng = np.random.default_rng(7)
    for list_len in (1, 2, 5, 12):
        prefs = [list(rng.permutation(20)[:list_len]) for _ in range(20)]
        matches = np.array([[p, prefs[p][rng.integers(0, list_len)]] for p in range(20)])
        value = rank_satisfaction_index(matches, prefs)
        assert 0.0 < value <= 1.0, (list_len, value)
        assert value >= 1.0 / list_len - TOL, "a top choice is the best score available"


def test_rsi_skips_a_partner_absent_from_the_list():
    """A missing partner contributes nothing; it is not silently scored at 1.0."""
    prefs = [[0, 1, 2], [1, 0, 2]]
    # Partner 99 is on nobody's list: the only pair is dropped, nothing is scored.
    assert np.isnan(rank_satisfaction_index(np.array([[0, 99]]), prefs))

    # One scorable pair (top choice of a 3-list) and one unscorable: the mean is
    # over the scorable pair alone, i.e. 1/3 — scoring the hole at 1.0 would give
    # (1/3 + 1)/2 = 2/3.
    mixed = np.array([[0, 0], [1, 99]])
    assert rank_satisfaction_index(mixed, prefs) == pytest.approx(1 / 3)


# ── algorithmic loss ───────────────────────────────────────────────────────────

def test_algorithmic_loss_known_value():
    assert algorithmic_loss(welfare_alg=60.0, welfare_full=100.0) == pytest.approx(0.4)


def test_algorithmic_loss_is_nan_without_a_usable_benchmark():
    for welfare_full in (0.0, float("nan"), float("inf"), float("-inf")):
        assert np.isnan(algorithmic_loss(welfare_alg=1.0, welfare_full=welfare_full)), (
            welfare_full
        )


def test_algorithmic_loss_is_not_clipped_at_zero():
    """The truncated matching can beat the benchmark; that is a result, not an error."""
    value = algorithmic_loss(welfare_alg=120.0, welfare_full=100.0)
    assert value == pytest.approx(-0.2)
    assert value < 0.0


def test_algorithmic_loss_rejects_positional_arguments():
    """Guards the keyword-only signature: a swapped call must not be expressible."""
    with pytest.raises(TypeError):
        algorithmic_loss(60.0, 100.0)


# ── pareto alpha ──────────────────────────────────────────────────────────────

def test_pareto_alpha_nan_on_degenerate_samples():
    assert np.isnan(pareto_alpha(np.array([1.0, 1.0, 1.0])))
    assert np.isnan(pareto_alpha(np.array([1.0, 0.0, 0.0])))


def test_pareto_alpha_needs_three_points():
    # Two points are a ratio, not a fit.
    assert np.isnan(pareto_alpha(np.array([1.0, 2.0])))
    assert np.isnan(pareto_alpha(np.array([3.0, 9.0])))
    # Three distinct points carry a fit.
    assert np.isfinite(pareto_alpha(np.array([1.0, 2.0, 4.0])))


def test_pareto_alpha_is_finite_on_a_clean_tail():
    assert np.isfinite(pareto_alpha(np.array([1.0, 2.0, 4.0, 8.0, 16.0])))


def test_pareto_alpha_recovers_a_known_shape():
    """Inverse-transform samples from Pareto(alpha=2, x_min=1); the MLE lands near 2."""
    rng = np.random.default_rng(20261019)
    alpha_true = 2.0
    u = rng.uniform(size=20_000)
    sample = u ** (-1.0 / alpha_true)

    estimate = pareto_alpha(sample)
    assert np.isfinite(estimate)
    assert estimate == pytest.approx(alpha_true, abs=0.1)


# ── blocking pair ratio ───────────────────────────────────────────────────────

def test_blocking_pair_ratio_does_not_divide_by_zero():
    assert blocking_pair_ratio(10, 0) == pytest.approx(10.0)
    assert np.isfinite(blocking_pair_ratio(0, 0))


def test_blocking_pair_ratio_is_pairs_over_matches():
    assert blocking_pair_ratio(6, 3) == pytest.approx(2.0)
    assert blocking_pair_ratio(0, 5) == 0.0


# ── §6 schema guard ───────────────────────────────────────────────────────────

def _schema_columns_from_doc() -> list[str]:
    """Column names, in order, from the §6 table in INTERFACES.md.

    Parsed from the document rather than imported from ``orchestrator.COLUMNS``,
    because comparing that constant against itself would pass no matter how far
    the code drifted from the spec.
    """
    text = (PROJECT_ROOT / "docs" / "INTERFACES.md").read_text(encoding="utf-8")
    start = text.index("## 6. Metrics DataFrame schema")
    section = text[start:].split("\n## ", 1)[0]

    columns: list[str] = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        columns.append(line.split("`")[1])
    return columns


def test_metrics_frame_matches_the_documented_schema():
    """A 2-epoch Simulation emits exactly the §6 columns, in the §6 order."""
    try:
        import _hypergamic  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip(
            "C++ extension not built — run: cmake --build build --config Release"
        )

    from src.python.orchestrator import Simulation

    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["population"]["n"] = 200
    cfg["simulation"]["n_epochs"] = 2

    df = Simulation(config=cfg).run(2)

    documented = _schema_columns_from_doc()
    assert len(documented) == 15, "the §6 table should list 15 columns"
    assert list(df.columns) == documented
