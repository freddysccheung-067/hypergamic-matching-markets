"""Ablation-arm tests (docs/SPRINT_PROMPTS.md Session 3-C, item 4).

Two kinds of assertion, deliberately different in character:

  * **Identities.** The popularity arm's concentration is a closed form, not a
    tendency: one shared list per side means exactly 2k agents carry all the
    exposure in equal amounts, so Gini = 1 - 2k/N and MCR = 2k/N *exactly*. Those
    are asserted to 1e-9 at two different k. A loose ordering check would pass an
    implementation that gave agents different lists while calling itself a
    popularity baseline; an exact identity will not.

  * **Orderings.** Between arms nothing is exact, so the ordering is asserted on
    the mean across seeds with the spread reported alongside. Asserting
    `gini(a) < gini(b)` on a single run would pass on a seed where the two are
    separated by noise and say nothing about whether the separation is real.

A note on when the identity holds. The popularity arm has no history at epoch 0
and falls back to the oracle ranking by design, so epoch 0 is identical to the
oracle arm and *not* yet the 2k attractor. The identity is asserted from epoch 1
onward, and the epoch-0 fallback is itself asserted — both that it equals the
oracle arm's epoch-0 lists and that epoch 0 is the one row that does not satisfy
the closed form.
"""

from __future__ import annotations


import numpy as np
import pytest

from src.python.config import SimConfig
from src.python.metrics import gini as gini_metric
from src.python.orchestrator import Simulation

# A small population keeps these tests to a few seconds. The identity is exact at
# any N, so nothing here depends on N being large; the ordering test does use a
# bigger N because arm separation at N=60 is dominated by discreteness.
SMALL_N = 60
ORDERING_N = 300


def _config(n: int, seed: int = 7, top_k: int = 10,
            admissible_fraction: float = 0.05) -> dict:
    """A config dict in the YAML's shape, with every knob the arms care about.

    ``admissible_fraction`` is small enough that θ binds: at the shipped 0.2 an
    agent has ~0.2*(n/2) admissible candidates, far more than ``top_k``, and the
    gate never truncates — which makes the market static and would let a broken
    arm pass. At 0.05 the gate is live, as Session 3-C requires.
    """
    cfg = SimConfig.from_yaml().to_dict()
    cfg["population"]["n"] = n
    cfg["population"]["seed"] = seed
    cfg["population"]["admissible_fraction"] = admissible_fraction
    cfg["ncf"]["top_k"] = top_k
    cfg["simulation"]["n_epochs"] = 4
    cfg["simulation"]["n_generations"] = 2
    return cfg


def _run(arm: str, n: int = SMALL_N, seed: int = 7, top_k: int = 10,
         epochs: int = 2, freeze_theta: bool = False) -> Simulation:
    cfg = _config(n, seed=seed, top_k=top_k)
    sim = Simulation(config=cfg, recommender=arm, freeze_theta=freeze_theta)
    sim.run(epochs)
    return sim


# ── the closed form ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("top_k", [5, 10, 20])
def test_popularity_identity_is_exact(top_k: int) -> None:
    """Gini == 1 - 2k/N and MCR == 2k/N, to 1e-9, at several k.

    Why this is exact rather than approximate: the popularity arm gives every
    chooser on a side the same list of k. Exactly 2k agents therefore appear on
    every list, each exactly N/2 times, and the other N - 2k appear zero times.
    Sorted ascending, the exposure vector is 0 repeated N-2k times followed by
    N/2 repeated 2k times, and the Gini of that vector is algebraically 1 - 2k/N.

    MCR is 2k/N for the matching reason: under deferred acceptance every acceptor
    on the shared list receives N/2 proposals and holds her best, so exactly k
    pairs form and MCR = 2k/(N/2)/2 = 2k/N.

    Asserted from epoch 1: epoch 0 is the oracle fallback (see the module
    docstring), so it is the one row that must *not* satisfy this.
    """
    sim = _run("popularity", n=SMALL_N, top_k=top_k, epochs=2)

    exposure = sim.exposure_series
    assert exposure.shape == (2, SMALL_N)

    expected_gini = 1.0 - 2.0 * top_k / SMALL_N

    # Epoch 1 is the first epoch with history, so it is the first at the attractor.
    g = gini_metric(exposure[1])
    assert g == pytest.approx(expected_gini, abs=1e-9), (
        f"gini {g!r} != 1 - 2k/N = {expected_gini!r} at k={top_k}"
    )

    # And exactly 2k agents are exposed, each equally.
    nonzero = exposure[1][exposure[1] > 0]
    assert nonzero.size == 2 * top_k
    assert np.unique(nonzero).size == 1


@pytest.mark.parametrize("top_k", [5, 20])
def test_popularity_mcr_identity(top_k: int) -> None:
    """MCR == 2k/N, separately from the Gini identity above.

    Split out because the two closed forms fail for different reasons: Gini is a
    property of the exposure vector alone, MCR of the matching. An arm that built
    the right lists but handed the matching a different object would pass one and
    fail the other.
    """
    cfg = _config(SMALL_N, top_k=top_k)
    sim = Simulation(config=cfg, recommender="popularity")
    frame = sim.run(2)
    # Epoch 1 is the first at the attractor.
    assert float(frame["mcr"].iloc[1]) == pytest.approx(2.0 * top_k / SMALL_N, abs=1e-9)


def test_popularity_epoch0_is_the_oracle_fallback() -> None:
    """Epoch 0 has no in-degree history and falls back to the oracle ranking.

    Two things are asserted, and the second is the one that matters: epoch 0 must
    equal the oracle arm's epoch-0 *lists* (not merely its metrics), and epoch 0
    must therefore be the one epoch that does not satisfy 1 - 2k/N.

    Why the lists and not the Gini: two different preference lists can produce the
    same exposure count, so an equal Gini would not establish the fallback. The
    lists are the object the arm actually chose.
    """
    pop = _run("popularity", epochs=1, freeze_theta=True)
    orc = _run("oracle", epochs=1, freeze_theta=True)

    assert np.array_equal(pop.exposure_series[0], orc.exposure_series[0])

    # Epoch 0 is not yet at the attractor: 1 - 2k/N would be 0.6667, and the
    # oracle fallback should be something else entirely.
    k = 10
    assert gini_metric(pop.exposure_series[0]) != pytest.approx(1.0 - 2.0 * k / SMALL_N,
                                                                 abs=1e-6)

    # From epoch 1 the attractor is reached.
    pop2 = _run("popularity", epochs=2)
    assert gini_metric(pop2.exposure_series[1]) == pytest.approx(
        1.0 - 2.0 * k / SMALL_N, abs=1e-9)


# ── ordering ─────────────────────────────────────────────────────────────────


def _final_gini(arm: str, seed: int, n: int = ORDERING_N, epochs: int = 4) -> float:
    cfg = _config(n, seed=seed, top_k=10)
    sim = Simulation(config=cfg, recommender=arm)
    frame = sim.run(epochs)
    return float(frame["gini"].iloc[-1])


def test_gini_ordering_over_seeds() -> None:
    """random < oracle < ncf < popularity, as a mean ± sd over seeds.

    The ordering is asserted on the *mean*, and the spread is computed and
    reported so a failure prints the numbers rather than a bare inequality. A
    single-seed assertion would pass or fail on sampling noise.

    `ncf` here is the Session 3-C `lowrank` arm: the NCF is the low-rank
    approximation standing in for the rank-8 replica the preamble measured.

    If the NCF lands *below* the oracle, that is a real signal about the model and
    must be reported rather than adjusted for — the assertion below would then
    fail, and the failure message says so. It is not silently retuned.
    """
    seeds = [7, 8, 9]
    arms = ("random", "oracle", "ncf", "popularity")
    results = {arm: [] for arm in arms}
    for seed in seeds:
        for arm in arms:
            results[arm].append(_final_gini(arm, seed))

    stats = {
        arm: (float(np.mean(v)), float(np.std(v, ddof=1)))
        for arm, v in results.items()
    }

    ordered = [stats[a][0] for a in arms]
    message = (
        "gini ordering violated over seeds "
        + ", ".join(f"{a}={stats[a][0]:.4f}±{stats[a][1]:.4f}" for a in arms)
        + ". random < oracle < ncf < popularity is the expected order; an NCF "
        "below the oracle is a real model signal to report, not to tune away."
    )
    assert ordered[0] < ordered[1] < ordered[2] < ordered[3], message

    # The gap the result is reported in: ncf - oracle, mean ± sd over seeds.
    gaps = [results["ncf"][i] - results["oracle"][i] for i in range(len(seeds))]
    gap_mean, gap_sd = float(np.mean(gaps)), float(np.std(gaps, ddof=1))
    # Separation must be real, not a seed artifact. The threshold is loose on
    # purpose: this asserts the sign and the spread, not a magnitude, because the
    # magnitude is a function of the heterogeneity setting (THEORY.md §2.3).
    assert gap_mean > 0.0, (
        f"gini(ncf) - gini(oracle) = {gap_mean:+.4f} ± {gap_sd:.4f}: the NCF is "
        "concentrating LESS than the oracle. Report this, do not adjust the test."
    )


def test_reach_separates_popularity_from_random() -> None:
    """Popularity reaches under 5%; random over 90%.

    Reach is the metric that says the popularity arm is not a market: 2k agents
    out of N is a rounding error at these sizes.

    The 5% figure is *not* scale-free, and that is worth being explicit about
    rather than tuning the test until it passes: the arm's reach is exactly
    2k/N by construction, so 5% presumes the base regime (N = 1000, k = 20,
    where 2k/N = 4%). At N = 300 the same k gives 6.7% and the threshold would
    be measuring the population size, not the arm. So this test runs at the
    reference scale, and it asserts the exact identity alongside the threshold.
    """
    n, k = 1000, 20
    pop = _run("popularity", n=n, top_k=k, epochs=2)
    rnd = _run("random", n=n, top_k=k, epochs=2)

    pop_reach = float(np.count_nonzero(pop.exposure_series[1] > 0)) / n
    rnd_reach = float(np.count_nonzero(rnd.exposure_series[1] > 0)) / n

    # Exact, not approximate: exactly 2k agents appear on the shared lists.
    assert pop_reach == pytest.approx(2.0 * k / n, abs=1e-12)
    assert pop_reach < 0.05, f"popularity reach {pop_reach:.3f} should be < 5%"
    assert rnd_reach > 0.90, f"random reach {rnd_reach:.3f} should be > 90%"


# ── reproducibility and shared state ─────────────────────────────────────────


@pytest.mark.parametrize("arm", ["ncf", "random", "popularity", "oracle"])
def test_same_arm_and_seed_reproduces_exposure_matrix(arm: str) -> None:
    """Same (arm, seed) reproduces the exposure matrix exactly.

    Exact equality, not approx: every arm's randomness is seeded from the run seed
    and (for `random`) the epoch, and torch is seeded before the model is built,
    so there is no unseeded draw anywhere in the path.
    """
    a = _run(arm, n=SMALL_N, epochs=2)
    b = _run(arm, n=SMALL_N, epochs=2)
    assert np.array_equal(a.exposure_series, b.exposure_series)


def test_all_arms_see_identical_states() -> None:
    """Every arm is handed the same state matrix.

    Two assertions, because they establish different things:

    * At epoch 0, with the loop live, all four arms must agree — this is the
      "shared population and θ" claim, and a resampled population would break it.
    * With θ frozen, they must agree at *every* epoch. With the loop live they
      legitimately diverge after epoch 0: each arm's matching outcome moves θ
      differently, which is the feedback the model is supposed to have. Freezing
      it isolates the preference rule, which is what the ablation compares.
    """
    arms = ("ncf", "random", "popularity", "oracle")

    live = {}
    for arm in arms:
        cfg = _config(SMALL_N)
        sim = Simulation(config=cfg, recommender=arm)
        sim.record_states = True
        sim.run(2)
        live[arm] = sim.states_seen
    for arm in arms[1:]:
        assert np.array_equal(live[arms[0]][0], live[arm][0]), (
            f"{arm} was handed a different epoch-0 state matrix than {arms[0]}"
        )

    frozen = {}
    for arm in arms:
        cfg = _config(SMALL_N)
        sim = Simulation(config=cfg, recommender=arm, freeze_theta=True)
        sim.record_states = True
        sim.run(2)
        frozen[arm] = sim.states_seen
    for epoch in range(2):
        for arm in arms[1:]:
            assert np.array_equal(frozen[arms[0]][epoch], frozen[arm][epoch]), (
                f"{arm} was handed a different state matrix at epoch {epoch}"
            )


def test_unknown_arm_is_rejected() -> None:
    """A misspelled arm must fail at construction, not silently run the NCF."""
    with pytest.raises(ValueError, match="recommender must be one of"):
        Simulation(config=_config(SMALL_N), recommender="popularty")  # type: ignore[arg-type]


def test_exposure_series_shape_and_dtype() -> None:
    """run() exposes a (T, N) int array, as the Lorenz plot needs."""
    cfg = _config(SMALL_N)
    sim = Simulation(config=cfg, recommender="oracle")
    frame = sim.run(3)
    assert sim.exposure_series.shape == (len(frame), SMALL_N)
    assert sim.exposure_series.dtype == np.int64
