"""Attribution tests (THEORY.md §9, docs/SPRINT_PROMPTS.md Session 3-D).

What these tests are for, stated because two of them would otherwise look
backwards:

**They document the artifact rather than denying it.** A test asserting that the
selected-set correlation is *zero* would be asserting that the market does not sort,
which is false. The tests assert that the population correlation is near zero —
the generative fact — while the selected-set correlation is measurably different
from it. That pair is what makes "this is selection, not a relationship" a checked
claim instead of a slogan.

**They are written to fail when the model changes underneath them.** The
leave-one-term-out numbers are measured, not derived; they depend on the trait
distributions, the weights, and the chemistry scale. The bounds below are set wide
enough to survive a toolchain difference and narrow enough to catch a retune that
makes the docs wrong. Each threshold is annotated with what it is protecting.

A note on `k`. The prompt that specified this session used k=1000, which at
``population.n: 1000`` is the entire candidate side — the "shortlist" is then the
whole market and every overlap is 1.0 by construction. Attribution is only
meaningful when k is smaller than the candidate pool, so the fixtures here use both
k=20 (the shipped ``ncf.top_k``) and the full pool, and the assertions are written
against the former.
"""

from __future__ import annotations

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

from src.python.attribution import (  # noqa: E402
    delta_power,
    leave_one_term_out_effect,
    selection_correlation,
)

# One population, sampled once. Every test here is a statement about a *market*,
# not about a random draw from a generator, so re-sampling per test would only add
# noise to numbers whose whole point is that they are reproducible.
N_AGENTS = 1000
SEED = 42
TOP_K = 20  # ncf.top_k in experiments/base.yaml
# k=1000 equals the candidate side at N=1000, so it is included deliberately: the
# full-pool case is where the market does not sort at all and every overlap must be
# 1.0. Asserting that pins the degenerate end and keeps the other bounds honest.


def base_config() -> dict:
    with open(PROJECT_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def config() -> dict:
    return base_config()


@pytest.fixture(scope="module")
def states(config):
    return _hypergamic.sample_population(N_AGENTS, config, SEED)


# ── The generative fact ──────────────────────────────────────────────────────

def test_status_and_attractiveness_are_sampled_independently(states, config):
    """The premise of the whole section: no generative link between the two.

    Sampled on one population, so this is a draw-level statement with a sampling
    error of about 1/sqrt(N) ~ 0.032. The tolerance is +/-0.05, which is roughly
    1.5 sd: wide enough that a legitimate reseed does not trip it, narrow enough
    that a sampler which started coupling the traits (a shared engine, a
    conditional draw) is caught immediately.

    This is the only test here that asserts near-zero, and it does so because it
    is measuring the *generator*. The tests below measure the *market*, where
    near-zero would be the wrong expectation.
    """
    corr = np.corrcoef(states[:, 2], states[:, 3])[0, 1]
    assert abs(corr) < 0.05, (
        f"status and attractiveness correlate at {corr:+.4f} in the sampled "
        "population; sample_population is supposed to draw them independently"
    )


def test_all_trait_pairs_are_near_zero_generatively(config):
    """Not just the status/attractiveness pair: every pair is independent.

    The *mean* over 20 seeds rather than the worst single draw. A single draw at
    N=1000 has a standard error of 0.032, so a per-draw bound loose enough to pass
    legitimately would be too loose to catch a real leak. Averaged over 20 draws
    the standard error of the mean falls to 0.007, which makes a +/-0.03 bound both
    safe and meaningful: a sampler that coupled two traits at 0.04 would fail it
    every time.

    Status is lognormal and so is skewed; that does not move the correlation, which
    is a statement about joint ranks, but it does mean this pair is worth checking
    separately from the symmetric ones.
    """
    means = {}
    for i, j in ((2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)):
        draws = []
        for seed in range(SEED, SEED + 20):
            s = _hypergamic.sample_population(N_AGENTS, config, seed)
            draws.append(np.corrcoef(s[:, i], s[:, j])[0, 1])
        means[f"{i}/{j}"] = float(np.mean(draws))

    worst = max(means.items(), key=lambda kv: abs(kv[1]))
    assert abs(worst[1]) < 0.03, (
        f"trait columns {worst[0]} correlate at {worst[1]:+.4f} averaged over 20 "
        f"seeds (all pairs: {means}); sample_population draws them independently"
    )


# ── The selection artifact ───────────────────────────────────────────────────

def test_selection_induces_a_correlation_the_population_does_not_have(states, config):
    """Top-k shortlists correlate the traits even though the draw does not.

    This documents the artifact the section is about. What is asserted is the
    *separation*: the selected-set correlation differs from the population one by
    more than sampling error. The direction is deliberately not pinned to
    positive — see the module docstring of src/python/attribution.py. With the
    market ranking on both traits and a shortlist far smaller than the candidate
    pool, the induced correlation is negative; a test asserting `> 0` would encode
    a guess rather than a measurement and would fail on any correct implementation.

    The n at top-20 is 1000 agents x 20 = 20,000 pairs, so the Pearson standard
    error is ~1/sqrt(20000) = 0.007. A 0.05 separation is seven standard errors and
    is not a flake; it is also loose enough to survive a toolchain difference in
    the sampled traits.
    """
    result = selection_correlation(states, config, TOP_K)["attractiveness~status"]
    population = result["population"]
    selected = result["selected"]

    assert abs(population) < 0.05, "the generative correlation should be near zero"
    assert abs(selected - population) > 0.05, (
        f"selected-set correlation {selected:+.4f} is not distinguishable from the "
        f"population value {population:+.4f}; either the market stopped sorting on "
        "both traits or selection_correlation is not selecting anything"
    )
    assert result["n_pairs"] == N_AGENTS * TOP_K


def test_selection_correlation_decays_as_the_shortlist_widens(states, config):
    """The artifact is strongest at the top and fades as the shortlist grows.

    This is the signature that distinguishes selection from a real relationship: a
    genuine generative correlation would be flat in top_n, because it is a property
    of the population and the shortlist only samples it. A selection artifact is
    largest where the market is most selective and decays toward the population
    value as the shortlist approaches the whole pool.

    Asserted on absolute magnitude, so the test does not depend on the sign.
    """
    mags = [
        abs(selection_correlation(states, config, top_n)["attractiveness~status"]["selected"])
        for top_n in (10, 50, 500)
    ]
    assert mags[0] > mags[-1], (
        f"|corr| at top-10 ({mags[0]:.4f}) should exceed |corr| at top-500 "
        f"({mags[-1]:.4f}); a flat profile would mean the correlation is generative, "
        "not selection-induced"
    )


def test_top_ten_varies_enough_across_markets_that_one_is_not_quotable(states, config):
    """A single market's top-10 is not the number; the average over markets is.

    What the measurement actually shows, over 20 independent populations at N=1000:

        per-market values span -0.06 to -0.38, mean -0.23, sd 0.10

    So the *sign* is stable — every one of the 20 draws is negative — while the
    *magnitude* varies by a factor of six between markets. That is the reason the
    docs report a mean and a spread rather than a single run: quoting the mildest
    market would understate the effect by more than 3x, and quoting the strongest
    would overstate it by 1.7x, with neither being wrong about its own population.

    The assertion is on that spread, and it is deliberately about magnitude only.
    A test demanding sign instability would fail on the real data, which would be
    the test being wrong rather than the data.

    Note on n: each market contributes 1000 agents x 10 = 10,000 pooled pairs, not
    10. The per-market variation here is *between markets*, not binomial noise on
    10 points, which is why it is 0.10 and not 0.3 — and why averaging seven
    markets tightens the estimate rather than merely smoothing it.
    """
    per_market = []
    for seed in range(SEED, SEED + 20):
        s = _hypergamic.sample_population(N_AGENTS, config, seed)
        per_market.append(
            selection_correlation(s, config, 10)["attractiveness~status"]["selected"]
        )

    arr = np.asarray(per_market)
    span = float(arr.max() - arr.min())
    assert span > 0.15, (
        f"per-market top-10 coefficients span only {span:.4f} "
        f"({arr.min():+.3f} to {arr.max():+.3f}); if they were that stable, a "
        "single market's value would be quotable and the docs would not need to "
        "average"
    )
    # The sign is stable at this configuration, which is what licenses the
    # "negative at small k" statement in docs/attribution.md. If a retune makes the
    # sign flicker across markets, that sentence becomes false and this fails.
    assert (arr < 0).all(), (
        f"top-10 sign is not stable: {np.round(arr, 3)}. docs/attribution.md states "
        "the induced correlation at small k is negative; re-measure and rewrite it"
    )


# ── Leave-one-term-out ──────────────────────────────────────────────────────

def test_the_two_linear_terms_drive_selection_and_the_other_two_do_not(states, config):
    """The central measured claim, in the form the data actually supports.

    Measured at k=20 over seven seeds (docs/attribution.md):

        status    0.748 ± 0.010     attract  0.607 ± 0.014
        distance  0.929 ± 0.005     indicator 0.918 ± 0.002

    So there are two groups, cleanly separated by 0.17 with spreads an order of
    magnitude smaller. The assertion is the separation, not the within-group
    ordering: **attractiveness moves more of the shortlist than status does**, which
    is the opposite of the intuitive reading and follows directly from beta = 2·alpha
    at the shipped weights. Pinning the ordering alpha-then-rest would encode a
    guess about the weights and fail on a correct implementation, so what is pinned
    is the gap.

    The prompt that specified this session expected status -> 0 overlap 0.000 and
    attractiveness -> 0 overlap 0.967. Neither reproduces: at the shipped
    configuration the two are 0.748 and 0.607. That is recorded in the README and in
    docs/attribution.md rather than corrected away.
    """
    effects = leave_one_term_out_effect(states, config, k=TOP_K)
    linear = max(effects["status"], effects["attract"])
    other = min(effects["distance"], effects["indicator"])

    assert linear < other - 0.10, (
        f"the linear terms (status {effects['status']:.3f}, attract "
        f"{effects['attract']:.3f}) should reorder substantially more of the "
        f"shortlist than distance ({effects['distance']:.3f}) and the indicator "
        f"({effects['indicator']:.3f})"
    )
    # Both linear terms must be doing real work, or the separation above could be
    # satisfied by two inert weights against two slightly-less-inert ones.
    assert effects["status"] < 0.90
    assert effects["attract"] < 0.90


def test_status_and_attractiveness_swap_rank_when_their_weights_swap(states, config):
    """The ordering is set by the weights, which is §9.2's qualification as a test.

    The previous test establishes that the linear terms dominate. This one
    establishes *why* — by swapping alpha and beta and checking the two overlaps
    change places. A leave-one-term-out number that did not respond to its own
    weight would not be measuring the weight.

    The chemistry is switched off here. The pairwise shock is unlearnable noise on
    every pair, and leaving it on blurs the comparison without changing the point;
    with it off the two terms are closer to mirror images.
    """
    quiet = base_config()
    quiet["hypergamy"]["shock_scale"] = 0.0

    ab = leave_one_term_out_effect(states, quiet, terms=("status", "attract"), k=TOP_K)
    swapped = dict(quiet["hypergamy"]) | {
        "alpha": quiet["hypergamy"]["beta"],
        "beta": quiet["hypergamy"]["alpha"],
    }
    quiet_swapped = dict(quiet)
    quiet_swapped["hypergamy"] = swapped
    ba = leave_one_term_out_effect(states, quiet_swapped, terms=("status", "attract"), k=TOP_K)

    # Attractiveness dominates at beta=2, alpha=1; status must dominate after the
    # swap, because alpha then carries the weight of 2.
    assert ab["attract"] < ab["status"], (
        f"at beta=2·alpha, removing attract ({ab['attract']:.3f}) should move more "
        f"of the shortlist than removing status ({ab['status']:.3f})"
    )
    assert ba["status"] < ba["attract"], (
        f"after swapping the weights, removing status ({ba['status']:.3f}) should "
        f"move more than removing attract ({ba['attract']:.3f}); the overlaps did "
        "not follow their weights, so they are not measuring them"
    )


def test_full_pool_ablation_is_inert_by_construction(states, config):
    """At k = the whole candidate side every overlap is exactly 1.0.

    The degenerate case, asserted on purpose. A full-market "shortlist" cannot be
    reordered by a weight change — it holds everyone in some order and overlap
    ignores order — so any number other than 1.0 here means the overlap is
    comparing something other than what it claims. This is the test that catches an
    off-by-one in the k clamp or a comparison over the wrong side.
    """
    effects = leave_one_term_out_effect(states, config, k=N_AGENTS)
    for term, value in effects.items():
        assert value == 1.0, f"{term}: full-pool overlap was {value}, expected exactly 1.0"


# ── Delta power ─────────────────────────────────────────────────────────────

def test_delta_has_power_at_the_shipped_weights(states, config):
    """delta reorders part of the women's shortlist at beta/alpha = 2.

    Measured 0.854 at k=20 over the seven report seeds. The 0.95 ceiling is the
    same floor tests/cpp/test_delta_power.cpp uses, so the two cannot drift apart
    without one of them failing.

    Women only: delta enters the acceptor's utility alone, so a man's ranking is
    invariant to it. Measured pooled across both sides this reads 0.923 — the
    men's half contributes a flat 1.0 to the average and makes a live indicator
    look nearly inert. That is the bug this restriction exists to prevent, and the
    final assertion below pins the difference.
    """
    overlap = delta_power(states, config, deltas=(0.0, 1.0), k=TOP_K)
    assert overlap < 0.95, (
        f"delta 0.0 and 1.0 agree on {overlap:.3f} of the shortlist; delta has gone "
        "inert. THEORY.md §2.2: the indicator needs beta/alpha >= 2"
    )
    # And it must actually move something: a value near 1.0 that squeaked under the
    # floor would pass the check above while delta does nothing.
    assert 1.0 - overlap > 0.02, (
        f"delta moved only {1.0 - overlap:.3f} of the women's shortlist; that is "
        "not power, it is noise at the edge of the measurement"
    )


def test_delta_warns_when_it_is_inert(states, config):
    """The guard the prompt asks for: 1.0 must be loud, not silent.

    Constructed rather than hoped for, and it takes two knobs off, not one. With
    beta = 0 and gamma = 0 the ranking is a monotone function of candidate status
    and the indicator is constant on every shortlist — but only when the pairwise
    chemistry is off as well. With shock_scale at its shipped 1.5 the independent
    shock on every ordered pair reorders a fifth of the shortlist regardless of
    the weights, and delta's contribution is drowned in it: measured 0.850 at
    beta = gamma = 0 *with* chemistry, against exactly 1.000 without. The
    chemistry is unlearnable noise by design (§2.3) and the guardrail is about the
    deterministic ranking, so the inert case is the quiet one.

    Both the warning and the returned value are asserted: a function that warned
    without returning 1.0, or returned 1.0 without warning, would each be half the
    guard.
    """
    inert = dict(config)
    inert["hypergamy"] = dict(config["hypergamy"], beta=0.0, gamma=0.0, shock_scale=0.0)

    with pytest.warns(RuntimeWarning, match="delta is inert"):
        overlap = delta_power(states, inert, deltas=(0.0, 1.0), k=TOP_K)
    assert overlap == 1.0
    assert overlap == 1.0


def test_delta_power_needs_two_deltas(states, config):
    """A single delta cannot be compared to anything; that is a caller error."""
    with pytest.raises(ValueError, match="at least two"):
        delta_power(states, config, deltas=(0.5,))


def test_men_shortlists_are_invariant_to_delta(states, config):
    """Why delta_power is women-only, asserted directly rather than assumed.

    The indicator is added inside `if (type == AgentType::W)` in
    `Agent::utility_toward`, so a man's ranking cannot depend on delta. This test
    states that as a fact about the built code: changing delta from 0.0 to 1.0 must
    leave every man's shortlist identical.

    It matters because pooling the two sides is otherwise a plausible-looking
    implementation that halves every delta figure and reports a live indicator as
    nearly inert. If this ever fails, the asymmetry has moved and delta_power's
    `side="W"` restriction is measuring the wrong thing.
    """
    from src.python.attribution import _top_k_by_agent, _with_weights  # noqa: PLC0415

    low = _top_k_by_agent(states, _with_weights(config, delta=0.0), TOP_K, side="M")
    high = _top_k_by_agent(states, _with_weights(config, delta=1.0), TOP_K, side="M")

    for row, (a, b) in enumerate(zip(low, high)):
        assert a == b, (
            f"agent {row}'s shortlist changed with delta, but delta only enters the "
            "acceptor's utility; either the indicator moved into the proposer branch "
            "or delta_power's women-only restriction is wrong"
        )


def test_women_shortlists_do_depend_on_delta(states, config):
    """The complement: the restriction must not exclude the side that responds."""
    from src.python.attribution import _top_k_by_agent, _with_weights  # noqa: PLC0415

    low = _top_k_by_agent(states, _with_weights(config, delta=0.0), TOP_K, side="W")
    high = _top_k_by_agent(states, _with_weights(config, delta=1.0), TOP_K, side="W")
    changed = sum(a != b for a, b in zip(low, high))
    assert changed > 0, (
        "no woman's shortlist responded to delta at the shipped weights; the "
        "indicator is inert and the guardrail is not testing anything"
    )


# ── Input handling ──────────────────────────────────────────────────────────

def test_unknown_term_is_rejected(states, config):
    """A misspelled term would otherwise silently ablate nothing and report 1.0."""
    with pytest.raises(ValueError, match="unknown term"):
        leave_one_term_out_effect(states, config, terms=("atractiveness",), k=TOP_K)


def test_functions_do_not_mutate_the_caller_states(states, config):
    """θ is overwritten internally, so the caller's matrix must survive intact.

    attribution.py ranks an unthresholded copy; if that copy ever became a view,
    every subsequent call in a script would run against a market whose gate had
    been switched off by the previous call, silently.
    """
    before = np.array(states, copy=True)
    leave_one_term_out_effect(states, config, terms=("status",), k=TOP_K)
    selection_correlation(states, config, TOP_K)
    delta_power(states, config, k=TOP_K)
    assert np.array_equal(states, before)


def test_the_theta_gate_is_ignored(states, config):
    """Calibrating θ changes nothing in these functions, by design.

    Attribution is about the ranking. If θ leaked into it, zeroing a weight would
    move both the ordering and the admissible set, and the overlap would measure
    the two together.
    """
    gated = np.array(states, copy=True)
    gated[:, 6] = -np.inf

    calibrated = leave_one_term_out_effect(states, config, k=TOP_K)
    open_market = leave_one_term_out_effect(gated, config, k=TOP_K)

    for term in calibrated:
        assert calibrated[term] == pytest.approx(open_market[term], abs=1e-12), (
            f"{term}: attribution changed when θ was removed, so θ is leaking into "
            "the ranking"
        )
