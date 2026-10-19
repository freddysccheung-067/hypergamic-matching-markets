#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "agent.h"
#include "population.h"
#include "simulation.h"
#include "utilities.h"

// ── Preference heterogeneity ─────────────────────────────────────────────────
// Two knobs, both off by default: per-agent taste multipliers (taste_sigma) and
// persistent pairwise chemistry (shock_scale). THEORY.md §2.3 is the write-up;
// this file is what pins the properties the rest of the project depends on.
//
// The single most important property is the negative one: with both knobs at 0
// nothing changes. That is asserted at the level of sampled traits and of the
// utility value itself, bit for bit, because "approximately unchanged" would not
// protect the published numbers.

namespace {

TraitDist trait(const std::string& name, double p1, double p2) {
    TraitDist d;
    d.name = name;
    d.p1 = p1;
    d.p2 = p2;
    return d;
}

PopConfig base_config(double taste_sigma = 0.0) {
    PopConfig cfg;
    cfg.status = trait("lognormal", 0.0, 0.8);
    cfg.attractiveness = trait("normal", 0.5, 0.15);
    cfg.sociability = trait("normal", 0.5, 0.15);
    cfg.age = trait("normal", 30.0, 6.0);
    cfg.admissible_fraction = 0.2;
    cfg.taste_sigma = taste_sigma;
    return cfg;
}

UtilityParams base_params(double shock_scale = 0.0, std::uint64_t seed = 42) {
    UtilityParams p{1.0, 2.0, 0.3, 0.5, 1.5};
    p.shock_scale = shock_scale;
    p.shock_seed = seed;
    return p;
}

}  // namespace

// ── The shock function itself ───────────────────────────────────────────────

TEST_CASE("pairwise_shock is deterministic in (seed, key_i, key_j)", "[heterogeneity]") {
    const double a = pairwise_shock(42, 1000, 2000);
    CHECK(pairwise_shock(42, 1000, 2000) == a);
    // Same ordered pair, recomputed from scratch after unrelated calls: the
    // function holds no state, so nothing another call did can move it.
    (void)pairwise_shock(7, 3, 4);
    (void)pairwise_shock(42, 2000, 1000);
    CHECK(pairwise_shock(42, 1000, 2000) == a);

    // Every input actually participates.
    CHECK(pairwise_shock(43, 1000, 2000) != a);
    CHECK(pairwise_shock(42, 1001, 2000) != a);
    CHECK(pairwise_shock(42, 1000, 2001) != a);
}

TEST_CASE("pairwise_shock is asymmetric", "[heterogeneity]") {
    // i's chemistry with j is not j's chemistry with i. This is what hashing the
    // ORDERED pair buys, and without it half the model's heterogeneity would
    // collapse into a symmetric pair effect.
    int n_asymmetric = 0;
    constexpr int kN = 2000;
    for (int i = 0; i < kN; ++i) {
        const double forward = pairwise_shock(42, static_cast<std::uint64_t>(i),
                                              static_cast<std::uint64_t>(i + 50000));
        const double reverse = pairwise_shock(42, static_cast<std::uint64_t>(i + 50000),
                                              static_cast<std::uint64_t>(i));
        if (forward != reverse) {
            ++n_asymmetric;
        }
    }
    // "The vast majority" — a collision in the final double would be a 1-in-2^64
    // event, so requiring essentially all of them is not a loose bound.
    CHECK(n_asymmetric > kN * 999 / 1000);
}

TEST_CASE("pairwise_shock has standard-logistic moments", "[heterogeneity]") {
    // sd pi/sqrt(3) = 1.8138, mean 0. Over 100k pairs the sampling error on the
    // mean is sd/sqrt(n) = 0.0057, so 0.05 is a ~9-sigma window; the sd's own
    // relative error is about 0.2%, so 3% is generous. Neither bound is tight
    // enough to hide a wrong distribution though: a uniform or normal shock misses
    // the sd by a factor of 3 or 1.8.
    constexpr int kN = 100000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int t = 0; t < kN; ++t) {
        // Spread the keys so consecutive samples are not consecutive hash inputs.
        const std::uint64_t i = static_cast<std::uint64_t>(t) * 2654435761ULL;
        const std::uint64_t j = static_cast<std::uint64_t>(t) * 40503ULL + 7ULL;
        const double x = pairwise_shock(2026, i, j);
        sum += x;
        sum_sq += x * x;
    }
    const double mean = sum / kN;
    const double var = sum_sq / kN - mean * mean;
    const double sd = std::sqrt(var);

    CAPTURE(mean, sd);
    CHECK(std::abs(mean) < 0.05);
    CHECK(std::abs(sd - kLogisticSd) / kLogisticSd < 0.03);
}

// ── Backward compatibility ──────────────────────────────────────────────────

TEST_CASE("taste_sigma 0 leaves the traits bit-identical", "[heterogeneity]") {
    // The new columns are drawn from their own stream, so the four traits are
    // untouched by the existence of the feature. Without that, every number the
    // project has ever published would move.
    const std::vector<Agent> a = sample_population(400, base_config(0.0), 7);
    const std::vector<Agent> b = sample_population(400, base_config(0.5), 7);

    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].status == b[i].status);
        CHECK(a[i].attractiveness == b[i].attractiveness);
        CHECK(a[i].sociability == b[i].sociability);
        CHECK(a[i].age == b[i].age);
        // And the key does not move with sigma either, so chemistry is comparable
        // between a homogeneous and a heterogeneous population.
        CHECK(a[i].key == b[i].key);
    }
}

TEST_CASE("taste_sigma 0 leaves the utility bit-identical", "[heterogeneity]") {
    std::vector<Agent> pop = sample_population(200, base_config(0.5), 11);
    // Sampled with heterogeneity ON and then forced to the homogeneous
    // multipliers, so this checks the arithmetic identity (1.0 * x is exactly x)
    // rather than only the trivially-defaulted case: if the new expression were
    // (alpha * m) * z instead of the old alpha * z, rounding could differ here
    // even though the values agree to a decimal.
    for (Agent& a : pop) {
        a.taste_alpha = 1.0;
        a.taste_beta = 1.0;
        a.taste_gamma = 1.0;
    }
    const TraitStats stats = compute_trait_stats(pop);
    const UtilityParams p = base_params(0.0);

    for (const Agent& a : pop) {
        for (const Agent& b : pop) {
            // Recompute the pre-heterogeneity expression by hand and require exact
            // equality: 1.0*x is x in IEEE-754, and the shock is skipped entirely
            // at scale 0, so anything short of bit equality is a real change.
            const double zs = standardize(b.status, stats.status_mean, stats.status_sd);
            const double za =
                standardize(b.attractiveness, stats.attractiveness_mean, stats.attractiveness_sd);
            double expected = p.alpha * zs + p.beta * za - p.gamma * weighted_distance(a, b, stats);
            if (a.type == AgentType::W) {
                const double zs_self = standardize(a.status, stats.status_mean, stats.status_sd);
                expected += p.lambda * ((zs >= zs_self + p.delta) ? 1.0 : 0.0);
            }
            CHECK(a.utility_toward(b, p, stats) == expected);
        }
    }
}

// ── Taste multipliers ───────────────────────────────────────────────────────

TEST_CASE("taste multipliers have mean 1 and log-sd taste_sigma", "[heterogeneity]") {
    constexpr int kN = 2000;
    constexpr double kSigma = 0.5;
    const std::vector<Agent> pop = sample_population(kN, base_config(kSigma), 3);

    auto check = [&pop](const char* name, auto field) {
        double sum = 0.0;
        double log_sum = 0.0;
        double log_sq = 0.0;
        for (const Agent& a : pop) {
            const double m = a.*field;
            sum += m;
            const double l = std::log(m);
            log_sum += l;
            log_sq += l * l;
        }
        const double mean = sum / kN;
        const double log_mean = log_sum / kN;
        const double log_sd = std::sqrt(log_sq / kN - log_mean * log_mean);
        CAPTURE(name, mean, log_mean, log_sd);
        // Mean 1 by the -sigma^2/2 drift correction: within 3% over 2000 draws
        // (the standard error of the mean is sigma/sqrt(n) = 1.1%).
        CHECK(std::abs(mean - 1.0) < 0.03);
        // log is exactly normal(mu, sigma), so this pins the dispersion knob
        // itself: within 5% of 0.5 over 2000 draws (se = 0.79%).
        CHECK(std::abs(log_sd - kSigma) / kSigma < 0.05);
    };

    check("alpha", &Agent::taste_alpha);
    check("beta", &Agent::taste_beta);
    check("gamma", &Agent::taste_gamma);
}

TEST_CASE("zero taste_sigma pins every multiplier at exactly 1", "[heterogeneity]") {
    const std::vector<Agent> pop = sample_population(500, base_config(0.0), 5);
    for (const Agent& a : pop) {
        REQUIRE(a.taste_alpha == 1.0);
        REQUIRE(a.taste_beta == 1.0);
        REQUIRE(a.taste_gamma == 1.0);
    }
}

TEST_CASE("keys are distinct, below 2^53, and survive float64", "[heterogeneity]") {
    constexpr int kN = 2000;
    const std::vector<Agent> pop = sample_population(kN, base_config(0.5), 9);

    std::set<std::uint64_t> keys;
    for (const Agent& a : pop) {
        // 2^53 is the first integer a double cannot represent. Staying below it is
        // what makes the state-matrix round trip lossless.
        REQUIRE(a.key < (std::uint64_t{1} << 53));
        CHECK(static_cast<std::uint64_t>(static_cast<double>(a.key)) == a.key);
        keys.insert(a.key);
    }
    // A collision would give two agents identical chemistry with everyone, which
    // is a modeling error the sampler should never produce at this size.
    CHECK(keys.size() == static_cast<std::size_t>(kN));
}

// ── The shock inside the utility ────────────────────────────────────────────

TEST_CASE("chemistry is persistent across call sites and epochs", "[heterogeneity]") {
    // The whole point: generate_swipes, calibrate_thresholds,
    // build_full_info_pref_lists and compute_welfare must all see the same L(i,j).
    // They do because none of them stores it — all four call utility_toward, and
    // the value is a pure function of the keys. This test pins that the function
    // returns the same number when called from different places, including after
    // the agent has been flattened and read back, which is the path the Python
    // layer takes.
    std::vector<Agent> pop = sample_population(40, base_config(0.5), 13);
    const UtilityParams p = base_params(1.5);
    const TraitStats stats = compute_trait_stats(pop);

    std::vector<double> before;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        for (std::size_t j = 0; j < pop.size(); ++j) {
            before.push_back(pop[i].utility_toward(pop[j], p, stats));
        }
    }

    const std::vector<double> flat = flatten_states(pop);
    const std::vector<Agent> round_tripped = unflatten_states(flat.data(), static_cast<int>(pop.size()));
    const TraitStats stats2 = compute_trait_stats(round_tripped);

    std::size_t at = 0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        for (std::size_t j = 0; j < pop.size(); ++j) {
            CHECK(round_tripped[i].utility_toward(round_tripped[j], p, stats2) == before[at]);
            ++at;
        }
    }
}

TEST_CASE("a fresh key gives fresh chemistry and disturbs nobody else", "[heterogeneity]") {
    // Generational turnover draws a new key for the replaced agent. Its chemistry
    // with everyone must change, and every pair that does not involve it must not.
    std::vector<Agent> pop = sample_population(20, base_config(0.5), 21);
    const UtilityParams p = base_params(1.5);
    const TraitStats stats = compute_trait_stats(pop);

    std::map<std::pair<std::int32_t, std::int32_t>, double> before;
    for (const Agent& a : pop) {
        for (const Agent& b : pop) {
            before[{a.id, b.id}] = a.utility_toward(b, p, stats);
        }
    }

    const std::int32_t replaced = 3;
    std::vector<Agent> after = pop;
    after[static_cast<std::size_t>(replaced)].key ^= 0xDEADBEEFULL;

    int n_changed_involving = 0;
    int n_changed_elsewhere = 0;
    for (const Agent& a : after) {
        for (const Agent& b : after) {
            const double now = a.utility_toward(b, p, stats);
            if (a.id == replaced || b.id == replaced) {
                if (now != before[{a.id, b.id}]) {
                    ++n_changed_involving;
                }
            } else if (now != before[{a.id, b.id}]) {
                ++n_changed_elsewhere;
            }
        }
    }
    CHECK(n_changed_elsewhere == 0);
    // Both directions for every other agent, minus the id==id diagonal case that
    // the XOR might coincidentally preserve.
    CHECK(n_changed_involving > 30);
}

TEST_CASE("shock_scale 0 skips the term rather than adding zero", "[heterogeneity]") {
    // A numeric no-op and a skipped branch differ only in cost, so the observable
    // assertion is that two agents with wildly different keys still get the same
    // utility from the same traits when the scale is 0.
    std::vector<Agent> pop = sample_population(10, base_config(0.5), 31);
    const TraitStats stats = compute_trait_stats(pop);
    const UtilityParams on = base_params(0.0);

    for (Agent& a : pop) {
        a.key = 0;
        a.taste_alpha = 1.0;
        a.taste_beta = 1.0;
        a.taste_gamma = 1.0;
        a.status = 1.0;
        a.attractiveness = 0.5;
        a.sociability = 0.5;
        a.age = 30.0;
        a.type = AgentType::M;
    }
    // Identical agents, but the shock would still differ had it been computed from
    // keys that were not equal. Set unequal keys explicitly.
    pop[1].key = 123456789ULL;
    const TraitStats s2 = compute_trait_stats(pop);
    CHECK(pop[0].utility_toward(pop[1], on, s2) == pop[0].utility_toward(pop[0], on, s2));
}

// ── Swipes under heterogeneity ──────────────────────────────────────────────

TEST_CASE("generate_swipes is thread-count independent with heterogeneity on",
          "[heterogeneity][simulation]") {
    constexpr int kN = 300;
    constexpr int kK = 4;

    const std::vector<Agent> pop = sample_population(kN, base_config(0.5), 42);
    const UtilityParams p = base_params(1.5);

    std::vector<Agent> calibrated = pop;
    // θ is per-agent and must be calibrated before the gate means anything; the
    // calibration itself now goes through the shock, which is the point.
    const TraitStats stats = compute_trait_stats(calibrated);
    calibrate_thresholds(calibrated, p, 0.2, stats);

    std::vector<Agent> recopy = pop;
    const TraitStats stats2 = compute_trait_stats(recopy);
    calibrate_thresholds(recopy, p, 0.2, stats2);
    // Calibration is deterministic given the population, even with the shock.
    for (std::size_t i = 0; i < calibrated.size(); ++i) {
        REQUIRE(calibrated[i].threshold == recopy[i].threshold);
    }

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto serial = generate_swipes(calibrated, p, stats, kK, 2024, 3);

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel = generate_swipes(calibrated, p, stats, kK, 2024, 3);

#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif

    REQUIRE(serial.size() == parallel.size());
    CHECK(serial == parallel);
}

TEST_CASE("heterogeneity spreads exposure across the market", "[heterogeneity]") {
    // The concentration result. Homogeneous utilities mean every agent's shortlist
    // is roughly the same shortlist, so the same few people are shortlisted by
    // everyone; per-agent tastes and pair-specific chemistry break that. Measured
    // at N=1000, k=20: reach ~60% and Gini < 0.85 with the base values, against
    // ~8% and ~0.96 without.
    //
    // This is the test that makes the control arm meaningful, so it asserts BOTH
    // worlds in their own right, not just their ordering: a version that returned
    // the same thing at either setting would pass a relative-only assertion
    // vacuously.
    constexpr int kN = 1000;
    constexpr int kK = 20;
    constexpr std::uint64_t kSeed = 42;

    auto measure = [](double taste_sigma, double shock_scale) {
        std::vector<Agent> pop = sample_population(kN, base_config(taste_sigma), kSeed);
        const UtilityParams p = base_params(shock_scale, kSeed);
        const TraitStats stats = compute_trait_stats(pop);
        calibrate_thresholds(pop, p, 0.2, stats);

        const auto swipes = generate_swipes(pop, p, stats, kK, kSeed, 0);

        std::map<std::int32_t, int> exposure;
        for (const auto& s : swipes) {
            if (s[2] == 1) {
                ++exposure[s[1]];
            }
        }
        const int reached = static_cast<int>(exposure.size());

        // Gini over the full population, agents with no shortlist appearance
        // included at zero — that is what makes visibility inequality visible.
        // Both sides belong in the denominator: every shortlist is written by one
        // side and read against the other, so men appear as items in women's
        // lists and women in men's, and each row has a real exposure count.
        std::vector<double> counts(static_cast<std::size_t>(kN), 0.0);
        for (const auto& [id, c] : exposure) {
            counts[static_cast<std::size_t>(id)] = static_cast<double>(c);
        }
        std::sort(counts.begin(), counts.end());
        double total = 0.0;
        double weighted = 0.0;
        for (int i = 0; i < kN; ++i) {
            total += counts[static_cast<std::size_t>(i)];
            weighted += static_cast<double>(i + 1) * counts[static_cast<std::size_t>(i)];
        }
        const double g = (2.0 * weighted) / (kN * total) - (kN + 1.0) / static_cast<double>(kN);
        return std::pair<int, double>{reached, g};
    };

    const auto [reach_hom, gini_hom] = measure(0.0, 0.0);
    const auto [reach_het, gini_het] = measure(0.5, 1.5);

    CAPTURE(reach_hom, gini_hom, reach_het, gini_het);
    // The heterogeneous market: at least 60% of agents appear on someone's
    // shortlist, and the oracle Gini is below 0.85. Measured 95.5% and 0.664.
    CHECK(reach_het >= 600);
    CHECK(gini_het < 0.85);
    // The homogeneous market, pinned in its own right rather than only as a
    // comparison: under 10% reach and Gini above 0.90. Measured 6.9% and 0.958.
    // Asserting the absolute values is what stops the test passing vacuously —
    // if a future change collapsed both arms to the same concentrated market, the
    // two relative comparisons below would still hold.
    CHECK(reach_hom < 100);
    CHECK(gini_hom > 0.90);
    // And the two worlds must be far apart, not merely ordered.
    CHECK(reach_het > 5 * std::max(reach_hom, 1));
    CHECK(gini_hom - gini_het > 0.20);
}
