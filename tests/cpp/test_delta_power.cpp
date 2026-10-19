#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "agent.h"
#include "population.h"

// ── The δ guardrail ──────────────────────────────────────────────────────────
// δ only has power when the indicator bisects the competitive set — that is, when
// men above the cutoff and men below it are both plausible candidates for the
// top-k. If status alone dominates, every man in a woman's top-k is above her
// cutoff (or every one is below), the indicator is a constant on the shortlist,
// and δ reorders nothing no matter what value it is given.
//
// Measured on the base trait distributions (lognormal(0, 0.8) status,
// normal(0.5, 0.15) attractiveness and sociability, normal(30, 6) age), 2000
// agents / 1000 candidates per side, k = 20, δ = 0.0 vs δ = 1.0, γ = 0.3,
// λ = 1.5:
//
//     alpha  beta  overlap
//     1.0    0.1   0.9978   inert
//     1.0    0.8   0.9661   nearly inert (the old pre-standardization default)
//     1.0    1.0   0.9374
//     1.0    1.5   0.8301
//     1.0    2.0   0.7615   some power — what base.yaml now ships
//     1.0    3.0   0.7366
//     0.6    1.5   0.6278
//     0.3    1.5   0.4994   strong
//
// Rule of thumb: δ has real power when beta/alpha >= 2. The numbers here were
// re-measured after standardization and are the ones the project should cite;
// the hidden [.measure] case at the bottom regenerates them after a retune.
//
// The two tests below are a pair on purpose. The first fails if someone retunes
// alpha/beta and re-kills δ; the second fails if the first is written in a way
// that can never fail — it pins the "δ is dead" end of the scale.

namespace {

constexpr int kAgents = 2000;      // 1000 M + 1000 W
constexpr int kCandidates = 1000;  // one side
constexpr int kTopK = 20;          // ncf.top_k in base.yaml
constexpr std::uint64_t kSeed = 20261019;

TraitDist trait(const std::string& name, double p1, double p2) {
    TraitDist d;
    d.name = name;
    d.p1 = p1;
    d.p2 = p2;
    return d;
}

// The trait distributions from experiments/base.yaml, so the guardrail is
// measured on the same population the experiments actually run. If base.yaml's
// trait block changes materially, re-run [.measure] and update the table above.
PopConfig base_pop_config() {
    PopConfig cfg;
    cfg.status = trait("lognormal", 0.0, 0.8);
    cfg.attractiveness = trait("normal", 0.5, 0.15);
    cfg.sociability = trait("normal", 0.5, 0.15);
    cfg.age = trait("normal", 30.0, 6.0);
    cfg.admissible_fraction = 0.2;
    return cfg;
}

std::vector<Agent> sample_base_population() {
    PopConfig cfg = base_pop_config();
    return sample_population(kAgents, cfg, kSeed);
}

// The top-k candidate ids for one user under one utility weighting, ranked by
// utility with the same id tie-break generate_swipes uses. θ is deliberately not
// applied: this measures whether δ can reorder the ranking at all, which is a
// property of the weights, not of where a particular woman's cutoff sits.
std::set<std::int32_t> top_k_ids(const std::vector<Agent>& pop, const Agent& user, int k,
                                 const UtilityParams& p, const TraitStats& stats) {
    std::vector<std::pair<double, std::int32_t>> scored;
    scored.reserve(pop.size());
    for (const Agent& cand : pop) {
        if (cand.type == user.type) {
            continue;
        }
        scored.emplace_back(user.utility_toward(cand, p, stats), cand.id);
    }
    const std::size_t n = std::min(static_cast<std::size_t>(k), scored.size());
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(n), scored.end(),
                      [](const auto& a, const auto& b) {
                          return a.first != b.first ? a.first > b.first : a.second < b.second;
                      });
    std::set<std::int32_t> ids;
    for (std::size_t i = 0; i < n; ++i) {
        ids.insert(scored[i].second);
    }
    return ids;
}

// Mean share of each woman's top-k that survives a δ change, averaged over all
// women. 1.0 means δ reorders nothing anywhere; 0.95 means 5% of the shortlist
// moves, which is the guardrail's floor.
double mean_top_k_overlap(const std::vector<Agent>& pop, const TraitStats& stats, double alpha,
                          double beta, double gamma, double lambda) {
    const UtilityParams delta_0{alpha, beta, gamma, 0.0, lambda};
    const UtilityParams delta_1{alpha, beta, gamma, 1.0, lambda};

    double total = 0.0;
    int n_women = 0;
    for (const Agent& user : pop) {
        if (user.type != AgentType::W) {
            continue;
        }
        const std::set<std::int32_t> a = top_k_ids(pop, user, kTopK, delta_0, stats);
        const std::set<std::int32_t> b = top_k_ids(pop, user, kTopK, delta_1, stats);
        std::vector<std::int32_t> shared;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(shared));
        total += static_cast<double>(shared.size()) / static_cast<double>(kTopK);
        ++n_women;
    }
    REQUIRE(n_women > 0);
    return total / static_cast<double>(n_women);
}

}  // namespace

TEST_CASE("at the base weights delta moves at least 5% of the top-k shortlist",
          "[simulation][hypergamy][guardrail]") {
    // The base config's weights: alpha=1.0 keeps status the dominant linear term
    // (the Matthew-effect story), beta=2.0 puts δ over the power threshold. If a
    // future retune drops beta back toward alpha, δ goes inert again and every
    // sweep over δ silently measures nothing — this test is the alarm.
    constexpr double kAlpha = 1.0;
    constexpr double kBeta = 2.0;
    constexpr double kGamma = 0.3;
    constexpr double kLambda = 1.5;

    const std::vector<Agent> pop = sample_base_population();
    const TraitStats stats = compute_trait_stats(pop);

    const double overlap = mean_top_k_overlap(pop, stats, kAlpha, kBeta, kGamma, kLambda);

    CAPTURE(overlap, kAlpha, kBeta);
    // Measured 0.7615 at these weights. The 0.95 floor is deliberately loose: a
    // different toolchain or a modest trait retune should not trip it, but a
    // revert to beta <= alpha (0.9661 at the old beta=0.8) fails both this check
    // and the 5% requirement below.
    CHECK(overlap < 0.95);
    // The effect must be real, not a floating-point artifact: at least 5% of the
    // shortlist differs.
    CHECK(1.0 - overlap > 0.05);
}

TEST_CASE("at beta well below alpha delta is inert, so the guardrail is not vacuous",
          "[simulation][hypergamy][guardrail]") {
    // The inverse test. With alpha=1.0 and beta=0.1, status dominates so heavily
    // that the indicator is constant on almost every shortlist and δ reorders
    // almost nothing. A guardrail that passed even here would be measuring
    // nothing, so this pair proves the threshold distinguishes "δ has power" from
    // "δ is dead".
    //
    // Measured 0.9978, not exactly 1.000: the residual 0.22% is the handful of
    // women in the top status tail whose cutoff s_w + δ genuinely bisects their
    // candidate set even at these weights — the same mechanism the guardrail
    // tests, confined to the extreme tail. It goes to exactly 1.000 only when the
    // distance term is removed as well (gamma = 0, beta = 0), where utility is a
    // strict monotone function of candidate status. The assertion is therefore a
    // near-inertness bound rather than an equality; the equality is pinned
    // separately below.
    constexpr double kAlpha = 1.0;
    constexpr double kBeta = 0.1;
    constexpr double kGamma = 0.3;
    constexpr double kLambda = 1.5;

    const std::vector<Agent> pop = sample_base_population();
    const TraitStats stats = compute_trait_stats(pop);

    const double overlap = mean_top_k_overlap(pop, stats, kAlpha, kBeta, kGamma, kLambda);

    CAPTURE(overlap, kAlpha, kBeta);
    CHECK(overlap > 0.99);
    CHECK(overlap > 0.95);  // i.e. it is on the FAILING side of the guardrail above
}

TEST_CASE("with status as the only signal delta reorders nothing at all",
          "[simulation][hypergamy][guardrail]") {
    // The exact end of the scale. With beta = gamma = 0 the utility is a strictly
    // increasing function of candidate status, so every woman's top-k is a status
    // prefix. That prefix is either entirely above her cutoff or entirely below
    // it, the indicator is constant on it, and changing δ changes nothing —
    // overlap is exactly 1.0, bit for bit.
    //
    // This is what makes the main guardrail meaningful: it can distinguish a
    // configuration where δ is structurally incapable of mattering from one where
    // it moves a quarter of every shortlist.
    const std::vector<Agent> pop = sample_base_population();
    const TraitStats stats = compute_trait_stats(pop);

    const double overlap = mean_top_k_overlap(pop, stats, /*alpha=*/1.0, /*beta=*/0.0,
                                              /*gamma=*/0.0, /*lambda=*/1.5);

    CAPTURE(overlap);
    CHECK(overlap == 1.0);
}

TEST_CASE("standardization makes alpha and beta comparable weights",
          "[population][standardize]") {
    // The bug this replaces: on raw traits, alpha=1.0 and beta=1.0 made status
    // about 11x the weight of attractiveness, because status sd was 1.31 and
    // attractiveness sd 0.15. After standardization both traits have unit sd, so
    // equal weights mean equal influence — up to the distance term, which is the
    // s-enters-twice coupling THEORY.md §2.1 documents.
    const std::vector<Agent> pop = sample_base_population();
    const TraitStats stats = compute_trait_stats(pop);

    auto spread = [&](double Agent::*field, double mean, double sd) {
        double lo = 0.0;
        double hi = 0.0;
        bool first = true;
        for (const Agent& a : pop) {
            const double z = standardize(a.*field, mean, sd);
            if (first) {
                lo = hi = z;
                first = false;
            }
            lo = std::min(lo, z);
            hi = std::max(hi, z);
        }
        return hi - lo;
    };

    const double status_spread = spread(&Agent::status, stats.status_mean, stats.status_sd);
    const double attractiveness_spread =
        spread(&Agent::attractiveness, stats.attractiveness_mean, stats.attractiveness_sd);

    CAPTURE(status_spread, attractiveness_spread);
    // Both are z-scores, so both spreads are O(1) and comparable — the raw scales
    // differed by two orders of magnitude (status ~40, attractiveness ~0.8).
    CHECK(status_spread > 2.0);
    CHECK(status_spread < 12.0);
    CHECK(attractiveness_spread > 1.5);
    CHECK(attractiveness_spread < 12.0);

    // Equal weights must rank by z(s) + z(a), not by raw s + a. Pinning one
    // concrete value catches a regression to raw-scale arithmetic.
    const UtilityParams equal{1.0, 1.0, 0.0, 0.0, 0.0};
    const Agent& user = pop.back();
    std::vector<std::pair<double, std::int32_t>> scored;
    for (const Agent& cand : pop) {
        if (cand.type != user.type) {
            scored.emplace_back(user.utility_toward(cand, equal, stats), cand.id);
        }
    }
    std::sort(scored.begin(), scored.end(), [](const auto& x, const auto& y) {
        return x.first > y.first;
    });
    const Agent& top = pop[static_cast<std::size_t>(scored.front().second)];
    const double expected = standardize(top.status, stats.status_mean, stats.status_sd) +
                            standardize(top.attractiveness, stats.attractiveness_mean,
                                        stats.attractiveness_sd);
    CHECK(scored.front().first == Catch::Approx(expected));
}

// Hidden by default (the leading dot keeps it out of the normal run). Run it after
// retuning the weights or the trait block to regenerate the table at the top:
//     test_delta_power.exe "[.measure]"
TEST_CASE("measure delta overlap across weight settings", "[.measure]") {
    const std::vector<Agent> pop = sample_base_population();
    const TraitStats stats = compute_trait_stats(pop);

    struct W {
        double alpha;
        double beta;
        double gamma;
    };
    const std::vector<W> grid = {{1.0, 0.1, 0.3},  {1.0, 0.8, 0.3},  {1.0, 1.0, 0.3},
                                 {1.0, 1.5, 0.3},  {1.0, 2.0, 0.3},  {1.0, 3.0, 0.3},
                                 {0.6, 1.5, 0.3},  {0.3, 1.5, 0.3},  {1.0, 0.0, 0.0}};
    for (const W& w : grid) {
        const double overlap = mean_top_k_overlap(pop, stats, w.alpha, w.beta, w.gamma, 1.5);
        WARN("alpha=" << w.alpha << " beta=" << w.beta << " gamma=" << w.gamma
                      << " overlap=" << overlap);
    }
}
