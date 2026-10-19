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

// The same population with the base heterogeneity knobs on: taste_sigma 0.5 and
// the multipliers it draws stay on the agents, and shock_scale reaches the
// utility through the params the tests pass in.
std::vector<Agent> sample_heterogeneous_population() {
    PopConfig cfg = base_pop_config();
    cfg.taste_sigma = 0.5;
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
//
// `shock_scale` is threaded into the utility rather than bolted on here, so the
// "with noise" case measures δ's power on exactly the ranking the experiment
// uses. The per-agent tastes come from the population and the pair-specific
// chemistry from the keys; both call sites see the same L(i,j) because it is a
// pure function of the ordered pair, which is what keeps this an apples-to-apples
// comparison rather than two different markets.
double mean_top_k_overlap(const std::vector<Agent>& pop, const TraitStats& stats, double alpha,
                          double beta, double gamma, double lambda, double shock_scale = 0.0) {
    UtilityParams delta_0{alpha, beta, gamma, 0.0, lambda};
    UtilityParams delta_1{alpha, beta, gamma, 1.0, lambda};
    delta_0.shock_scale = shock_scale;
    delta_1.shock_scale = shock_scale;
    delta_0.shock_seed = kSeed;
    delta_1.shock_seed = kSeed;

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

TEST_CASE("the delta guardrail still holds at the base heterogeneity settings",
          "[simulation][hypergamy][guardrail][heterogeneity]") {
    // Noise erodes δ's power: the indicator has to out-shout a shock on every
    // pair, so the shortlist moves less than it would in the noise-free market.
    // That is not a bug, but it means the guardrail has to be checked at the real
    // values rather than assumed to carry over — a retune of taste_sigma or
    // shock_scale upward can quietly push δ back over the 0.95 floor and make
    // every sweep over it measure nothing.
    //
    // Measured here on the built code, women only, at base.yaml's settings
    // (taste_sigma 0.5, k=20, N=2000, seed 20261019), with shock_scale set to give
    // each ratio: 0.779 at 0, 0.778 at 0.5, 0.830 at 1.0, 0.904 at 2.0.
    //
    // Noise does erode delta's power: 0.779 noise-free to 0.830 at the shipped
    // ratio, and 0.904 by ratio 2.0. The last figure is close enough to the 0.95
    // floor that a further retune of shock_scale upward would cross it — which is
    // exactly why the guardrail is checked at the real values instead of assumed.
    // The binding case at the shipped settings is 0.830.
    constexpr double kAlpha = 1.0;
    constexpr double kBeta = 2.0;
    constexpr double kGamma = 0.3;
    constexpr double kLambda = 1.5;
    constexpr double kShockScale = 1.5;  // hypergamy.shock_scale in base.yaml

    const std::vector<Agent> pop = sample_heterogeneous_population();
    const TraitStats stats = compute_trait_stats(pop);

    const double overlap =
        mean_top_k_overlap(pop, stats, kAlpha, kBeta, kGamma, kLambda, kShockScale);

    CAPTURE(overlap, kShockScale);
    // The same floor the noise-free guardrail uses: the claim is that δ still has
    // power, not that it has the same amount of it.
    CHECK(overlap < 0.95);
    // And it must still do something, not merely sit under a floor by luck.
    CHECK(1.0 - overlap > 0.02);
}

TEST_CASE("noise erodes delta monotonically, so the knob has a ceiling",
          "[simulation][hypergamy][guardrail][heterogeneity]") {
    // δ's power falls as the chemistry grows: the indicator has to out-shout a
    // shock on every pair, and a larger shock wins more often. This pins the
    // direction and the rough size, and — more usefully — states in a test what
    // the docs say: there is a shock_scale above which δ goes inert, and it is
    // reachable from the shipped value.
    //
    // Measured, women only, N=2000, k=20: 0.779 at ratio 0, 0.778 at 0.5, 0.830
    // at 1.0, 0.904 at 2.0. The sd used to convert a ratio to a scale is the
    // base-config noise-free utility sd.
    constexpr double kAlpha = 1.0;
    constexpr double kBeta = 2.0;
    constexpr double kGamma = 0.3;
    constexpr double kLambda = 1.5;
    constexpr double kSignalSd = 2.70;  // measured at taste_sigma 0.5

    const std::vector<Agent> pop = sample_heterogeneous_population();
    const TraitStats stats = compute_trait_stats(pop);

    auto at_ratio = [&](double ratio) {
        const double scale = ratio * kSignalSd / 1.8138;
        return mean_top_k_overlap(pop, stats, kAlpha, kBeta, kGamma, kLambda, scale);
    };

    const double r0 = at_ratio(0.0);
    const double r1 = at_ratio(1.0);
    const double r2 = at_ratio(2.0);

    CAPTURE(r0, r1, r2);
    // Monotone in the noise ratio. Strict on the second step, which is where the
    // effect is large enough to be well clear of sampling noise.
    CHECK(r0 <= r1);
    CHECK(r1 < r2);
    // The shipped ratio is still under the floor; ratio 2.0 is not far from it.
    CHECK(r1 < 0.95);
    CHECK(r2 < 0.95);
    CHECK(r2 > 0.88);
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
