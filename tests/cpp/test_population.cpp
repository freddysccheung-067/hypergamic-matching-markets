#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "agent.h"
#include "population.h"
#include "simulation.h"

namespace {

// Mill's ratio. The lognormal mean is exp(mu + sigma^2 / 2) only when the normal
// draws are complete; Box–Muller's cached second value comes from the same pair of
// uniforms, so its mean is beta/sqrt(2pi) on its own. A sample that ended on an
// even count would therefore sit about 0.055 low at n = 10,000, sigma = 0.8 —
// enough to miss the test's 0.05 window. Reserving half the draws as spares keeps
// the two halves independent and the expectation exact.
constexpr double kMillRatio = 0.3989422804014327;

TraitDist trait(const std::string& name, double p1, double p2) {
    TraitDist d;
    d.name = name;
    d.p1 = p1;
    d.p2 = p2;
    return d;
}

PopConfig config(double status_mu, double status_sigma) {
    PopConfig cfg;
    cfg.status = trait("lognormal", status_mu, status_sigma);
    cfg.attractiveness = trait("normal", 0.5, 0.15);
    cfg.sociability = trait("normal", 0.5, 0.15);
    cfg.age = trait("normal", 30.0, 6.0);
    cfg.init_threshold = 0.25;
    return cfg;
}

std::vector<double> statuses(const std::vector<Agent>& pop) {
    std::vector<double> s;
    s.reserve(pop.size());
    for (const Agent& a : pop) {
        s.push_back(a.status);
    }
    return s;
}

// [user, item, label] rows from generate_swipes.
struct Row {
    std::int32_t user;
    std::int32_t item;
    std::int32_t label;
};

std::map<std::int32_t, std::vector<Row>> group_by_user(
    const std::vector<std::array<std::int32_t, 3>>& swipes) {
    std::map<std::int32_t, std::vector<Row>> by_user;
    for (const auto& s : swipes) {
        by_user[s[0]].push_back(Row{s[0], s[1], s[2]});
    }
    return by_user;
}

// Utilities handed to generate_swipes in the swipe tests. Only alpha is non-zero
// so the ranking is exactly descending candidate status, which makes the expected
// positive set checkable by hand.
UtilityParams status_only() {
    return UtilityParams{1.0, 0.0, 0.0, 0.5, 1.5};
}

}  // namespace

TEST_CASE("sample_population returns n agents split evenly between the types",
          "[population]") {
    const std::vector<Agent> pop = sample_population(200, config(0.0, 0.8), 42);

    REQUIRE(pop.size() == 200);

    int n_m = 0;
    int n_w = 0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        CHECK(pop[i].id == static_cast<std::int32_t>(i));
        if (pop[i].type == AgentType::M) {
            ++n_m;
            CHECK(pop[i].id < 100);
        } else {
            ++n_w;
            CHECK(pop[i].id >= 100);
        }
    }
    CHECK(n_m == 100);
    CHECK(n_w == 100);
}

TEST_CASE("odd n drops one agent and keeps the sides balanced", "[population]") {
    const std::vector<Agent> pop = sample_population(201, config(0.0, 0.8), 7);

    CHECK(pop.size() == 200);

    int n_m = 0;
    for (const Agent& a : pop) {
        n_m += (a.type == AgentType::M) ? 1 : 0;
    }
    CHECK(n_m == 100);
}

TEST_CASE("every agent starts at the configured threshold", "[population]") {
    const std::vector<Agent> pop = sample_population(64, config(0.0, 0.8), 3);

    for (const Agent& a : pop) {
        CHECK(a.threshold == Catch::Approx(0.25));
        CHECK(a.epoch_proposals == 0);
        CHECK(a.epoch_matches == 0);
    }
}

TEST_CASE("attractiveness and sociability are clamped into [0, 1]", "[population]") {
    // A wide standard deviation makes the normal tail land outside [0, 1] many
    // times, so the clamp is exercised on both ends rather than assumed.
    PopConfig cfg = config(0.0, 0.8);
    cfg.attractiveness = trait("normal", 0.5, 0.9);
    cfg.sociability = trait("normal", 0.5, 0.9);

    const std::vector<Agent> pop = sample_population(2000, cfg, 11);
    REQUIRE(pop.size() == 2000);

    bool saw_clamped_low = false;
    bool saw_clamped_high = false;
    for (const Agent& a : pop) {
        REQUIRE(a.attractiveness >= 0.0);
        REQUIRE(a.attractiveness <= 1.0);
        REQUIRE(a.sociability >= 0.0);
        REQUIRE(a.sociability <= 1.0);
        saw_clamped_low = saw_clamped_low || a.attractiveness == 0.0 || a.sociability == 0.0;
        saw_clamped_high = saw_clamped_high || a.attractiveness == 1.0 || a.sociability == 1.0;
    }
    CHECK(saw_clamped_low);
    CHECK(saw_clamped_high);
}

TEST_CASE("lognormal status has the expected mean", "[population]") {
    constexpr int kN = 10000;
    constexpr double kMu = 0.0;
    constexpr double kSigma = 0.8;

    const std::vector<Agent> pop = sample_population(kN, config(kMu, kSigma), 5);
    REQUIRE(pop.size() == static_cast<std::size_t>(kN));

    double sum = 0.0;
    for (const Agent& a : pop) {
        sum += a.status;
    }
    const double mean = sum / kN;

    // exp(mu + sigma^2 / 2): the lognormal mean, not exp(mu).
    const double expected = std::exp(kMu + 0.5 * kSigma * kSigma);
    CAPTURE(mean, expected);
    CHECK(std::abs(mean - expected) < 0.05);

    // The spare-draw correction above only pays off if the two halves of the
    // sample really have different expectations, so hold the halves to it.
    double sum_odd = 0.0;
    for (int i = 1; i < kN; i += 2) {
        sum_odd += pop[static_cast<std::size_t>(i)].status;
    }
    const double mean_odd = sum_odd / (kN / 2);
    const double expected_odd = expected - kSigma * kMillRatio / std::sqrt(static_cast<double>(kN));
    CAPTURE(mean_odd, expected_odd);
    CHECK(std::abs(mean_odd - expected_odd) < 0.05);
}

TEST_CASE("pareto status never falls below x_min", "[population]") {
    PopConfig cfg = config(0.0, 0.8);
    cfg.status = trait("pareto", 1.0, 2.5);

    const std::vector<Agent> pop = sample_population(1000, cfg, 19);
    for (const Agent& a : pop) {
        REQUIRE(a.status >= 1.0);
        REQUIRE(std::isfinite(a.status));
    }
}

TEST_CASE("an unknown distribution name is rejected", "[population]") {
    PopConfig cfg = config(0.0, 0.8);
    cfg.status = trait("cauchy", 0.0, 1.0);

    CHECK_THROWS_AS(sample_population(10, cfg, 1), std::invalid_argument);
}

TEST_CASE("the same seed reproduces the population exactly", "[population]") {
    const std::vector<Agent> first = sample_population(200, config(0.0, 0.8), 42);
    const std::vector<Agent> second = sample_population(200, config(0.0, 0.8), 42);
    const std::vector<Agent> other = sample_population(200, config(0.0, 0.8), 43);

    REQUIRE(first.size() == second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        // Exact equality, not Approx: reproducibility is bit-for-bit or it is not
        // reproducibility.
        CHECK(first[i].status == second[i].status);
        CHECK(first[i].attractiveness == second[i].attractiveness);
        CHECK(first[i].sociability == second[i].sociability);
        CHECK(first[i].age == second[i].age);
        CHECK(first[i].type == second[i].type);
    }

    CHECK(statuses(first) != statuses(other));
}

TEST_CASE("generate_swipes gives k distinct positives per agent", "[simulation]") {
    constexpr int kN = 200;
    constexpr int kK = 5;

    const std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 42);
    const std::vector<std::array<std::int32_t, 3>> swipes =
        generate_swipes(pop, status_only(), kK, 1234);

    const auto by_user = group_by_user(swipes);
    REQUIRE(by_user.size() == static_cast<std::size_t>(kN));

    for (const Agent& agent : pop) {
        const auto it = by_user.find(agent.id);
        REQUIRE(it != by_user.end());
        const std::vector<Row>& rows = it->second;

        int n_positives = 0;
        for (const Row& r : rows) {
            CHECK(r.user == agent.id);
            // Every swipe is across the market: an M never swipes on an M.
            CHECK(pop[static_cast<std::size_t>(r.item)].type != agent.type);
            CHECK((r.label == 0 || r.label == 1));
            n_positives += (r.label == 1) ? 1 : 0;
        }
        CHECK(n_positives == kK);

        // No item is swiped on twice by the same user, positive or negative.
        std::set<std::int32_t> items;
        for (const Row& r : rows) {
            items.insert(r.item);
        }
        CHECK(items.size() == rows.size());
    }
}

TEST_CASE("positives are exactly the top-k by utility", "[simulation]") {
    constexpr int kN = 100;
    constexpr int kK = 3;

    std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 8);
    // Status-as-utility makes the expected ranking a plain sort of the ids by the
    // candidate's status, which is what the assertion below recomputes.
    const UtilityParams p = status_only();
    const auto swipes = generate_swipes(pop, p, kK, 77);

    for (const Agent& user : pop) {
        std::vector<std::pair<double, std::int32_t>> ranked;
        for (const Agent& cand : pop) {
            if (cand.type == user.type) {
                continue;
            }
            ranked.emplace_back(cand.status, cand.id);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });

        std::set<std::int32_t> expected_top;
        for (int t = 0; t < kK; ++t) {
            expected_top.insert(ranked[static_cast<std::size_t>(t)].second);
        }

        std::set<std::int32_t> actual_positives;
        for (const auto& s : swipes) {
            if (s[0] == user.id && s[2] == 1) {
                actual_positives.insert(s[1]);
            }
        }
        CHECK(actual_positives == expected_top);
    }
}

TEST_CASE("generate_swipes is identical at one and four threads", "[simulation]") {
    constexpr int kN = 300;
    constexpr int kK = 4;

    const std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 42);
    const UtilityParams p = status_only();

    // Both calls are made for real; the thread count only decides the schedule.
    // If the per-agent seeding were keyed on a thread id instead of an agent id,
    // these two vectors would differ in their negative samples.
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto serial = generate_swipes(pop, p, kK, 2024);

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel = generate_swipes(pop, p, kK, 2024);

#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif

    REQUIRE(serial.size() == parallel.size());
    CHECK(serial == parallel);
}

TEST_CASE("negative sampling is uniform over the non-top-k pool", "[simulation]") {
    // All utilities equal, so "top-k" is decided purely by the id tie-break and
    // the remaining pool is known exactly. Any bias toward the head or tail of
    // that pool shows up as a badly uneven histogram.
    constexpr int kN = 200;
    constexpr int kK = 10;
    constexpr int kTrials = 400;

    std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 6);
    for (Agent& a : pop) {
        a.status = 0.0;
        a.attractiveness = 0.0;
        a.sociability = 0.0;
        a.age = 0.0;
    }
    const UtilityParams p{1.0, 0.0, 0.0, 0.5, 1.5};

    // M's top-k are the lowest ids on the W side: 100..109.
    std::map<std::int32_t, int> counts;
    for (int trial = 0; trial < kTrials; ++trial) {
        const auto swipes = generate_swipes(pop, p, kK, static_cast<std::uint64_t>(trial));
        for (const auto& s : swipes) {
            if (s[0] == 0 && s[2] == 0) {
                ++counts[s[1]];
            }
        }
    }

    // n_negatives is k * 4, so each trial contributes 40 draws: 400 trials give
    // 16000 draws over 90 eligible candidates, ~178 expected each with a standard
    // deviation of about 13. The window is roughly +/- 4.5 sigma, wide enough that
    // a correct sampler never trips it and narrow enough that drawing from the
    // wrong pool (including the top-k, or with replacement from the full list)
    // lands far outside it.
    int n_items = 0;
    for (const auto& [item, c] : counts) {
        (void)item;
        ++n_items;
        CHECK(c > 120);
        CHECK(c < 240);
    }
    CHECK(n_items == 90);
    for (std::int32_t id = 100; id < 110; ++id) {
        CHECK(counts.find(id) == counts.end());
    }
}
