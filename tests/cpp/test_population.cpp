#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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

namespace {

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

// A threshold below every utility. Tests that care about *ranking* rather than
// about the θ gate set this, so the admissible set is the whole market and the
// expected top-k is checkable by hand.
void admit_everyone(std::vector<Agent>& pop) {
    for (Agent& a : pop) {
        a.threshold = kUncalibratedThreshold;
    }
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

int count_positives(const std::vector<std::array<std::int32_t, 3>>& swipes) {
    int n = 0;
    for (const auto& s : swipes) {
        n += (s[2] == 1) ? 1 : 0;
    }
    return n;
}

// The standardization stats for a population, which is what every production
// call site passes. Recomputed rather than cached because these tests build a
// fresh population per case and the stats have to follow it.
TraitStats stats_of(const std::vector<Agent>& pop) { return compute_trait_stats(pop); }

}  // namespace

// ── population sampling ──────────────────────────────────────────────────────

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
    // The tolerance is a real one: four standard errors of the sample mean under
    // the lognormal, not a hand-tuned window.
    //
    //   E[X]    = exp(mu + sigma^2 / 2)
    //   sd(X)   = exp(mu + sigma^2 / 2) * sqrt(exp(sigma^2) - 1)
    //   se(mean)= sd(X) / sqrt(n)
    //
    // Both halves of a Box–Muller pair are iid standard normals, so there is no
    // biased half and no truncated-normal correction to apply — and each agent
    // consumes exactly four normal draws in this configuration, so the parity of
    // an agent's index separates nothing at all.
    constexpr int kN = 10000;
    constexpr double kMu = 0.0;
    constexpr double kSigma = 0.8;

    const std::vector<Agent> pop = sample_population(kN, config(kMu, kSigma), 5);
    REQUIRE(pop.size() == static_cast<std::size_t>(kN));

    double sum = 0.0;
    for (const Agent& a : pop) {
        sum += a.status;
    }
    const double mean = sum / static_cast<double>(kN);

    const double expected = std::exp(kMu + 0.5 * kSigma * kSigma);
    const double sd = expected * std::sqrt(std::exp(kSigma * kSigma) - 1.0);
    const double tolerance = 4.0 * sd / std::sqrt(static_cast<double>(kN));

    CAPTURE(mean, expected, sd, tolerance);
    CHECK(std::abs(mean - expected) < tolerance);
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

// ── flat state matrix ───────────────────────────────────────────────────────

TEST_CASE("flatten and unflatten round-trip bit for bit", "[population][states]") {
    // Every field is overwritten with a value that differs from its default and
    // from its neighbor's, so a column written to the wrong offset or a field
    // dropped by the conversion shows up as a mismatch rather than as a value that
    // happens to coincide.
    std::vector<Agent> pop(8);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        const double k = static_cast<double>(i) + 1.0;
        Agent& a = pop[i];
        a.id = static_cast<std::int32_t>(100 + i);
        a.type = (i % 2 == 0) ? AgentType::M : AgentType::W;
        a.status = 0.1 * k * k;
        a.attractiveness = 1.0 - 0.11 * k;
        a.sociability = 0.05 * k + 0.001;
        a.age = 18.0 + 3.5 * k;
        a.threshold = -0.25 * k;
    }

    const std::vector<double> flat = flatten_states(pop);
    REQUIRE(flat.size() == pop.size() * 7);

    // Spot-check the column layout itself against docs/INTERFACES.md §1, so a
    // transposed row in flatten_states is caught even if unflatten_states
    // transposes it back.
    for (std::size_t i = 0; i < pop.size(); ++i) {
        const double* row = &flat[i * 7];
        CHECK(row[0] == static_cast<double>(pop[i].id));
        CHECK(row[1] == ((pop[i].type == AgentType::W) ? 1.0 : 0.0));
        CHECK(row[2] == pop[i].status);
        CHECK(row[3] == pop[i].attractiveness);
        CHECK(row[4] == pop[i].sociability);
        CHECK(row[5] == pop[i].age);
        CHECK(row[6] == pop[i].threshold);
    }

    const std::vector<Agent> back = unflatten_states(flat.data(), static_cast<int>(pop.size()));
    REQUIRE(back.size() == pop.size());

    for (std::size_t i = 0; i < pop.size(); ++i) {
        CHECK(back[i].id == pop[i].id);
        CHECK(back[i].type == pop[i].type);
        // Bit equality, not Approx: the conversion is a copy, so any drift is a bug.
        CHECK(back[i].status == pop[i].status);
        CHECK(back[i].attractiveness == pop[i].attractiveness);
        CHECK(back[i].sociability == pop[i].sociability);
        CHECK(back[i].age == pop[i].age);
        CHECK(back[i].threshold == pop[i].threshold);
    }
}

TEST_CASE("flatten preserves non-finite thresholds", "[population][states]") {
    // θ defaults to -infinity before calibration, so the conversion has to carry
    // that value rather than normalize it. The config helper sets 0.25 for the
    // other tests, so this one builds a PopConfig that leaves init_threshold at
    // its default and checks that the default really is the uncalibrated value.
    PopConfig cfg = config(0.0, 0.8);
    cfg.init_threshold = kUncalibratedThreshold;

    const std::vector<Agent> pop = sample_population(16, cfg, 2);
    for (const Agent& a : pop) {
        REQUIRE(a.threshold == kUncalibratedThreshold);
    }

    const std::vector<double> flat = flatten_states(pop);
    const std::vector<Agent> back = unflatten_states(flat.data(), 16);
    for (const Agent& a : back) {
        CHECK(std::isinf(a.threshold));
        CHECK(a.threshold < 0.0);
    }
}

// ── threshold calibration ───────────────────────────────────────────────────

TEST_CASE("calibration admits roughly the requested fraction of candidates",
          "[population][threshold]") {
    constexpr int kN = 400;
    constexpr int kCandidates = kN / 2;
    const UtilityParams p = status_only();

    std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 31);
    const TraitStats stats = stats_of(pop);
    calibrate_thresholds(pop, p, 0.25, stats);

    // Count, per agent, how many candidates clear its calibrated θ.
    double total_fraction = 0.0;
    for (const Agent& user : pop) {
        int admissible = 0;
        for (const Agent& cand : pop) {
            if (cand.type == user.type) {
                continue;
            }
            admissible += (user.utility_toward(cand, p, stats) >= user.threshold) ? 1 : 0;
        }
        total_fraction += static_cast<double>(admissible) / kCandidates;
    }
    const double mean_fraction = total_fraction / kN;

    CAPTURE(mean_fraction);
    // The quantile lands on an actual candidate, so the realised fraction differs
    // from 0.25 by at most one candidate per agent.
    CHECK(std::abs(mean_fraction - 0.25) < 0.02);
}

TEST_CASE("calibration admits exactly lround(f*n) candidates when utilities differ",
          "[population][threshold]") {
    // Pins the order-statistic index arithmetic, which is where an off-by-one
    // would live. Utilities are made distinct on both sides so "the top f share"
    // is unambiguous; the tie case is a documented limitation, not tested here.
    for (const int n_side : {5, 10, 20, 100}) {
        const int n = 2 * n_side;
        std::vector<Agent> base(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            Agent& a = base[static_cast<std::size_t>(i)];
            a.id = i;
            a.type = (i < n_side) ? AgentType::M : AgentType::W;
            a.attractiveness = 0.0;
            a.sociability = 0.0;
            a.age = 0.0;
            a.status = static_cast<double>(i % n_side) * 0.37 + 0.01 * i;
        }
        const UtilityParams p{1.0, 0.0, 0.0, 0.0, 0.0};

        for (const double fraction : {0.05, 0.1, 0.2, 0.5, 1.0}) {
            std::vector<Agent> pop = base;
            const TraitStats stats = stats_of(pop);
            calibrate_thresholds(pop, p, fraction, stats);

            const int expected =
                std::clamp(static_cast<int>(std::lround(fraction * n_side)), 1, n_side);

            for (const Agent& user : pop) {
                int admitted = 0;
                for (const Agent& cand : pop) {
                    if (cand.type == user.type) {
                        continue;
                    }
                    admitted += (user.utility_toward(cand, p, stats) >= user.threshold) ? 1 : 0;
                }
                CAPTURE(n_side, fraction, expected, admitted);
                CHECK(admitted == expected);
            }
        }
    }
}

TEST_CASE("a calibrated theta leaves some candidates on both sides of it",
          "[population][threshold]") {
    // The bug this guards against: a θ that is inert because it sits below every
    // utility (admit all) or above every utility (admit none).
    std::vector<Agent> pop = sample_population(200, config(0.0, 0.8), 12);
    const TraitStats stats = stats_of(pop);
    calibrate_thresholds(pop, status_only(), 0.2, stats);

    for (const Agent& user : pop) {
        int above = 0;
        int below = 0;
        for (const Agent& cand : pop) {
            if (cand.type == user.type) {
                continue;
            }
            (user.utility_toward(cand, status_only(), stats) >= user.threshold ? above : below) += 1;
        }
        CHECK(above > 0);
        CHECK(below > 0);
    }
}

// ── swipe generation ────────────────────────────────────────────────────────

TEST_CASE("generate_swipes gives k distinct positives per agent", "[simulation]") {
    constexpr int kN = 200;
    constexpr int kK = 5;

    std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 42);
    admit_everyone(pop);
    const std::vector<std::array<std::int32_t, 3>> swipes =
        generate_swipes(pop, status_only(), stats_of(pop), kK, 1234);

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
    admit_everyone(pop);
    // Status-as-utility makes the expected ranking a plain sort of the ids by the
    // candidate's status, which is what the assertion below recomputes. The
    // ranking is unchanged by standardization because z is a strictly increasing
    // affine function of the raw trait — only the *gaps* between utilities move,
    // not their order.
    const UtilityParams p = status_only();
    const auto swipes = generate_swipes(pop, p, stats_of(pop), kK, 77);

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

TEST_CASE("theta gates acceptance and the positive set shrinks as it rises",
          "[simulation][threshold]") {
    constexpr int kN = 200;
    constexpr int kK = 20;  // larger than the smallest admissible set below, so
                            // θ — not k — is what bounds the positive count
    const UtilityParams p = status_only();

    // The two extreme thresholds first: -inf admits the whole market, +inf
    // admits none. Before the gate was wired in these produced identical output,
    // because the positives were top-k by utility and ignored θ entirely.
    std::vector<Agent> open_all = sample_population(kN, config(0.0, 0.8), 55);
    admit_everyone(open_all);
    const int n_open = count_positives(generate_swipes(open_all, p, stats_of(open_all), kK, 9));

    std::vector<Agent> shut = sample_population(kN, config(0.0, 0.8), 55);
    for (Agent& a : shut) {
        a.threshold = std::numeric_limits<double>::infinity();
    }
    const int n_shut = count_positives(generate_swipes(shut, p, stats_of(shut), kK, 9));

    CAPTURE(n_open, n_shut);
    CHECK(n_open > 0);
    CHECK(n_shut == 0);

    // A rising θ must not increase the number of positives, and must eventually
    // shrink it. The ladder is expressed as admissible fractions of each agent's
    // own candidate pool, which is what calibration actually controls.
    std::vector<int> counts;
    for (const double fraction : {1.0, 0.10, 0.05, 0.02}) {
        std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 55);
        const TraitStats stats = stats_of(pop);
        calibrate_thresholds(pop, p, fraction, stats);
        counts.push_back(count_positives(generate_swipes(pop, p, stats, kK, 9)));
    }

    CAPTURE(counts);
    for (std::size_t i = 1; i < counts.size(); ++i) {
        CHECK(counts[i] < counts[i - 1]);
    }
}

TEST_CASE("a threshold above the k-th utility truncates the positive set",
          "[simulation][threshold]") {
    // The gate sits before the top-k, so its effect is to truncate: the positives
    // are the top-min(k, |admissible|) by utility. It can never reorder the ones
    // that survive, because the top-k of a filtered set is a prefix of the global
    // top-k. What this test pins down is that the truncation is real — that θ
    // above the k-th utility costs an agent positives rather than being ignored —
    // and that what survives is the better part of the list.
    constexpr int kN = 200;
    constexpr int kK = 3;
    constexpr int kCandidates = kN / 2;
    const UtilityParams p = status_only();

    std::vector<Agent> open_all = sample_population(kN, config(0.0, 0.8), 44);
    admit_everyone(open_all);
    const TraitStats open_stats = stats_of(open_all);
    const auto open_swipes = generate_swipes(open_all, p, open_stats, kK, 21);

    // 0.02 of 100 candidates rounds to two admissible, below k = 3.
    std::vector<Agent> tight = sample_population(kN, config(0.0, 0.8), 44);
    const TraitStats tight_stats = stats_of(tight);
    calibrate_thresholds(tight, p, 2.0 / kCandidates, tight_stats);
    const auto tight_swipes = generate_swipes(tight, p, tight_stats, kK, 21);

    std::map<std::int32_t, std::set<std::int32_t>> open_pos;
    std::map<std::int32_t, std::set<std::int32_t>> tight_pos;
    for (const auto& s : open_swipes) {
        if (s[2] == 1) {
            open_pos[s[0]].insert(s[1]);
        }
    }
    for (const auto& s : tight_swipes) {
        if (s[2] == 1) {
            tight_pos[s[0]].insert(s[1]);
        }
    }

    REQUIRE(tight_pos.size() == open_pos.size());
    for (const auto& [user, items] : tight_pos) {
        // Two positives instead of three, and both are a subset of the open run.
        CHECK(items.size() == 2);
        for (const std::int32_t item : items) {
            CHECK(open_pos[user].count(item) == 1);
        }
    }

    // Mean utility of the positives must rise when the low end is cut.
    auto mean_utility = [&](const std::vector<Agent>& pop, const TraitStats& stats,
                            const std::map<std::int32_t, std::set<std::int32_t>>& pos) {
        double sum = 0.0;
        int n = 0;
        for (const auto& [user, items] : pos) {
            for (const std::int32_t item : items) {
                sum += pop[static_cast<std::size_t>(user)].utility_toward(
                    pop[static_cast<std::size_t>(item)], p, stats);
                ++n;
            }
        }
        return n > 0 ? sum / n : 0.0;
    };

    const double u_open = mean_utility(open_all, open_stats, open_pos);
    const double u_tight = mean_utility(tight, tight_stats, tight_pos);
    CAPTURE(u_open, u_tight);
    CHECK(u_tight > u_open);
}

TEST_CASE("generate_swipes is identical at one and four threads", "[simulation]") {
    constexpr int kN = 300;
    constexpr int kK = 4;

    const std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 42);
    const UtilityParams p = status_only();
    const TraitStats stats = stats_of(pop);

    // Both calls are made for real; the thread count only decides the schedule.
    // If the per-agent seeding were keyed on a thread id instead of an agent id,
    // these two vectors would differ in their negative samples.
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto serial = generate_swipes(pop, p, stats, kK, 2024, 3);

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel = generate_swipes(pop, p, stats, kK, 2024, 3);

#ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif

    REQUIRE(serial.size() == parallel.size());
    CHECK(serial == parallel);
}

TEST_CASE("the epoch is part of the per-agent seed", "[simulation]") {
    // Negatives must move between epochs while staying reproducible: the same
    // (seed, epoch) pair has to reproduce exactly, and a new epoch has to produce
    // a different negative set. Otherwise the NCF trains on a frozen sample.
    constexpr int kN = 200;
    constexpr int kK = 4;

    std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 42);
    admit_everyone(pop);
    const UtilityParams p = status_only();
    const TraitStats stats = stats_of(pop);

    const auto epoch0 = generate_swipes(pop, p, stats, kK, 2024, 0);
    const auto epoch0_again = generate_swipes(pop, p, stats, kK, 2024, 0);
    const auto epoch1 = generate_swipes(pop, p, stats, kK, 2024, 1);

    // Same epoch: bit-identical, negatives included.
    CHECK(epoch0 == epoch0_again);

    // Positives are a function of (traits, θ) and must not move; negatives are
    // sampled and must.
    std::vector<std::array<std::int32_t, 3>> pos0;
    std::vector<std::array<std::int32_t, 3>> pos1;
    std::vector<std::array<std::int32_t, 3>> neg0;
    std::vector<std::array<std::int32_t, 3>> neg1;
    for (const auto& s : epoch0) {
        (s[2] == 1 ? pos0 : neg0).push_back(s);
    }
    for (const auto& s : epoch1) {
        (s[2] == 1 ? pos1 : neg1).push_back(s);
    }
    CHECK(pos0 == pos1);
    CHECK(neg0 != neg1);
}

TEST_CASE("negative sampling is uniform over the non-positive pool", "[simulation]") {
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
        a.threshold = kUncalibratedThreshold;
    }
    const UtilityParams p{1.0, 0.0, 0.0, 0.5, 1.5};
    // Every trait is constant across the population here, so every sd is 0 and
    // standardize() leaves each value untouched — the stats are the identity in
    // practice as well as in intent. That is the documented degenerate case, and
    // it keeps this test's hand-computed expectations valid.
    const TraitStats stats = stats_of(pop);

    // M's top-k are the lowest ids on the W side: 100..109.
    std::map<std::int32_t, int> counts;
    for (int trial = 0; trial < kTrials; ++trial) {
        const auto swipes =
            generate_swipes(pop, p, stats, kK, static_cast<std::uint64_t>(trial));
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

// ── the project's core hypothesis ───────────────────────────────────────────

TEST_CASE("delta changes who W swipes on: mean status of her positives rises",
          "[simulation][hypergamy]") {
    // Same population, same seed, same utility weights — only δ moves. If δ did
    // not reach the ranking, all three runs would produce the same positive set
    // and the hypergamy hypothesis would be untestable.
    //
    // Why beta > alpha here. U_W is alpha*z(s) + beta*z(a) - gamma*d + lambda*1[.].
    // The indicator is non-decreasing in status, so when alpha*z(s) dominates the
    // utility is monotone in status across the whole market, and adding lambda to
    // the men above the cutoff is a *constant* on the top-k — it shifts every
    // value in the shortlist by the same amount and reorders nothing. Measured on
    // this population at the old raw-scale base weights (alpha=1.0, beta=0.8),
    // raising δ from 0.0 to 1.0 moved 0 of 199 women's positive sets. Weighting
    // attractiveness over status gives status the span to matter: delta then
    // decides which high-status men are worth the distance, and the effect is
    // large and monotone.
    //
    // Since standardization, the same mechanism holds but the threshold moved: δ
    // has real power when beta/alpha >= 2, and base.yaml now ships alpha=1.0,
    // beta=2.0 for exactly that reason (docs/THEORY.md §2.2).
    constexpr int kN = 400;
    constexpr int kK = 5;
    constexpr double kAlpha = 0.3;
    constexpr double kBeta = 1.5;
    constexpr double kGamma = 0.3;
    constexpr double kLambda = 1.5;

    std::vector<double> mean_status;

    for (const double delta : {0.0, 0.5, 1.0}) {
        std::vector<Agent> pop = sample_population(kN, config(0.0, 0.8), 101);
        const UtilityParams p{kAlpha, kBeta, kGamma, delta, kLambda};
        const TraitStats stats = stats_of(pop);
        calibrate_thresholds(pop, p, 0.25, stats);

        const auto swipes = generate_swipes(pop, p, stats, kK, 500);

        double sum = 0.0;
        int n = 0;
        for (const auto& s : swipes) {
            const Agent& user = pop[static_cast<std::size_t>(s[0])];
            if (s[2] == 1 && user.type == AgentType::W) {
                sum += pop[static_cast<std::size_t>(s[1])].status;
                ++n;
            }
        }
        REQUIRE(n > 0);
        mean_status.push_back(sum / n);
    }

    CAPTURE(mean_status);
    CHECK(mean_status[1] > mean_status[0]);
    CHECK(mean_status[2] > mean_status[1]);
}
