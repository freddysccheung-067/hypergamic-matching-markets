#include "population.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// 2^-53: multiplying a 53-bit integer by this is the exact inverse of taking the
// top 53 bits of engine() output, so the result lands uniformly in [0, 1) with no
// rounding at all. Spelled as a hex float literal because the decimal form is not
// exactly representable.
constexpr double kInvTwoTo53 = 0x1.0p-53;

// A uniform on [0, 1) built straight from the engine.
//
// std::generate_canonical is the obvious choice and the wrong one here: the
// standard leaves both the number of engine calls and the rounding
// implementation-defined, so the same seed yields different populations under
// MSVC and libstdc++. The project's results are meant to survive a compiler
// change, so the mapping is spelled out instead. mt19937_64 returns 64 bits; the
// top 53 are exactly the mantissa width of a double.
double uniform01(std::mt19937_64& engine) {
    return static_cast<double>(engine() >> 11) * kInvTwoTo53;
}

// A uniform on (0, 1]. Used wherever the value is about to be logged or raised to
// a negative power, both of which are undefined or infinite at exactly 0.
double uniform_open_closed(std::mt19937_64& engine) {
    return 1.0 - uniform01(engine);
}

// Box–Muller rather than std::normal_distribution, for the same reason as above:
// the standard library's algorithm is implementation-defined, so the same seed
// would produce a different population under a different toolchain. Two uniforms
// per pair of normals, with the second kept for the next call.
class NormalSource {
public:
    explicit NormalSource(std::mt19937_64& engine) : engine_(engine) {}

    double operator()() {
        if (spare_ready_) {
            spare_ready_ = false;
            return spare_;
        }
        const double u1 = uniform_open_closed(engine_);
        const double u2 = uniform01(engine_);
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = kTwoPi * u2;
        spare_ = radius * std::sin(angle);
        spare_ready_ = true;
        return radius * std::cos(angle);
    }

private:
    std::mt19937_64& engine_;
    bool spare_ready_ = false;
    double spare_ = 0.0;
};

// Validate every trait before the loop, so a typo in the config fails on an
// empty population rather than halfway through.
void validate(const TraitDist& d, const char* field) {
    if (d.name != "normal" && d.name != "lognormal" && d.name != "pareto") {
        throw std::invalid_argument(std::string("sample_population: unsupported distribution '") +
                                    d.name + "' for trait '" + field +
                                    "' (expected normal, lognormal or pareto)");
    }
    if (d.name == "pareto" && (d.p1 <= 0.0 || d.p2 <= 0.0)) {
        throw std::invalid_argument(std::string("sample_population: pareto trait '") + field +
                                    "' requires x_min > 0 and alpha > 0");
    }
}

double draw(const TraitDist& d, std::mt19937_64& engine, NormalSource& normal) {
    if (d.name == "normal") {
        return d.p1 + d.p2 * normal();
    }
    if (d.name == "lognormal") {
        return std::exp(d.p1 + d.p2 * normal());
    }
    // Pareto inverse CDF. u is drawn on (0, 1] so x >= x_min and the result is
    // never infinite.
    return d.p1 * std::pow(uniform_open_closed(engine), -1.0 / d.p2);
}

// Column offsets for the flat matrix, in the order docs/INTERFACES.md §1 fixes.
constexpr std::size_t kCols = 7;

}  // namespace

std::vector<Agent> sample_population(int n, const PopConfig& cfg, std::uint64_t seed) {
    validate(cfg.status, "status");
    validate(cfg.attractiveness, "attractiveness");
    validate(cfg.sociability, "sociability");
    validate(cfg.age, "age");

    if (n < 0) {
        n = 0;
    }
    if (n % 2 != 0) {
        std::cerr << "sample_population: n=" << n
                  << " is odd; dropping the last agent so the two sides are equal\n";
        --n;
    }

    const int half = n / 2;
    std::vector<Agent> pop(static_cast<std::size_t>(n));

    // The only generator in the function: every trait draw for every agent comes
    // from this stream, so (n, cfg, seed) is fully reproducible.
    std::mt19937_64 engine(seed);
    NormalSource normal(engine);

    for (int i = 0; i < n; ++i) {
        Agent& a = pop[static_cast<std::size_t>(i)];
        a.id = i;
        a.type = (i < half) ? AgentType::M : AgentType::W;
        a.status = draw(cfg.status, engine, normal);
        a.attractiveness = std::clamp(draw(cfg.attractiveness, engine, normal), 0.0, 1.0);
        a.sociability = std::clamp(draw(cfg.sociability, engine, normal), 0.0, 1.0);
        a.age = draw(cfg.age, engine, normal);
        a.threshold = cfg.init_threshold;
    }
    return pop;
}

std::vector<double> flatten_states(const std::vector<Agent>& pop) {
    std::vector<double> out(pop.size() * kCols);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        const Agent& a = pop[i];
        double* row = &out[i * kCols];
        row[0] = static_cast<double>(a.id);
        row[1] = (a.type == AgentType::W) ? 1.0 : 0.0;
        row[2] = a.status;
        row[3] = a.attractiveness;
        row[4] = a.sociability;
        row[5] = a.age;
        row[6] = a.threshold;
    }
    return out;
}

std::vector<Agent> unflatten_states(const double* data, int n) {
    if (data == nullptr || n <= 0) {
        return {};
    }
    std::vector<Agent> pop(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double* row = data + static_cast<std::size_t>(i) * kCols;
        Agent& a = pop[static_cast<std::size_t>(i)];
        a.id = static_cast<std::int32_t>(row[0]);
        a.type = (row[1] == 1.0) ? AgentType::W : AgentType::M;
        a.status = row[2];
        a.attractiveness = row[3];
        a.sociability = row[4];
        a.age = row[5];
        a.threshold = row[6];
    }
    return pop;
}

void calibrate_thresholds(std::vector<Agent>& pop, const UtilityParams& p,
                          double admissible_fraction) {
    // θ must leave at least one admissible candidate: a theta above every utility
    // would make the agent swipe on nothing, which is a different failure from
    // the one this replaces, not a fix for it.
    if (!(admissible_fraction > 0.0)) {
        admissible_fraction = 1.0;
    }
    if (admissible_fraction > 1.0) {
        admissible_fraction = 1.0;
    }

    std::vector<double> utilities;
    for (Agent& a : pop) {
        utilities.clear();
        for (const Agent& cand : pop) {
            if (cand.type == a.type) {
                continue;
            }
            utilities.push_back(a.utility_toward(cand, p));
        }
        if (utilities.empty()) {
            continue;  // a one-sided market has nothing to calibrate against
        }

        const std::size_t n_cand = utilities.size();
        std::size_t n_admissible = static_cast<std::size_t>(
            std::lround(admissible_fraction * static_cast<double>(n_cand)));
        n_admissible = std::clamp<std::size_t>(n_admissible, 1u, n_cand);

        // The admissible candidates are the top n_admissible utilities, so θ is
        // the smallest utility still admitted. Taking the value of an actual
        // candidate (rather than interpolating between two) keeps θ on the scale
        // the comparison in generate_swipes actually uses.
        //
        // nth_element, not sort: only this one order statistic is needed, so the
        // O(n) selection makes calibration O(N²) like generate_swipes rather than
        // O(N² log N).
        const std::size_t kth = n_cand - n_admissible;
        std::nth_element(utilities.begin(), utilities.begin() + static_cast<std::ptrdiff_t>(kth),
                         utilities.end());
        a.threshold = utilities[kth];
    }
}
