#include "population.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Box–Muller rather than std::normal_distribution. The standard library's
// algorithm is implementation-defined, so the same seed would produce a
// different population under MSVC than under libstdc++ — and the results in this
// project are meant to survive a change of compiler. Two uniforms per pair of
// normals, with the second kept for the next call.
class NormalSource {
public:
    explicit NormalSource(std::mt19937_64& engine) : engine_(engine) {}

    double operator()() {
        if (spare_ready_) {
            spare_ready_ = false;
            return spare_;
        }
        // generate_canonical returns [0, 1); 1 - u lies in (0, 1], so the log
        // below is always finite.
        const double u1 = 1.0 - std::generate_canonical<double, 53>(engine_);
        const double u2 = std::generate_canonical<double, 53>(engine_);
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
    const double u = 1.0 - std::generate_canonical<double, 53>(engine);
    return d.p1 * std::pow(u, -1.0 / d.p2);
}

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
        a.epoch_proposals = 0;
        a.epoch_matches = 0;
    }
    return pop;
}
