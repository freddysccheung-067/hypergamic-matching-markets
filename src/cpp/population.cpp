#include "population.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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
// The count lives in population.h as kStateCols so the shape guards in
// bindings.cpp and this file cannot drift apart.
constexpr std::size_t kCols = kStateCols;

// A lognormal multiplier with mean exactly 1: exp(-sigma^2/2 + sigma*z).
//
// The -sigma^2/2 is what makes the mean 1 rather than exp(sigma^2/2), so the knob
// changes the dispersion of tastes and not their level. At sigma == 0 this returns
// exactly 1.0 — neither branch calls exp — which is what keeps the no-heterogeneity
// path bit-identical.
double taste_multiplier(double sigma, NormalSource& normal) {
    if (sigma == 0.0) {
        return 1.0;
    }
    return std::exp(-0.5 * sigma * sigma + sigma * normal());
}

// The top 52 bits of an engine output. 52 rather than 53 because the value has to
// survive a round trip through a float64 with room to spare for any later
// arithmetic: 2^53 is the first integer a double cannot represent exactly, and
// staying one bit below it makes that boundary unreachable rather than merely
// unlikely. Never returns a value at or above 2^53.
std::uint64_t draw_key(std::mt19937_64& engine) {
    return engine() >> 12;
}

// ── Beta draws (THEORY.md §10.1) ──────────────────────────────────────────────
// Two gamma draws divided by their sum: X ~ Gamma(a, 1), Y ~ Gamma(b, 1) give
// X/(X+Y) ~ Beta(a, b). Written out rather than taken from std::gamma_distribution
// for the same reason as every other draw in this file — the standard library's
// algorithm and its engine-call count are implementation-defined, and these draws
// decide the published intent and value columns.
//
// Marsaglia & Tsang (2000), which is the standard method for a >= 1: the boosted
// exponential dominates and the squeeze rejects in a few iterations. The a < 1
// case uses the standard boost X ~ Gamma(a+1) * U^(1/a), which is exact in law
// and needs one extra uniform. Both branches consume a data-dependent number of
// engine calls, which is why the traits below get their own streams rather than
// sharing one: a variable-length draw interleaved into the trait stream would
// shift every later trait and destroy the bit-identical guarantee.
double gamma_draw(double shape, std::mt19937_64& engine, NormalSource& normal) {
    if (shape < 1.0) {
        // Boost to shape + 1 and take the a-th root of a uniform on (0, 1].
        const double u = uniform_open_closed(engine);
        return gamma_draw(shape + 1.0, engine, normal) * std::pow(u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        // The engine draws happen before the acceptance test either way, so the
        // rejection loop is the only place the call count varies.
        const double x = normal();
        const double v = 1.0 + c * x;
        if (v <= 0.0) {
            continue;
        }
        const double v3 = v * v * v;
        const double u = uniform01(engine);
        const double x2 = x * x;
        // The cheap squeeze first: it accepts most candidates for shape >= 1.
        if (u < 1.0 - 0.0331 * x2 * x2) {
            return d * v3;
        }
        if (u > 0.0 && std::log(u) < 0.5 * x2 + d * (1.0 - v3 + std::log(v3))) {
            return d * v3;
        }
    }
}

double beta_draw(double a, double b, std::mt19937_64& engine, NormalSource& normal) {
    const double x = gamma_draw(a, engine, normal);
    const double y = gamma_draw(b, engine, normal);
    const double s = x + y;
    // Both shapes are validated positive at entry, so s > 0 in exact arithmetic.
    // A guard rather than an assert: at a = b = 2 the probability of seeing this
    // is nil, but returning 0.5 beats returning NaN into a state column.
    return (s > 0.0) ? x / s : 0.5;
}

}  // namespace

std::vector<Agent> sample_population(int n, const PopConfig& cfg, std::uint64_t seed) {
    validate(cfg.status, "status");
    validate(cfg.attractiveness, "attractiveness");
    validate(cfg.sociability, "sociability");
    validate(cfg.age, "age");

    // The Beta shapes are validated before the loop for the same reason as the
    // trait distributions: a bad shape must fail on an empty population rather
    // than halfway through. Both shapes must be strictly positive — a zero shape
    // makes one of the gamma draws identically zero, so the ratio is degenerate.
    for (const auto& [a, b, name] :
         {std::tuple{cfg.intent_beta_a, cfg.intent_beta_b, "intent_beta"},
          std::tuple{cfg.values_beta_a, cfg.values_beta_b, "values_beta"}}) {
        if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
            throw std::invalid_argument(std::string("sample_population: population.") + name +
                                        "_a and " + name + "_b must be finite and > 0");
        }
    }
    if (!(cfg.community_prob >= 0.0) || !(cfg.community_prob <= 1.0)) {
        throw std::invalid_argument(
            "sample_population: population.community_prob must be in [0, 1]");
    }

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

    // Two independent streams for the heterogeneity columns, both distinct from
    // the trait stream. Separate rather than continuing the first because the
    // trait draws are pinned: a population sampled with taste_sigma = 0 has to
    // reproduce the pre-heterogeneity traits bit for bit, and interleaving new
    // draws into that stream would move every subsequent trait.
    //
    // Keys and multipliers get their own engines as well, so that the keys do not
    // depend on taste_sigma: at sigma = 0 no normal is consumed, and a shared
    // engine would therefore shift every later key relative to a sigma = 0.5 run.
    // With two streams an agent's chemistry is the same at every sigma, and the
    // taste knob moves only the multipliers.
    //
    // The tags differ from any agent engine in simulation.cpp, so a key can never
    // collide with a swipe-stream state by construction rather than by luck.
    const std::uint32_t seed_lo = static_cast<std::uint32_t>(seed);
    const std::uint32_t seed_hi = static_cast<std::uint32_t>(seed >> 32);
    // Spelled at 32 bits. Both tags below were written as 35-bit literals, which MSVC
    // warns about twice per build for the narrowing to uint32_t. The compiler was
    // already taking the low 32 bits, so writing that value directly is bit-identical
    // and the swipe-log digests pinned in tests/python/test_heterogeneity.py do not
    // move.
    const std::uint32_t key_tag[3] = {seed_lo, seed_hi, 0xA57E5EEDu};
    std::seed_seq key_seq(std::begin(key_tag), std::end(key_tag));
    std::mt19937_64 key_engine(key_seq);

    const std::uint32_t taste_tag[3] = {seed_lo, seed_hi, 0xA5AC4EE5u};
    std::seed_seq taste_seq(std::begin(taste_tag), std::end(taste_tag));
    std::mt19937_64 taste_engine(taste_seq);
    NormalSource taste_normal(taste_engine);

    // A third stream for the institutional columns (THEORY.md §10.1). Same
    // argument as the two above, and one more specific to it: a Beta draw consumes
    // a data-dependent number of uniforms, so sharing the trait stream would make
    // every later trait depend on the Beta shapes. Here the knob moves only the
    // three new columns.
    // Spelled within 32 bits, unlike the two older tags: those literals are 36-bit
    // and MSVC warns on the narrowing. The value only has to differ from every
    // other tag in this file and from any agent engine in simulation.cpp.
    const std::uint32_t signal_tag[3] = {seed_lo, seed_hi, 0x51E9A1C0u};
    std::seed_seq signal_seq(std::begin(signal_tag), std::end(signal_tag));
    std::mt19937_64 signal_engine(signal_seq);
    NormalSource signal_normal(signal_engine);

    for (int i = 0; i < n; ++i) {
        Agent& a = pop[static_cast<std::size_t>(i)];
        a.id = i;
        a.type = (i < half) ? AgentType::M : AgentType::W;
        a.status = draw(cfg.status, engine, normal);
        a.attractiveness = std::clamp(draw(cfg.attractiveness, engine, normal), 0.0, 1.0);
        a.sociability = std::clamp(draw(cfg.sociability, engine, normal), 0.0, 1.0);
        a.age = draw(cfg.age, engine, normal);
        a.threshold = cfg.init_threshold;

        // Fixed order — key first, then the three multipliers — and key from its
        // own stream, so a key never depends on taste_sigma. A replaced agent
        // (generational turnover) draws a fresh key, which is what gives it fresh
        // chemistry without disturbing anyone else's: every other pair is a
        // function of two keys, neither of which moved.
        a.key = draw_key(key_engine);
        a.taste_alpha = taste_multiplier(cfg.taste_sigma, taste_normal);
        a.taste_beta = taste_multiplier(cfg.taste_sigma, taste_normal);
        a.taste_gamma = taste_multiplier(cfg.taste_sigma, taste_normal);

        // Institutional columns, from the third stream and in a fixed order so a
        // knob moves only its own column. intent is rescaled to (0, 1]: a Beta
        // draw is never exactly 0 in exact arithmetic, but the smallest double
        // keeps the invariant true rather than merely almost true, and it matters
        // because tau enters a strict inequality against a cost.
        a.intent = std::max(beta_draw(cfg.intent_beta_a, cfg.intent_beta_b, signal_engine,
                                      signal_normal),
                            std::numeric_limits<double>::min());
        a.values = beta_draw(cfg.values_beta_a, cfg.values_beta_b, signal_engine, signal_normal);
        a.community =
            (uniform01(signal_engine) < cfg.community_prob) ? std::int32_t{1} : std::int32_t{0};
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
        row[7] = a.taste_alpha;
        row[8] = a.taste_beta;
        row[9] = a.taste_gamma;
        // Exact: key < 2^52, so every bit is inside a double's 53-bit mantissa.
        row[10] = static_cast<double>(a.key);
        row[11] = a.intent;
        row[12] = a.values;
        // Exact: the flag is 0 or 1, both representable.
        row[13] = static_cast<double>(a.community);
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
        a.taste_alpha = row[7];
        a.taste_beta = row[8];
        a.taste_gamma = row[9];
        // The round trip is exact for any key below 2^53; a hand-built matrix with
        // a larger value in this column would round here, which is why the sampler
        // draws 52 bits rather than 53.
        a.key = static_cast<std::uint64_t>(row[10]);
        a.intent = row[11];
        a.values = row[12];
        // Round-tripping through a double then narrowing: a hand-built matrix with
        // 0.5 here would become 0, which is the safe direction (not eligible for a
        // subgraph) rather than silently admitting an agent nobody flagged.
        a.community = (row[13] == 1.0) ? std::int32_t{1} : std::int32_t{0};
    }
    return pop;
}

void calibrate_thresholds(std::vector<Agent>& pop, const UtilityParams& p,
                          double admissible_fraction, const TraitStats& stats) {
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
            utilities.push_back(a.utility_toward(cand, p, stats));
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
