#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "agent.h"

// ── Trait distributions ───────────────────────────────────────────────────────
// One entry per block under population.traits.<name> in experiments/base.yaml.
// The meaning of p1/p2 depends on name, which is why they are not called
// mean/std:
//
//   "normal"     p1 = mean,  p2 = standard deviation
//   "lognormal"  p1 = mu,    p2 = sigma of the *underlying* normal, so the
//                sample mean is exp(mu + sigma^2 / 2)
//   "pareto"     p1 = x_min, p2 = alpha (the tail index), sampled by inverse
//                CDF: x = x_min * U^(-1/alpha), U uniform on (0, 1]
struct TraitDist {
    std::string name;
    double p1 = 0.0;
    double p2 = 1.0;
};

struct PopConfig {
    TraitDist status;
    TraitDist attractiveness;
    TraitDist sociability;
    TraitDist age;

    // Target share of candidates an agent should find admissible at t=0. This is
    // a *fraction*, not a θ: calibrate_thresholds() turns it into a per-agent
    // utility quantile. 0.2 matches experiments/base.yaml — the top fifth of
    // candidates.
    //
    // It defaults to a real number rather than to "off" on purpose. A default
    // that failed to calibrate would leave θ at -infinity, admitting every
    // candidate, which is exactly the inert-gate bug this replaces — and it
    // would do so silently, because a swipe log of k positives per agent looks
    // entirely healthy whether θ is 20% or 100%.
    double admissible_fraction = 0.2;

    // Absolute θ, used only when the caller asks for one. kUncalibratedThreshold
    // means "not set" — the sampler then leaves θ neutral and calibration runs.
    // Any other value is used verbatim.
    double init_threshold = kUncalibratedThreshold;
};

// Draws n agents. The first n/2 are M with ids 0..n/2-1 and the remainder are W
// with ids n/2..n-1, matching states[:, 1] in docs/INTERFACES.md §1.
//
// n must be even: an odd n drops the last agent and logs a warning to stderr, so
// the two sides stay balanced. Every draw comes from one std::mt19937_64 seeded
// with `seed`. Uniforms are derived from engine() output directly rather than via
// std::generate_canonical, whose number of engine calls and rounding are
// implementation-defined (docs/INTERFACES.md §3).
//
// What that does and does not buy, stated precisely because the difference is
// easy to overclaim:
//
//   Guaranteed   The uniform stream is (engine() >> 11) * 2^-53, which is exact in
//                IEEE-754, and the engine state is fully specified by the standard
//                given a specified seed_seq. So the *draws themselves* are
//                identical on every conforming implementation.
//   NOT conveyed  std::exp/std::log/std::sqrt/std::pow are required only to be
//                within a few ulp, not correctly rounded, so libm differences can
//                move a sampled trait in its last bits. Byte-identical trait
//                values are therefore guaranteed within one toolchain, and across
//                thread counts and runs; they are not guaranteed across compilers.
//
// Swipe generation is unaffected by that caveat: it works in integers and
// comparisons, so negatives are bit-identical across toolchains.
//
// Attractiveness and sociability are clamped to [0, 1] after sampling; status and
// age are left unbounded, per §3 of the interface contract.
std::vector<Agent> sample_population(int n, const PopConfig& cfg, std::uint64_t seed);

// ── Flat state matrix ─────────────────────────────────────────────────────────
// The (N, 7) bridge required by docs/INTERFACES.md §1 and §2. Row layout, exactly
// as the contract states it:
//
//   col 0 id | 1 type | 2 status | 3 attractiveness | 4 sociability | 5 age | 6 threshold
//
// Row-major, 7 columns, no padding: flatten_states() returns n*7 doubles and
// unflatten_states() reads the same layout back. The two are exact inverses, and
// the conversion is lossless because every field is stored in a double that can
// represent it exactly (ids are small integers; type is 0.0 or 1.0).
std::vector<double> flatten_states(const std::vector<Agent>& pop);

// Reads n rows of 7 doubles. The caller keeps ownership of `data`; the returned
// vector owns its own copy.
std::vector<Agent> unflatten_states(const double* data, int n);

// ── Threshold calibration ────────────────────────────────────────────────────
// Sets each agent's θ to the (1 - admissible_fraction) quantile of that agent's
// own utilities toward the opposite side.
//
// Why a per-agent quantile and not one number for the whole population: θ is
// compared against utility_toward(), whose scale is fixed by alpha/beta/gamma and
// by how status is distributed, not by the agent. A constant θ therefore admits
// an arbitrary and population-dependent share of candidates, and at the default
// weights it admits almost all of them or almost none — which is exactly the
// inert-threshold bug this replaces. A quantile makes "roughly admissible_fraction
// of candidates are swipeable at epoch 0" true by construction, for every agent.
//
// admissible_fraction is clamped to (0, 1], and the admitted count is
// clamp(lround(admissible_fraction * n_candidates), 1, n_candidates). An agent
// with no candidates on the other side keeps whatever θ it already had.
//
// Known limitation — ties. θ is set to an actual utility value rather than
// interpolated between order statistics, which keeps it on the scale the gate
// compares against, but it means a tie group straddling the boundary is admitted
// whole. With distinct utilities the admitted count is exact; with heavy ties it
// can overshoot the target substantially (a 2-of-10 target admits a 7-member tie
// group). Continuous traits make exact ties measure-zero in practice, but a
// discretised population would need a tie-breaking rule.
void calibrate_thresholds(std::vector<Agent>& pop, const UtilityParams& p,
                          double admissible_fraction);
