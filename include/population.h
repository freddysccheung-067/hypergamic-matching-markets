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
    double init_threshold = 0.0;
};

// Draws n agents. The first n/2 are M with ids 0..n/2-1 and the remainder are W
// with ids n/2..n-1, matching states[:, 1] in docs/INTERFACES.md §1.
//
// n must be even: an odd n drops the last agent and logs a warning to stderr, so
// the two sides stay balanced. Every draw comes from one std::mt19937_64 seeded
// with `seed`, making (n, cfg, seed) reproduce the population bit for bit.
//
// Attractiveness and sociability are clamped to [0, 1] after sampling; status and
// age are left unbounded, per §3 of the interface contract.
std::vector<Agent> sample_population(int n, const PopConfig& cfg, std::uint64_t seed);
