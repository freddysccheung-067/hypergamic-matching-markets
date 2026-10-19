#pragma once

#include <cmath>
#include <cstdint>

// ── Persistent pairwise "chemistry" ──────────────────────────────────────────
// A standard logistic shock L(i, j) attached to every utility, present so the
// market has taste variation a recommender cannot learn (THEORY.md §2.3).
//
// Three properties are load-bearing, and each is a test:
//
//   Deterministic and persistent. L is a pure function of (seed, key_i, key_j),
//   so generate_swipes, calibrate_thresholds, build_full_info_pref_lists and
//   compute_welfare all see the same value for the same pair. If any of them
//   disagreed, the swipe log and the welfare number would describe two different
//   markets and no result would mean anything.
//
//   Asymmetric. L(i, j) != L(j, i): i's chemistry with j is not j's chemistry
//   with i. That is what the ORDERED pair buys, and the mix below is deliberately
//   not commutative.
//
//   Computed, not stored. No N x N matrix anywhere: the value is a splitmix64
//   chain over the three keys, mapped to a uniform strictly inside (0, 1) and
//   then through the logistic quantile. O(1) memory per pair.
//
// Why a logistic and not a normal: the logistic has closed-form sd pi/sqrt(3) =
// 1.8138 and a heavier tail, which is the right shape for a preference shock —
// a small minority of pairs are strongly drawn to each other, and a normal
// understates that. The scale knob is shock_scale, in absolute utility units.

// splitmix64 finalizer (Steele, Lea & Flood). A bijection on 64-bit words with
// good avalanche, which is all that is needed for decorrelation here: it is not
// cryptography and is not claimed to be.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// L(i, j). Order matters: the key_i and key_j steps are separate mixes, so
// swapping them changes the result for almost every pair.
//
// The uniform mapping is ((h >> 11) + 0.5) * 2^-53 rather than (h >> 11) * 2^-53.
// The 0.5 keeps u strictly inside (0, 1) — the plain form can return exactly
// 0.0, whose logit is -infinity — and costs nothing in distributional terms
// beyond shrinking the support by half an ulp at each end.
inline double pairwise_shock(std::uint64_t seed, std::uint64_t key_i, std::uint64_t key_j) {
    const std::uint64_t h = splitmix64(splitmix64(splitmix64(seed) ^ key_i) ^ key_j);
    const double u = (static_cast<double>(h >> 11) + 0.5) * 0x1.0p-53;
    return std::log(u / (1.0 - u));
}

// sd of a standard logistic. Named because the noise-to-signal ratio in the
// orchestrator's diagnostic is shock_scale * kLogisticSd / utility_sd, and a
// literal 1.8138 in a log line is the kind of number nobody can later verify.
inline constexpr double kLogisticSd = 1.8137993642342179;  // pi / sqrt(3)
