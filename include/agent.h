#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "utilities.h"

// ── Agent ─────────────────────────────────────────────────────────────────────
// The in-memory mirror of one row of the state matrix in docs/INTERFACES.md §1:
// id, type, status, attractiveness, sociability, age, threshold, and the four
// preference-heterogeneity columns (taste multipliers and the chemistry key).
// The matrix is flat because the hot loop touches 10^5 rows per epoch; this
// struct exists so the matching and simulation code can work with named fields
// instead of column indices.

// Same encoding as states[:, 1]: 0.0 = M (proposer side), 1.0 = W (acceptor side).
enum class AgentType : std::uint8_t { M = 0, W = 1 };

// Weights for the utility functions in docs/THEORY.md §2. generate_swipes and
// compute_welfare must use the *same* five values — if they disagree, every
// welfare number in the project is meaningless (docs/INTERFACES.md §2.2).
//
// The last two members are trailing so every existing five-field aggregate
// initializer in the tests still compiles, and they default to "off": scale 0.0
// skips the chemistry term entirely and seed 0 is only ever consulted when the
// scale is non-zero.
struct UtilityParams {
    double alpha;        // weight on the candidate's status
    double beta;         // weight on the candidate's attractiveness
    double gamma;        // weight on the feature-space distance penalty
    double delta;        // status surplus W requires before the indicator fires
    double lambda;       // size of W's indicator bonus

    // Persistent pairwise chemistry (THEORY.md §2.3). Absolute utility units, not
    // a ratio — the orchestrator logs the implied noise-to-signal ratio so a
    // retune of alpha/beta cannot silently change what a given scale means.
    double shock_scale = 0.0;
    // Seeds the shock hash. Lives here rather than on the Agent because it is a
    // property of the run, not of the individual; the per-agent part is key_i.
    std::uint64_t shock_seed = 0;
};

// ── Signalling (THEORY.md §10.2) ──────────────────────────────────────────────
// The proposal decision is  tau_i * V_ij * p > c.  These are the two numbers on
// the right-hand side, and this is deliberately a separate struct rather than two
// more members of UtilityParams:
//
//   * c is spent, not disliked. It is charged to the proposer's welfare but
//     never enters utility_toward, which compute_welfare also reads. Putting the
//     cost inside the utility would silently change every welfare number in the
//     project (THEORY.md §10.2, "Implementation note").
//   * the existing five-field aggregate initialisers in the tests keep compiling
//     untouched, which is the reason UtilityParams has only trailing defaults.
//
// p is the chance a proposal is accepted. It is not a modeling constant: the
// orchestrator measures it per side from the previous epoch's empirical accept
// rate and passes it in, so the cost stays responsive to the market rather than
// decorative. Epoch 0 has no history and uses 0.5, stated wherever it is used.
struct SignalParams {
    double cost = 0.0;         // c, in utility_toward's units
    double accept_prob = 0.5;  // p, in [0, 1]
};

// ── Trait standardization ─────────────────────────────────────────────────────
// Raw traits live on incompatible scales. At the base config, status is
// lognormal(0, 0.8) — sd 1.31, max near 44 — while attractiveness is
// normal(0.5, 0.15) clipped to [0, 1], sd 0.15. A weight of 1.0 on status and
// 0.8 on attractiveness therefore weights status 11.2x, not 1.25x, and δ is
// expressed in log-status units, which no longer mean anything interpretable.
//
// So every trait is z-scored before it enters the utility:
//
//     z(t) = (t - mean_t) / sd_t
//
// using population-level statistics computed once after sampling. The raw values
// stay on the Agent, in the state matrix and in every report — this struct is
// only the lens the utility looks through.
//
// δ is therefore measured in standard deviations of log-status. That is the
// single most important consequence: a δ of 1.0 means "one population standard
// deviation above her own status", not "one raw log-status unit".
struct TraitStats {
    double status_mean = 0.0;
    double status_sd = 1.0;
    double attractiveness_mean = 0.0;
    double attractiveness_sd = 1.0;
    double sociability_mean = 0.0;
    double sociability_sd = 1.0;
    double age_mean = 0.0;
    double age_sd = 1.0;

    // The identity transform: z(t) == t. Used by callers that hold no sampled
    // population — hand-built test fixtures, and any code path that wants the
    // old raw-scale behavior. Default-constructed TraitStats is this, so an
    // Agent utility call that forgets to pass stats degrades to raw units
    // rather than to a division by an arbitrary constant.
    static TraitStats identity() { return TraitStats{}; }
};

// z-score one value. A degenerate trait (sd == 0, a constant column) is left
// unstandardized rather than divided by zero: z is 0 for every agent, which
// contributes nothing to the utility and nothing to the distance, which is the
// correct behavior for a trait that carries no information.
inline double standardize(double value, double mean, double sd) {
    return (sd > 0.0) ? (value - mean) / sd : value;
}

// Age is the only trait measured in years; the other three live at unit scale.
// Dividing the age gap by this before squaring keeps one decade of age comparable
// to a full-unit trait gap instead of letting age dominate the distance.
//
// After standardization this divisor is redundant for utility purposes — a
// z-scored age gap is already dimensionless — but weighted_distance() applies it
// only on the raw path, so the two paths stay separately meaningful. Keeping the
// constant documents that age's raw scale was always a special case.
inline constexpr double kAgeScale = 10.0;

// θ before calibration: the agent admits every candidate. sample_population
// stamps this, and calibrate_thresholds() in population.h replaces it with a
// quantile of the agent's own utility distribution. -inf (rather than 0.0) is the
// neutral default because utilities are unbounded reals and several of them are
// negative — a 0.0 default would silently drop part of the market before anyone
// had calibrated anything.
inline constexpr double kUncalibratedThreshold =
    -std::numeric_limits<double>::infinity();

struct Agent;

// Weighted Euclidean distance d(v_a, v_b) over (status, attractiveness,
// sociability, age), as defined in docs/THEORY.md §2.1. Declared before Agent so
// the inline member below can call it; defined after the struct.
//
// Traits are standardized first (see TraitStats above), so every component is a
// gap in population standard deviations and the distance is scale-free. The
// former division of the age gap by kAgeScale is gone: it existed to stop raw
// years (sd ≈ 6) from swamping raw attractiveness (sd ≈ 0.15), and once all four
// traits are z-scored there is no scale left to compensate for.
inline double weighted_distance(const Agent& a, const Agent& b, const TraitStats& stats);

struct Agent {
    std::int32_t id = 0;
    AgentType type = AgentType::M;

    // State vector v_i = [status, attractiveness, sociability, age].
    double status = 0.0;
    double attractiveness = 0.0;
    double sociability = 0.0;
    double age = 0.0;

    // Selectivity θ_i, in the same units as utility_toward. generate_swipes admits
    // a candidate only when utility_toward(candidate) >= threshold, so θ is a
    // utility cutoff, not a probability: it must sit inside the utility range of
    // the agent's candidate pool, which is what calibrate_thresholds() arranges.
    // Mutated in place each epoch by update_threshold().
    double threshold = kUncalibratedThreshold;

    // ── Preference heterogeneity ─────────────────────────────────────────────
    // Per-agent taste multipliers (THEORY.md §2.3). Drawn once at sampling from
    // lognormal(-sigma^2/2, sigma), so the mean is 1 for every sigma: turning the
    // knob on rescales the *spread* of preferences, not their average strength.
    // Both defaults are exactly 1.0, and 1.0 * x is bit-identical to x, so a
    // population sampled at taste_sigma = 0 produces the same utilities — and the
    // same swipe log — as one sampled before these fields existed.
    double taste_alpha = 1.0;  // multiplies alpha: how much i weights status
    double taste_beta = 1.0;   // multiplies beta: how much i weights attractiveness
    double taste_gamma = 1.0;  // multiplies gamma: how much i penalises distance

    // Per-agent identifier for the chemistry hash. Drawn as the top 52 bits of an
    // engine output, so it is below 2^53 and round-trips exactly through the
    // float64 state matrix. 0 is a legal key but is never drawn deliberately.
    std::uint64_t key = 0;

    // ── Institutional state (THEORY.md §10.1) ────────────────────────────────
    // Appended after the heterogeneity block and, like it, drawn by
    // sample_population rather than chosen by a caller. None of the three enters
    // utility_toward: they change behavior only through the proposal decision and
    // the Environment B subgraph, which is what keeps every pre-existing welfare
    // number untouched at the default knobs.
    //
    // intent tau_i, in (0, 1]. Private willingness to commit, never observable.
    // Strictly positive by construction so no agent is exactly priced out at any
    // finite cost, and rescaled from a Beta draw at sampling. The 1.0 default is
    // the neutral one for a hand-built fixture: at c > 0 it proposes to every
    // candidate, which is the behavior a fixture that never set intent means.
    double intent = 1.0;
    // values g_i, in [0, 1]. Value alignment; observable only inside B, where
    // the g_min entry filter reads it. 0.0 is the neutral default and is below
    // any positive filter, so a fixture must opt in explicitly.
    double values = 0.0;
    // community k_i in {0, 1}: 1 = eligible for the Environment B subgraph.
    // Stored as int32 because it is a flag, not a continuous quantity, and the
    // flat matrix narrows it back to 0.0/1.0 exactly.
    std::int32_t community = 0;

    // U_i(j) from docs/THEORY.md §2.3:
    //   W evaluating a man : m_a*alpha*z(s_j) + m_b*beta*z(a_j) - m_g*gamma*d
    //                        + lambda*1[z(s_j) >= z(s_i) + delta] + s*L(i,j)
    //   M evaluating a woman: m_a*alpha*z(s_j) + m_b*beta*z(a_j) - m_g*gamma*d
    //                        + s*L(i,j)
    // The asymmetry is exactly the indicator term; nothing else differs by type.
    // The multipliers are per-agent, so the same candidate is worth different
    // things to different agents even before the shock.
    //
    // `stats` is required, not defaulted. A default would let a call site forget
    // to standardize and silently fall back to raw units — which is the exact bug
    // this parameter exists to remove, and it would show up as weights that look
    // reasonable and behave as though status were 11x the others. Hand-built
    // fixtures that genuinely want raw arithmetic pass TraitStats::identity().
    double utility_toward(const Agent& other, const UtilityParams& p,
                          const TraitStats& stats) const {
        const double zs_other = standardize(other.status, stats.status_mean, stats.status_sd);
        const double za_other = standardize(other.attractiveness, stats.attractiveness_mean,
                                            stats.attractiveness_sd);
        // The multiplier multiplies the weight, and the product is then applied to
        // the trait. At the default 1.0 this is exactly the old expression: IEEE
        // multiplication by 1.0 is exact, so no test that pins a utility value can
        // see the difference.
        double u = (taste_alpha * p.alpha) * zs_other + (taste_beta * p.beta) * za_other -
                   (taste_gamma * p.gamma) * weighted_distance(*this, other, stats);
        if (type == AgentType::W) {
            // δ is in standard deviations of log-status: the cutoff is "her own
            // standardized status, plus δ sd". On the raw scale δ was in raw
            // log-status units, which is why it was inert at the defaults — the
            // indicator was constant across the competitive set.
            const double zs_self = standardize(status, stats.status_mean, stats.status_sd);
            const double cutoff = zs_self + p.delta;
            u += p.lambda * (zs_other >= cutoff ? 1.0 : 0.0);
        }
        // Skip rather than multiply by zero: a hash and a log per pair is real
        // work, and at shock_scale = 0 the term must cost nothing at all
        // (THEORY.md §2.3, and the before/after timing in the session report).
        if (p.shock_scale != 0.0) {
            u += p.shock_scale * pairwise_shock(p.shock_seed, key, other.key);
        }
        return u;
    }

    // θ ← θ + η·(matches/proposals − θ). A zero-proposal epoch carries no
    // information about this agent's match probability, so θ is left alone.
    //
    // The per-epoch proposal and match counts are passed in, not stored: the
    // caller owns them for the epoch it is closing, and keeping a second copy on
    // the agent created two sources of truth for one quantity.
    void update_threshold(std::uint32_t proposals, std::uint32_t matches, double eta) {
        if (proposals == 0) {
            return;
        }
        const double yield =
            static_cast<double>(matches) / static_cast<double>(proposals);
        threshold += eta * (yield - threshold);
    }
};

inline double weighted_distance(const Agent& a, const Agent& b, const TraitStats& stats) {
    const double ds = standardize(a.status, stats.status_mean, stats.status_sd) -
                      standardize(b.status, stats.status_mean, stats.status_sd);
    const double da = standardize(a.attractiveness, stats.attractiveness_mean,
                                  stats.attractiveness_sd) -
                      standardize(b.attractiveness, stats.attractiveness_mean,
                                  stats.attractiveness_sd);
    const double dc = standardize(a.sociability, stats.sociability_mean, stats.sociability_sd) -
                      standardize(b.sociability, stats.sociability_mean, stats.sociability_sd);
    const double dt = standardize(a.age, stats.age_mean, stats.age_sd) -
                      standardize(b.age, stats.age_mean, stats.age_sd);
    return std::sqrt(ds * ds + da * da + dc * dc + dt * dt);
}

// ── Population statistics ─────────────────────────────────────────────────────
// Mean and sd of each trait across the whole population, computed once after
// sampling. These are the numbers utility_toward() and weighted_distance()
// standardize with, so they must be derived from the same population those
// functions are called on — a stats struct from a different population would
// quietly reweight every trait.
//
// Whole population rather than per side: both sides are drawn from the same
// trait distributions, so the two conventions agree statistically, and one set
// of numbers is one thing to reason about.
//
// Sample sd (n-1 denominator), not population sd: the traits are draws from a
// distribution, so the population of agents is a sample from it. At the sizes
// this model runs (N >= 100) the difference is below 1e-3 and is not worth a
// second code path, but the choice is stated rather than left implicit.
//
// A trait whose sd is zero (a constant column) is reported with sd = 0.0;
// standardize() then leaves it unscaled instead of dividing by zero.
inline TraitStats compute_trait_stats(const std::vector<Agent>& pop) {
    TraitStats s;
    const std::size_t n = pop.size();
    if (n == 0) {
        return s;
    }

    auto moments = [&pop, n](double Agent::*field) {
        double mean = 0.0;
        for (const Agent& a : pop) {
            mean += a.*field;
        }
        mean /= static_cast<double>(n);
        double ss = 0.0;
        for (const Agent& a : pop) {
            const double d = a.*field - mean;
            ss += d * d;
        }
        const double sd =
            (n > 1) ? std::sqrt(ss / static_cast<double>(n - 1)) : 0.0;
        return std::pair<double, double>{mean, std::isfinite(sd) ? sd : 0.0};
    };

    const auto [s_mean, s_sd] = moments(&Agent::status);
    s.status_mean = s_mean;
    s.status_sd = s_sd;
    const auto [a_mean, a_sd] = moments(&Agent::attractiveness);
    s.attractiveness_mean = a_mean;
    s.attractiveness_sd = a_sd;
    const auto [c_mean, c_sd] = moments(&Agent::sociability);
    s.sociability_mean = c_mean;
    s.sociability_sd = c_sd;
    const auto [t_mean, t_sd] = moments(&Agent::age);
    s.age_mean = t_mean;
    s.age_sd = t_sd;
    return s;
}
