#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

// ── Agent ─────────────────────────────────────────────────────────────────────
// The in-memory mirror of one row of the state matrix in docs/INTERFACES.md §1:
// id, type, status, attractiveness, sociability, age, threshold. The matrix is
// flat because the hot loop touches 10^5 rows per epoch; this struct exists so
// the matching and simulation code can work with named fields instead of column
// indices.

// Same encoding as states[:, 1]: 0.0 = M (proposer side), 1.0 = W (acceptor side).
enum class AgentType : std::uint8_t { M = 0, W = 1 };

// Weights for the utility functions in docs/THEORY.md §2. generate_swipes and
// compute_welfare must use the *same* five values — if they disagree, every
// welfare number in the project is meaningless (docs/INTERFACES.md §2.2).
struct UtilityParams {
    double alpha;   // weight on the candidate's status
    double beta;    // weight on the candidate's attractiveness
    double gamma;   // weight on the feature-space distance penalty
    double delta;   // status surplus W requires before the indicator fires
    double lambda;  // size of W's indicator bonus
};

// Age is the only trait measured in years; the other three live at unit scale.
// Dividing the age gap by this before squaring keeps one decade of age comparable
// to a full-unit trait gap instead of letting age dominate the distance.
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
inline double weighted_distance(const Agent& a, const Agent& b);

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

    // U_i(j) from docs/THEORY.md §2:
    //   W evaluating a man : alpha*s_j + beta*a_j - gamma*d + lambda*1[s_j >= s_i + delta]
    //   M evaluating a woman: alpha*s_j + beta*a_j - gamma*d
    // The asymmetry is exactly the indicator term; nothing else differs by type.
    double utility_toward(const Agent& other, const UtilityParams& p) const {
        double u = p.alpha * other.status + p.beta * other.attractiveness -
                   p.gamma * weighted_distance(*this, other);
        if (type == AgentType::W) {
            const double cutoff = status + p.delta;
            u += p.lambda * (other.status >= cutoff ? 1.0 : 0.0);
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

inline double weighted_distance(const Agent& a, const Agent& b) {
    const double ds = a.status - b.status;
    const double da = a.attractiveness - b.attractiveness;
    const double dc = a.sociability - b.sociability;
    const double dt = (a.age - b.age) / kAgeScale;
    return std::sqrt(ds * ds + da * da + dc * dc + dt * dt);
}
