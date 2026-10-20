#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "agent.h"

// ── Preference lists ─────────────────────────────────────────────────────────
// Both sides' preference lists are indexed densely, per docs/INTERFACES.md §2.3:
// entry i is the i-th agent *of that side*, and the integers inside a list are
// indexes into the other side, not global agent ids. The reason is that
// deferred_acceptance() receives the lists and nothing else, so with global ids
// it could not recover an unmatched acceptor's position — an acceptor nobody
// proposes to appears in no list at all.
//
// Lists are ordered by descending preference. Ties are broken by ascending index,
// so the order is total even on a population with equal utilities.

// Full-information preference lists: every agent ranks every agent of the other
// side by utility, so the result admits the benchmark of THEORY.md §6.1.
//
// O(N² log N) time and O(N²) memory — at N = 5,000 each side's lists are 25M
// integers, about 100 MB. docs/INTERFACES.md §4 caps it there and says to run it
// once per configuration, not once per epoch.
struct PrefLists {
    std::vector<std::vector<std::int32_t>> proposers;  // indexed by proposer, holds acceptors
    std::vector<std::vector<std::int32_t>> acceptors;  // indexed by acceptor, holds proposers
};

PrefLists build_full_info_pref_lists(const std::vector<Agent>& pop, const UtilityParams& p,
                                     const TraitStats& stats);

// ── Deferred acceptance ──────────────────────────────────────────────────────
struct MatchResult {
    // [proposer index, acceptor index] pairs.
    std::vector<std::array<std::int32_t, 2>> matches;
    std::vector<std::int32_t> unmatched_proposers;
    std::vector<std::int32_t> unmatched_acceptors;

    // Mutually-preferring pairs the matching leaves blockable. Zero whenever the
    // lists are complete, so a full-information run is also a check on the
    // implementation: a non-zero count there is a bug, not a finding.
    std::int32_t n_blocking = 0;

    // Attempts made (per proposer) and received (per acceptor), rejections
    // included. Proposers occupy [0, n_prop), acceptors [n_prop, n_prop + n_acc).
    // This is the denominator update_thresholds() needs and cannot recover from
    // matches: a rejected proposal leaves no trace in the matching.
    std::vector<std::int32_t> proposals;
};

// Gale-Shapley with proposers as the proposing side (THEORY.md §6).
//
// `truncated` selects the acceptability rule. When true — the NCF-mediated case
// — an acceptor rejects a proposer who does not appear in her list, because the
// truncated list is what she knows about; this is the standard stable-matching
// model with incomplete lists and is where the congestion story comes from. When
// false every proposer is acceptable, ranked last if somehow unlisted, which is
// the full-information benchmark.
//
// Complexity O(N·k) for lists of length k, per docs/INTERFACES.md §4.
MatchResult deferred_acceptance(const std::vector<std::vector<std::int32_t>>& prop_prefs,
                                const std::vector<std::vector<std::int32_t>>& acc_prefs,
                                bool truncated);

// Counts pairs that would mutually prefer each other to the given matching,
// judged against *these* preference lists. Zero on a complete-list matching.
//
// Exported because it is what BPR (docs/INTERFACES.md §6) actually needs. The
// n_blocking inside MatchResult is computed against the lists deferred_acceptance
// was handed, and a truncated run is stable with respect to those by
// construction — so it is always zero there and would make bpr an identically
// zero column. Measuring the instability that truncation introduces means asking
// how many blocking pairs exist under the *true* ordering, which requires the
// full lists and the algorithm's own matching. Callers build the full lists once
// per population (§4) and pass them here each epoch.
std::int32_t count_blocking_pairs(
    const std::vector<std::vector<std::int32_t>>& prop_prefs,
    const std::vector<std::vector<std::int32_t>>& acc_prefs,
    const std::vector<std::array<std::int32_t, 2>>& matches);

// ── Welfare ──────────────────────────────────────────────────────────────────
struct Welfare {
    double total = 0.0;
    double proposer = 0.0;
    double acceptor = 0.0;
    std::int32_t n_pairs = 0;
    // The value-created term of THEORY.md §10.4: each side's utility weighted by
    // its own intent, Σ τ_m U_m(f) + τ_f U_f(m). Accumulated proposer-then-acceptor
    // in the same order as `total`, so at τ ≡ 1 the two are bit-identical (1.0 * x
    // is exact in IEEE arithmetic). That is the identity the tests pin.
    double value_created = 0.0;
    double value_proposer = 0.0;
    double value_acceptor = 0.0;
};

// Σ over matched pairs of U_prop(acc) + U_acc(prop), plus the intent-weighted
// value-created term. Uses the same weights and standardization as
// generate_swipes — docs/INTERFACES.md §2.2 calls a disagreement between the two
// the most likely source of a silent bug in the project, so it is worth stating
// that both go through Agent::utility_toward(). The signal-cost and
// defection-loss terms of §10.4 are not functions of the matching alone and are
// added by the binding.
Welfare compute_welfare(const std::vector<Agent>& pop,
                        const std::vector<std::array<std::int32_t, 2>>& matches,
                        const UtilityParams& p, const TraitStats& stats);

// ── Threshold step ───────────────────────────────────────────────────────────
// θ ← θ + η·(matches/proposals − θ) per agent, skipping agents with no
// proposals (THEORY.md §3). n_proposals and n_matches are indexed by *row* in
// pop, not by the dense per-side index the preference lists use, because this
// writes Agent::threshold and the caller has already mapped back.
//
// The scale mismatch this encodes — the target is a yield in [0, 1] while θ is a
// utility cutoff — is deliberate and documented in THEORY.md §3 as recorded
// rather than repaired. Reproducing the documented step is the contract; quietly
// rescaling it here would change every result the project reports.
void update_thresholds(std::vector<Agent>& pop, const std::vector<std::int32_t>& n_proposals,
                       const std::vector<std::int32_t>& n_matches, double eta);
