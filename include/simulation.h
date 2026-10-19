#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "agent.h"

// ── Swipe generation ──────────────────────────────────────────────────────────
// Builds the implicit-feedback log the NCF model trains on.
//
// Each agent swipes right on its top-k *admissible* candidates of the opposite
// type: candidates are ranked by utility_toward, and those whose utility falls
// below the agent's own θ are dropped before the top-k is taken. θ is therefore a
// live gate on behavior, not a counter kept for reporting — it is what makes the
// adaptive-expectation loop in docs/THEORY.md §3 close, since a rising θ shrinks
// the positive set and the NCF's target moves with it.
//
// Because the admissible set depends on θ, the positives change from epoch to
// epoch. Negatives also differ per epoch: the per-agent engine is seeded from
// (rng_seed, agent id, k, epoch). See docs/INTERFACES.md §2 for the seeding rule.
//
// Negatives: k * n_negatives candidates drawn uniformly without replacement from
// the pool of candidates that are *not* admissible-and-selected. That pool
// includes the admissible ones outside the top-k, so a dropped-for-low-utility
// candidate can come back as a negative — correct, since the label records what
// the agent did, not what it could have done.
//
// n_negatives comes from config["ncf"]["n_negatives"] (bindings.cpp) and defaults
// to 4. It is a training-dynamics knob, not a modeling constant: the
// negative:positive ratio changes how hard the NCF's implicit-feedback problem is
// and therefore how sharply it funnels exposure, which is exactly what the
// experiment matrix varies. It is deliberately NOT part of the per-agent seed, so
// a run at n_negatives = 4 reproduces the pre-parameter output bit for bit.
//
// Output rows are [user_id, item_id, label] with label 1 for a positive and 0 for
// a negative, as fixed by docs/INTERFACES.md §2. Positives precede negatives
// within each agent's rows, and agents appear in id order.
//
// The count is not n*k*(1+4) without qualification: an agent whose admissible
// pool or candidate pool is smaller than k contributes fewer rows. That only
// happens in small test populations, but it means callers must read the returned
// vector rather than assume its length.
//
// Determinism: seeded per agent from its own id, never from a thread id or from
// shared state, so the output is identical at any thread count — the exact
// regression the thread-safety test is there to catch.
std::vector<std::array<std::int32_t, 3>> generate_swipes(const std::vector<Agent>& pop,
                                                         const UtilityParams& p,
                                                         const TraitStats& stats, int k,
                                                         std::uint64_t rng_seed,
                                                         std::uint64_t epoch = 0,
                                                         int n_negatives = 4);
