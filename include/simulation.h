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
// live gate on behaviour, not a counter kept for reporting — it is what makes the
// adaptive-expectation loop in docs/THEORY.md §3 close, since a rising θ shrinks
// the positive set and the NCF's target moves with it.
//
// Because the admissible set depends on θ, the positives change from epoch to
// epoch. Negatives also differ per epoch: the per-agent engine is seeded from
// (rng_seed, agent id, k, epoch). See docs/INTERFACES.md §2 for the seeding rule.
//
// Negatives: k * 4 candidates drawn uniformly without replacement from the pool
// of candidates that are *not* admissible-and-selected. That pool includes the
// admissible ones outside the top-k, so a dropped-for-low-utility candidate can
// come back as a negative — correct, since the label records what the agent did,
// not what it could have done.
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
                                                         const UtilityParams& p, int k,
                                                         std::uint64_t rng_seed,
                                                         std::uint64_t epoch = 0);
