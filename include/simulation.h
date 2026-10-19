#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "agent.h"

// ── Swipe generation ──────────────────────────────────────────────────────────
// Builds the implicit-feedback log the NCF model trains on. Each agent swipes
// right on its own top-k candidates of the opposite type — ranked by
// utility_toward, so W's indicator term shapes her positives — and k * 4
// negatives are drawn uniformly from the candidates outside that top-k.
//
// Output rows are [user_id, item_id, label] with label 1 for a positive and 0 for
// a negative, as fixed by docs/INTERFACES.md §2. Positives precede negatives
// within each agent's rows, and agents appear in id order.
//
// The count is not n*k*(1+4) without qualification: an agent whose candidate pool
// is smaller than k (or than the requested negatives) contributes fewer rows. That
// only happens in tiny test populations, but it means callers must read the
// returned vector rather than assume its length.
//
// Determinism: every agent seeds its own engine from (rng_seed, agent id, k), so
// the output is independent of the thread count. Seeding from a thread id, as one
// might expect, would make a four-thread run differ from a one-thread run — the
// exact regression the thread-safety test is there to catch.
std::vector<std::array<std::int32_t, 3>> generate_swipes(const std::vector<Agent>& pop,
                                                         const UtilityParams& p, int k,
                                                         std::uint64_t rng_seed);
