#include "simulation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr int kNegativesPerPositive = 4;

// A candidate paired with the utility it earns. Carrying the score next to the id
// is what lets partial_sort do the ranking in one pass over the population.
struct Scored {
    double utility;
    std::int32_t id;
};

// Descending utility, ties broken by ascending id. Without the tie-break a
// population with many equal utilities (the all-zero-weight tests) would rank
// differently depending on the input order partial_sort happens to produce.
bool ranks_above(const Scored& a, const Scored& b) {
    if (a.utility != b.utility) {
        return a.utility > b.utility;
    }
    return a.id < b.id;
}

// Per-agent engine. The seed mixes in the agent id rather than a thread id so the
// same population yields the same swipes at any thread count, and so no two
// threads ever touch the same engine.
std::mt19937_64 agent_engine(std::uint64_t seed, std::int32_t agent_id, int k) {
    std::seed_seq seq{static_cast<std::uint32_t>(seed),
                      static_cast<std::uint32_t>(seed >> 32),
                      static_cast<std::uint32_t>(agent_id),
                      static_cast<std::uint32_t>(k)};
    return std::mt19937_64(seq);
}

// Uniform integer on [lo, hi], inclusive. std::uniform_int_distribution is not
// used here for the same reason std::normal_distribution is avoided in
// population.cpp: its mapping is implementation-defined, and these draws decide
// which negatives a run reports.
std::size_t uniform_index(std::mt19937_64& engine, std::size_t lo, std::size_t hi) {
    const std::uint64_t span = static_cast<std::uint64_t>(hi - lo) + 1u;
    const std::uint64_t draw = engine() % span;
    return lo + static_cast<std::size_t>(draw);
}

}  // namespace

std::vector<std::array<std::int32_t, 3>> generate_swipes(const std::vector<Agent>& pop,
                                                         const UtilityParams& p, int k,
                                                         std::uint64_t rng_seed) {
    if (k <= 0 || pop.empty()) {
        return {};
    }

    const std::size_t n_agents = pop.size();
    const std::size_t stride =
        static_cast<std::size_t>(k) * (1u + static_cast<std::size_t>(kNegativesPerPositive));

    // Fixed-size private block per agent: no thread ever writes where another
    // writes, and no push_back happens inside the parallel region. Blocks that
    // go unused are trimmed during the compaction pass below.
    std::vector<std::array<std::int32_t, 3>> blocks(n_agents * stride);
    std::vector<std::size_t> counts(n_agents, 0);

    std::int64_t agent_count = static_cast<std::int64_t>(n_agents);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::int64_t a = 0; a < agent_count; ++a) {
        const std::size_t i = static_cast<std::size_t>(a);
        const Agent& user = pop[i];

        // Score every candidate of the opposite side. The population is balanced,
        // so this is the other half of it.
        std::vector<Scored> candidates;
        candidates.reserve(n_agents);
        for (std::size_t j = 0; j < n_agents; ++j) {
            if (pop[j].type == user.type) {
                continue;
            }
            candidates.push_back(Scored{user.utility_toward(pop[j], p), pop[j].id});
        }

        std::mt19937_64 engine = agent_engine(rng_seed, user.id, k);

        const std::size_t n_pos =
            std::min(static_cast<std::size_t>(k), candidates.size());
        // partial_sort, not sort: the rest of the candidate list is only needed
        // as an unordered pool of negatives, so ordering it would be wasted work.
        std::partial_sort(candidates.begin(), candidates.begin() + n_pos, candidates.end(),
                          ranks_above);

        // Negatives come from [n_pos, end): everything partial_sort left outside
        // the top-k. Drawing k*4 distinct ones by partially shuffling that tail
        // means no item appears twice for the same user, which keeps the label
        // matrix well-defined.
        const std::size_t n_neg = std::min(
            static_cast<std::size_t>(k) * kNegativesPerPositive, candidates.size() - n_pos);
        for (std::size_t t = 0; t < n_neg; ++t) {
            const std::size_t swap_at = uniform_index(engine, n_pos + t, candidates.size() - 1);
            std::swap(candidates[n_pos + t], candidates[swap_at]);
        }

        std::array<std::int32_t, 3>* block = &blocks[i * stride];
        std::size_t written = 0;
        for (std::size_t t = 0; t < n_pos; ++t) {
            block[written++] = std::array<std::int32_t, 3>{user.id, candidates[t].id, 1};
        }
        for (std::size_t t = 0; t < n_neg; ++t) {
            block[written++] =
                std::array<std::int32_t, 3>{user.id, candidates[n_pos + t].id, 0};
        }
        counts[i] = written;
    }

    // Compact in agent order. This is the step that makes the row order identical
    // whatever order the threads finished in.
    std::size_t total = 0;
    for (const std::size_t c : counts) {
        total += c;
    }

    std::vector<std::array<std::int32_t, 3>> out;
    out.reserve(total);
    for (std::size_t i = 0; i < n_agents; ++i) {
        const std::array<std::int32_t, 3>* block = &blocks[i * stride];
        out.insert(out.end(), block, block + counts[i]);
    }
    return out;
}
