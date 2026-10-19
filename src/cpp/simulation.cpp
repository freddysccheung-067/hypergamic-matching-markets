#include "simulation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// A candidate paired with the utility it earns. Carrying the score next to the id
// is what lets the partition below split admissible from inadmissible in one pass
// and then rank only the admissible part.
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
// threads ever touch the same engine. The epoch is mixed in so that an unchanged
// population still produces fresh negatives in the next epoch; without it the
// negative set would repeat forever and the NCF would memorise it.
std::mt19937_64 agent_engine(std::uint64_t seed, std::int32_t agent_id, int k,
                             std::uint64_t epoch) {
    std::seed_seq seq{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(agent_id),
        static_cast<std::uint32_t>(k),
        static_cast<std::uint32_t>(epoch),
        static_cast<std::uint32_t>(epoch >> 32),
    };
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
                                                         const UtilityParams& p,
                                                         const TraitStats& stats, int k,
                                                         std::uint64_t rng_seed,
                                                         std::uint64_t epoch,
                                                         int n_negatives) {
    if (k <= 0 || pop.empty()) {
        return {};
    }
    // A negative ratio is meaningless and would underflow the unsigned stride
    // arithmetic below into a gigantic allocation. Treat it as "no negatives",
    // which is a legal configuration: an all-positive log.
    if (n_negatives < 0) {
        n_negatives = 0;
    }

    const std::size_t n_agents = pop.size();
    const std::size_t stride =
        static_cast<std::size_t>(k) * (1u + static_cast<std::size_t>(n_negatives));

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
            candidates.push_back(Scored{user.utility_toward(pop[j], p, stats), pop[j].id});
        }

        // θ gates acceptance: a candidate is swipeable only if its utility clears
        // the user's current threshold. Partitioning first keeps the ranking below
        // proportional to the admissible set rather than the whole market.
        //
        // The comparison is written `!(u >= theta)` so that a NaN utility is
        // treated as inadmissible instead of slipping through an inverted test.
        const double theta = user.threshold;
        const auto first_inadmissible =
            std::partition(candidates.begin(), candidates.end(), [theta](const Scored& s) {
                return s.utility >= theta;
            });
        const std::size_t n_admissible = static_cast<std::size_t>(
            std::distance(candidates.begin(), first_inadmissible));

        std::mt19937_64 engine = agent_engine(rng_seed, user.id, k, epoch);

        // Positives are the top-k of the admissible set, so both k and θ bound
        // how many an agent gives.
        const std::size_t n_pos = std::min(static_cast<std::size_t>(k), n_admissible);
        std::partial_sort(candidates.begin(), candidates.begin() + n_pos, first_inadmissible,
                          ranks_above);

        // Negatives come from everything that is not a reported positive. That
        // pool is [n_pos, end): the admissible candidates outside the top-k
        // followed by every inadmissible one, in whatever order the partition
        // left them.
        const std::size_t n_neg = std::min(
            static_cast<std::size_t>(k) * static_cast<std::size_t>(n_negatives),
            candidates.size() - n_pos);
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
