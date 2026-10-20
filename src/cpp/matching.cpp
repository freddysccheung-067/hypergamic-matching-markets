#include "matching.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// One side's preference lists, flattened so a rank lookup is a binary search
// instead of a linear scan. Preference order is not id order, so the flat block
// is sorted by id and carries the original position alongside.
//
// Memory is O(sum of list lengths), which for the truncated case is O(N·k) and
// for the full-information case O(N²) — the same order the lists themselves
// already cost, so this does not change the limits in INTERFACES.md §4.
class RankIndex {
public:
    RankIndex(const std::vector<std::vector<std::int32_t>>& lists, int n_other) {
        offset_.reserve(lists.size() + 1);
        offset_.push_back(0);
        std::vector<std::pair<std::int32_t, std::int32_t>> block;
        for (const auto& list : lists) {
            block.clear();
            block.reserve(list.size());
            for (std::size_t r = 0; r < list.size(); ++r) {
                const std::int32_t id = list[r];
                if (id < 0 || id >= n_other) {
                    throw std::invalid_argument(
                        "preference list contains id " + std::to_string(id) +
                        " outside the other side's range [0, " + std::to_string(n_other) + ")");
                }
                block.emplace_back(id, static_cast<std::int32_t>(r));
            }
            std::sort(block.begin(), block.end());
            // A repeated id would make the binary search ambiguous and let a
            // proposer be considered twice; refuse rather than guess.
            for (std::size_t i = 1; i < block.size(); ++i) {
                if (block[i].first == block[i - 1].first) {
                    throw std::invalid_argument("preference list repeats id " +
                                                std::to_string(block[i].first));
                }
            }
            for (const auto& [id, rank] : block) {
                ids_.push_back(id);
                ranks_.push_back(rank);
            }
            offset_.push_back(static_cast<std::int32_t>(ids_.size()));
        }
    }

    // Position of `id` in that agent's list, or -1 when the list does not
    // contain it — which, in the truncated regime, means "not acceptable".
    std::int32_t rank_of(std::int32_t outer, std::int32_t id) const {
        const auto lo = ids_.begin() + offset_[static_cast<std::size_t>(outer)];
        const auto hi = ids_.begin() + offset_[static_cast<std::size_t>(outer) + 1];
        const auto it = std::lower_bound(lo, hi, id);
        if (it == hi || *it != id) {
            return -1;
        }
        return ranks_[static_cast<std::size_t>(it - ids_.begin())];
    }

    bool contains(std::int32_t outer, std::int32_t id) const {
        return rank_of(outer, id) >= 0;
    }

private:
    std::vector<std::int32_t> offset_;
    std::vector<std::int32_t> ids_;
    std::vector<std::int32_t> ranks_;
};

// Rank used for "acceptable but unlisted", which can only happen in the
// full-information regime. Larger than any real rank.
constexpr std::int32_t kRankUnlisted = std::numeric_limits<std::int32_t>::max();

// Rank used for "not matched". Anything acceptable beats it, including an
// unlisted proposer, which is what makes an unmatched proposer blocking against
// every acceptor who ranks him above her partner.
constexpr std::int32_t kRankUnmatched = std::numeric_limits<std::int32_t>::max();

// Counts mutually-preferring pairs the matching leaves unrealized.
//
// The definition has to work from the two lists alone, and it has to be zero on
// complete lists — otherwise it is measuring a bug rather than instability. So a
// pair (p, a) blocks when
//
//   * they are not matched to each other,
//   * a is unmatched or ranks p above her current partner,
//   * p finds a acceptable — she is in his list — and
//   * p is unmatched, or p lists a above his current partner.
//
// The acceptability clause is what makes one rule serve both regimes. With
// complete lists every acceptance is acceptable to him, so it is vacuous and
// the test reduces to the textbook one; Gale-Shapley admits none of these,
// which is the property the full-information test asserts. With truncated lists
// it is the definition from the incomplete-list stable-matching model: a pair
// neither side knows about cannot block.
//
// The case that clause hides is the one worth understanding, because it is the
// information loss itself rather than a violation of it. a lists p highly, but
// p's recommendation list never contained a, so he never proposed and ends
// unmatched while she settles for someone worse. That is a genuine welfare cost
// and it does NOT show up here — both sides are being individually consistent
// with what they were shown. It shows up in algorithmic loss and in RSI instead.
//
// Iterating acceptors' lists is enough to find every blocking pair: a needs p to
// be acceptable, and being in a's list is what acceptable means for her.
std::int32_t count_blocking_pairs(const RankIndex& acc_index, const RankIndex& prop_index,
                                  const std::vector<std::vector<std::int32_t>>& acc_prefs,
                                  const std::vector<std::int32_t>& partner_prop,
                                  const std::vector<std::int32_t>& partner_acc) {
    std::int32_t blocking = 0;
    for (std::size_t a = 0; a < acc_prefs.size(); ++a) {
        const std::int32_t ai = static_cast<std::int32_t>(a);
        const std::int32_t partner = partner_acc[a];
        const std::int32_t partner_rank_w = (partner < 0) ? kRankUnmatched
                                                          : acc_index.rank_of(ai, partner);
        for (const std::int32_t p : acc_prefs[a]) {
            if (p == partner) {
                continue;
            }
            const std::int32_t rank_w = acc_index.rank_of(ai, p);
            if (rank_w > partner_rank_w) {
                continue;  // she does not prefer him
            }
            const std::int32_t rank_m = prop_index.rank_of(p, ai);
            if (rank_m < 0) {
                continue;  // not acceptable to him, so the pair cannot block
            }
            const std::int32_t his_partner = partner_prop[static_cast<std::size_t>(p)];
            if (his_partner < 0) {
                ++blocking;  // he could only improve on being unmatched
                continue;
            }
            const std::int32_t his_partner_rank = prop_index.rank_of(p, his_partner);
            if (rank_m < his_partner_rank) {
                ++blocking;
            }
        }
    }
    return blocking;
}

}  // namespace

PrefLists build_full_info_pref_lists(const std::vector<Agent>& pop, const UtilityParams& p,
                                     const TraitStats& stats) {
    std::vector<std::int32_t> proposer_rows;
    std::vector<std::int32_t> acceptor_rows;
    for (const Agent& a : pop) {
        if (a.type == AgentType::M) {
            proposer_rows.push_back(a.id);
        } else {
            acceptor_rows.push_back(a.id);
        }
    }

    // A dense index is a position in these vectors, so id and index coincide
    // only when the population is grouped by type and ordered by id — which
    // sample_population arranges but a caller need not. The lookup goes through
    // the row, never through the id.
    const std::size_t n_prop = proposer_rows.size();
    const std::size_t n_acc = acceptor_rows.size();
    std::vector<std::int32_t> prop_slot(pop.size(), -1);
    std::vector<std::int32_t> acc_slot(pop.size(), -1);
    for (std::size_t i = 0; i < n_prop; ++i) {
        prop_slot[static_cast<std::size_t>(proposer_rows[i])] = static_cast<std::int32_t>(i);
    }
    for (std::size_t i = 0; i < n_acc; ++i) {
        acc_slot[static_cast<std::size_t>(acceptor_rows[i])] = static_cast<std::int32_t>(i);
    }

    PrefLists lists;
    lists.proposers.assign(n_prop, {});
    lists.acceptors.assign(n_acc, {});

    struct Scored {
        double utility;
        std::int32_t slot;
    };
    auto rank_against = [&pop, &p, &stats](const Agent& self, const std::vector<std::int32_t>& rows,
                                           const std::vector<std::int32_t>& slot_of_row,
                                           std::vector<std::int32_t>& out) {
        std::vector<Scored> scored;
        scored.reserve(rows.size());
        for (const std::int32_t row : rows) {
            const Agent& cand = pop[static_cast<std::size_t>(row)];
            scored.push_back(Scored{self.utility_toward(cand, p, stats),
                                    slot_of_row[static_cast<std::size_t>(row)]});
        }
        // Descending utility, ties by ascending slot. The tie-break keeps the
        // order total, so two agents with identical utilities still get a
        // well-defined list and the output does not depend on sort internals.
        std::sort(scored.begin(), scored.end(), [](const Scored& x, const Scored& y) {
            if (x.utility != y.utility) {
                return x.utility > y.utility;
            }
            return x.slot < y.slot;
        });
        out.clear();
        out.reserve(scored.size());
        for (const Scored& s : scored) {
            out.push_back(s.slot);
        }
    };

    for (std::size_t i = 0; i < n_prop; ++i) {
        rank_against(pop[static_cast<std::size_t>(proposer_rows[i])], acceptor_rows, acc_slot,
                     lists.proposers[i]);
    }
    for (std::size_t i = 0; i < n_acc; ++i) {
        rank_against(pop[static_cast<std::size_t>(acceptor_rows[i])], proposer_rows, prop_slot,
                     lists.acceptors[i]);
    }
    return lists;
}

MatchResult deferred_acceptance(const std::vector<std::vector<std::int32_t>>& prop_prefs,
                                const std::vector<std::vector<std::int32_t>>& acc_prefs,
                                bool truncated) {
    const auto n_prop = static_cast<std::int32_t>(prop_prefs.size());
    const auto n_acc = static_cast<std::int32_t>(acc_prefs.size());

    const RankIndex prop_index(prop_prefs, n_acc);
    const RankIndex acc_index(acc_prefs, n_prop);

    std::vector<std::int32_t> partner_prop(static_cast<std::size_t>(n_prop), -1);
    std::vector<std::int32_t> partner_acc(static_cast<std::size_t>(n_acc), -1);

    MatchResult result;
    result.proposals.assign(static_cast<std::size_t>(n_prop) + static_cast<std::size_t>(n_acc), 0);

    // Free proposers, as an explicit stack. A proposer returns to it only when
    // displaced, so he works down his own list monotonically — the reason the
    // loop is O(N·k) rather than quadratic in the number of proposals.
    std::vector<std::int32_t> free_proposers;
    free_proposers.reserve(static_cast<std::size_t>(n_prop));
    for (std::int32_t p = n_prop - 1; p >= 0; --p) {
        free_proposers.push_back(p);
    }

    std::vector<std::size_t> next(static_cast<std::size_t>(n_prop), 0);

    while (!free_proposers.empty()) {
        const std::int32_t p = free_proposers.back();
        free_proposers.pop_back();
        const auto& list = prop_prefs[static_cast<std::size_t>(p)];
        if (next[static_cast<std::size_t>(p)] >= list.size()) {
            continue;  // list exhausted, unmatched; recorded below
        }

        const std::int32_t a = list[next[static_cast<std::size_t>(p)]++];
        result.proposals[static_cast<std::size_t>(p)] += 1;
        result.proposals[static_cast<std::size_t>(n_prop) + static_cast<std::size_t>(a)] += 1;

        if (truncated && !acc_index.contains(a, p)) {
            // She does not know he exists. This is the whole point of the
            // truncated arm: the rejection is informational, not preferential.
            free_proposers.push_back(p);
            continue;
        }

        const std::int32_t current = partner_acc[static_cast<std::size_t>(a)];
        if (current < 0) {
            partner_acc[static_cast<std::size_t>(a)] = p;
            partner_prop[static_cast<std::size_t>(p)] = a;
            continue;
        }

        // Unlisted is acceptable only in the full-information regime, and then
        // only below every listed proposer. In the truncated regime an unlisted
        // proposer was already turned away above, so rank_new here is >= 0.
        const std::int32_t rank_new = acc_index.rank_of(a, p);
        const std::int32_t rank_current = acc_index.rank_of(a, current);
        const std::int32_t effective_new = (rank_new < 0) ? kRankUnlisted : rank_new;
        const std::int32_t effective_current = (rank_current < 0) ? kRankUnlisted : rank_current;

        if (effective_new < effective_current) {
            partner_acc[static_cast<std::size_t>(a)] = p;
            partner_prop[static_cast<std::size_t>(p)] = a;
            partner_prop[static_cast<std::size_t>(current)] = -1;
            free_proposers.push_back(current);
        } else {
            // Keeps her current partner; the proposer stays free and tries the
            // next woman on his list.
            free_proposers.push_back(p);
        }
    }

    for (std::int32_t a = 0; a < n_acc; ++a) {
        const std::int32_t p = partner_acc[static_cast<std::size_t>(a)];
        if (p >= 0) {
            result.matches.push_back(std::array<std::int32_t, 2>{p, a});
        } else {
            result.unmatched_acceptors.push_back(a);
        }
    }
    for (std::int32_t p = 0; p < n_prop; ++p) {
        if (partner_prop[static_cast<std::size_t>(p)] < 0) {
            result.unmatched_proposers.push_back(p);
        }
    }

    // Matches are emitted acceptor-major; sort to proposer-major so the array is
    // stable against changes in acceptor count and reads in the order the pairs
    // were formed.
    std::sort(result.matches.begin(), result.matches.end(),
              [](const std::array<std::int32_t, 2>& x, const std::array<std::int32_t, 2>& y) {
                  return x[0] < y[0];
              });

    result.n_blocking = count_blocking_pairs(acc_index, prop_index, acc_prefs, partner_prop,
                                             partner_acc);
    return result;
}

std::int32_t count_blocking_pairs(
    const std::vector<std::vector<std::int32_t>>& prop_prefs,
    const std::vector<std::vector<std::int32_t>>& acc_prefs,
    const std::vector<std::array<std::int32_t, 2>>& matches) {
    const auto n_prop = static_cast<std::int32_t>(prop_prefs.size());
    const auto n_acc = static_cast<std::int32_t>(acc_prefs.size());
    const RankIndex prop_index(prop_prefs, n_acc);
    const RankIndex acc_index(acc_prefs, n_prop);

    std::vector<std::int32_t> partner_prop(static_cast<std::size_t>(n_prop), -1);
    std::vector<std::int32_t> partner_acc(static_cast<std::size_t>(n_acc), -1);
    for (const auto& pair : matches) {
        const std::int32_t p = pair[0];
        const std::int32_t a = pair[1];
        if (p < 0 || a < 0 || p >= n_prop || a >= n_acc) {
            throw std::invalid_argument("count_blocking_pairs: match index out of range");
        }
        if (partner_prop[static_cast<std::size_t>(p)] >= 0 ||
            partner_acc[static_cast<std::size_t>(a)] >= 0) {
            // A matching is a partial function on each side. Two entries for one
            // agent would corrupt the partner lookups above and the count would
            // be meaningless rather than merely wrong.
            throw std::invalid_argument("count_blocking_pairs: an agent appears twice");
        }
        partner_prop[static_cast<std::size_t>(p)] = a;
        partner_acc[static_cast<std::size_t>(a)] = p;
    }
    return count_blocking_pairs(acc_index, prop_index, acc_prefs, partner_prop, partner_acc);
}

Welfare compute_welfare(const std::vector<Agent>& pop,
                        const std::vector<std::array<std::int32_t, 2>>& matches,
                        const UtilityParams& p, const TraitStats& stats) {
    // Row lookup by dense index, the same mapping the preference lists use.
    std::vector<std::int32_t> prop_rows;
    std::vector<std::int32_t> acc_rows;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        if (pop[i].type == AgentType::M) {
            prop_rows.push_back(static_cast<std::int32_t>(i));
        } else {
            acc_rows.push_back(static_cast<std::int32_t>(i));
        }
    }

    Welfare w;
    for (const auto& pair : matches) {
        const std::int32_t pi = pair[0];
        const std::int32_t ai = pair[1];
        if (pi < 0 || ai < 0 || static_cast<std::size_t>(pi) >= prop_rows.size() ||
            static_cast<std::size_t>(ai) >= acc_rows.size()) {
            throw std::invalid_argument("compute_welfare: match index out of range");
        }
        const Agent& m = pop[static_cast<std::size_t>(prop_rows[static_cast<std::size_t>(pi)])];
        const Agent& f = pop[static_cast<std::size_t>(acc_rows[static_cast<std::size_t>(ai)])];
        const double u_m = m.utility_toward(f, p, stats);
        const double u_f = f.utility_toward(m, p, stats);
        w.proposer += u_m;
        w.acceptor += u_f;
        w.value_proposer += m.intent * u_m;
        w.value_acceptor += f.intent * u_f;
        ++w.n_pairs;
    }
    w.total = w.proposer + w.acceptor;
    w.value_created = w.value_proposer + w.value_acceptor;
    return w;
}

void update_thresholds(std::vector<Agent>& pop, const std::vector<std::int32_t>& n_proposals,
                       const std::vector<std::int32_t>& n_matches, double eta) {
    if (n_proposals.size() != pop.size() || n_matches.size() != pop.size()) {
        throw std::invalid_argument(
            "update_thresholds: n_proposals and n_matches must be indexed by row and have "
            "length equal to the population");
    }
    for (std::size_t i = 0; i < pop.size(); ++i) {
        const auto proposals = static_cast<std::uint32_t>(std::max<std::int32_t>(n_proposals[i], 0));
        const auto matches =
            static_cast<std::uint32_t>(std::max<std::int32_t>(n_matches[i], 0));
        pop[i].update_threshold(proposals, matches, eta);
    }
}
