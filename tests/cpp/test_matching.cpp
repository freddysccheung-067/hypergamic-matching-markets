#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "matching.h"

namespace {

using Lists = std::vector<std::vector<std::int32_t>>;
using Pairs = std::vector<std::array<std::int32_t, 2>>;

bool contains(const std::vector<std::int32_t>& v, std::int32_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// The properties the orchestrator relies on, checked on every result below.
void check_structure(const Lists& prop, const Lists& acc, const MatchResult& r, bool truncated) {
    const auto n_prop = static_cast<std::int32_t>(prop.size());
    const auto n_acc = static_cast<std::int32_t>(acc.size());

    std::set<std::int32_t> used_p;
    std::set<std::int32_t> used_a;
    for (const auto& m : r.matches) {
        REQUIRE(m[0] >= 0);
        REQUIRE(m[0] < n_prop);
        REQUIRE(m[1] >= 0);
        REQUIRE(m[1] < n_acc);
        REQUIRE(used_p.insert(m[0]).second);
        REQUIRE(used_a.insert(m[1]).second);
        REQUIRE(contains(prop[static_cast<std::size_t>(m[0])], m[1]));
        if (truncated) {
            REQUIRE(contains(acc[static_cast<std::size_t>(m[1])], m[0]));
        }
    }

    std::set<std::int32_t> up(r.unmatched_proposers.begin(), r.unmatched_proposers.end());
    std::set<std::int32_t> ua(r.unmatched_acceptors.begin(), r.unmatched_acceptors.end());
    REQUIRE(up.size() == r.unmatched_proposers.size());
    REQUIRE(ua.size() == r.unmatched_acceptors.size());
    for (std::int32_t p = 0; p < n_prop; ++p) {
        REQUIRE((used_p.count(p) == 0) == (up.count(p) == 1));
    }
    for (std::int32_t a = 0; a < n_acc; ++a) {
        REQUIRE((used_a.count(a) == 0) == (ua.count(a) == 1));
    }
    REQUIRE(r.matches.size() + r.unmatched_proposers.size() == prop.size());
    REQUIRE(r.matches.size() + r.unmatched_acceptors.size() == acc.size());

    REQUIRE(r.proposals.size() == prop.size() + acc.size());
    std::int64_t made = 0;
    std::int64_t received = 0;
    for (std::size_t p = 0; p < prop.size(); ++p) {
        REQUIRE(r.proposals[p] >= 0);
        REQUIRE(static_cast<std::size_t>(r.proposals[p]) <= prop[p].size());
        made += r.proposals[p];
    }
    for (std::size_t a = 0; a < acc.size(); ++a) {
        received += r.proposals[prop.size() + a];
    }
    REQUIRE(made == received);
}

struct Lcg {
    std::uint64_t s;
    std::uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(s >> 33);
    }
    std::uint32_t below(std::uint32_t n) { return next() % n; }
};

std::vector<std::int32_t> random_perm(Lcg& g, std::int32_t n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), 0);
    for (std::int32_t i = n - 1; i > 0; --i) {
        std::swap(v[static_cast<std::size_t>(i)], v[g.below(static_cast<std::uint32_t>(i) + 1)]);
    }
    return v;
}

Lists random_lists(Lcg& g, std::int32_t n_owner, std::int32_t n_other, std::int32_t k) {
    Lists out;
    for (std::int32_t i = 0; i < n_owner; ++i) {
        auto perm = random_perm(g, n_other);
        perm.resize(static_cast<std::size_t>(std::min(k, n_other)));
        out.push_back(std::move(perm));
    }
    return out;
}

// Cyclic instance: every proposer gets his first choice and every acceptor her last. It has
// three stable matchings; this is the proposer-optimal, acceptor-pessimal extreme of them.
const Lists kCyclicProp = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
const Lists kCyclicAcc = {{1, 2, 0}, {2, 0, 1}, {0, 1, 2}};

// Crossed instance used for the truncation cases.
const Lists kTextProp = {{0, 1, 2}, {1, 0, 2}, {0, 1, 2}};
const Lists kTextAcc = {{1, 0, 2}, {0, 1, 2}, {0, 1, 2}};

Lists truncate(const Lists& l, std::size_t k) {
    Lists out = l;
    for (auto& v : out) {
        if (v.size() > k) v.resize(k);
    }
    return out;
}

}  // namespace

TEST_CASE("3x3 cyclic instance yields the proposer-optimal stable matching", "[matching]") {
    for (const bool truncated : {false, true}) {
        const MatchResult r = deferred_acceptance(kCyclicProp, kCyclicAcc, truncated);
        const Pairs expected = {{0, 0}, {1, 1}, {2, 2}};
        REQUIRE(r.matches == expected);
        REQUIRE(r.unmatched_proposers.empty());
        REQUIRE(r.unmatched_acceptors.empty());
        REQUIRE(r.n_blocking == 0);
        REQUIRE(r.proposals == std::vector<std::int32_t>{1, 1, 1, 1, 1, 1});
        check_structure(kCyclicProp, kCyclicAcc, r, truncated);
    }
}

TEST_CASE("crossed instance resolves by displacement to the known matching", "[matching]") {
    const MatchResult r = deferred_acceptance(kTextProp, kTextAcc, false);
    const Pairs expected = {{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(r.matches == expected);
    REQUIRE(r.proposals == std::vector<std::int32_t>{1, 1, 3, 2, 2, 1});
    check_structure(kTextProp, kTextAcc, r, false);
}

TEST_CASE("truncating both sides to k=2 leaves a proposer unmatched", "[matching]") {
    const Lists tp = truncate(kTextProp, 2);
    const Lists ta = truncate(kTextAcc, 2);
    const MatchResult r = deferred_acceptance(tp, ta, true);

    const Pairs expected = {{0, 0}, {1, 1}};
    REQUIRE(r.matches == expected);
    REQUIRE(r.unmatched_proposers == std::vector<std::int32_t>{2});
    REQUIRE(r.unmatched_acceptors == std::vector<std::int32_t>{2});
    REQUIRE(r.proposals[2] == 2);
    check_structure(tp, ta, r, true);
}

TEST_CASE("count_blocking_pairs is zero on the full-information matching", "[matching][blocking]") {
    for (const bool truncated : {false, true}) {
        const MatchResult r = deferred_acceptance(kTextProp, kTextAcc, truncated);
        REQUIRE(count_blocking_pairs(kTextProp, kTextAcc, r.matches) == 0);
        const MatchResult c = deferred_acceptance(kCyclicProp, kCyclicAcc, truncated);
        REQUIRE(count_blocking_pairs(kCyclicProp, kCyclicAcc, c.matches) == 0);
    }
    // The counter is not vacuous: with nobody matched, every pair blocks.
    REQUIRE(count_blocking_pairs(kTextProp, kTextAcc, Pairs{}) == 9);
}

TEST_CASE("truncated matching is unstable under the full lists", "[matching][blocking]") {
    const Lists tp = truncate(kTextProp, 2);
    const Lists ta = truncate(kTextAcc, 2);
    const MatchResult r = deferred_acceptance(tp, ta, true);

    // Stable against the lists it was given, by construction.
    REQUIRE(r.n_blocking == 0);
    REQUIRE(count_blocking_pairs(tp, ta, r.matches) == 0);

    // Under the true ordering (m2, w2) blocks: both unmatched, each prefers the other.
    REQUIRE(count_blocking_pairs(kTextProp, kTextAcc, r.matches) >= 1);
}

TEST_CASE("empty inputs return an empty result", "[matching][edge]") {
    for (const bool truncated : {false, true}) {
        SECTION("no agents at all") {
            const MatchResult r = deferred_acceptance({}, {}, truncated);
            REQUIRE(r.matches.empty());
            REQUIRE(r.unmatched_proposers.empty());
            REQUIRE(r.unmatched_acceptors.empty());
            REQUIRE(r.proposals.empty());
            REQUIRE(r.n_blocking == 0);
            REQUIRE(count_blocking_pairs({}, {}, {}) == 0);
        }
        SECTION("agents with empty lists") {
            const Lists prop(3);
            const Lists acc(2);
            const MatchResult r = deferred_acceptance(prop, acc, truncated);
            REQUIRE(r.matches.empty());
            REQUIRE(r.unmatched_proposers == std::vector<std::int32_t>{0, 1, 2});
            REQUIRE(r.unmatched_acceptors == std::vector<std::int32_t>{0, 1});
            REQUIRE(r.proposals == std::vector<std::int32_t>(5, 0));
            REQUIRE(r.n_blocking == 0);
            REQUIRE(count_blocking_pairs(prop, acc, {}) == 0);
        }
        SECTION("one side empty") {
            const Lists acc(2);
            const MatchResult r = deferred_acceptance({}, acc, truncated);
            REQUIRE(r.matches.empty());
            REQUIRE(r.unmatched_acceptors == std::vector<std::int32_t>{0, 1});
            REQUIRE(r.proposals == std::vector<std::int32_t>(2, 0));
        }
    }
}

TEST_CASE("unequal sides: 5 proposers, 3 acceptors", "[matching][edge]") {
    const Lists prop(5, std::vector<std::int32_t>{0, 1, 2});
    const Lists acc(3, std::vector<std::int32_t>{0, 1, 2, 3, 4});

    for (const bool truncated : {false, true}) {
        const MatchResult r = deferred_acceptance(prop, acc, truncated);
        REQUIRE(r.matches.size() == 3);
        REQUIRE(r.unmatched_proposers.size() == 2);
        REQUIRE(r.unmatched_acceptors.empty());
        check_structure(prop, acc, r, truncated);
    }
}

TEST_CASE("identical proposer lists terminate and give a valid matching", "[matching][edge]") {
    const Lists prop = {{0, 1, 2}, {0, 1, 2}};
    const Lists acc = {{1, 0}, {1, 0}, {1, 0}};
    for (const bool truncated : {false, true}) {
        const MatchResult r = deferred_acceptance(prop, acc, truncated);
        const Pairs expected = {{0, 1}, {1, 0}};
        REQUIRE(r.matches == expected);
        REQUIRE(r.unmatched_proposers.empty());
        REQUIRE(r.unmatched_acceptors == std::vector<std::int32_t>{2});
        check_structure(prop, acc, r, truncated);
    }
}

TEST_CASE("malformed lists are rejected rather than silently accepted", "[matching][edge]") {
    REQUIRE_THROWS_AS(deferred_acceptance(Lists{{0, 0}}, Lists{{0}}, false), std::invalid_argument);
    REQUIRE_THROWS_AS(deferred_acceptance(Lists{{1}}, Lists{{0}}, false), std::invalid_argument);
    REQUIRE_THROWS_AS(deferred_acceptance(Lists{{0}}, Lists{{-1}}, false), std::invalid_argument);
    REQUIRE_THROWS_AS(count_blocking_pairs(kTextProp, kTextAcc, Pairs{{0, 0}, {0, 1}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(count_blocking_pairs(kTextProp, kTextAcc, Pairs{{3, 0}}),
                      std::invalid_argument);
}

TEST_CASE("proposals is sized n_prop + n_acc and bounded by list lengths", "[matching][proposals]") {
    const Lists tp = truncate(kTextProp, 2);
    const Lists ta = truncate(kTextAcc, 2);
    const MatchResult r = deferred_acceptance(tp, ta, true);

    REQUIRE(r.proposals.size() == tp.size() + ta.size());
    const std::int32_t proposer_total =
        std::accumulate(r.proposals.begin(), r.proposals.begin() + 3, 0);
    REQUIRE(proposer_total <= static_cast<std::int32_t>(tp.size()) * 2);
    for (std::size_t p = 0; p < tp.size(); ++p) {
        REQUIRE(static_cast<std::size_t>(r.proposals[p]) <= tp[p].size());
    }
    const std::int32_t acceptor_total =
        std::accumulate(r.proposals.begin() + 3, r.proposals.end(), 0);
    REQUIRE(acceptor_total == proposer_total);
}

TEST_CASE("random instances satisfy the structural properties", "[matching][random]") {
    Lcg g{12345};
    for (int trial = 0; trial < 300; ++trial) {
        const auto n_prop = static_cast<std::int32_t>(g.below(12));
        const auto n_acc = static_cast<std::int32_t>(g.below(12));
        const auto k = static_cast<std::int32_t>(1 + g.below(8));

        {
            const Lists prop = random_lists(g, n_prop, n_acc, n_acc);
            const Lists acc = random_lists(g, n_acc, n_prop, n_prop);
            const MatchResult r = deferred_acceptance(prop, acc, false);
            check_structure(prop, acc, r, false);
            REQUIRE(r.n_blocking == 0);
            REQUIRE(count_blocking_pairs(prop, acc, r.matches) == 0);
            REQUIRE(r.matches.size() == static_cast<std::size_t>(std::min(n_prop, n_acc)));
        }
        {
            const Lists prop = random_lists(g, n_prop, n_acc, k);
            const Lists acc = random_lists(g, n_acc, n_prop, k);
            const MatchResult r = deferred_acceptance(prop, acc, true);
            check_structure(prop, acc, r, true);
            REQUIRE(r.n_blocking == 0);
            REQUIRE(count_blocking_pairs(prop, acc, r.matches) == 0);
        }
    }
}
