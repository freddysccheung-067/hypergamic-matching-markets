#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "agent.h"

namespace {

Agent make_agent(AgentType type, double status, double attractiveness,
                 double sociability, double age) {
    Agent a;
    a.type = type;
    a.status = status;
    a.attractiveness = attractiveness;
    a.sociability = sociability;
    a.age = age;
    return a;
}

// alpha = beta = gamma = 0 leaves the indicator as the only term that differs
// between any two candidates, so a utility gap is the indicator's contribution.
UtilityParams indicator_only(double delta, double lambda) {
    return UtilityParams{0.0, 0.0, 0.0, delta, lambda};
}

}  // namespace

TEST_CASE("W's indicator fires above her cutoff and not below it", "[agent]") {
    const Agent woman = make_agent(AgentType::W, 0.4, 0.5, 0.5, 30.0);
    const Agent man_surplus = make_agent(AgentType::M, 1.00, 0.5, 0.5, 30.0);
    const Agent man_below = make_agent(AgentType::M, 0.80, 0.5, 0.5, 30.0);

    // δ = 0.5 puts the cutoff at s_w + δ = 0.9: 1.00 clears it, 0.80 does not.
    const UtilityParams p = indicator_only(0.5, 1.5);

    const double u_surplus = woman.utility_toward(man_surplus, p);
    const double u_below = woman.utility_toward(man_below, p);

    CHECK(u_surplus == Catch::Approx(1.5));
    CHECK(u_below == Catch::Approx(0.0));
    CHECK(u_surplus - u_below == Catch::Approx(p.lambda));
    CHECK(u_surplus > u_below);
}

TEST_CASE("W's indicator is inclusive at the cutoff", "[agent]") {
    const Agent woman = make_agent(AgentType::W, 0.4, 0.5, 0.5, 30.0);
    const Agent man_exact = make_agent(AgentType::M, 0.9, 0.5, 0.5, 30.0);
    const UtilityParams p = indicator_only(0.5, 2.0);

    CHECK(woman.utility_toward(man_exact, p) == Catch::Approx(2.0));
}

TEST_CASE("M's utility does not depend on delta", "[agent]") {
    const Agent man = make_agent(AgentType::M, 0.4, 0.6, 0.5, 32.0);
    const Agent woman = make_agent(AgentType::W, 0.7, 0.55, 0.45, 29.0);

    // Non-zero alpha/beta/gamma so the comparison covers the full formula and
    // not just an all-zero degenerate case.
    UtilityParams p{1.0, 0.8, 0.3, /*delta=*/0.0, 1.5};
    const double u_delta_0 = man.utility_toward(woman, p);

    p.delta = 2.0;
    const double u_delta_2 = man.utility_toward(woman, p);

    CHECK(u_delta_0 == Catch::Approx(u_delta_2));
}

TEST_CASE("M's utility matches the document formula", "[agent]") {
    const Agent man = make_agent(AgentType::M, 0.2, 0.6, 0.5, 30.0);
    const Agent woman = make_agent(AgentType::W, 0.4, 0.6, 0.5, 30.0);
    const UtilityParams p{1.0, 0.8, 0.3, 0.5, 1.5};

    // Distance is the 0.2 status gap; attractiveness, sociability and age match.
    const double expected = 1.0 * 0.4 + 0.8 * 0.6 - 0.3 * 0.2;
    CHECK(man.utility_toward(woman, p) == Catch::Approx(expected));
}

TEST_CASE("threshold rises to 1.0 when every proposal matches", "[agent]") {
    Agent a;
    a.threshold = 0.0;

    for (int i = 0; i < 100; ++i) {
        a.update_threshold(10, 10, 0.1);
    }

    CHECK(a.threshold == Catch::Approx(1.0).margin(1e-3));
}

TEST_CASE("threshold falls to 0.0 when no proposal matches", "[agent]") {
    Agent a;
    a.threshold = 1.0;

    for (int i = 0; i < 100; ++i) {
        a.update_threshold(10, 0, 0.1);
    }

    CHECK(a.threshold == Catch::Approx(0.0).margin(1e-3));
}

TEST_CASE("threshold is untouched when there were no proposals", "[agent]") {
    Agent a;
    a.threshold = 0.42;

    a.update_threshold(0, 0, 0.1);

    CHECK(a.threshold == Catch::Approx(0.42));
}
