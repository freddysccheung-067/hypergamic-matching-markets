#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "agent.h"
#include "matching.h"
#include "population.h"
#include "simulation.h"

// The Python surface fixed by docs/INTERFACES.md §2. The C++ core keeps its own
// signatures (named fields, std::vector<Agent>); the flat (N, kStateCols) float64
// matrix the contract specifies exists only at this boundary, where numpy lives.

namespace py = pybind11;

namespace {

// Read a utility block from the config dict. The five weights are required rather
// than defaulted: a silently missing weight would change every welfare number in
// the project, which is the failure mode §2.2 warns about.
//
// shock_scale is the exception and defaults to 0.0. It is a modeling knob whose
// "off" value is meaningful and is the historical behavior, so a config written
// before this key existed must keep running unchanged. shock_seed is not a config
// key at all: it is population.seed, so a run's chemistry follows the run's seed
// without a second number anyone could forget to set.
UtilityParams params_from(const py::dict& config) {
    if (!config.contains("hypergamy")) {
        throw std::invalid_argument("config is missing the 'hypergamy' block");
    }
    const py::dict h = config["hypergamy"].cast<py::dict>();
    auto get = [&h](const char* key) -> double {
        if (!h.contains(key)) {
            throw std::invalid_argument(std::string("hypergamy block is missing '") + key + "'");
        }
        return h[key].cast<double>();
    };
    double shock_scale = 0.0;
    if (h.contains("shock_scale")) {
        shock_scale = h["shock_scale"].cast<double>();
    }
    if (shock_scale < 0.0) {
        throw std::invalid_argument("hypergamy.shock_scale must be >= 0; got " +
                                    std::to_string(shock_scale));
    }

    UtilityParams p{get("alpha"), get("beta"), get("gamma"), get("delta"), get("lambda")};
    p.shock_scale = shock_scale;
    if (config.contains("population")) {
        const py::dict pop = config["population"].cast<py::dict>();
        if (pop.contains("seed")) {
            p.shock_seed = pop["seed"].cast<std::uint64_t>();
        }
    }
    return p;
}

// The proposal-decision parameters (THEORY.md §10.2). c is read from the
// hypergamy block; p is per side and comes from an optional "signal" block the
// orchestrator fills with the previous epoch's empirical accept rate, one entry
// per proposing side. Both are optional and default to c = 0, p = 0.5, which is
// the "no signalling cost" behavior every config written before this session had
// and the epoch-0 rule for p.
//
// Why p lives in its own block rather than in hypergamy: hypergamy's five weights
// are pinned and required, and a knob that changes every epoch is not a model
// parameter but a state variable. Keeping them apart means a config read by an
// older tool still resolves every weight it knows.
SignalSide signal_from(const py::dict& config) {
    SignalSide sides;
    double cost = 0.0;
    if (config.contains("hypergamy")) {
        const py::dict h = config["hypergamy"].cast<py::dict>();
        if (h.contains("signal_cost")) {
            cost = h["signal_cost"].cast<double>();
            if (!(cost >= 0.0)) {
                throw std::invalid_argument("hypergamy.signal_cost must be >= 0; got " +
                                            std::to_string(cost));
            }
        }
    }
    for (SignalParams& s : sides) {
        s.cost = cost;
    }
    if (config.contains("signal")) {
        const py::dict sig = config["signal"].cast<py::dict>();
        // accept_prob applies to both sides when given; the per-side keys override
        // it. The single key exists so a hand-written config or a test can set one
        // number without inventing a split it does not have information for.
        if (sig.contains("accept_prob")) {
            const double p_both = sig["accept_prob"].cast<double>();
            if (!(p_both >= 0.0) || !(p_both <= 1.0)) {
                throw std::invalid_argument("signal.accept_prob must be in [0, 1]; got " +
                                            std::to_string(p_both));
            }
            sides[0].accept_prob = p_both;
            sides[1].accept_prob = p_both;
        }
        for (const auto& [key, slot] :
             {std::pair{"accept_prob_m", std::size_t{0}},
              std::pair{"accept_prob_w", std::size_t{1}}}) {
            if (sig.contains(key)) {
                const double p = sig[key].cast<double>();
                if (!(p >= 0.0) || !(p <= 1.0)) {
                    throw std::invalid_argument(std::string("signal.") + key +
                                                " must be in [0, 1]; got " + std::to_string(p));
                }
                sides[slot].accept_prob = p;
            }
        }
    }
    return sides;
}

TraitDist trait_from(const py::dict& d, const char* field) {
    if (!d.contains("dist") || !d.contains("mean") || !d.contains("std")) {
        throw std::invalid_argument(std::string("population.traits.") + field +
                                    " needs dist, mean and std");
    }
    TraitDist t;
    t.name = d["dist"].cast<std::string>();
    t.p1 = d["mean"].cast<double>();
    t.p2 = d["std"].cast<double>();
    return t;
}

PopConfig pop_config_from(const py::dict& config) {
    PopConfig cfg;
    if (!config.contains("population")) {
        throw std::invalid_argument("config is missing the 'population' block");
    }
    const py::dict pop = config["population"].cast<py::dict>();
    if (!pop.contains("traits")) {
        throw std::invalid_argument("config is missing 'population.traits'");
    }
    const py::dict traits = pop["traits"].cast<py::dict>();
    for (const char* field : {"status", "attractiveness", "sociability", "age"}) {
        if (!traits.contains(field)) {
            throw std::invalid_argument(std::string("population.traits is missing '") + field +
                                        "'");
        }
    }
    cfg.status = trait_from(traits["status"].cast<py::dict>(), "status");
    cfg.attractiveness = trait_from(traits["attractiveness"].cast<py::dict>(), "attractiveness");
    cfg.sociability = trait_from(traits["sociability"].cast<py::dict>(), "sociability");
    cfg.age = trait_from(traits["age"].cast<py::dict>(), "age");

    // θ is an absolute utility cutoff, so it is calibrated per population rather
    // than taken from the config as a raw number. These two keys are independent:
    // admissible_fraction sets the calibration target, init_threshold bypasses
    // calibration entirely. THEORY.md §3 documents why the default is a quantile
    // and not a constant.
    if (pop.contains("admissible_fraction")) {
        cfg.admissible_fraction = pop["admissible_fraction"].cast<double>();
    }
    // Absent means 0.0, the "no heterogeneity" default: a config predating this
    // key must sample exactly the population it used to.
    if (pop.contains("taste_sigma")) {
        cfg.taste_sigma = pop["taste_sigma"].cast<double>();
        if (cfg.taste_sigma < 0.0) {
            throw std::invalid_argument("population.taste_sigma must be >= 0; got " +
                                        std::to_string(cfg.taste_sigma));
        }
    }

    // Institutional columns (THEORY.md §10.1). All optional and defaulted to the
    // shipped values, so a config predating these keys samples exactly the traits
    // it used to: these draws live on their own stream and touch no trait column.
    auto get_opt = [&pop](const char* key, double& out) {
        if (pop.contains(key)) {
            out = pop[key].cast<double>();
        }
    };
    get_opt("intent_beta_a", cfg.intent_beta_a);
    get_opt("intent_beta_b", cfg.intent_beta_b);
    get_opt("values_beta_a", cfg.values_beta_a);
    get_opt("values_beta_b", cfg.values_beta_b);
    get_opt("community_prob", cfg.community_prob);
    // Shapes are validated again inside sample_population, which is the function
    // that cannot run without them; this check just fails with a Python-level
    // message naming the key rather than a std::invalid_argument from deep inside.
    for (const auto& [a, b, name] :
         {std::tuple{cfg.intent_beta_a, cfg.intent_beta_b, "intent_beta"},
          std::tuple{cfg.values_beta_a, cfg.values_beta_b, "values_beta"}}) {
        if (!(a > 0.0) || !(b > 0.0)) {
            throw std::invalid_argument(std::string("population.") + name +
                                        "_a and " + name + "_b must be > 0");
        }
    }
    if (!(cfg.community_prob >= 0.0) || !(cfg.community_prob <= 1.0)) {
        throw std::invalid_argument("population.community_prob must be in [0, 1]; got " +
                                    std::to_string(cfg.community_prob));
    }
    return cfg;
}

// A caller passing a raw init_threshold is asking for an absolute cutoff; honor
// it instead of calibrating. Kept separate from pop_config_from because it is a
// decision about behavior, not about parsing.
bool has_explicit_threshold(const py::dict& config) {
    if (!config.contains("population")) {
        return false;
    }
    const py::dict pop = config["population"].cast<py::dict>();
    return pop.contains("init_threshold");
}

// config["ncf"]["n_negatives"], defaulting to 4 when the block or the key is
// absent. Defaulting rather than requiring is deliberate and asymmetric with the
// hypergamy weights above: those are required because a missing one silently
// changes every welfare number, whereas this one only changes training hardness
// and 4 is the documented default (docs/INTERFACES.md §2). A run that omits the
// key therefore reproduces the historical behavior exactly.
int n_negatives_from(const py::dict& config) {
    if (!config.contains("ncf")) {
        return 4;
    }
    const py::dict ncf = config["ncf"].cast<py::dict>();
    if (!ncf.contains("n_negatives")) {
        return 4;
    }
    const int n = ncf["n_negatives"].cast<int>();
    if (n < 0) {
        throw std::invalid_argument("ncf.n_negatives must be >= 0; got " + std::to_string(n));
    }
    return n;
}

// Validate an incoming (N, kStateCols) float64 array and hand back a contiguous
// copy. The width comes from population.h's kStateCols rather than a literal, so
// this guard and flatten_states cannot disagree — a guard left at the old width
// would reject every array the sampler produces.
py::array_t<double, py::array::c_style | py::array::forcecast> require_states(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& states) {
    const auto info = states.request();
    if (info.ndim != 2 || info.shape[1] != static_cast<py::ssize_t>(kStateCols)) {
        throw std::invalid_argument("states must be a C-contiguous float64 array of shape (N, " +
                                    std::to_string(kStateCols) + ")");
    }
    return states;
}

}  // namespace

PYBIND11_MODULE(_hypergamic, m) {
    m.doc() = "Hypergamic matching-market core (see docs/INTERFACES.md §2)";

    m.def(
        "sample_population",
        [](int n, const py::dict& config, std::uint64_t seed) {
            PopConfig cfg = pop_config_from(config);

            // Two mutually exclusive modes:
            //
            //   population.init_threshold given -> an absolute utility cutoff,
            //       used verbatim. This is the escape hatch for a caller that
            //       wants a specific θ rather than a target admissibility.
            //   otherwise -> θ is calibrated after the draws, so that "roughly
            //       population.admissible_fraction of candidates are swipeable"
            //       holds at the start of a run. The fraction is not a threshold
            //       and must not reach the sampler as one.
            const bool explicit_theta = has_explicit_threshold(config);
            const double admissible_fraction = cfg.admissible_fraction;
            cfg.init_threshold =
                explicit_theta ? config["population"]["init_threshold"].cast<double>()
                               : kUncalibratedThreshold;

            std::vector<Agent> pop = sample_population(n, cfg, seed);
            // Standardization statistics, computed once from the sampled traits.
            // The raw values stay on the population and in the state matrix; these
            // numbers are only what the utility divides by (agent.h TraitStats).
            // θ is calibrated against the same use of them, so the quantile and the
            // gate it later feeds are on one scale.
            const TraitStats stats = compute_trait_stats(pop);
            if (!explicit_theta) {
                calibrate_thresholds(pop, params_from(config), admissible_fraction, stats);
            }
            const std::vector<double> flat = flatten_states(pop);
            py::array_t<double> out({static_cast<py::ssize_t>(pop.size()),
                                     static_cast<py::ssize_t>(kStateCols)});
            std::copy(flat.begin(), flat.end(), out.mutable_data());
            return out;
        },
        py::arg("n"), py::arg("config"), py::arg("seed"),
        "Draw a population and return the (n, kStateCols) float64 state matrix of "
        "docs/INTERFACES.md §1.");

    m.def(
        "generate_swipes",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           const py::dict& config, std::uint64_t epoch) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const auto* data = static_cast<const double*>(info.ptr);
            const int n = static_cast<int>(info.shape[0]);

            std::vector<Agent> pop = unflatten_states(data, n);
            const UtilityParams p = params_from(config);
            const int k = config["ncf"]["top_k"].cast<int>();
            const int n_negatives = n_negatives_from(config);
            const std::uint64_t base_seed = config["population"]["seed"].cast<std::uint64_t>();

            // Recomputed from the states rather than stashed at sample time: the
            // matrix is the only thing crossing the boundary, and recomputing from
            // it keeps utility_toward here identical to the one sample_population
            // calibrated θ with. Traits are not mutated by update_thresholds, so
            // an evolving θ does not move these.
            const TraitStats stats = compute_trait_stats(pop);

            const auto rows =
                generate_swipes(pop, p, stats, k, base_seed, epoch, n_negatives,
                                signal_from(config));

            py::array_t<std::int32_t> out(
                {static_cast<py::ssize_t>(rows.size()), py::ssize_t{3}});
            std::int32_t* o = out.mutable_data();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                o[3 * i + 0] = rows[i][0];
                o[3 * i + 1] = rows[i][1];
                o[3 * i + 2] = rows[i][2];
            }
            return out;
        },
        py::arg("states"), py::arg("config"), py::arg("epoch"),
        "Return an (S, 3) int32 array of [user_id, item_id, label] rows "
        "(docs/INTERFACES.md §2). Positives are top-k among candidates whose "
        "utility clears the user's θ; the per-agent RNG is seeded from "
        "(seed, agent_id, k, epoch).");

    m.def(
        "admissible_prefs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           const py::dict& config) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);

            const RankedAdmissible lists = build_admissible_prefs(
                pop, params_from(config), compute_trait_stats(pop));

            // Ragged list[list[int]], the same convention as
            // build_full_info_pref_lists (INTERFACES.md §3).
            py::list proposers(lists.proposers.size());
            for (std::size_t i = 0; i < lists.proposers.size(); ++i) {
                proposers[static_cast<py::ssize_t>(i)] = py::cast(lists.proposers[i]);
            }
            py::list acceptors(lists.acceptors.size());
            for (std::size_t i = 0; i < lists.acceptors.size(); ++i) {
                acceptors[static_cast<py::ssize_t>(i)] = py::cast(lists.acceptors[i]);
            }
            return py::make_tuple(proposers, acceptors);
        },
        py::arg("states"), py::arg("config"),
        "Every θ-admissible candidate per agent, ranked by descending utility "
        "(ties by ascending dense index), NOT truncated at k. Returns "
        "(proposer_lists, acceptor_lists) of dense per-side indexes (INTERFACES.md "
        "§2.3). The ablation arms differ only in what they draw from this: oracle "
        "takes the first k, random samples k without replacement from the whole "
        "admissible set, popularity ignores it. Truncating it here would collapse "
        "random into oracle.");

    m.def(
        "flatten_states",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& states) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            py::array_t<double> out({info.shape[0], static_cast<py::ssize_t>(kStateCols)});
            std::copy_n(static_cast<const double*>(info.ptr), info.shape[0] * kStateCols,
                        out.mutable_data());
            return out;
        },
        py::arg("states"),
        "Identity on an already-flat (N, kStateCols) matrix; exists so callers can "
        "normalize an arbitrary input without branching.");

    m.def(
        "calibrate_thresholds",
        // Deliberately NOT forcecast: this binding writes states[:, 6] in place,
        // and forcecast would silently copy a float32 or non-contiguous input,
        // leaving the caller's θ column untouched while reporting success — the
        // inert-threshold bug all over again, one layer up. A dtype or layout
        // mismatch has to be an error the caller can see.
        // Taken untyped on purpose. A py::array_t<double> argument converts a
        // float32 input into a float64 temporary *before* the lambda body runs,
        // so any dtype check performed there inspects the copy and always passes
        // — and the write below would land in that temporary, leaving the
        // caller's θ column untouched while reporting success. Checking the
        // caller's own object is the only way to catch it.
        [](py::array states, const py::dict& config, double admissible_fraction) {
            const auto info = states.request();
            if (!states.dtype().is(py::dtype::of<double>())) {
                throw std::invalid_argument(
                    "states must be float64; a float32 (or other) array would be "
                    "converted to a temporary and the in-place θ update lost");
            }
            if (info.ndim != 2 || info.shape[1] != static_cast<py::ssize_t>(kStateCols)) {
                throw std::invalid_argument(
                    "states must be a C-contiguous float64 array of shape (N, " +
                    std::to_string(kStateCols) + ")");
            }
            if ((info.strides[0] !=
                 static_cast<py::ssize_t>(kStateCols) * static_cast<py::ssize_t>(sizeof(double))) ||
                (info.strides[1] != static_cast<py::ssize_t>(sizeof(double)))) {
                throw std::invalid_argument(
                    "states must be C-contiguous; a strided view would be copied "
                    "and the in-place θ update lost");
            }
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);
            calibrate_thresholds(pop, params_from(config), admissible_fraction,
                                 compute_trait_stats(pop));
            const std::vector<double> flat = flatten_states(pop);
            std::copy(flat.begin(), flat.end(), static_cast<double*>(info.ptr));
        },
        py::arg("states"), py::arg("config"), py::arg("admissible_fraction") = 0.2,
        "Set states[:, 6] in place to each agent's (1 - admissible_fraction) "
        "utility quantile.");

    m.def(
        "build_full_info_pref_lists",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           const py::dict& config) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);

            const PrefLists lists =
                build_full_info_pref_lists(pop, params_from(config), compute_trait_stats(pop));

            // Ragged lists, per INTERFACES.md §3: a list[list[int]] of Python
            // ints, not an offset array. Simple to debug, and the matching loop
            // is not the bottleneck.
            py::list proposers(lists.proposers.size());
            for (std::size_t i = 0; i < lists.proposers.size(); ++i) {
                proposers[static_cast<py::ssize_t>(i)] =
                    py::cast(lists.proposers[i]);
            }
            py::list acceptors(lists.acceptors.size());
            for (std::size_t i = 0; i < lists.acceptors.size(); ++i) {
                acceptors[static_cast<py::ssize_t>(i)] =
                    py::cast(lists.acceptors[i]);
            }
            return py::make_tuple(proposers, acceptors);
        },
        py::arg("states"), py::arg("config"),
        "Full-information preference lists: (proposer_prefs, acceptor_prefs), "
        "dense indexes on both sides (docs/INTERFACES.md §2.3). O(N^2) memory; "
        "do not call above N=5000 (§4).");

    m.def(
        "deferred_acceptance",
        [](const std::vector<std::vector<std::int32_t>>& prop_prefs,
           const std::vector<std::vector<std::int32_t>>& acc_prefs, bool truncated) {
            const MatchResult r = deferred_acceptance(prop_prefs, acc_prefs, truncated);

            py::array_t<std::int32_t> matches(
                {static_cast<py::ssize_t>(r.matches.size()), py::ssize_t{2}});
            std::int32_t* m = matches.mutable_data();
            for (std::size_t i = 0; i < r.matches.size(); ++i) {
                m[2 * i + 0] = r.matches[i][0];
                m[2 * i + 1] = r.matches[i][1];
            }

            py::array_t<std::int32_t> unmatched_p(
                {static_cast<py::ssize_t>(r.unmatched_proposers.size())});
            std::copy(r.unmatched_proposers.begin(), r.unmatched_proposers.end(),
                      unmatched_p.mutable_data());

            py::array_t<std::int32_t> unmatched_a(
                {static_cast<py::ssize_t>(r.unmatched_acceptors.size())});
            std::copy(r.unmatched_acceptors.begin(), r.unmatched_acceptors.end(),
                      unmatched_a.mutable_data());

            py::array_t<std::int32_t> proposals(
                {static_cast<py::ssize_t>(r.proposals.size())});
            std::copy(r.proposals.begin(), r.proposals.end(), proposals.mutable_data());

            py::dict out;
            out["matches"] = matches;
            out["unmatched_proposers"] = unmatched_p;
            out["unmatched_acceptors"] = unmatched_a;
            out["n_blocking"] = r.n_blocking;
            out["proposals"] = proposals;
            return out;
        },
        py::arg("prop_prefs"), py::arg("acc_prefs"), py::arg("truncated"),
        "Gale-Shapley deferred acceptance, proposers proposing. Returns the dict "
        "of docs/INTERFACES.md §2.1. Indexes are dense per-side positions (§2.3); "
        "'proposals' counts attempts made and received, rejections included.");

    m.def(
        "count_blocking_pairs",
        [](const std::vector<std::vector<std::int32_t>>& prop_prefs,
           const std::vector<std::vector<std::int32_t>>& acc_prefs,
           py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> matches) {
            const auto minfo = matches.request();
            if (minfo.ndim != 2 || minfo.shape[1] != 2) {
                throw std::invalid_argument("matches must be an int32 array of shape (P, 2)");
            }
            const auto* mdata = static_cast<const std::int32_t*>(minfo.ptr);
            std::vector<std::array<std::int32_t, 2>> pairs(
                static_cast<std::size_t>(minfo.shape[0]));
            for (py::ssize_t i = 0; i < minfo.shape[0]; ++i) {
                pairs[static_cast<std::size_t>(i)] = std::array<std::int32_t, 2>{
                    mdata[2 * i + 0], mdata[2 * i + 1]};
            }
            return count_blocking_pairs(prop_prefs, acc_prefs, pairs);
        },
        py::arg("prop_prefs"), py::arg("acc_prefs"), py::arg("matches"),
        "Blocking pairs in `matches` judged against these preference lists. BPR "
        "must be counted against the full-information lists, not the truncated "
        "ones: a truncated run is stable with respect to its own lists by "
        "construction, so that count is always zero.");

    // ── Noise-to-signal diagnostic ────────────────────────────────────────────
    // shock_scale is in absolute utility units, so its *meaning* moves whenever
    // alpha/beta are retuned: the same 1.5 is a huge shock at one weighting and a
    // whisper at another. This returns the denominator of the ratio the
    // orchestrator logs once at construction,
    //
    //     noise/signal = shock_scale * 1.8138 / utility_sd(states, config)
    //
    // so the number stays checkable across retunes instead of silently drifting.
    m.def(
        "utility_sd",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           const py::dict& config) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop = unflatten_states(static_cast<const double*>(info.ptr), n);

            // The noise-free utility: a copy of the params with the shock switched
            // off, so the shock term is skipped rather than multiplied by zero.
            UtilityParams p = params_from(config);
            p.shock_scale = 0.0;

            const TraitStats stats = compute_trait_stats(pop);

            std::vector<std::int32_t> prop_rows;
            std::vector<std::int32_t> acc_rows;
            for (int i = 0; i < n; ++i) {
                (pop[static_cast<std::size_t>(i)].type == AgentType::M ? prop_rows : acc_rows)
                    .push_back(i);
            }

            // A deterministic, well-spread sample of ordered cross-side pairs,
            // capped at 4000 utility evaluations. Not random: with the same states
            // and config this must return the same number on every call, or a
            // logged ratio could not be compared between two runs that differ only
            // in noise.
            //
            // The index pair comes from an R2 low-discrepancy sequence (the
            // additive recurrence with the golden-ratio conjugate) rather than from
            // a fixed stride. A stride of n_pairs/4000 over a row-major enumeration
            // advances the item index by the same amount every step, so with 500
            // candidates the sample touches only a handful of distinct item columns
            // and the sd it reports is of a slice, not of the market — measured
            // 2.86 that way against 2.39 for a proper sample.
            //
            // BOTH orientations are emitted. M's utility and W's differ by the
            // indicator term, so M->W alone understates the spread: measured 2.19
            // (M->W), 2.57 (W->M), 2.39 pooled at the base weights. What the ratio
            // needs is the spread of whichever utility the shock is added to, which
            // is the pooled one.
            constexpr std::size_t kMaxSamples = 4000;
            const double kGoldenConj = 0.6180339887498949;

            const std::size_t n_prop = prop_rows.size();
            const std::size_t n_acc = acc_rows.size();
            if (n_prop == 0 || n_acc == 0) {
                return 0.0;
            }
            const std::size_t n_values = std::min<std::size_t>(kMaxSamples, 2 * n_prop * n_acc);
            // Two values per point, so a point count of n_values/2 fills the budget
            // with exactly balanced orientations.
            const std::size_t n_points = std::max<std::size_t>(n_values / 2, 1u);

            std::vector<double> samples;
            samples.reserve(2 * n_points);
            for (std::size_t s = 0; s < n_points; ++s) {
                const double u = (static_cast<double>(s) + 0.5) / static_cast<double>(n_points);
                const double v = (static_cast<double>(s) + 0.5) * kGoldenConj;
                const double frac = v - std::floor(v);
                const std::size_t ip = std::min(
                    n_prop - 1, static_cast<std::size_t>(u * static_cast<double>(n_prop)));
                const std::size_t ia = std::min(
                    n_acc - 1, static_cast<std::size_t>(frac * static_cast<double>(n_acc)));
                const Agent& m = pop[static_cast<std::size_t>(prop_rows[ip])];
                const Agent& w = pop[static_cast<std::size_t>(acc_rows[ia])];
                samples.push_back(m.utility_toward(w, p, stats));
                samples.push_back(w.utility_toward(m, p, stats));
            }

            // Sample sd (n-1).
            const double mean =
                std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
            double ss = 0.0;
            for (const double u : samples) {
                const double d = u - mean;
                ss += d * d;
            }
            const double denom = static_cast<double>(samples.size() - 1);
            return denom > 0.0 ? std::sqrt(ss / denom) : 0.0;
        },
        py::arg("states"), py::arg("config"),
        "Standard deviation of the noise-free utility over a deterministic sample "
        "of at most 4000 ordered pairs. The orchestrator logs "
        "shock_scale * 1.8138 / utility_sd as the implied noise-to-signal ratio.");

    m.def(
        "utility_for_pairs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> pairs,
           const py::dict& config) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);

            const auto pinfo = pairs.request();
            if (pinfo.ndim != 2 || pinfo.shape[1] != 2) {
                throw std::invalid_argument("pairs must be an int32 array of shape (P, 2)");
            }
            const auto* pdata = static_cast<const std::int32_t*>(pinfo.ptr);
            const py::ssize_t np = pinfo.shape[0];

            const UtilityParams p = params_from(config);
            const TraitStats stats = compute_trait_stats(pop);

            py::array_t<double> out(np);
            double* o = out.mutable_data();
            for (py::ssize_t i = 0; i < np; ++i) {
                const std::int32_t u = pdata[2 * i + 0];
                const std::int32_t v = pdata[2 * i + 1];
                // A row id outside the matrix is a caller bug, and returning NaN
                // would let it flow into a comparison silently (NaN compares false,
                // so a refusual would look like a decision). Throw instead.
                if (u < 0 || u >= n || v < 0 || v >= n) {
                    throw std::invalid_argument("pairs contains a row id outside [0, N)");
                }
                o[i] = pop[static_cast<std::size_t>(u)].utility_toward(
                    pop[static_cast<std::size_t>(v)], p, stats);
            }
            return out;
        },
        py::arg("states"), py::arg("pairs"), py::arg("config"),
        "U_i(j) for each ordered pair of row ids, on the same standardized scale "
        "generate_swipes uses. Callers apply the §10.2 proposal decision to a list "
        "they already hold (the NCF shortlist) without rebuilding the whole "
        "candidate market in Python.");

    m.def(
        "mean_admissible_utility",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           const py::dict& config) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);

            const UtilityParams p = params_from(config);
            const TraitStats stats = compute_trait_stats(pop);

            // V̄ per chooser: the mean utility over candidates that clear the
            // chooser's own θ, which is the pool a proposal decision is actually
            // made against (THEORY.md §10.2). Deliberately not the whole market:
            // utility_toward is unbounded and many of its values are negative, so
            // a whole-market mean is pulled toward zero by pairs the gate already
            // refused, and a cost compared against it would almost never bite.
            //
            // Returned as (mean, count) per row rather than a bare mean, so the
            // caller can pool two sides by their pool sizes instead of averaging
            // averages — a side whose agents have very different admissible-set
            // sizes would otherwise be weighted wrongly.
            py::array_t<double> means(static_cast<py::ssize_t>(n));
            py::array_t<double> counts(static_cast<py::ssize_t>(n));
            double* m_out = means.mutable_data();
            double* c_out = counts.mutable_data();

            for (int i = 0; i < n; ++i) {
                const Agent& self = pop[static_cast<std::size_t>(i)];
                double sum = 0.0;
                std::size_t k = 0;
                for (const Agent& cand : pop) {
                    if (cand.type == self.type) {
                        continue;
                    }
                    const double u = self.utility_toward(cand, p, stats);
                    // Same gate and same NaN handling as generate_swipes: a NaN
                    // utility is not admissible.
                    if (u >= self.threshold) {
                        sum += u;
                        ++k;
                    }
                }
                // NaN rather than 0.0 for an agent with no admissible candidate:
                // its mean does not exist, and 0.0 would silently drag a side's
                // pooled average toward a value no agent actually has.
                m_out[i] = (k > 0) ? sum / static_cast<double>(k) : std::nan("");
                c_out[i] = static_cast<double>(k);
            }
            return py::make_tuple(means, counts);
        },
        py::arg("states"), py::arg("config"),
        "Per-agent (mean utility over θ-admissible candidates, admissible count). "
        "Feed both to metrics.separating_threshold to get τ* = c / (V̄·p) per side "
        "(THEORY.md §10.2).");

    m.def(
        "compute_welfare",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> matches,
           const py::dict& config, double signal_cost, std::int64_t n_proposals,
           double defection_loss) {
            // Both terms are subtracted, so a negative value would add a cost as a
            // gain. Refused rather than clamped.
            if (!(signal_cost >= 0.0) || n_proposals < 0 || !(defection_loss >= 0.0) ||
                !std::isfinite(signal_cost) || !std::isfinite(defection_loss)) {
                throw std::invalid_argument(
                    "compute_welfare: signal_cost, n_proposals and defection_loss must be "
                    "finite and >= 0");
            }
            const auto arr = require_states(states);
            const auto info = arr.request();
            const int n = static_cast<int>(info.shape[0]);
            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);

            const auto minfo = matches.request();
            if (minfo.ndim != 2 || minfo.shape[1] != 2) {
                throw std::invalid_argument("matches must be an int32 array of shape (P, 2)");
            }
            const auto* mdata = static_cast<const std::int32_t*>(minfo.ptr);
            std::vector<std::array<std::int32_t, 2>> pairs(
                static_cast<std::size_t>(minfo.shape[0]));
            for (py::ssize_t i = 0; i < minfo.shape[0]; ++i) {
                pairs[static_cast<std::size_t>(i)] = std::array<std::int32_t, 2>{
                    mdata[2 * i + 0], mdata[2 * i + 1]};
            }

            const Welfare w = compute_welfare(pop, pairs, params_from(config),
                                              compute_trait_stats(pop));
            py::dict out;
            out["welfare"] = w.total;
            out["welfare_proposer"] = w.proposer;
            out["welfare_acceptor"] = w.acceptor;
            out["n_pairs"] = w.n_pairs;
            // THEORY.md §10.4, term by term. The signal cost is c per proposal; the
            // defection loss needs the survival draw, so it is resolved upstream and
            // only subtracted here. One place forms the total.
            const double cost = signal_cost * static_cast<double>(n_proposals);
            out["value_created"] = w.value_created;
            out["value_proposer"] = w.value_proposer;
            out["value_acceptor"] = w.value_acceptor;
            out["signal_cost"] = cost;
            out["defection_loss"] = defection_loss;
            out["welfare_total"] = w.value_created - cost - defection_loss;
            return out;
        },
        py::arg("states"), py::arg("matches"), py::arg("config"),
        py::arg("signal_cost") = 0.0, py::arg("n_proposals") = 0,
        py::arg("defection_loss") = 0.0,
        "Social welfare over a matching (docs/INTERFACES.md §2.2). `welfare` is the "
        "unweighted sum of U_prop(acc) + U_acc(prop), unchanged by §10.4. "
        "`welfare_total` is §10.4's W = value_created - signal_cost - defection_loss "
        "with value_created intent-weighted; at intent == 1, c = 0 and no dissolved "
        "match it equals `welfare` exactly.");

    m.def(
        "update_thresholds",
        // Untyped and stride-checked for the same reason as calibrate_thresholds:
        // a py::array_t<double> argument would forcecast into a temporary and the
        // write to states[:, 6] would land there instead of in the caller's array.
        [](py::array states, py::array n_proposals, py::array n_matches, double eta) {
            if (!states.dtype().is(py::dtype::of<double>())) {
                throw std::invalid_argument(
                    "states must be float64; a converted copy would take the θ update "
                    "and the caller's column would stay unchanged");
            }
            const auto info = states.request();
            if (info.ndim != 2 || info.shape[1] != static_cast<py::ssize_t>(kStateCols)) {
                throw std::invalid_argument(
                    "states must be a C-contiguous float64 array of shape (N, " +
                    std::to_string(kStateCols) + ")");
            }
            if ((info.strides[0] !=
                 static_cast<py::ssize_t>(kStateCols) * static_cast<py::ssize_t>(sizeof(double))) ||
                (info.strides[1] != static_cast<py::ssize_t>(sizeof(double)))) {
                throw std::invalid_argument(
                    "states must be C-contiguous; a strided view would be copied and "
                    "the in-place θ update lost");
            }
            const int n = static_cast<int>(info.shape[0]);

            auto read_counts = [n](py::array& arr, const char* name) {
                if (!arr.dtype().is(py::dtype::of<std::int32_t>())) {
                    throw std::invalid_argument(std::string(name) + " must be int32");
                }
                const auto c = arr.request();
                if (c.ndim != 1 || c.shape[0] != n) {
                    throw std::invalid_argument(std::string(name) +
                                                " must be 1-D with one entry per state row");
                }
                const auto* d = static_cast<const std::int32_t*>(c.ptr);
                return std::vector<std::int32_t>(d, d + n);
            };
            const std::vector<std::int32_t> proposals = read_counts(n_proposals, "n_proposals");
            const std::vector<std::int32_t> matched = read_counts(n_matches, "n_matches");

            std::vector<Agent> pop =
                unflatten_states(static_cast<const double*>(info.ptr), n);
            update_thresholds(pop, proposals, matched, eta);
            const std::vector<double> flat = flatten_states(pop);
            std::copy(flat.begin(), flat.end(), static_cast<double*>(info.ptr));
        },
        py::arg("states"), py::arg("n_proposals"), py::arg("n_matches"), py::arg("eta"),
        "θ ← θ + η·(matches/proposals − θ) per agent, in place on states[:, 6]. "
        "Counts are indexed by row; agents with no proposals are left alone.");
}
