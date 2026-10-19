#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "agent.h"
#include "matching.h"
#include "population.h"
#include "simulation.h"

// The Python surface fixed by docs/INTERFACES.md §2. The C++ core keeps its own
// signatures (named fields, std::vector<Agent>); the flat (N, 7) float64 matrix
// the contract specifies exists only at this boundary, where numpy lives.

namespace py = pybind11;

namespace {

// Read a utility block from the config dict. Every key is required rather than
// defaulted: a silently missing weight would change every welfare number in the
// project, which is the failure mode §2.2 warns about.
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
    return UtilityParams{get("alpha"), get("beta"), get("gamma"), get("delta"), get("lambda")};
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

// Validate an incoming (N, 7) float64 array and hand back a contiguous copy.
py::array_t<double, py::array::c_style | py::array::forcecast> require_states(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& states) {
    const auto info = states.request();
    if (info.ndim != 2 || info.shape[1] != 7) {
        throw std::invalid_argument("states must be a C-contiguous float64 array of shape (N, 7)");
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
            py::array_t<double> out({static_cast<py::ssize_t>(pop.size()), py::ssize_t{7}});
            std::copy(flat.begin(), flat.end(), out.mutable_data());
            return out;
        },
        py::arg("n"), py::arg("config"), py::arg("seed"),
        "Draw a population and return the (n, 7) float64 state matrix of "
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
                generate_swipes(pop, p, stats, k, base_seed, epoch, n_negatives);

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
        "flatten_states",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& states) {
            const auto arr = require_states(states);
            const auto info = arr.request();
            py::array_t<double> out({info.shape[0], py::ssize_t{7}});
            std::copy_n(static_cast<const double*>(info.ptr), info.shape[0] * 7,
                        out.mutable_data());
            return out;
        },
        py::arg("states"),
        "Identity on an already-flat (N, 7) matrix; exists so callers can normalize "
        "an arbitrary input without branching.");

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
            if (info.ndim != 2 || info.shape[1] != 7) {
                throw std::invalid_argument(
                    "states must be a C-contiguous float64 array of shape (N, 7)");
            }
            if ((info.strides[0] != 7 * static_cast<py::ssize_t>(sizeof(double))) ||
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

    m.def(
        "compute_welfare",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> states,
           py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> matches,
           const py::dict& config) {
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
            return out;
        },
        py::arg("states"), py::arg("matches"), py::arg("config"),
        "Social welfare over a matching: sum of U_prop(acc) + U_acc(prop) "
        "(docs/INTERFACES.md §2.2). Weights and standardization are the same as "
        "generate_swipes.");

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
            if (info.ndim != 2 || info.shape[1] != 7) {
                throw std::invalid_argument(
                    "states must be a C-contiguous float64 array of shape (N, 7)");
            }
            if ((info.strides[0] != 7 * static_cast<py::ssize_t>(sizeof(double))) ||
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
