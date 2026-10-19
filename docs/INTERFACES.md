# Interface Contract

The C++ core and the Python layer are built in **separate Claude Code tabs that do not share context**. This file is the only synchronisation point between them. Freeze it before Day 2. If a signature has to change, change it here first, commit, and tell the other tab to re-read this file.

## 1. State matrix

Every agent lives in one contiguous array.

```
states: np.float64, C-contiguous, shape (N, 7)
```

| col | field | semantics |
|-----|-------|-----------|
| 0 | `id` | integer stored as float; always equals row index |
| 1 | `type` | `0.0` = M (proposer side), `1.0` = W (acceptor side) |
| 2 | `status` | socio-economic score, lognormal |
| 3 | `attractiveness` | clipped to `[0, 1]` at sampling |
| 4 | `sociability` | clipped to `[0, 1]` at sampling |
| 5 | `age` | years |
| 6 | `threshold` | selectivity θ, starts at a config default, **mutated in place** by `update_thresholds` |

One flat array rather than a list of structs: the hot loop touches 10⁵ rows per epoch and the cost of marshalling per-agent dicts through pybind11 dwarfs the arithmetic.

## 2. Exported functions

All are exposed as `_hypergamic.<name>`.

```python
sample_population(n: int, config: dict, seed: int) -> np.ndarray[float64, (n, 7)]
#   θ in column 6 is calibrated, not a config constant: by default each agent's
#   θ is set to the (1 - population.admissible_fraction) quantile of its own
#   utilities toward the opposite side, so that roughly that share of candidates
#   is swipeable at epoch 0. Setting population.init_threshold to a number
#   overrides calibration with that absolute utility cutoff.

generate_swipes(states, config: dict, epoch: int) -> np.ndarray[int32, (S, 3)]
#   columns: [user_id, item_id, label]; label 1 = right swipe
#   S is not fixed: up to k positives per agent plus sampled negatives
#
#   Seeding rule. Each agent gets its own engine, seeded from the 5-tuple
#       (config["population"]["seed"], agent_id, config["ncf"]["top_k"], epoch)
#   fed through std::seed_seq. Consequences, all of which are tested:
#     * the output is independent of the thread count (the seed never contains a
#       thread id or any shared state), so a one-thread and a four-thread run are
#       bit-identical;
#     * the same (seed, epoch) reproduces exactly;
#     * a new epoch produces a different *negative* sample while the positives —
#       a function of traits and θ only — are unchanged.
#
#   θ enters here: a candidate is swipeable only if
#       utility_toward(user, candidate, config["hypergamy"]) >= states[user, 6]
#   Positives are the top-k *among admissible candidates*, so θ can only ever
#   truncate the list, never reorder it (the top-k of a filtered set is a prefix
#   of the global top-k). Negatives are drawn uniformly without replacement from
#   every candidate not reported as a positive, admissible-but-not-selected ones
#   included.

build_full_info_pref_lists(states, config: dict)
    -> tuple[list[list[int]], list[list[int]]]
#   (proposer_prefs, acceptor_prefs); O(N²) — see limits in §4

deferred_acceptance(prop_prefs: list[list[int]],
                    acc_prefs: list[list[int]],
                    truncated: bool) -> dict

compute_welfare(states, matches: np.ndarray[int32, (P, 2)], config: dict) -> dict

update_thresholds(states, n_proposals: np.ndarray[int32, (N,)],
                  n_matches: np.ndarray[int32, (N,)], eta: float) -> None
#   mutates states[:, 6] in place
```

### 2.1 `deferred_acceptance` return

```python
{
  "matches":             np.int32, shape (P, 2),  # [proposer_id, acceptor_id]
  "unmatched_proposers": np.int32, shape (U,),
  "unmatched_acceptors": np.int32, shape (V,),
  "n_blocking":          int,
}
```

### 2.2 `compute_welfare` return

```python
{
  "welfare":           float,  # Σ over matched pairs of U_prop(acc) + U_acc(prop)
  "welfare_proposer":  float,
  "welfare_acceptor":  float,
  "n_pairs":           int,
}
```

Uses the same `alpha, beta, gamma, delta, lambda` from `config["hypergamy"]` as `generate_swipes`. If the two disagree, every welfare number in the project is meaningless — this is the single most likely source of a silent bug.

## 3. Numerical conventions

- float64 everywhere except `labels`, `ids` and match arrays (int32).
- `label = 1` for a right swipe; `0` for a sampled negative.
- Utilities are unbounded reals, never clipped. Only `attractiveness` and `sociability` are clipped, and only at sampling.
- Population sampling uses one `std::mt19937_64` seeded from `seed`; swipe generation uses one engine per agent, seeded from `(seed, agent_id, k, epoch)` as described in §2. There is no shared stream between the two phases, so `(seed, epoch, config)` fully determines a run.
- Uniforms are derived from `engine()` output directly — `(engine() >> 11) * 2^-53` — rather than through `std::generate_canonical`, and normals come from Box–Muller on those uniforms rather than through `std::normal_distribution`. Both library facilities are implementation-defined in their number of engine calls and rounding, which would have made `(n, cfg, seed)` reproduce only within one toolchain. The residual caveat: `std::exp`, `std::log` and `std::pow` are only required to be correctly rounded to within a few ulp, so cross-*compiler* identity of a sampled trait is not guaranteed by the standard; what the code does guarantee is that no *engine-level* ambiguity remains. Byte-identical results are guaranteed within one toolchain and across thread counts.
- Preference lists are `list[list[int]]` (ragged Python lists), not offset arrays. Simpler to debug; the O(N·k) matching loop is not the bottleneck.

## 4. Complexity and scale limits

| call | time | memory | safe N |
|------|------|--------|--------|
| `sample_population` | O(N) + calibration | O(N) | any |
| `calibrate_thresholds` | O(N² log N) | O(N) | ≤ 50,000 |
| `flatten_states` / `unflatten_states` | O(N) | O(N) | any |
| `generate_swipes` | O(N²) | O(N·k) | ≤ 50,000 |
| `build_full_info_pref_lists` | O(N² log N) | **O(N²)** | ≤ 5,000 |
| `deferred_acceptance` (truncated) | O(N·k) | O(N·k) | ≤ 500,000 |
| `compute_welfare` | O(P) | O(P) | any |

The full-information benchmark is quadratic in memory: at N=5,000 each side's preference lists are 25M integers ≈ 100 MB. Do not run it above N=5,000, and run it once per configuration rather than once per epoch. State this limit in the notebook so the comparison is honest.

## 5. Proposer direction — a design decision worth defending

M proposes; W accepts. Gale-Shapley returns the **proposer-optimal** stable matching, so this produces the M-optimal / W-pessimal outcome. Hypergamy is imposed on W's utility function, so M-proposing is the *conservative* choice: it understates W's dissatisfaction relative to a W-proposing run. Worth one sentence in the README's limitations — a reviewer will ask, and the answer is good.

## 6. Metrics DataFrame schema

`Simulation.run()` returns exactly these columns, one row per epoch. `visualization.py` reads only these — adding a column is free, renaming one breaks the plots.

| column | dtype | definition |
|--------|-------|------------|
| `epoch` | int | absolute epoch index across all generations |
| `generation` | int | which turnover cycle |
| `delta` | float | copied from config, for post-hoc grouping |
| `seed` | int | |
| `gini` | float | over per-agent exposure counts |
| `mcr` | float | `2 · n_matched / N` |
| `rsi` | float | mean normalised partner rank, in `(0, 1]` |
| `bpr` | float | `n_blocking / max(n_matched, 1)` |
| `welfare_alg` | float | |
| `welfare_full` | float | `NaN` when the benchmark was not run |
| `algorithmic_loss` | float | `NaN` when the benchmark was not run |
| `n_matched` | int | |
| `n_blocking` | int | |
| `pareto_alpha` | float | fitted on cumulative match counts; `NaN` except on generation-final rows |
| `mean_swipes_per_agent` | float | |

## 7. Module layout

```
src/cpp/agent.{h,cpp}        state + utility + threshold step
src/cpp/population.{h,cpp}   distribution sampling
src/cpp/matching.{h,cpp}     deferred acceptance + blocking-pair count
src/cpp/simulation.{h,cpp}   epoch loop, OpenMP
src/cpp/bindings.cpp         everything in §2
src/python/ncf.py            GMF+MLP model, training, top-k
src/python/orchestrator.py   Simulation class, one-epoch pipeline
src/python/metrics.py        Gini, MCR, RSI, Δ, Pareto MLE
src/python/visualization.py  plots
```

Python imports the extension as `from _hypergamic import ...`. The built module lands in `build/`, so either add `build/` to `PYTHONPATH` or install with `cmake --install build --prefix .`; pick one on Day 1 and write it in the README.
