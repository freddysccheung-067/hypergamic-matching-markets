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
| 6 | `threshold` | selectivity θ, in **standardized utility units** after calibration, **mutated in place** by `update_thresholds` |

Column 6 is a raw number on the utility scale, and that scale is now the standardized one:
a utility of $\alpha\tilde{s} + \beta\tilde{a} - \gamma d + \lambda\mathbb{1}[\cdot]$ where
every trait has been z-scored against population statistics. θ is **not** a probability
and must not be read as one, even though `update_thresholds` drives it toward a yield in
$[0,1]$ — see THEORY.md §3 for why those two quantities share a column. Consumers that
print θ should label it as a utility cutoff, never as a swipe rate.

One flat array rather than a list of structs: the hot loop touches 10⁵ rows per epoch and the cost of marshalling per-agent dicts through pybind11 dwarfs the arithmetic.

### 1.1 Columns 2–5 are modeling constructs

None of these columns holds a measured quantity. They are draws from distributions chosen for
tractability, and every downstream result inherits that status. See THEORY.md §1.1 for the full
argument; the parts that bind this interface:

- **Column 3 (`attractiveness`) is the weakest of the four.** It is not measurable on an interval
  scale, the scales used to elicit it are ordinal, and self-reported values cluster near the top
  (a 7/10 for self-described average) so their usable variance is small. The model treats it as a
  real number with a roughly Gaussian population spread — an assumption of convenience, not an
  empirical claim.
- **Only rank order matters, not units.** Traits are standardized before entering the utility, so
  a caller may pass any monotone rescaling of a trait and get the same ranking. What breaks the
  model is a trait with no variance, not a trait on the wrong scale.
- **Do not compare these columns to real data.** Any empirical calibration is a separate exercise
  requiring a named reference population, which this project does not have.

Consumers of the state matrix — the NCF trainer, the metrics layer, the visualization — should
treat columns 2–5 as opaque coordinates and never attach real-world units to them in a caption or
a README sentence.

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
#   S is not fixed: up to k positives per agent plus n_negatives * k sampled
#   negatives, where n_negatives = config["ncf"]["n_negatives"] (default 4 when
#   the key or the ncf block is absent). S is also bounded by the candidate pool:
#   an agent contributes min(k, |admissible|) positives and at most
#   min(n_negatives * k, |candidates| - n_positives) negatives.
#
#   Traits are standardized before the utility is computed (THEORY.md §2.0), using
#   population statistics recomputed from `states` on entry. Both this call and
#   sample_population's calibration therefore see the same utility scale, which is
#   what makes θ comparable across the two.
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
#   n_negatives is deliberately NOT in the seed, so changing it leaves each
#   agent's negative draw sequence a prefix of what it was and a run at
#   n_negatives = 4 reproduces the pre-parameter output exactly.
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
#   (proposer_prefs, acceptor_prefs); O(N²) — see limits in §4. Both lists use the
#   index space of §2.3, so the result feeds deferred_acceptance directly.

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
  "proposals":           np.int32, shape (n_prop + n_acc,),  # by global agent id
}
```

`P` is the number of matched pairs; `U` and `V` are the unmatched counts. The first four keys
are what the rest of this file refers to and are not to be renamed.

`proposals` is additive, and it is what makes `update_thresholds` callable at all: the
Robbins–Monro step in THEORY.md §3 divides *matches* by *proposals*, and a proposal count
cannot be recovered from the matching afterwards, because a rejected proposal leaves no trace
in `matches`. It counts attempts *made* against a proposer's id and *received* against an
acceptor's id, rejections included — the update target is a yield, so the denominator has to
be every attempt. Length is `n_prop + n_acc`; see §2.3 for the index space.

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

### 2.3 Index space of preference lists

`prop_prefs` and `acc_prefs` are indexed **densely**, not by global agent id:

- `prop_prefs[p]` is the preference list of the `p`-th proposer, and `acc_prefs[a]` that of the
  `a`-th acceptor, each listed in descending preference order.
- The integer ids *inside* the lists are the other side's **global** ids from `states[:, 0]`
  (column 0 of the matrix), which is the same id space `generate_swipes` reports. So a list
  entry and a `matches` column are directly comparable.
- `n_prop` is the number of rows with `states[:, 1] == 0.0` and `n_acc` those with `== 1.0`.
  `sample_population` emits M first, so in a sampled population id `i` is proposer `i` for
  `i < n_prop`; callers must not rely on that and should derive the counts by counting types.

The dense outer index keeps the lists a plain `list[list[int]]` however the ids inside are
numbered, and it is why the return arrays are sized by count rather than by max id.

`compute_welfare` takes no preference lists, so it is unaffected; it reads `states[:, 0]` as the
id space for `matches` and looks the rows up by that column.

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
| `rsi` | float | mean normalized partner rank, in `(0, 1]` |
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
