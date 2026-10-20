# Interface Contract

The C++ core and the Python layer are built in **separate Claude Code tabs that do not share context**. This file is the only synchronisation point between them. Freeze it before Day 2. If a signature has to change, change it here first, commit, and tell the other tab to re-read this file.

## 1. State matrix

Every agent lives in one contiguous array.

```
states: np.float64, C-contiguous, shape (N, 14)
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
| 7 | `taste_alpha` | per-agent multiplier on α; lognormal, mean 1, spread `population.taste_sigma` |
| 8 | `taste_beta` | per-agent multiplier on β; same distribution |
| 9 | `taste_gamma` | per-agent multiplier on γ; same distribution |
| 10 | `key` | per-agent chemistry key: integer below 2^53, exact in float64 |
| 11 | `intent` | τ, in `(0, 1]`; Beta(`intent_beta_a`, `intent_beta_b`), rescaled so no agent is exactly 0 |
| 12 | `values` | g, in `[0, 1]`; Beta(`values_beta_a`, `values_beta_b`) |
| 13 | `community` | k, exactly `0.0` or `1.0`; Bernoulli(`population.community_prob`) |

Columns 7–10 were appended when preference heterogeneity was added (THEORY.md §2.3). They are
trailing so every consumer that indexes 0–6 keeps working, and they are drawn by
`sample_population`, not chosen by a caller: editing them means overriding the model. With
`population.taste_sigma = 0` the three multipliers are exactly 1.0 and the utility is
bit-identical to the pre-heterogeneity one.

Columns 11–13 were appended in the same way for the institutional model (THEORY.md §10.1).
None of the three enters `utility_toward`, so none changes any pre-existing number — they act
only through the proposal decision (§2, `generate_swipes`) and, later, the Environment B entry
filter. They are drawn from a **separate engine stream** from the traits and the preference
heterogeneity columns, so the four trait columns and `key` are bit-identical at every setting of
`intent_beta_*`, `values_beta_*` and `community_prob`. `intent` is strictly positive by
construction: the comparison against a cost is a strict inequality, and an agent with exactly
zero intent would be indistinguishable from an agent that never considered a proposal.

`key` is re-drawn whenever an agent is re-sampled (generational turnover), which gives a
replaced agent fresh chemistry without disturbing any other pair — every other pair is a
function of two keys that did not move. It is below 2^53 by construction, which is what makes
the float64 round trip lossless; a hand-built matrix with a larger value in this column would
round silently.

Column 6 is a raw number on the utility scale, and that scale is now the standardized one:
a utility of $m_\alpha\alpha\tilde{s} + m_\beta\beta\tilde{a} - m_\gamma\gamma d
+ \lambda\mathbb{1}[\cdot] + \sigma_{\text{shock}}L(i,j)$ where every trait has been z-scored
against population statistics. θ is **not** a probability
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
sample_population(n: int, config: dict, seed: int) -> np.ndarray[float64, (n, 11)]
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
#
#   The proposal decision (THEORY.md §10.2). With c = config["hypergamy"]
#   ["signal_cost"] > 0, a top-k candidate is reported as a positive only if
#
#       tau_user * utility_toward(user, candidate) * p_side > c
#
#   where p_side = config["signal"]["accept_prob_m" | "accept_prob_w"], keyed by
#   the proposer's side. The candidates that fail are still reported, with label
#   0, in the same agent's block after the positives and before the sampled
#   negatives — the pair was considered and declined, so the log records the
#   refusal rather than making it indistinguishable from a candidate never seen.
#   They are not eligible to be sampled as negatives again.
#
#   c <= 0 skips the decision entirely, by design rather than by optimization: a
#   literal test would reject every pair with a negative utility (0 > 0 is false)
#   and break the c = 0 pooling regime. At c = 0 every admissible type proposes,
#   which is the pooling equilibrium of §10.2, and the swipe log is bit-identical
#   to the pre-signalling one. When neither key is present the defaults are c = 0
#   and p = 0.5, so a config written before this session runs unchanged.

build_full_info_pref_lists(states, config: dict)
    -> tuple[list[list[int]], list[list[int]]]
#   (proposer_prefs, acceptor_prefs); O(N²) — see limits in §4. Both lists use the
#   index space of §2.3, so the result feeds deferred_acceptance directly.

deferred_acceptance(prop_prefs: list[list[int]],
                    acc_prefs: list[list[int]],
                    truncated: bool) -> dict

compute_welfare(states, matches: np.ndarray[int32, (P, 2)], config: dict) -> dict

utility_sd(states, config: dict) -> float
#   Standard deviation of the *noise-free* utility over a deterministic sample of
#   at most 4000 ordered cross-side pairs, both orientations included. shock_scale
#   is in absolute utility units and alpha/beta get retuned, so the orchestrator
#   logs the implied ratio  shock_scale * 1.8138 / utility_sd  once at
#   construction; this binding is its denominator. Deterministic: the same states
#   and config always give the same number, whatever shock_scale says.

update_thresholds(states, n_proposals: np.ndarray[int32, (N,)],
                  n_matches: np.ndarray[int32, (N,)], eta: float) -> None
#   mutates states[:, 6] in place
```

### 2.1 `deferred_acceptance` return

```python
{
  "matches":             np.int32, shape (P, 2),  # [proposer_index, acceptor_index]
  "unmatched_proposers": np.int32, shape (U,),
  "unmatched_acceptors": np.int32, shape (V,),
  "n_blocking":          int,
  "proposals":           np.int32, shape (n_prop + n_acc,),  # proposers first
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
- The integer ids *inside* `prop_prefs[p]` are **acceptor indexes** `0..n_acc-1`; the ids inside
  `acc_prefs[a]` are **proposer indexes** `0..n_prop-1`. Same space as the outer index, just the
  other side of it.
- `n_prop` is the number of rows with `states[:, 1] == 0.0` and `n_acc` those with `== 1.0`, so
  proposer index `p` is the `p`-th type-0 row in matrix order and acceptor index `a` the `a`-th
  type-1 row. `sample_population` emits M first, so in a sampled population index `p` is also row
  `p`; callers must not rely on that and should derive the mapping by counting types.

Dense indexes rather than global ids because `deferred_acceptance` is handed the lists and nothing
else: with global ids it could not recover an unmatched acceptor's position, since an acceptor no
one proposes to appears in no list at all. Every consumer that needs a row — the orchestrator, and
`compute_welfare` internally — maps an index to a row through the type column, which is a lookup
it already has the matrix for.

`matches` and the two unmatched arrays therefore hold **index pairs**, not global ids.
`update_thresholds` is the exception: its `n_proposals` and `n_matches` are indexed by **row**,
because it writes `states[:, 6]` and the caller has already mapped back.

`compute_welfare` takes no preference lists; it maps the index pair to rows through column 1 and
reads the utilities off those rows.

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

M proposes; W accepts. Gale-Shapley returns the **proposer-optimal** stable matching, which is
simultaneously the **acceptor-pessimal** one: among all stable matchings every M does at least as
well, and every W at most as well, as in any other. So M-proposing hands each W her *worst* stable
partner. Measured on the cyclic 3×3 in `tests/cpp/test_matching.cpp`: M-proposing leaves every
woman with her last choice (sum of women's ranks 6), W-proposing gives every woman her first
(sum 0).

Consequences for how results may be read:

- **W-side rank satisfaction is a lower bound across stable matchings, not a central estimate.**
  Hypergamy lives in W's utility, so this choice lands on exactly the side the project is about.
- The full-information versus NCF comparison uses the same direction in both arms, so the *gap*
  between them (algorithmic loss) is largely unaffected; the absolute W-side level is what is
  pessimistic.
- Never compare a W-side level against a W-proposing baseline without saying so.
- A W-proposing arm is a cheap sensitivity check if a reviewer asks: swap the two list arguments.

An earlier version of this section called M-proposing "conservative" because it "understates W's
dissatisfaction". That was backwards, and is corrected here. The README's limitations section
states the correct version.

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
| `mean_swipes_per_agent` | float | share of `N` swipes generated; **0.0** for every arm that does not build a swipe log |
| `arm` | str | which ablation arm produced the row (Session 3-C): `ncf`, `random`, `popularity` or `oracle` |
| `reach` | float | share of agents with non-zero exposure |
| `top1_share` | float | share of exposure held by the top `ceil(0.01 N)` agents |
| `mean_proposal_intent` | float | E[τ \| proposal]: mean `intent` (col 11) over the proposers of this epoch's label-1 rows; `NaN` for an arm with no swipe log |
| `separating_threshold` | float | analytic τ* = c / (V̄·p) for this epoch, pooled across sides by proposal count; `inf` when a side's V̄ or p is non-positive |
| `proposal_cost_total` | float | c × (number of label-1 rows): the signal-cost term of the welfare sum |

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
