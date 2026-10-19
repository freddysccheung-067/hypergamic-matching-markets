# Sprint Prompts — 7-Day Execution Guide

Each block below is a standalone prompt. Paste it verbatim into a fresh Claude Code tab.
Work them in order. Each session should end with a passing build and committed code.

## How to use this with multiple tabs

Your coding tabs do **not** share context with each other. Two rules make that workable:

1. **Start every prompt with the contract line.** Each block already opens with
   `Read docs/INTERFACES.md first.` — that file is the single source of truth for the C++/Python
   boundary. If a tab changes a signature, it must change `INTERFACES.md` and commit, before the
   other tab picks up the work.
2. **Never run two dependent sessions in parallel.** `1-C` needs `1-B`. `2-A` (pure Python) can run
   alongside `1-C`, because the NCF model only needs arrays, not the extension.

**Windows commands.** This machine is Windows 11 with no C++ compiler yet — read
`docs/PREREQUISITES.md` and install MSVC Build Tools before Session 1-A. All build commands below
are PowerShell-safe. There is no `nproc`; use `--parallel`.

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

---

## DAY 1 — Project Scaffolding & C++ Agent Core

### Session 1-A: Toolchain verification

Read docs/PREREQUISITES.md and docs/INTERFACES.md before starting.

```
Read docs/PREREQUISITES.md first.

Current state (verified 2026-10-18): the MSVC toolchain works, a venv already exists at
.venv/ with all requirements installed, and build/Release/_hypergamic.cp312-win_amd64.pyd
imports cleanly. Re-verify rather than rebuild from scratch.

The repo already has a working CMakeLists.txt, requirements.txt, .gitignore, and stub
placeholder files. Do not rewrite them.

1. Activate the venv. Never install into the global interpreter — it shares packages with
   other projects on this machine, and requirements.txt pins older versions that would
   downgrade them:
   .\.venv\Scripts\Activate.ps1
   python -c "import sys; print(sys.executable)"
   Confirm the path contains `.venv`. If it does not, stop and fix that first.

2. Confirm CMake is configured against the venv interpreter. Reconfigure with it explicit:
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"
   The output must print:
     -- Python interpreter : ...\.venv\Scripts\python.exe
     -- OpenMP: enabled (/openmp) ...
   If OpenMP says NOT found, that is acceptable — record it and continue.

3. Build and find the module:
   cmake --build build --config Release --parallel
   Get-ChildItem -Recurse build -Filter "_hypergamic*.pyd" | Select-Object FullName

4. Verify the import path every later session relies on:
   python -c "import sys; sys.path.insert(0, 'build/Release'); import _hypergamic; print('ok')"

5. Report: the exact .pyd path, whether OpenMP was enabled, and the venv interpreter path.

Do not commit src/cpp/stub.cpp or src/cpp/bindings.cpp — 1-B replaces them.
```

---

### Session 1-B: Agent State & Utility

Read docs/INTERFACES.md first — §1 defines the state matrix and §2 the exported surface.

```
Working in the hypergamic-matching-markets repo. The MSVC toolchain is verified and
`python -c "import sys; sys.path.insert(0,'build/Release'); import _hypergamic"` works on the stub.

Read docs/INTERFACES.md first. Implement the Agent abstraction to match it exactly.

include/agent.h
- enum class AgentType : uint8_t { M = 0, W = 1 };
- struct Agent:
    int32_t id
    AgentType type
    double status, attractiveness, sociability, age   // state vector v_i
    double threshold                                   // selectivity θ_i
    int32_t epoch_proposals, epoch_matches             // for Bayesian update
  - double utility_toward(const Agent& other, const UtilityParams& p) const
      * For type W evaluating M: add p.lambda * (other.status >= this->status + p.delta ? 1.0 : 0.0)
      * For type M evaluating W: omit the indicator term entirely
      * Both subtract p.gamma * weighted_distance(v_self, v_other)
  - void update_threshold(uint32_t proposals, uint32_t matches, double eta)
      * θ += η * (matches/proposals − θ); return early if proposals == 0

- struct UtilityParams { double alpha, beta, gamma, delta, lambda; };

src/cpp/agent.cpp — implement both members. Header-only inline is fine if you prefer;
if so, delete the .cpp and note it, since CMake globs src/cpp/*.cpp.

Write tests/cpp/test_agent.cpp with Catch2:
- A W agent with δ=0.5 scores an M with status 1.0 above her own 0.4+0.5 higher than
  the same M with status 0.80 — isolating the indicator term.
- An M agent's utility is unchanged when δ varies from 0 to 2 (asymmetry holds).
- update_threshold with matches=proposals converges θ → 1.0 over 100 steps; with
  matches=0 converges θ → 0.0; with proposals=0 leaves θ untouched.

Build and test:
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release --parallel
  ctest --test-dir build -C Release --output-on-failure

All tests must pass. Report the test output verbatim.
```

---

### Session 1-C: Population Manager

Read docs/INTERFACES.md first — §2 defines `sample_population`'s exact shape and dtype.

```
Continuing in hypergamic-matching-markets. Agent is implemented and tests pass.

Read docs/INTERFACES.md first. Implement population generation in C++.

include/population.h + src/cpp/population.cpp
- struct TraitDist { std::string name; double p1, p2; };  // p1/p2 = mean/std or x_min/alpha
- struct PopConfig { TraitDist status, attractiveness, sociability, age; double init_threshold; };
- std::vector<Agent> sample_population(int n, const PopConfig& cfg, uint64_t seed)
    * n must be even; if odd, drop one agent and log a warning
    * First n/2 are AgentType::M (ids 0..n/2-1), rest are AgentType::W
    * Supported dist names: "normal", "lognormal", "pareto"
      - pareto via inverse CDF: x = x_min * pow(U, -1.0/alpha), U ~ Uniform(0,1)
    * Clamp attractiveness and sociability to [0, 1] AFTER sampling
    * threshold initialized to cfg.init_threshold for every agent
    * One std::mt19937_64 seeded with `seed`, used for every draw — the same
      (n, cfg, seed) must reproduce the population bit for bit

src/cpp/simulation.cpp
- np-style function: std::vector<std::array<int32_t,3>> generate_swipes(
      const std::vector<Agent>& pop, const UtilityParams& p, int k, uint64_t rng_seed)
    * Each agent swipes right on its own top-k candidates by utility_toward
    * Determine top-k with std::partial_sort, not a full sort
    * Then sample 4 negatives per positive uniformly from candidates NOT in the top-k
    * Emit [user_id, item_id, label] with label 1 for positives, 0 for negatives
    * Wrap the outer per-agent loop in #pragma omp parallel for when _OPENMP is defined

  THREAD SAFETY — get this right the first time, it is a silent-corruption bug:
    * Do NOT push_back into a shared vector from inside the parallel region. Either
      pre-size the output to n_agents * k * (1 + n_negatives) and write at a computed
      index, or give each thread a private buffer and concatenate after the loop in
      thread-index order. The second keeps the output deterministic across thread counts.
    * Do NOT share one std::mt19937_64 across threads — it is not thread-safe and the run
      stops being reproducible. Derive a per-thread seed from rng_seed and the thread id
      (e.g. std::seed_seq{rng_seed, thread_id}), then use a thread-local engine.
    * A test asserting "same seed → identical output" must pass at BOTH omp_set_num_threads(1)
      and (4). If it only passes at 1, you still have the race.

Write tests/cpp/test_population.cpp:
- sample_population(200, cfg, 42) returns 200 agents, exactly 100 of each type
- Every attractiveness and sociability value lies in [0, 1]
- Mean status of a lognormal(0, 0.8) sample of 10,000 lies within 0.05 of exp(0.32)
- Calling twice with the same seed yields identical status vectors; a different seed differs
- generate_swipes produces exactly k positives per agent and no duplicate items within an agent

Build and run ctest as in 1-B. All tests must pass.
```

---

## DAY 1.5 — Corrections to Sessions 1-B and 1-C

Run this before Day 2. It closes a gap that would otherwise make the whole closed loop open,
and fixes three accuracy problems in the existing code.

```
Read docs/INTERFACES.md and docs/THEORY.md §2–3 first.

Session 1-B and 1-C are built and green, but review found five issues. Fix them in order.

1. THRESHOLD IS INERT — the most important fix.
   Agent::threshold is updated every epoch by update_threshold(), but utility_toward() never
   reads it and generate_swipes() never reads it. Consequences:
     - the adaptive-expectation mechanism in THEORY.md §3 has no effect on behavior;
     - positives are top-k by utility, which depends only on traits, so the SAME positives are
       produced on every epoch regardless of seed;
     - the "closed loop" is therefore open, and the NCF trains on a static target.
   Fix: make theta gate acceptance. In generate_swipes, after ranking, drop candidates whose
   utility falls below the user's theta before taking the top-k, i.e. a candidate is only
   swipeable if utility_toward(...) >= user.threshold. Calibrate theta's initial value so that
   at epoch 0 roughly the intended fraction of candidates is admissible — otherwise every agent
   either accepts all or accepts none.
   Add a test: with theta = -infinity and theta = +infinity the positive sets differ, and the
   number of positives shrinks as theta rises.
   Document the calibration in THEORY.md §3 in one sentence.

2. generate_swipes signature diverges from the contract.
   INTERFACES.md §2 specifies generate_swipes(states, config, epoch); the implementation is
   generate_swipes(pop, p, k, rng_seed). Reconcile them: keep the C++ signature but ensure the
   bound Python function accepts and uses `epoch`, deriving the per-agent seed from
   (base_seed, epoch, agent_id) so that negatives differ per epoch while remaining
   reproducible. Update INTERFACES.md §2 to state the seeding rule explicitly.

3. No flat-matrix conversion exists.
   INTERFACES.md §1 and §2 require sample_population -> float64 (n, 7). The C++ side returns
   std::vector<Agent>. Add:
     std::vector<double> flatten_states(const std::vector<Agent>&)   // row-major, 7 cols
     std::vector<Agent> unflatten_states(const double* data, int n)
   Row layout exactly as §1: id, type, status, attractiveness, sociability, age, threshold.
   Add a round-trip test: flip every field, convert both ways, require bit equality.

4. generate_canonical is not portable.
   population.cpp claims (n, cfg, seed) reproduces the population "bit for bit" across
   compilers, but std::generate_canonical is implementation-defined — the number of engine
   calls and the rounding differ between MSVC and libstdc++. Either derive uniforms manually
   from engine() output (e.g. (engine() >> 11) * 0x1.0p-53) so the claim holds, or delete the
   cross-compiler claim and scope reproducibility to a single toolchain. Prefer the former;
   the project's results are meant to survive a compiler change.

5. Remove the dead per-epoch counters.
   Agent::epoch_proposals and Agent::epoch_matches are written by sample_population and never
   read; update_threshold takes both as arguments instead. Delete the fields, or use them —
   do not leave two sources of truth for the same quantity.

Also correct two test problems in tests/cpp/test_population.cpp:

6. The lognormal mean test's "Mill's ratio" reasoning is wrong.
   Box–Muller returns two iid standard normals (cos and sine halves), so BOTH have the same
   expectation; there is no truncated normal and no Mill's-ratio correction. Worse, the test
   as written is non-discriminating: the claimed correction is ~0.003 while the tolerance is
   0.05, so it passes regardless. Additionally, each agent consumes exactly four normal draws,
   so `status` always receives the cos half and the odd/even split separates nothing.
   Replace it with a test that actually has power: check the sample mean against
   exp(mu + sigma^2/2) with a tolerance derived from the standard error of the lognormal
   (sigma_X / sqrt(n), where sigma_X = exp(mu + sigma^2/2) * sqrt(exp(sigma^2) - 1)), at
   roughly 4 standard errors. Delete the spurious half-sample assertion.

7. Add the missing hypergamy test.
   Nothing currently verifies that delta changes WHO gets swiped — the core hypothesis of the
   project. Add: with the same population and seed, the mean status of W's positives rises
   monotonically as delta increases over {0.0, 0.5, 1.0}.

Build and run the full suite when done:
   cmake --build build --config Release --parallel
   ctest --test-dir build -C Release --output-on-failure
```

---

## DAY 1.6 — Trait scaling, config wiring, and the δ power guardrail

Run before Day 2. Measured on this machine, at `experiments/base.yaml`'s weights, over
N=200,000 draws:

| term | sd |
|------|-----|
| `α·status` (α=1.0) | 1.338 |
| `β·attractiveness` (β=0.8) | 0.120 |
| `−γ·distance` (γ=0.3) | 0.326 |

Status outweighs attractiveness by **11.2×** in the ranking, despite the weights reading 1.0 vs
0.8. And δ is **completely inert**: raising it from 0.0 to 1.0 changes 0 of the top-1000, because
every man in the shortlist already clears the cutoff, so the indicator adds the same `+λ` to all
of them and reorders nothing.

The `δ` fix is not a scaling fix — δ stays inert under z-scoring too at these weights. The cause is
arithmetic: for the indicator to reorder anything, the cutoff `s_w + δ` must fall *inside* the
competitive set, which requires non-status traits to be able to compensate for a status deficit.
Measured across a grid, that happens at **β/α ≥ ~2**. Both changes are needed, and they answer
different questions.

```
Read docs/THEORY.md §2 and docs/INTERFACES.md §1–§3 first.

Two independent problems. Do them in this order.

1. STANDARDIZE TRAITS FOR UTILITY.
   Raw traits have incompatible scales: status is lognormal(0, 0.8) with sd 1.31 and a max
   near 44; attractiveness is normal(0.5, 0.15) clipped to [0,1], sd 0.15. So α=1.0 and
   β=0.8 do not mean "status slightly more important than attractiveness" — they mean
   11.2× more important.

   Fix: z-score every trait to zero mean and unit standard deviation *for the purposes of
   utility computation*, using population-level statistics computed once after sampling.
   Keep the raw values on the Agent for reporting, plotting and the state matrix.

   Implementation notes:
   - Compute (mean, sd) per trait across the whole population, once, after
     sample_population(). Store them on the Population or pass them in a small struct.
   - utility_toward() and weighted_distance() use the standardized values.
   - The distance term becomes scale-free, which also softens the "s enters twice"
     collinearity documented in THEORY.md §2.1.
   - δ is now expressed in standard deviations of log-status. Document that in THEORY.md
     and in the base.yaml comment.
   - Guard sd == 0 (a degenerate constant trait) by leaving that trait unstandardized
     rather than dividing by zero.

   Do NOT change the raw state matrix layout in INTERFACES.md §1. Column 6 (θ) stays in
   utility units, which are now standardized units — state that explicitly.

2. WIRE ncf.n_negatives THROUGH, AND ADD THE MISSING KEYS.
   experiments/base.yaml declares ncf.n_negatives: 4, but simulation.cpp hardcodes
   constexpr int kNegativesPerPositive = 4 and bindings.cpp never reads the key. Editing
   the YAML does nothing today.

   Chosen fix: thread it through, do not delete it. The negative:positive ratio changes NCF
   training dynamics and is exactly the kind of knob the experiment matrix needs.

   - generate_swipes gains a `int n_negatives` parameter (default 4).
   - bindings.cpp reads config["ncf"]["n_negatives"], defaulting to 4 when absent.
   - INTERFACES.md §2's signature note mentions it.

   Also, in experiments/base.yaml:
   - add population.admissible_fraction: 0.2 (the code already defaults to 0.2, but the key
     is invisible today and the docstring points at a key that does not exist)
   - remove the stray CJK character in the delta comment on the hypergamy block
   - set beta: 2.0 (see item 3) and note why in a comment

3. MAKE delta ABLE TO MATTER — a default-config change, not a code change.
   With standardized traits, measured top-1000 overlap between δ=0.0 and δ=1.0:

     alpha=1.0 beta=0.8  -> 1.000  (inert)
     alpha=1.0 beta=2.0  -> 0.915  (some power)
     alpha=1.0 beta=3.0  -> 0.803
     alpha=0.6 beta=1.5  -> 0.783
     alpha=0.3 beta=1.5  -> 0.478  (strong)

   Rule of thumb: δ has real power when beta/alpha >= 2.

   Set the base config to alpha=1.0, beta=2.0 so that status remains the dominant linear
   term (preserving the Matthew-effect story) while δ is no longer structurally dead. Add a
   comment recording the measured overlap so the choice is auditable.

   ADD A GUARDRAIL TEST so this cannot silently regress:
     with the base config's weights, the top-k computed at delta=0.0 and delta=1.0 must
     differ for at least 5% of a 1000-candidate shortlist. Assert the measured overlap is
     below 0.95. If someone later retunes alpha/beta and re-kills delta, this fails loudly.

   Also add the inverse test, so the guardrail is not vacuous: at alpha=1.0, beta=0.1,
   overlap must be 1.000 — proving the test distinguishes "δ has power" from "δ is dead"
   and is not just always-passing.

4. UPDATE THE DOCS WITH WHAT WAS MEASURED.
   - THEORY.md §2: note that traits are standardized before utility computation, and that
     δ is therefore in standard-deviation units.
   - THEORY.md: add a short "δ has power only when β/α ≥ 2" note with the mechanism —
     the indicator is a constant unless the cutoff bisects the competitive set, which
     requires non-status traits to partly compensate for status.
   - Keep the existing note that the raw-scale version made δ inert; it explains why the
     change exists.

Build and run the full suite:
   cmake --build build --config Release --parallel
   ctest --test-dir build -C Release --output-on-failure

Every existing test must still pass. The θ calibration tests will need their
utility-scale assumptions revisited — θ is now a quantile of standardized-unit utilities,
which changes its numeric range but not its behavior. If a calibration test hardcodes a
raw-scale number, fix the test, not the calibration.
```

---

## DAY 2 — NCF Engine & Python Integration

### Session 2-A: Neural Collaborative Filtering

Pure Python — no extension needed. This can run in parallel with Session 1-C.

Read docs/INTERFACES.md first — §6 defines the epochs metrics schema, §2 the swipe-log columns.

```
In hypergamic-matching-markets/src/python/. The C++ population and swipe logging layer is being
built in another tab; you do not need it. Work only from docs/INTERFACES.md.

Read docs/INTERFACES.md first. Implement ncf.py — Neural Collaborative Filtering in PyTorch.

class NCFModel(nn.Module):
  - __init__(self, n_users, n_items, embed_dim=64, mlp_layers=(128, 64, 32)):
      * GMF branch: Embedding(n_users, embed_dim), Embedding(n_items, embed_dim)
      * MLP branch: Embedding(n_users, embed_dim), Embedding(n_items, embed_dim)
        then Linear layers with ReLU, final Linear(mlp_layers[-1] + embed_dim, 1)
      * Do NOT put a Sigmoid in forward() — train with BCEWithLogitsLoss, which is
        numerically stable; apply torch.sigmoid() only in the user-facing prediction
  - forward(self, user_ids, item_ids) → logits, shape (B,)
  - predict_proba(self, user_ids, item_ids) → sigmoid of forward

class NCFTrainer:
  - __init__(self, model, lr=1e-3, l2_reg=1e-5, device=None)
      * device defaults to cuda if available, else cpu
  - train_epoch(self, interactions: np.ndarray) → float (mean loss)
      * interactions shape (B, 3) int32 — [user_id, item_id, label]; label 1 = right swipe
      * BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
      * Negative sampling: for each positive (label==1), sample 4 random (user, item) pairs
        not present in the log. The C++ layer already emits labeled negatives; use those
        as-is and only top up if a batch is short of them.
  - top_k_recommendations(self, user_ids, candidate_ids, k) → dict[int, list[int]]
      * Compute the full score matrix ONCE: batch users × candidates through the model,
        not one forward pass per user. At N=100k a per-user loop is the bottleneck.
      * torch.topk, then convert to Python lists
      * Exclude self from each user's own candidate list

Write tests/python/test_ncf.py:
- predict_proba returns values strictly in (0, 1)
- Loss decreases over 20 epochs on 500 synthetic interactions (assert loss[-1] < loss[0])
- top_k_recommendations returns exactly k items per user, no duplicates, and never the
  user's own id
- Same seed twice → identical recommendations

Run: python -m pytest tests/python/test_ncf.py -v
```

---

### Session 2-B: Pybind11 Bindings & Orchestrator

Read docs/INTERFACES.md first — this is the session that implements the contract. If anything in
§2 does not fit the implementation, change INTERFACES.md and commit that change on its own.

```
Continuing in hypergamic-matching-markets. Both the C++ core and the NCF layer exist.

Read docs/INTERFACES.md first and implement §1 and §2 exactly.

1. Write src/cpp/bindings.cpp exposing the contract surface:
   - sample_population(n, config, seed) → np.ndarray float64 (n, 7)
   - generate_swipes(states, config, epoch) → np.ndarray int32 (S, 3)
   - build_full_info_pref_lists(states, config) → (list[list[int]], list[list[int]])
   - deferred_acceptance(prop_prefs, acc_prefs, truncated) → dict per §2.1
   - compute_welfare(states, matches, config) → dict per §2.2
   - update_thresholds(states, n_proposals, n_matches, eta) → None, mutates states[:, 6]
   Use py::array_t<double, py::array::c_style | py::array::forcecast> for inputs and return
   py::array_t with an explicit shape. Never return a raw std::vector<double>.

2. Write src/python/config.py:
   - dataclass SimConfig loaded from experiments/base.yaml
   - Nested dataclasses for population, hypergamy, ncf, simulation, output
   - Validate on load: n must be even; top_k < n/2; delta >= 0

3. Write src/python/orchestrator.py:
   class Simulation:
     - __init__(self, config=None, config_path="experiments/base.yaml")
     - run(self, n_epochs) → pd.DataFrame with EXACTLY the columns in INTERFACES.md §6
     - Per-epoch pipeline:
         1. generate_swipes → log
         2. NCFTrainer.train_epoch(log)
         3. top_k_recommendations → truncated lists
         4. deferred_acceptance(prop, acc, truncated=True)
         5. compute_welfare + metrics
         6. update_thresholds(states, proposals, matches, eta)

4. Smoke test: N=200, 10 epochs. Assert the DataFrame has 10 rows, the exact §6 columns,
   no NaN in gini/mcr/rsi, and gini in [0, 1].
```

---

## DAY 3 — Gale-Shapley & Full Matching Pipeline

### Session 3-A: Gale-Shapley Engine

Read docs/INTERFACES.md first — §2.1 fixes the return shape and §5 explains the proposer direction.

```
In hypergamic-matching-markets. Focus: production-quality Gale-Shapley in C++.

Read docs/INTERFACES.md first.

Implement include/matching.h + src/cpp/matching.cpp:

struct MatchResult {
  std::vector<std::pair<int32_t,int32_t>> matches;   // (proposer_id, acceptor_id)
  std::vector<int32_t> unmatched_proposers;
  std::vector<int32_t> unmatched_acceptors;
  int32_t n_blocking;
};

MatchResult deferred_acceptance(
    const std::vector<std::vector<int32_t>>& proposer_prefs,  // ranked acceptor ids
    const std::vector<std::vector<int32_t>>& acceptor_prefs,  // ranked proposer ids
    bool truncated);

int32_t count_blocking_pairs(const MatchResult&, const std::vector<std::vector<int32_t>>&,
                             const std::vector<std::vector<int32_t>>&);

Implementation requirements:
- Deferred acceptance semantics: each acceptor holds at most one tentative match and
  rejects a strictly worse proposer immediately. A rejected proposer advances to their
  next preference.
- Pre-compute acceptor rank tables (acceptor_id → rank of each proposer) once, as a
  std::vector<std::vector<int32_t>> of size |acceptors| × |proposers|. The naive version
  scans the list on every rejection, which is O(n·k²) and will show up in Day 5 profiling.
- truncated=true: a proposer who exhausts their list is unmatched. truncated=false:
  lists must be complete; assert this and give a clear message if not.
- count_blocking_pairs: for each pair (p, a) not matched together, check whether p prefers
  a to their partner AND a prefers p to theirs. Use the rank tables.

Write tests/cpp/test_matching.cpp:
- The classic 3×3 textbook instance (m1: w1>w2>w3, etc.) yields the known stable matching
- Truncated k=2 lists leave at least one proposer unmatched on a 6×6 instance
- count_blocking_pairs == 0 for the full-information textbook matching
- count_blocking_pairs > 0 for a truncated run on the same instance
- Empty preference lists return an empty MatchResult without crashing
- Unequal set sizes (|prop|=5, |acc|=3) are handled: extra proposers unmatched

Build and run ctest. All tests must pass.
```

---

### Session 3-B: Metrics Layer

Read docs/INTERFACES.md first — §6 fixes the exact column names these functions feed.

```
In hypergamic-matching-markets/src/python/. Gale-Shapley is implemented and tested.

Read docs/INTERFACES.md first.

Write src/python/metrics.py:

def gini_coefficient(exposures: np.ndarray) -> float
    Sorted-array formula: G = (2 * Σ i·x_(i)) / (n · Σ x) − (n+1)/n, for sorted x.
    Return 0.0 if the total is 0.

def market_clearing_rate(n_matched: int, n_total: int) -> float

def rank_satisfaction_index(matches, pref_lists: dict[int, list[int]]) -> float
    Mean rank of the partner within the proposer's list, normalized by list length.
    Unmatched agents excluded. Return float('nan') if no matches.

def algorithmic_loss(welfare_full: float, welfare_alg: float) -> float
    Δ = (SW* − SW_alg) / SW*. Return float('nan') if welfare_full is 0 or NaN.

def blocking_pair_ratio(n_blocking: int, n_matched: int) -> float

def fit_pareto(match_counts: np.ndarray) -> tuple[float, float]
    MLE: alpha = n / Σ ln(x_i / x_min), x_min = min positive count.
    Requires scipy.stats; return (nan, nan) if fewer than 3 positive observations —
    a silent NaN is far better than a garbage alpha propagating into the plots.

def epoch_metrics(...) -> dict
    Returns a flat dict with EXACTLY the keys in INTERFACES.md §6, in that order.

Write tests/python/test_metrics.py:
- gini_coefficient([1,1,1,1]) == 0.0 (within 1e-12)
- gini_coefficient([0,0,0,4]) ≈ 0.75
- gini_coefficient([0,0,0,0]) == 0.0 and does not divide by zero
- algorithmic_loss(welfare_full=100, welfare_alg=60) == 0.4
- algorithmic_loss handles welfare_full=0 by returning NaN, not raising
- rank_satisfaction_index lands in (0, 1]
- fit_pareto on [1,1,1] returns NaN rather than an exception

Run python -m pytest tests/python/test_metrics.py -v
```

---

### Session 3-C: The ablation arm — what makes the Matthew effect attributable

Read docs/INTERFACES.md §6 and docs/THEORY.md §7.1 first.

```
In hypergamic-matching-markets. Metrics exist; the recommendation loop does not yet have a
control condition.

THE PROBLEM THIS SOLVES. A rising Gini under the NCF proves nothing on its own. At the base
config, α·status has ~11× the spread of β·attractiveness, so positives concentrate on
high-status men even if the recommender is a coin flip. Reporting Gini(NCF) alone would
attribute a preference-driven effect to the algorithm.

Measured on this population at base weights, a single chooser's baseline utility already
puts 78% of its variance in the status term. So the honest claim needs a control:

    Matthew effect  :=  Gini(NCF) - Gini(random)

Both arms share population, seed, utilities, θ calibration and the matching algorithm.
Only the exposure mechanism differs. Implement all four arms.

1. Add a `recommender` field to Simulation, one of:
     "ncf"      — the trained model (already built in 2-B)
     "random"   — exposure sampled uniformly from the candidate pool
     "popularity" — rank by raw popularity (in-degree from the previous epoch's swipes),
                    no personalisation. This is the arm that isolates *personalisation*
                    from *the Matthew effect*: a popularity recommender manufactures
                    concentration with no ML at all.
     "oracle"   — rank by true utility, no learning. Upper bound on match quality; shows
                    what a perfect recommender would do to concentration.

   Each arm consumes the same (population, seed, theta). The random arm must still respect
   θ — it samples among admissible candidates — otherwise the arms differ in two ways at
   once and the comparison is confounded.

2. Add `exposure_counts` to the epoch output: an int array of length N, the number of times
   each agent appeared in anyone's recommended list that epoch. Gini runs over this, per
   THEORY.md §7.1. Expose it from Simulation.run() as a per-epoch matrix so the Lorenz plot
   can show before/after rather than only the final state.

3. Write scripts/run_ablation.py:
   - Runs all four arms at the same seed and config
   - Emits one CSV per arm plus a combined summary with columns
     [arm, epoch, gini, gini_excess_over_random, mcr, rsi, mean_exposure_top1pct]
   - gini_excess_over_random is the headline number

4. Write tests/python/test_ablation.py:
   - The random arm produces Gini below the NCF arm's Gini at the same seed
   - The popularity arm produces a higher Gini than random, without training anything
   - Oracle arm produces the highest mean rank satisfaction of all four
   - Reproducibility: same seed, same arm, identical exposure matrix

5. Document in THEORY.md under §7: state plainly that Gini alone is not evidence of
   algorithmic amplification, that the random arm is the control, and that
   gini_excess_over_random is the quantity the project's central claim rests on.

Report: the four Gini values at epoch 50 and the excess over random.
```

---

### Session 3-D: Attribution — can an observed status gap be read as attractiveness?

Read docs/THEORY.md §1–2 first.

```
In hypergamic-matching-markets. This session adds the analysis that stops the project
overclaiming, and it is the one a sceptical reviewer will reach for first.

MEASURED FACT, established before this session: status and attractiveness are drawn
independently in sample_population(). Measured correlation on a 200k draw: -0.0031.
There is no generative link. But the SELECTED set does correlate, purely because the
market sorts on both:

    corr(status, attractiveness) among top-1000 by utility : -0.0088
    corr(status, attractiveness) among top-10            : +0.4471

That is selection-induced correlation, not a causal relationship. Anyone reading matched
pairs and concluding "high-status people are also better looking" is reading a collider.

1. Add src/python/attribution.py with:

   def leave_one_term_out_effect(states, config, terms=("status","attract","distance","indicator"),
                                 k=1000) -> dict[str, float]:
       """For each term, recompute the top-k with that term's weight zeroed and return the
       overlap with the baseline top-k (Jaccard or share-in-common). Measured on this
       population at base weights: status->0 overlap 0.000, attract->0 0.967,
       distance->0 0.863, indicator->0 1.000. Low overlap = that term drives selection."""

   def selection_correlation(states, config, top_n) -> dict[str, float]:
       """corr between each pair of traits within the selected set, versus the population."""

   def delta_power(states, config, deltas=(0.0,0.5,1.0), k=1000) -> float:
       """Share of the top-k held in common between delta=0 and the largest delta.
       1.0 means delta is inert. Guard: warn loudly if it is 1.0."""

2. Write scripts/run_attribution.py emitting a small markdown table into
   docs/attribution.md, containing:
   - the generative correlation for each trait pair (near zero, by construction)
   - the selection-induced correlation at top-10 / top-100 / top-1000
   - the leave-one-term-out overlaps
   - the delta-power number for the configured weights

3. Write tests/python/test_attribution.py:
   - generative correlation between status and attractiveness is within +/-0.01
   - selection-induced correlation at top-10 EXCEEDS the generative correlation
     (this documents the artifact rather than asserting it is absent)
   - leave-one-term-out: zeroing alpha drops overlap below 0.2; zeroing the indicator
     leaves overlap above 0.9 at the base config.

4. In THEORY.md, add §9 "Identification limits". State three things plainly:
   - traits are sampled independently, so no trait causes another in this model
   - the model cannot therefore answer "does status reflect attractiveness"; it can only
     answer "how much does each term drive selection", which is what leave-one-term-out
     measures
   - correlation among matched pairs is a selection artifact and must never be reported as
     a finding about individuals

This section is what keeps the README honest. Without it the project reads as though it
demonstrates something about real people rather than about a specified mechanism.
```

---

## DAY 4 — Parameter Sweeps & Longitudinal Dynamics

### Session 4-A: Experiment Configuration & Multi-Generational Loop

Read docs/INTERFACES.md first — §6 is the DataFrame this session must produce.

```
In hypergamic-matching-markets. The full single-epoch pipeline works end-to-end.

Read docs/INTERFACES.md first.

1. experiments/base.yaml ALREADY EXISTS and is committed. Do not rewrite it.
   Read it, then extend it only if something is genuinely missing. The fields you need:
   population.{n, seed, traits.*}, hypergamy.{alpha, beta, gamma, delta, lambda},
   ncf.{embed_dim, mlp_layers, lr, l2_reg, pos_weight, top_k, n_negatives},
   simulation.{n_epochs, bayesian_eta, n_generations, turnover_fraction},
   output.{dir, save_every}.

2. Extend src/python/orchestrator.py Simulation class:
   - run_multigen(self) → pd.DataFrame
     For each generation g in range(n_generations):
       - run n_epochs epochs, tagging every row with generation=g
       - After the generation's final epoch: replace the bottom turnover_fraction of agents
         by match rate with fresh agents re-sampled from the original distributions.
         Fresh agents get the config's initial threshold, not a carried-over one.
       - Fit Pareto alpha on cumulative match counts for that generation's final row only;
         every other row gets NaN. Set epoch as the absolute index across generations so
         time never resets in the plots.
   - Keep run() as the single-generation path; run_multigen() calls it n_generations times.

3. scripts/run_sweep.py ALREADY EXISTS and is committed. Verify it still matches the
   Simulation constructor signature (config=dict), and fix the script rather than the
   orchestrator if they disagree. Run it, do not rewrite it.

4. Test: N=100, 5 epochs, n_generations=2, delta=[0.0, 0.5, 1.0]. Confirm 3 CSVs appear in
   data/processed/, each with the §6 columns, and that generation, delta and seed are
   populated on every row.
```

---

### Session 4-B: Full-Information Benchmark

Read docs/INTERFACES.md first — §2.2 defines compute_welfare and §4 states the size limit.

```
In hypergamic-matching-markets. Parameter sweep and multi-gen loop work.

Read docs/INTERFACES.md first.

1. Add to src/python/orchestrator.py:
   def run_full_information_benchmark(self) -> dict:
       - Re-sample the population with the SAME seed as the main run, so the comparison
         is like-for-like
       - Call _hypergamic.build_full_info_pref_lists(states, config)
       - Call deferred_acceptance(prop, acc, truncated=False)
       - Call compute_welfare for the resulting matching
       - Return {"welfare_full": float, "mcr_full": float, "n_blocking_full": int}
       - Cache the result on self: it is quadratic in memory and must run ONCE per
         configuration, not once per epoch (INTERFACES.md §4). Hard-fail above N=5000
         with a clear message rather than exhausting RAM.

2. Wire into the epoch pipeline: after NCF-mediated matching, fill welfare_full and
   algorithmic_loss from the cached benchmark. Before it has run, both are NaN.

3. §6 columns welfare_alg, welfare_full and algorithmic_loss must already exist from
   Session 2-B; populate them rather than adding them again.

4. Write tests/python/test_benchmark.py:
   - Full-information matching with N=50 produces zero blocking pairs
   - algorithmic_loss in [0, 1] for all epochs once the benchmark has run
   - Full-info clearing rate >= NCF-mediated clearing rate
   - Requesting the benchmark at N=6000 raises rather than OOM-ing

Run python -m pytest tests/ -v. All tests must pass before Day 5.
```

---

## DAY 5 — Visualization & Performance Profiling

### Session 5-A: Visualization Suite

Read docs/INTERFACES.md first — §6 fixes the column names; read only those and never guess.

```
In hypergamic-matching-markets. The full metrics pipeline works and sweep CSVs are in data/processed/.

Read docs/INTERFACES.md §6 first — your plots must read exactly those column names.

Write src/python/visualization.py. matplotlib, dark theme, color-blind-safe series colors.
Use plt.rcParams rather than plt.style.use('dark_background') so the figures still read
correctly when embedded in a light-mode README (the README is the main consumer).

Reuse one shared helper:
  def _fig_ax(ax) -> tuple[Figure, Axes]   # create a 1-panel figure when ax is None

Functions:

def plot_gini_evolution(metrics_df, ax=None) -> plt.Axes
    Gini vs epoch. Vertical dashed line at every generation boundary
    (where generation increments). If delta varies across rows, one line per delta.

def plot_lorenz(exposures, ax=None, label=None) -> plt.Axes
    Lorenz curve from an exposure-count array, plus the 45° equality diagonal.
    Keep this as a primitive: plot_matthew_effect calls it twice on one axes, because
    two side-by-side axes make the before/after comparison much harder to read.

def plot_matthew_effect(before_exposures, after_exposures, ax=None) -> plt.Axes
    Both Lorenz curves on one axes, annotated with each Gini value.

def plot_clearing_rate(metrics_df, ax=None) -> plt.Axes
    mcr vs epoch, reference line at 1.0.

def plot_algorithmic_loss(metrics_df, ax=None) -> plt.Axes
    algorithmic_loss vs epoch. Skip NaN rows silently — the early epochs have no benchmark.

def plot_pareto_evolution(metrics_df, ax=None) -> plt.Axes
    pareto_alpha vs generation, dropping NaN. Declining alpha = heavier tail concentration.

def plot_sweep_comparison(sweep_results: dict[float, pd.DataFrame], metric: str) -> plt.Figure
    One violin per delta value, final-epoch values of `metric`. Raise a clear error if
    `metric` is not a column rather than plotting an empty axes.

def save_dashboard(metrics_df, output_path="docs/figures/dashboard.png") -> pathlib.Path
    Name it save_*, not create_*: it writes a file. 4 panels — Gini, MCR, algorithmic_loss,
    Pareto — on a 2×2 grid. Create parent directories. Return the path.

Verification:
  python -c "import src.python.visualization as v; print([n for n in dir(v) if n.startswith(('plot_','save_'))])"
  Then render one dashboard from a real sweep CSV and open the PNG. Check that no axis
  label is clipped and the dark background does not bleed into a white page.
```

---

### Session 5-B: Performance Profiling & Optimization

```
In hypergamic-matching-markets. Target: N=100,000 agents × 100 epochs in under 10 minutes.

1. Profile with N=5,000, 20 epochs (PowerShell-safe; there is no `time` builtin):
   python -m cProfile -o profile.out scripts/run_sweep.py --delta-values 0.5 --output-dir data/processed/
   python -c "import pstats; p=pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(20)"
   Report the top-5 bottlenecks by cumulative time. cProfile will not see inside the
   pybind11 extension — expect `_hypergamic.*` to appear as a single opaque frame. If it
   dominates and you need a C++-side breakdown, say so; do not guess at the split.

2. Apply targeted optimizations, most likely first:

   Python/NCF (almost certainly the bottleneck — the model trains every epoch):
   - Compute the full score matrix in ONE batched forward pass instead of per-user.
     At N=5000 that is 5000 sequential forwards; batched it is a handful of matmuls.
   - torch.topk on the score matrix instead of sorting.
   - Reuse the optimizer and model across epochs; never rebuild them.

   C++ side (only if `_hypergamic` dominates):
   - Pre-compute the distance matrix once per epoch, outside the OpenMP loop.
   - reserve() preference-list vectors from the known top-k.
   - Flatten the 2D preference lists to 1D + offsets if matching shows up.

   Do not add SIMD intrinsics or hand-rolled cache blocking. At these sizes the win is
   in allocation and Python-boundary crossings, and hand-vectorised code is a source of
   silent numerical bugs that will cost more than it saves.

3. Re-profile. Report the speedup factor as a ratio of wall-clock, measured the same way.

4. Scale test: N=10,000, 50 epochs, under 3 minutes:
   Measure-Command { python scripts/run_sweep.py --delta-values 0.5 --output-dir data/processed/ }
```

---

## DAY 6 — Testing, CI, and Polish

### Session 6-A: Test Suite Completion

Read docs/INTERFACES.md first — §6 is what the integration test asserts against.

```
In hypergamic-matching-markets. All core modules are implemented.

Read docs/INTERFACES.md first.

1. Run python -m pytest tests/ --tb=short -q. Fix any failures.

2. Add tests/python/test_integration.py — end-to-end:
   - N=100, 5 epochs, delta=0.5, n_generations=2
   - DataFrame has 10 rows and exactly the §6 columns in order
   - No NaN in gini, mcr, rsi, bpr, n_matched
   - gini in [0, 1]; mcr in (0, 1]; rsi in (0, 1]
   - len(matches) <= N//2
   - pareto_alpha is non-NaN exactly on generation-final rows and NaN elsewhere

3. Add tests/python/test_properties.py — statistical invariants. These are slow; mark them
   with @pytest.mark.slow and keep N small (<=200, <=20 epochs) so the suite stays usable.
   - Stronger hypergamy raises Gini: run delta in {0.0, 0.5, 1.0} with the same seed,
     assert gini_mean is monotonically non-decreasing across the three.
   - Full-info clearing rate >= NCF-mediated clearing rate on the same seed.
   - Reproducibility: two runs with the same seed produce identical gini series.

4. .github/workflows/ci.yml ALREADY EXISTS and is committed. Read it and reconcile it with
   this machine: it currently pins Python 3.11, but this project is developed on 3.12.10 and
   requirements.txt is pinned for 3.12. Change the CI Python version to "3.12" so CI and
   local agree — a green CI running a different interpreter than you do is worse than no CI.
   Do not otherwise rewrite the workflow.

5. Run python -m ruff check src/ scripts/ tests/python/ and fix all warnings.
```

---

### Session 6-B: Notebook Demo

Read docs/INTERFACES.md first — §6 for the metrics columns the plots consume.

```
In hypergamic-matching-markets/notebooks/. Core simulation and tests are complete.

Read docs/INTERFACES.md first. Create notebooks/demo.ipynb.

Constraints:
- The notebook imports from src.python.*; add a first cell that inserts the repo root
  and the extension directory onto sys.path so it runs from any working directory.
- Keep total runtime under 3 minutes. N=500, 50 epochs. Do not run a 1000-agent sweep here.
- Save every figure to docs/figures/ with save_dashboard() / fig.savefig(), because the
  README embeds them. dpi=150, bbox_inches='tight'.

Cells:

1 (markdown) Title + one-paragraph abstract. Write it in your own voice: what hypergamy is,
  why an algorithmic mediator changes the outcome, and what this simulation measures. No
  filler, no "in today's fast-paced world".

2 Setup — sys.path, imports, load experiments/base.yaml.

3 (markdown) "Where the population starts" + Cell 4.

4 2×2 histogram grid: status, attractiveness, sociability, age, split by group.
  Title "Population at Initialization".

5 (markdown) "Running the market" + Cell 6.

6 sim.run(n_epochs=50) with a tqdm bar; show metrics_df.tail().

7 save_dashboard(metrics_df → docs/figures/dashboard.png); display the image.

8 Lorenz curves from epoch 1 vs epoch 50 → docs/figures/matthew.png. This is the headline
  figure: label both Gini values on the axes.

9 Parameter sweep over delta in [0.0, 0.3, 0.6, 1.0, 1.5]; overlay Gini trajectories.
  Annotate delta=0 as the no-hypergamy baseline.

10 run_multigen with n_generations=4 → Pareto alpha bar chart.

11 (markdown) Interpretation. Plain English: which mechanism produced which curve, what
  the algorithmic loss means, and — importantly — the limits of the model. Name at least
  two ways this is not a real dating market.

Run Kernel > Restart & Run All. No errors, all figures visible, all PNGs written.
```

---

## DAY 7 — README, Documentation & GitHub

### Session 7-A: README

```
In hypergamic-matching-markets/. Everything is implemented, tested, and the notebook demo runs.

Rewrite README.md. It currently holds a placeholder; replace it wholesale.

Voice:
- Write like the person who built this. Direct, specific, willing to say what the model
  cannot show.
- Banned: "leverages", "comprehensive", "seamlessly", "state-of-the-art", "delve",
  "in today's fast-paced world", and any sentence that could appear in any other repo.
- Prefer concrete numbers to adjectives: "Gini rises from 0.21 to 0.68 over 50 epochs at
  delta=1.0" beats "significant inequality emerges".

Structure:

# Hypergamic Matching Markets
Badge row (shields.io, flat-square): CI status, Python 3.12, C++17, MIT.
Link the CI badge to the real workflow path.

Opening paragraph: what hypergamy is, and the specific question — what happens to a
matching market when an engagement-optimizing recommender sits between the two sides.

## The Model
Three layers in prose, then the mermaid diagram from docs/ARCHITECTURE.md, then the
utility function in LaTeX. Explain the indicator term in one sentence: it is the whole
asymmetry.

## Results
2–3 figures from docs/figures/ (dashboard, matthew, sweep). Under each, one sentence
stating what it shows. Then 3–4 bullets of headline numbers pulled from the actual
notebook output — no invented figures.

## Installation
Windows-first, since that is the development environment. Point at docs/PREREQUISITES.md
for the compiler. Use PowerShell-safe commands:
    git clone https://github.com/freddysccheung-067/hypergamic-matching-markets
    cd hypergamic-matching-markets
    python -m pip install -r requirements.txt
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --parallel
Add a short note that the extension lands in build/Release and how to put it on sys.path.

## Usage
The 4-line Simulation snippet and the sweep command.

## Limitations
Be honest and specific: synthetic utilities with no learned real-world weights; the
full-information benchmark is capped at N=5000 for memory reasons; M-proposing
Gale-Shapley returns the proposer-optimal matching, so W's dissatisfaction is
understated (docs/INTERFACES.md §5); no dynamic entry/exit beyond generational turnover.
This section is what separates a serious project from a demo.

## Documentation
Links to THEORY.md, ARCHITECTURE.md, INTERFACES.md, PREREQUISITES.md.

## License
MIT
```

---

### Session 7-B: Final Commit Sequence

```
In hypergamic-matching-markets/. All code, tests, notebook, and README are complete.

Read this whole block before running anything.

IMPORTANT — commit history already exists. A `chore: initial project scaffold` commit is
already on `master` with the docs, CMakeLists, requirements, .gitignore, CI config and
config files. Do NOT re-add those. Also: this repo is on branch `master`, not `main`, and
there may already be a remote. Check first:

    git status
    git remote -v
    git log --oneline

Rules for the sequence below:
- Each commit must be non-empty. If `git status` shows nothing staged for a step, that
  content was already committed earlier — skip it rather than forcing an empty commit.
- Never stage a directory and then stage a file from inside it in a later step; order
  matters. Stage specific paths, and check `git status` between steps.
- Commit messages: imperative mood, present tense, no mention of AI tools.
- Before the final push: confirm with the user. Pushing is irreversible from their side.

Sequence (skip any step that stages nothing):

    git add include/ src/cpp/agent.cpp src/cpp/agent.h
    git commit -m "feat: add agent state vector and asymmetric utility functions"

    git add src/cpp/population.cpp include/population.h
    git commit -m "feat: sample heterogeneous population from configurable distributions"

    git add src/cpp/simulation.cpp include/simulation.h
    git commit -m "feat: generate swipe logs with parallel top-k evaluation"

    git add src/cpp/matching.cpp include/matching.h tests/cpp/
    git commit -m "feat: implement Gale-Shapley deferred acceptance with truncated lists"

    git add src/cpp/bindings.cpp
    git commit -m "build: expose simulation core to Python through pybind11"

    git add src/python/ncf.py tests/python/test_ncf.py
    git commit -m "feat: add neural collaborative filtering with GMF and MLP branches"

    git add src/python/config.py src/python/orchestrator.py src/python/__init__.py
    git commit -m "feat: close the loop between swipes, recommendations and matching"

    git add src/python/metrics.py tests/python/test_metrics.py
    git commit -m "feat: add Gini, clearing rate, rank satisfaction and algorithmic loss"

    git add src/python/visualization.py docs/figures/
    git commit -m "feat: add plotting suite and four-panel dashboard"

    git add tests/python/test_integration.py tests/python/test_properties.py
    git commit -m "test: add end-to-end and statistical invariant tests"

    git add notebooks/demo.ipynb
    git commit -m "docs: add walkthrough notebook with results"

    git add README.md docs/
    git commit -m "docs: write README and methodology documentation"

Then, only after confirming with the user:

    git remote add origin https://github.com/freddysccheung-067/hypergamic-matching-markets.git
    git branch -M main
    git push -u origin main

If `origin` already exists, `git remote set-url origin <url>` instead of `add`.
If the remote already has commits, do NOT force-push — report the situation instead.
```

---

## Verification checkpoints

Run these at the end of each day. If one fails, fix it before moving on — later sessions assume
the earlier layers work.

| After | Check |
|-------|-------|
| Day 1 | `ctest --test-dir build -C Release --output-on-failure` is green; `import _hypergamic` works |
| Day 2 | 10-epoch smoke run produces a DataFrame with the exact §6 columns and no NaN in `gini`/`mcr`/`rsi` |
| Day 3 | Textbook Gale-Shapley test passes; `count_blocking_pairs == 0` on full information |
| Day 4 | Sweep writes one CSV per delta; `algorithmic_loss` is finite once the benchmark runs |
| Day 5 | `save_dashboard` writes a readable PNG; N=10,000 × 50 epochs completes under 3 min |
| Day 6 | `pytest tests/` green; CI green on push; notebook runs clean top to bottom |
| Day 7 | README renders on GitHub — check the LaTeX and the mermaid diagram actually display |

## Two things that will bite you

1. **The welfare mismatch.** `generate_swipes` and `compute_welfare` must read the *same*
   `UtilityParams` from config. If one drifts, every `algorithmic_loss` number is wrong and
   nothing will look obviously broken. Add an assertion in the orchestrator that both use the
   identical config dict.

2. **`algorithmic_loss` should not be assumed positive.** It compares an NCF-mediated matching
   against the full-information optimum, so it should be ≥ 0 — but with a stochastic NCF and
   a finite sample, it can dip slightly negative in early epochs. If your tests assert `>= 0`
   strictly and they fail, that is a real signal about the pipeline, not a flaky test. Log it
   and investigate rather than clamping it.
