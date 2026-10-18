# Sprint Prompts — 7-Day Execution Guide

Each block below is a standalone prompt. Paste it verbatim into a fresh Claude Code tab.
Work them in order. Each session should end with a passing build and committed code.

---

## DAY 1 — Project Scaffolding & C++ Agent Core

### Session 1-A: Build System

```
Set up the build system for a C++17 / Python dual-language simulation project.
Project root already has: src/cpp/, src/python/, include/, tests/cpp/, tests/python/, models/, data/raw/, data/processed/, notebooks/.

1. Write CMakeLists.txt at the project root:
   - Require CMake 3.18+, C++17
   - Find Python3 (Development, Interpreter) and Pybind11
   - Find OpenMP; add -fopenmp to compile options when found
   - Add compile options: -O3 -march=native -Wall -Wextra
   - Build target: shared library `hypergamic` (all .cpp in src/cpp/ except bindings.cpp)
   - Build target: Python extension `_hypergamic` linking hypergamic + pybind11 (bindings.cpp)
   - Enable CTest; add subdirectory tests/cpp/

2. Write requirements.txt with pinned versions:
   torch==2.3.1, numpy==1.26.4, scipy==1.13.1, matplotlib==3.9.0,
   networkx==3.3, pytest==8.2.2, pybind11==2.12.0, pyyaml==6.0.1,
   tqdm==4.66.4, pandas==2.2.2

3. Write a .gitignore covering: build/, __pycache__, *.pyc, *.so, *.pyd,
   .pytest_cache, models/*.pt (keep models/.gitkeep), data/raw/*.csv,
   data/processed/*.npz

4. Verify cmake .. && make -j$(nproc) completes without errors on an empty src/cpp/ stub.
```

---

### Session 1-B: Agent State & Utility

```
Working in the hypergamic-matching-markets repo. Build system (CMake + Pybind11) is already set up.

Implement the Agent abstraction in C++:

include/agent.h
- Struct AgentType enum: MALE, FEMALE
- Struct Agent:
    int id
    AgentType type
    double status, attractiveness, sociability, age   // state vector v_i
    double threshold                                   // selectivity θ_i
    int epoch_proposals, epoch_matches                 // for Bayesian update
  - Member: double utility_toward(const Agent& other, double alpha, double beta,
                                   double gamma, double delta) const
      - For FEMALE evaluating MALE: add lambda * indicator(other.status >= this->status + delta)
      - For MALE evaluating FEMALE: no status indicator term
  - Member: void update_threshold(double eta)
      - Implements Bayesian step: θ += η * (matches/proposals − θ)
      - Guard against division by zero proposals

src/cpp/agent.cpp — implement both members

Write tests/cpp/test_agent.cpp using Catch2 (header-only, fetch via CMake FetchContent):
- Female with δ=0.5: utility returns higher value when male status > female + 0.5
- update_threshold converges toward true match rate over 100 synthetic epochs

Run cmake .. && make -j && ctest --output-on-failure. All tests must pass.
```

---

### Session 1-C: Population Manager

```
Continuing in hypergamic-matching-markets. Agent is implemented and tests pass.

Implement population generation in C++:

include/population.h + src/cpp/population.cpp
- Class Population:
    std::vector<Agent> agents
    Constructor: Population(int n, const PopConfig& cfg, uint64_t seed)
    - Splits n into two equal halves: MALE and FEMALE
    - Samples each trait from distribution specified in PopConfig:
        * "normal": std::normal_distribution
        * "lognormal": std::lognormal_distribution  
        * "pareto": use inverse CDF  (x = x_min * U^(-1/alpha) where U ~ Uniform(0,1))
    - Clamps attractiveness and sociability to [0, 1]
    - Assigns sequential IDs starting at 0

- Struct PopConfig: holds per-trait distribution name + params (mean/std or alpha/x_min)

include/simulation.h + src/cpp/simulation.cpp
- Function: SwipeLog run_epoch(Population& pop, const SimConfig& cfg, int epoch)
    - For each FEMALE agent, compute utilities toward all MALE agents, swipe right on top-k by utility
    - For each MALE agent, swipe right on top-k FEMALE agents by their utility function
    - Returns vector of (user_id, item_id, label) triples (label=1 for swipes, sample negatives)
    - Parallelise outer loop with #pragma omp parallel for

Write tests: population of 200 agents has correct size, trait distributions within 3σ of params,
SwipeLog has exactly k*N positive entries (no duplicates per user).
```

---

## DAY 2 — NCF Engine & Python Integration

### Session 2-A: Neural Collaborative Filtering

```
In hypergamic-matching-markets/src/python/. C++ population and swipe logging work.

Implement ncf.py — Neural Collaborative Filtering model in PyTorch:

class NCFModel(nn.Module):
  - __init__(self, n_users, n_items, embed_dim=64, mlp_layers=[128,64,32]):
      * GMF branch: Embedding(n_users, embed_dim), Embedding(n_items, embed_dim)
      * MLP branch: Embedding(n_users, embed_dim), Embedding(n_items, embed_dim)
        then Linear layers with ReLU, final concat → Linear(mlp_layers[-1] + embed_dim, 1) → Sigmoid
  - forward(self, user_ids, item_ids) → predicted scores in [0, 1]

class NCFTrainer:
  - __init__(self, model, lr=1e-3, l2_reg=1e-5)
  - train_epoch(self, interactions: np.ndarray) → float (mean loss)
      * interactions shape: (B, 3) — [user_id, item_id, label]
      * Loss: BCELoss with sample weighting (positives weighted by pos_weight=4.0)
      * Negative sampling: for each positive, sample 4 random negatives
  - top_k_recommendations(self, user_ids: np.ndarray, all_item_ids: np.ndarray, k: int)
      → dict[int, list[int]]  (user_id → ranked item list, length k)

Write tests/python/test_ncf.py:
- Model forward pass produces values in [0, 1]
- After 20 training epochs on 500 synthetic interactions, loss decreases monotonically
- top_k_recommendations returns exactly k items per user with no duplicates

Run pytest tests/python/test_ncf.py -v.
```

---

### Session 2-B: Pybind11 Bindings & Orchestrator

```
Continuing in hypergamic-matching-markets. Both C++ and NCF Python layers are implemented.

1. Write src/cpp/bindings.cpp exposing via Pybind11:
   - sample_population(n: int, config_dict: dict, seed: int) → np.ndarray shape (n, 5)
     columns: [id, type, status, attractiveness, sociability] (float64)
   - run_epoch(states: np.ndarray, config_dict: dict, epoch: int) → np.ndarray shape (M, 3)
     returns swipe log [user_id, item_id, label]
   - gale_shapley(pref_m: np.ndarray, pref_w: np.ndarray, k: int) → np.ndarray shape (n_matched, 2)
     returns matched pairs [m_id, w_id]; shape (0, 2) if no matches
   Use py::array_t<double> with c_style for zero-copy where possible.

2. Write src/python/orchestrator.py:
   class Simulation:
     - __init__(self, config_path: str)  loads experiments/base.yaml
     - run(self, n_epochs: int) → pd.DataFrame of per-epoch metrics
     - Single epoch pipeline:
         1. C++ run_epoch → swipe_log (numpy)
         2. NCFTrainer.train_epoch(swipe_log)
         3. NCFTrainer.top_k_recommendations → truncated pref lists
         4. C++ gale_shapley(pref_m, pref_w, k) → matching
         5. Compute metrics (Gini, MCR, RSI) → dict
         6. C++ feedback: update agent thresholds

3. Run a smoke test: N=200 agents, 10 epochs, verify metrics DataFrame has 10 rows with no NaN.
```

---

## DAY 3 — Gale-Shapley & Full Matching Pipeline

### Session 3-A: Gale-Shapley Engine

```
In hypergamic-matching-markets. Focus: production-quality Gale-Shapley in C++.

Implement include/matching.h + src/cpp/matching.cpp:

struct MatchResult {
  std::vector<std::pair<int,int>> matches;   // (proposer_id, acceptor_id)
  std::vector<int> unmatched_proposers;
  std::vector<int> unmatched_acceptors;
  int n_blocking_pairs;
};

MatchResult deferred_acceptance(
    const std::vector<std::vector<int>>& proposer_prefs,  // proposer i → ranked list of acceptor ids
    const std::vector<std::vector<int>>& acceptor_prefs,  // acceptor j → ranked list of proposer ids
    bool truncated   // if true, agents with exhausted lists go unmatched
);

int count_blocking_pairs(const MatchResult& result,
    const std::vector<std::vector<int>>& prop_prefs,
    const std::vector<std::vector<int>>& acc_prefs);

Implementation requirements:
- Standard proposal loop with deferred acceptance semantics
- Each acceptor holds at most one tentative match and rejects a worse proposer immediately
- When truncated=true, a proposer who exhausts their list is marked unmatched
- count_blocking_pairs: O(n * k) scan — for each matched pair, check if any unmatched
  or lower-ranked partner prefers them mutually

Write tests/cpp/test_matching.cpp:
- Classic 3-man 3-woman example with known stable matching
- Truncated lists: verify some agents go unmatched when lists have k=2
- count_blocking_pairs returns 0 for the known-stable matching

Run ctest --output-on-failure.
```

---

### Session 3-B: Metrics Layer

```
In hypergamic-matching-markets/src/python/. Gale-Shapley is implemented and tested.

Write src/python/metrics.py with these functions (all take numpy arrays or dicts):

def gini_coefficient(exposures: np.ndarray) -> float:
    """Exposure counts per agent → G in [0,1]. Use the sorted-array formula."""

def market_clearing_rate(matches: np.ndarray, n_total: int) -> float:
    """Fraction of agents who received a match."""

def rank_satisfaction_index(matches: np.ndarray, pref_lists: dict[int, list[int]]) -> float:
    """Mean rank of partner in proposer's preference list, normalised by list length."""

def algorithmic_loss(welfare_full: float, welfare_alg: float) -> float:
    """Δ = (SW* − SW_alg) / SW*, returns 0.0 if welfare_full == 0."""

def blocking_pair_ratio(n_blocking: int, n_matched: int) -> float:

def fit_pareto(match_counts: np.ndarray) -> tuple[float, float]:
    """MLE fit of Pareto distribution; returns (alpha, x_min)."""

def epoch_metrics(swipe_log, matches, pref_lists, welfare_full, welfare_alg, n_agents) -> dict:
    """Compute and return all metrics for one epoch as a flat dict."""

Write tests/python/test_metrics.py:
- gini_coefficient([1,1,1,1]) == 0.0 and gini_coefficient([0,0,0,4]) == 0.75
- algorithmic_loss handles welfare_full=0 without raising
- rank_satisfaction_index returns value in (0, 1]
Run pytest -v.
```

---

## DAY 4 — Parameter Sweeps & Longitudinal Dynamics

### Session 4-A: Experiment Configuration & Multi-Generational Loop

```
In hypergamic-matching-markets. Full single-epoch pipeline works end-to-end.

1. Write experiments/base.yaml — canonical config:
   population:
     n: 1000
     seed: 42
     traits:
       status:       {dist: lognormal, mean: 0.0, std: 0.8}
       attractiveness: {dist: normal,   mean: 0.5, std: 0.15}
       sociability:    {dist: normal,   mean: 0.5, std: 0.15}
       age:            {dist: normal,   mean: 30,  std: 6}
   hypergamy:
     delta: 0.5          # status surplus threshold
     alpha: 1.0          # status weight in female utility
     beta:  0.8          # attractiveness weight
     gamma: 0.3          # distance penalty
     lambda: 1.5         # indicator bonus
   ncf:
     embed_dim: 64
     mlp_layers: [128, 64, 32]
     lr: 0.001
     pos_weight: 4.0
     top_k: 20
   simulation:
     n_epochs: 100
     bayesian_eta: 0.1
     n_generations: 5    # full population turnover cycles
   output:
     dir: data/processed/

2. Extend src/python/orchestrator.py Simulation class:
   - run_multigen(self) → pd.DataFrame
     For each generation g in range(n_generations):
       - run n_epochs epochs
       - After final epoch: replace bottom 20% match-rate agents with fresh agents
         (re-sampled from original distributions)
       - Track per-generation Pareto α fit on match_counts
       - Log generation number in metrics DataFrame

3. Write scripts/run_sweep.py:
   - Accept CLI args: --delta-values (comma-separated floats), --output-dir
   - For each delta, deep-copy base config, run Simulation.run(), save metrics CSV
   - Print sweep progress with tqdm

4. Test: smoke run with N=100, 5 epochs, delta=[0.0, 0.5, 1.0]. Confirm 3 CSVs created.
```

---

### Session 4-B: Full-Information Benchmark

```
In hypergamic-matching-markets. Parameter sweep and multi-gen loop work.

1. Add to src/python/orchestrator.py:
   def run_full_information_benchmark(self) -> dict:
       - Sample population (same seed)
       - Compute true utility matrix: all N/2 × N/2 pairs
       - Build complete preference lists from utility ranks (no NCF)
       - Run C++ deferred_acceptance with truncated=false
       - Return welfare SW*, match rate, blocking pairs

2. Add to epoch pipeline: after NCF-mediated matching, also compute:
   - welfare_full from benchmark (run once, cache)
   - algorithmic_loss per epoch

3. Add to metrics DataFrame columns: welfare_alg, welfare_full, algorithmic_loss

4. Write tests/python/test_benchmark.py:
   - Full-information matching with N=50 produces zero blocking pairs
   - algorithmic_loss in [0, 1] for all epochs
   - Full-info clearing rate >= NCF-mediated clearing rate (should always hold)

Run pytest tests/ -v. All tests must pass before Day 5.
```

---

## DAY 5 — Visualization & Performance Profiling

### Session 5-A: Visualization Suite

```
In hypergamic-matching-markets. Full metrics pipeline is working with sweep data in data/processed/.

Write src/python/visualization.py. All plots use matplotlib with a dark theme (plt.style.use('dark_background')).
Use the color sequence: ['#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8'] for multi-series.

Functions:

def plot_gini_evolution(metrics_df: pd.DataFrame, ax=None) -> plt.Axes:
    """Line plot of Gini coefficient vs epoch. Mark generation boundaries with vertical dashed lines."""

def plot_matthew_effect(before_exposures, after_exposures, ax=None) -> plt.Axes:
    """Side-by-side Lorenz curves showing exposure inequality before and after NCF."""

def plot_clearing_rate(metrics_df: pd.DataFrame, ax=None) -> plt.Axes:
    """MCR vs epoch, with reference line at 1.0 (full clearing)."""

def plot_algorithmic_loss(metrics_df: pd.DataFrame, ax=None) -> plt.Axes:
    """Δ vs epoch — how much welfare the NCF recommendation costs per epoch."""

def plot_pareto_evolution(pareto_alphas: list[float], ax=None) -> plt.Axes:
    """Pareto α per generation — declining α means heavier tail concentration."""

def plot_sweep_comparison(sweep_results: dict[float, pd.DataFrame], metric: str) -> plt.Figure:
    """One subplot per delta value, final-epoch metric distributions as violin plots."""

def create_dashboard(metrics_df: pd.DataFrame, output_path: str):
    """4-panel figure: Gini + MCR (top row), algorithmic_loss + Pareto (bottom). Save to output_path."""

Run: python -c "from src.python.visualization import create_dashboard; print('OK')" — no import errors.
```

---

### Session 5-B: Performance Profiling & Optimization

```
In hypergamic-matching-markets. Target: N=100,000 agents × 100 epochs in under 10 minutes.

1. Profile the current pipeline with N=5,000, 20 epochs:
   python -m cProfile -o profile.out scripts/run_sweep.py --delta-values 0.5 --output-dir data/processed/
   python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative'); p.print_stats(20)"
   Report top-5 bottlenecks.

2. Apply targeted optimizations:

   C++ side (only if utility eval or matching is in top-5):
   - Pre-compute distance matrix once per epoch outside the OpenMP loop
   - Use std::vector reserve() for preference lists to avoid reallocations
   - Flatten 2D preference arrays to 1D + offset array for cache locality

   Python/NCF side (if train_epoch or top_k_recommendations is top bottleneck):
   - Batch top-k inference: compute full score matrix in one matmul instead of per-user
     ( user_embed @ item_embed.T for GMF branch, batch MLP forward )
   - Use torch.topk() instead of sorting full score vector
   - Pin NCF DataLoader to use num_workers=2

3. Re-run profile after optimizations. Report speedup factor.

4. Scale test: N=10,000 agents, 50 epochs. Must complete in under 3 minutes wall-clock.
   Log: time python scripts/run_sweep.py --delta-values 0.5 ...
```

---

## DAY 6 — Testing, CI, and Polish

### Session 6-A: Test Suite Completion

```
In hypergamic-matching-markets. All core modules implemented.

Audit test coverage and fill gaps:

1. Run pytest tests/ --tb=short -q. Fix any failures.

2. Add tests/python/test_integration.py — end-to-end smoke test:
   - N=100 agents, 5 epochs, delta=0.5
   - Verify: metrics DataFrame has 5 rows, all expected columns present, no NaN
   - Verify: Gini in [0, 1], MCR in (0, 1], RSI in (0, 1]
   - Verify: len(matches) <= N//2

3. Add tests/python/test_properties.py — statistical invariants:
   - Higher delta (stronger hypergamy) produces higher Gini (run 3 delta values, verify ordering)
   - Full-info benchmark always produces MCR >= NCF-mediated MCR
   - Pareto α at generation 5 <= Pareto α at generation 1 (inequality grows)

4. Write .github/workflows/ci.yml:
   - Trigger on push and pull_request to main
   - Jobs: lint (ruff), test-python (pytest tests/python/), test-cpp (cmake + ctest)
   - Cache pip and cmake build artifacts
   - Python 3.11, Ubuntu-latest

5. Run ruff check src/ and fix all warnings.
```

---

### Session 6-B: Notebook Demo

```
In hypergamic-matching-markets/notebooks/. Core simulation and tests are complete.

Create notebooks/demo.ipynb — a narrative Jupyter notebook that tells the full story:

Cell 1 (markdown): Title, one-paragraph abstract (use your own words — explain hypergamy,
the NCF feedback loop, and what this simulation reveals).

Cell 2: Import all modules, load base.yaml config.

Cell 3: Show population distributions — 2x2 grid of histograms for status, attractiveness,
sociability, age split by demographic group. Title: "Population at Initialization".

Cell 4: Run 50 epochs (N=500 for notebook speed). Show tqdm progress bar.

Cell 5: create_dashboard() output — the 4-panel figure.

Cell 6: Matthew Effect plot — exposure Lorenz curves at epoch 1, 25, 50.

Cell 7: Parameter sweep — run delta in [0.0, 0.3, 0.6, 1.0, 1.5], plot Gini trajectories
on one axes. Annotate: "δ=0 = no hypergamy (baseline)".

Cell 8: Multi-generational Pareto evolution bar chart.

Cell 9 (markdown): Interpretation section — what the numbers mean in plain English.
Discuss Matthew Effect, algorithmic amplification, structural singlehood.

Run all cells from scratch (Kernel > Restart & Run All). No errors, all outputs visible.
```

---

## DAY 7 — README, Documentation & GitHub

### Session 7-A: Animated README

```
In hypergamic-matching-markets/. Everything is implemented, tested, and the notebook demo runs.

Write README.md. Requirements:
- Zero generic AI phrases ("leverages", "comprehensive", "seamlessly", "state-of-the-art")
- Write like a curious researcher who built this — direct, specific, a bit proud
- Use GitHub-native LaTeX rendering ($...$) for inline math, $$...$$ for display
- Dark-mode-friendly badge row at the top (shields.io): build status, Python 3.11, C++17, MIT License, last commit

Structure:

# Hypergamic Matching Markets
[badge row]

[1-paragraph hook: what hypergamy is, why it matters when paired with algorithmic recommendation]

## The Model
Brief prose description of the three-layer architecture. Use the mermaid diagram from ARCHITECTURE.md.
State the utility function in LaTeX.

## Key Results
Show 2-3 figures inline (reference the notebook outputs saved to docs/figures/ during demo run).
Bullet the headline findings: "At δ=1.0, Gini reaches X after 50 epochs..."

## Installation
\`\`\`bash
# Prerequisites: CMake 3.18+, Python 3.11+, a C++17 compiler
git clone https://github.com/freddysccheung-067/hypergamic-matching-markets
cd hypergamic-matching-markets
pip install -r requirements.txt
cmake -B build && cmake --build build -j$(nproc)
\`\`\`

## Usage
Show the 4-line run command and the dashboard output figure.

## Theory
Link to docs/THEORY.md and docs/ARCHITECTURE.md.

## License
MIT
```

---

### Session 7-B: Final Commit Sequence

```
In hypergamic-matching-markets/. All code, tests, notebook, and README are complete.

Perform the final commit sequence. Commit message style: imperative mood, present tense,
no mention of AI tools. Each commit should be atomic.

git add src/cpp/ include/ CMakeLists.txt
git commit -m "feat: implement agent state vector and utility functions"

git add tests/cpp/
git commit -m "test: add C++ unit tests for agent utility and matching"

git add src/cpp/simulation.cpp src/cpp/population.cpp
git commit -m "feat: add population manager and parallel epoch loop"

git add src/cpp/matching.cpp
git commit -m "feat: implement Gale-Shapley deferred acceptance engine"

git add src/cpp/bindings.cpp
git commit -m "build: expose simulation core to Python via Pybind11"

git add src/python/
git commit -m "feat: add NCF recommendation engine with GMF+MLP architecture"

git add src/python/orchestrator.py
git commit -m "feat: integrate ABM, NCF, and matching into simulation loop"

git add src/python/metrics.py
git commit -m "feat: implement Gini, MCR, RSI, and algorithmic-loss metrics"

git add src/python/visualization.py
git commit -m "feat: add visualization suite and multi-panel dashboard"

git add scripts/ experiments/
git commit -m "feat: add parameter sweep script and base experiment config"

git add tests/python/ .github/
git commit -m "test: integration suite and GitHub Actions CI"

git add notebooks/demo.ipynb
git commit -m "docs: add full-pipeline demo notebook"

git add README.md docs/
git commit -m "docs: complete README with theory, architecture, and results"

git remote add origin https://github.com/freddysccheung-067/hypergamic-matching-markets.git
git branch -M main
git push -u origin main
```
