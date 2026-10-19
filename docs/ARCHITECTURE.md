# Architecture

## System Overview

A closed-loop simulation that fuses three computational layers. Each epoch runs the same pipeline: agents form preferences → an ML engine mediates visibility → a matching algorithm clears the market → outcomes feed back into agent expectations.

```mermaid
flowchart TD
    A[Population Init<br/>N agents, sets M and W] --> B[Agent-Based Model<br/>utility + Bayesian updates]
    B --> C[Interaction Log<br/>swipes, likes, passes]
    C --> D[NCF Engine<br/>GMF + MLP → score y_ij]
    D --> E[Truncated Preference Lists<br/>top-k per agent]
    E --> F{Gale-Shapley<br/>variant}
    F -->|Full information| G[Benchmark μ*]
    F -->|ML-mediated| H[Constrained μ_alg]
    G --> I[Algorithmic Loss Δ]
    H --> I
    H --> J[Metrics: Gini, MCR, R, BPR]
    J --> K[Feedback to agents<br/>threshold updates]
    K --> B
```

## Dual-Language Split

Performance-critical loops live in C++; tensor ops and visualization live in Python. They talk through Pybind11.

| Layer | Language | Why |
|-------|----------|-----|
| Agent state + utility | C++ | Tight inner loop, called O(N²) per epoch |
| Population init + distributions | C++ | One-time cost, keeps data local to the engine |
| Gale-Shapley matching | C++ | Graph traversal, cache-friendly adjacency |
| Simulation epoch loop | C++ (OpenMP) | Parallelizable across agents |
| NCF model | Python (PyTorch) | Mature autograd, easy training loop |
| Metrics + statistics | Python (NumPy/SciPy) | Gini, Pareto fits, rank satisfaction |
| Visualization | Python (Matplotlib) | Plots and animations |

## Data Flow Per Epoch

```
 ┌─────────────┐   state vectors   ┌─────────────┐
 │  Population │ ───────────────▶ │  Utility    │
 │  (C++)      │                  │  Engine (C++)│
 └─────────────┘                  └──────┬──────┘
        ▲                                │
        │ threshold update                │ utilities
        │                                ▼
 ┌───────┴──────┐                  ┌─────────────┐
 │  Feedback    │ ◀── metrics ──── │  Swipe Log  │
 │  (C++)       │                  │  (CSV/npz)  │
 └─────────────┘                  └──────┬──────┘
                                         │
                                ┌────────▼─────────┐
                                │  NCF (Python)     │
                                │  train + predict  │
                                └────────┬─────────┘
                                         │ top-k scores
                                ┌────────▼─────────┐
                                │  Preference Lists │
                                │  P_i (truncated)  │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │  Gale-Shapley    │
                                │  (C++)            │
                                └────────┬─────────┘
                                         │ matching μ_alg
                                    metrics
```

## Module Boundaries

### C++ side (`src/cpp/`, headers in `include/`)
- `agent.{h,cpp}` — state vector, utility function, threshold update
- `population.{h,cpp}` — distribution sampling, population container
- `matching.{h,cpp}` — Gale-Shapley (full + truncated variants), blocking-pair check
- `simulation.{h,cpp}` — epoch orchestration, OpenMP parallel regions
- `bindings.cpp` — Pybind11 surface exposing the above to Python

### Python side (`src/python/`)
- `ncf.py` — GMF+MLP model, training loop, top-k inference
- `orchestrator.py` — drives one epoch: call C++ sim → log → train NCF → match
- `metrics.py` — Gini, clearing rate, rank satisfaction, algorithmic loss, Pareto fit
- `visualization.py` — static plots and epoch animations
- `config.py` — dataclass holding all hyperparameters (α, β, γ, δ, k, N, T, η)

## Interface Contract

**[docs/INTERFACES.md](INTERFACES.md) is the authoritative contract.** It fixes the state matrix,
every exported signature, numerical conventions, the metrics DataFrame schema, and the complexity
limits. This section only summarizes the shape of the boundary; if the two ever disagree, the
contract file wins and this one is stale.

```cpp
sample_population(n, config, seed)            -> float64 (n, 7)
generate_swipes(states, config, epoch)        -> int32   (S, 3)   [user, item, label]
build_full_info_pref_lists(states, config)    -> (prop_prefs, acc_prefs)   // O(N²) memory
deferred_acceptance(prop, acc, truncated)     -> {matches, unmatched_*, n_blocking}
compute_welfare(states, matches, config)      -> {welfare, welfare_proposer, ...}
update_thresholds(states, n_prop, n_match, eta) -> None   // mutates states[:, 6]
```

Python passes NumPy arrays in and gets NumPy arrays out, using `py::array_t` with
`py::array::c_style` so no copy is made at the boundary.

The reason a separate file exists: the C++ and Python layers are written in different Claude Code
sessions that share no context. The contract is the only thing keeping the two halves compatible.

## Scalability Targets

| Population | Epochs | Wall-clock target | Bottleneck |
|-----------|--------|-------------------|------------|
| 1,000 | 50 | < 30 s | NCF training |
| 10,000 | 100 | < 5 min | Utility eval + NCF |
| 100,000 | 100 | < 10 min | Memory bandwidth, NCF inference |

## Configuration Surface

All experiment parameters live in one YAML file (`experiments/base.yaml`), loaded into the `config.py` dataclass. Parameter sweeps override individual keys. This keeps experiments reproducible and diffable.
