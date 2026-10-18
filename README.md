# Hypergamic Matching Markets

![Build](https://img.shields.io/github/actions/workflow/status/freddysccheung-067/hypergamic-matching-markets/ci.yml?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)
![C++](https://img.shields.io/badge/C%2B%2B-17-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

A computational simulation of how hypergamous mate-selection preferences interact with algorithmic recommendation systems to produce macro-level market failure in dating ecosystems.

The simulation fuses three layers: an Agent-Based Model of heterogeneous agents with asymmetric utility functions, a Neural Collaborative Filtering engine that mediates visibility, and a Gale-Shapley market-clearing algorithm. Together they expose how small preference asymmetries are non-linearly amplified into severe exposure inequality, declining match rates, and structural demographic imbalance.

---

## The Model

Each agent $a_i$ carries a state vector $\mathbf{v}_i = [s_i, a_i, c_i, t_i]^\top$ — socio-economic status, attractiveness, sociability, and age. Hypergamy enters through the utility function agents in $W$ use when evaluating candidates in $M$:

$$U_w(m) = \alpha \cdot s_m + \beta \cdot a_m - \gamma \cdot d(\mathbf{v}_w, \mathbf{v}_m) + \lambda \cdot \mathbb{1}[s_m \geq s_w + \delta]$$

The indicator term rewards status surplus $\delta > 0$, encoding the preference asymmetry. Agents in $M$ use a symmetric utility with no status threshold. Each epoch, a Neural Collaborative Filtering engine (GMF + MLP) trains on swipe logs and generates truncated preference lists, which are passed to a modified Gale-Shapley algorithm to clear the market.

See [docs/THEORY.md](docs/THEORY.md) for full mathematical formulation and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.

---

## Key Metrics

| Metric | What it measures |
|--------|-----------------|
| Gini coefficient $G$ | Exposure inequality across the agent population |
| Market clearing rate | Fraction of agents who receive a match |
| Rank satisfaction index $R$ | How highly agents rank their actual partner |
| Algorithmic loss $\Delta$ | Welfare gap between full-information and NCF-mediated matching |
| Pareto $\alpha$ (per generation) | Tail concentration in cumulative match distribution |

---

## Installation

```bash
# Prerequisites: CMake 3.18+, Python 3.11+, C++17 compiler, OpenMP
git clone https://github.com/freddysccheung-067/hypergamic-matching-markets
cd hypergamic-matching-markets
pip install -r requirements.txt
cmake -B build && cmake --build build -j$(nproc)
```

## Usage

```python
from src.python.orchestrator import Simulation

sim = Simulation(config="experiments/base.yaml")
metrics = sim.run(n_epochs=100)
```

Run a parameter sweep over the hypergamy coefficient:

```bash
python scripts/run_sweep.py --delta-values 0.0,0.5,1.0,1.5 --output-dir data/processed/
```

---

## Repository Structure

```
src/cpp/        Agent state, utility, Gale-Shapley engine (C++17 + OpenMP)
src/python/     NCF model, orchestrator, metrics, visualization (PyTorch)
experiments/    YAML configs for reproducible runs
scripts/        Parameter sweep runner
tests/          C++ (Catch2) and Python (pytest) test suites
notebooks/      Full-pipeline demo notebook
docs/           Mathematical formulation and architecture diagrams
```

## License

MIT
