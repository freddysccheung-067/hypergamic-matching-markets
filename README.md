# Hypergamic Matching Markets

![Build](https://img.shields.io/github/actions/workflow/status/freddysccheung-067/hypergamic-matching-markets/ci.yml?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)
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

Requires CMake 3.18+, Python 3.9+, and a C++17 compiler. See
[docs/PREREQUISITES.md](docs/PREREQUISITES.md) for the Windows toolchain setup — MSVC Build Tools
are needed and are not installed by default.

```powershell
git clone https://github.com/freddysccheung-067/hypergamic-matching-markets
cd hypergamic-matching-markets
python -m pip install -r requirements.txt
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

The compiled extension lands in `build\Release`. Put that directory on `PYTHONPATH` before
importing `src.python`:

```powershell
python -c "import sys; sys.path.insert(0, 'build/Release'); import _hypergamic; print('ok')"
```

## Usage

```python
from src.python.orchestrator import Simulation

sim = Simulation(config_path="experiments/base.yaml")
metrics = sim.run(n_epochs=100)
```

Run a parameter sweep over the hypergamy coefficient:

```powershell
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

## Limitations

This is a model of a mechanism, not a measurement of a population. The distinction matters most
for attractiveness, so it gets its own note.

**Every trait is synthetic.** Status, attractiveness, sociability and age are drawn from
distributions chosen for tractability. Nothing here estimates a real quantity, and no output says
anything about an actual person.

**Attractiveness is not measurable on the scale the model uses.** Three separate problems, and the
model only absorbs the first:

- *Self-report compression.* People cluster near the top of the scale — a 7/10 rather than a 5/10
  for someone who feels average. A shift shared by everyone is an intercept and cancels in a
  ranking comparison, so this is less damaging than it looks. What it destroys is usable variance:
  if everyone reports 7, the term carries no information.
- *Rank scales are ordinal, not interval.* A 7 and a 5 are labels, not measurements. Treating them
  as reals assumes comparable gaps between them, which nothing in how they are elicited supports.
- *There is no absolute scale.* A 7 in one city, age group or app is a 5 in another. Any empirical
  calibration would have to name its reference population.

Because traits are standardized before entering the utility, only the *rank order* of
attractiveness matters — the specific 0–1 scale and the self-report question cancel. What the
model cannot survive is a distribution with no variance, which heavy compression would produce.
That is why the default is not a self-report-shaped distribution: a term that barely varies cannot
test anything. The results should be read as an upper bound on how much attractiveness can
influence selection, not as an estimate of how much it does.

**The Matthew effect needs its control.** A rising Gini under the recommender proves nothing on
its own — preferences alone concentrate attention on high-status agents. The claim this project
supports is `Gini(NCF) − Gini(random)`, measured against a random-exposure arm sharing the same
population, seed and utilities. Gini(NCF) reported alone is not evidence of algorithmic
amplification.

**Correlation among matched pairs is a selection artifact.** Traits are drawn independently, so no
trait causes another. The market sorts on several at once, which makes the survivors correlate —
strongly at the top of the distribution. That is a collider, not a finding about individuals.

**Other limits.** The full-information benchmark is capped near N=5000 for memory reasons.
M-proposing Gale-Shapley returns the proposer-optimal matching, which understates the acceptor
side's dissatisfaction. Generational turnover is the only population dynamic; there is no entry,
exit or preference learning from outside the simulation.

## License

MIT
