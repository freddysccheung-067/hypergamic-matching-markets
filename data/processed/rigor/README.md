# Rigor outputs: intervals, power, convergence and sensitivity

Generated 2026-10-24/25 on branch `rigor/statistical-hardening`. This directory
exists so that every number in `docs/STATS.md` can be traced to a file, and so
that the statistical claims the thesis rests on can be checked without re-running
the simulation.

## What replaced what

Every headline in this repository was a mean ± sd over five seeds. An sd is a
spread, not an interval, and at five seeds it is itself badly estimated
(`docs/STATS.md` §2 measures how badly). This directory holds the interval
version: BCa bootstrap over seeds, 10,000 resamples, with the percentile and
Student-t intervals beside it on every row.

## Contents

| path | what it holds | produced by |
|---|---|---|
| `intervals.csv` | the headline table: 30 rows, one per (quantity, regime), each with BCa + percentile + t | `scripts/run_intervals.py` |
| `power_analysis.csv` | 5-seed vs 20-seed spread, and the seeds each claim would need | `scripts/run_power_analysis.py` |
| `power_widths.csv` | interval width by seed count, against the 1/√n law | same |
| `power_conclusions.csv` | does the interval exclude zero, at each n, over every block | same |
| `multiple_comparisons.csv` | 40 cells, BH and Holm, the optimum's corrected interval | `scripts/run_multiple_comparisons.py` |
| `sensitivity_summary.csv` | the 40 institutional cells with per-cell t, p, q | same |
| `sensitivity.csv` | the 12 parameter-sweep cells with the four claim verdicts | `scripts/summarise_sensitivity.py` |
| `sensitivity/` | the raw cells and their README | `scripts/run_sensitivity.py` |
| `long_horizon/` | the committed 300-epoch run, **archival: does not reproduce** | `scripts/run_ablation.py` |
| `long_horizon_1t/` | the 300-epoch run the paper actually quotes, regenerated at 1 thread | same, `experiments/long_horizon_1t.yaml` |
| `long_horizon_*.csv` | the gap series, decay fits, asymptote, stationarity, for the regenerated series; the committed one's live inside `long_horizon/` | `scripts/analyse_long_horizon.py` |
| `base/` | 20 seeds, full metrics table, all four arms | `scripts/run_rigor_base.py` |
| `validation.csv` | Table 1's seven rows, recomputed rather than transcribed | `scripts/run_validation.py` |
| `fresh/` | seeds 100–119, the pre-registered confirmation set | `scripts/run_rigor_base.py` |
| `exploratory_1t/` | seeds 42–61 regenerated in the same session, the comparison arm for `fresh/` | same |
| `interventions/` | C3: control, frozen ranker, warm start | `scripts/run_interventions.py` |
| `latefreeze/` | E3b: the ranker frozen at epochs 20 and 40 | same |
| `c4_shock0.5/` | 300 epochs at `shock_scale` 0.5 | `scripts/run_ablation.py` |
| `c2_fresh_inert/` | the H4 grid on seeds 100–119 at the inert gate | `scripts/run_institutional.py --config experiments/c2_inert.yaml` |
| `h4_shipped_gate/` | the H4 grid at the shipped gate — no cell beats A | same, shipped config |
| `c2_fresh_institutional/` | superseded: the grid on fresh seeds at the *wrong* gate; kept, not quoted | same |

## Added in the revision of 2026-10-25

Five of the directories above exist because an earlier result turned out to rest
on a regime, a seed set, or a code version that the paper had not examined. Each
has its own README stating what was run, at what gate and thread count, and what
the result does not license:

- `fresh/` and `exploratory_1t/` answer the seed-selection objection, in one
  session at one thread count because the committed `base/` no longer reproduces
  (see its README).
- `interventions/` and `latefreeze/` test the coverage-learning mechanism. Both
  report a failed test: the frozen arm is flat by construction, and the warm start
  leaves two thirds of the decay unexplained.
- `h4_shipped_gate/` refutes H4 where the paper actually runs. It is the one
  directory here whose run was **not** pre-registered; `docs/PREREGISTRATION.md`
  E6 says so.

`docs/PREREGISTRATION.md` carries the prediction, or the admission that there was
none, for every one of them.

## Reproducing

Every bootstrap is seeded from the constant `20261024`, so the summary tables
regenerate byte for byte from the CSVs they read.

The simulations underneath them are a weaker promise, and the earlier version of
this paragraph overstated it. What `population.seed` guarantees is that a run
repeats itself at a fixed thread count, not that a checkout today reproduces a file
committed before the code it ran on changed. For the two directories here, it does
not: re-running `ncf` seed 42 on 2026-10-25 reproduces neither `base/` nor
`long_horizon/` at 1, 8 or 24 threads, while the same code reproduces
`../ablation_h100/` exactly at 24. See `docs/INTERFACES.md` §3.1 for the three claims
this repository makes about reproducibility and which of them apply where.

## Thread count

**24, for the simulation outputs, and not separately verified.** Every directory under
`data/processed/rigor/` was generated before `ncf.num_threads` existed, so each run
took whatever count torch read from the machine; the development machine reports 24
and that is the assumption recorded here rather than a measurement. It is recorded at
all because the count is part of a run's identity (`docs/INTERFACES.md` §3.1(b)): at
100 epochs the exposure Gini moves by 0.007 to 0.015 across 1, 4 and 8 threads, which
is why the number matters and why an unverified one is stated as unverified.

The analysis outputs under this directory that carry no simulation in them
(`intervals.csv`, `power_*.csv`, `multiple_comparisons.csv`, `sensitivity*.csv` and
the `long_horizon_*.csv` fits) are bootstrap or OLS products over the simulation
CSVs. Their thread count is whatever the CSVs they read had, and they regenerate
from those CSVs without re-simulating, so they are not affected by the current
machine's core count.

The one file here that *is* about threads, `thread_sensitivity.csv`, records its own
count in a column: seeds 42 to 44 at 1, 4 and 8 threads, one row per (seed, threads).
It was generated by hand for the measurement rather than by a committed script, and
re-running seed 42 at 1 thread on 2026-10-25 reproduces its 0.6976498 exactly.

## Which regime is which

| regime | φ | epochs | directory |
|---|---|---|---|
| `inert20` | 0.2 | 20 | `../ablation/` |
| `live20` | 0.02 | 20 | `../ablation_live/` |
| `live100` | 0.02 | 100 | `../ablation_h100/` |
| `long300` | 0.02 | 300 | `long_horizon/` |
| `base20` | 0.02 | 100 | `base/` (20 seeds, full metrics) |

The first three predate this branch and are unchanged; the last two were produced
here. `base20` duplicates `live100`'s regime deliberately — it is the same run
extended to 20 seeds so §2 can compare seed counts on one regime rather than
across two.

## What these do not support

The intervals are intervals for a *model's* seed-averaged quantities. They say
nothing about any real population: the model has no empirical calibration and
`docs/THESIS.md` says so. A narrow interval means the simulation is reproducible
across populations, which is a statement about the code and the model, not about
the world.
