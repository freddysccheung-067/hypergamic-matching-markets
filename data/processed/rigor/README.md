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
| `long_horizon/` | the 300-epoch run and its README | `scripts/run_ablation.py` |
| `long_horizon_*.csv` | the gap series, decay fits, asymptote, stationarity | `scripts/analyse_long_horizon.py` |
| `base/` | 20 seeds, full metrics table, all four arms | `scripts/run_rigor_base.py` |

## Reproducing

Every bootstrap is seeded from the constant `20261024`, so the tables regenerate
byte for byte. Every simulation respects `population.seed`, so the underlying
runs regenerate identically too — with the one caveat `docs/INTERFACES.md` §3
already states: cross-compiler identity of a sampled trait is not guaranteed,
only cross-thread-count identity within one toolchain.

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
