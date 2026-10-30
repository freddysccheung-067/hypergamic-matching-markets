# horizon/e13a_dose/

Written by `scripts/analyze_extras.py` from the run records in this directory. The design of each experiment is its entry in `docs/PREREGISTRATION.md`.

| registration | question | seeds | rounds | commits | CPU hours |
|---|---|---|---|---|---|
| R2-E13a | training dose: 0.5, 2, 4 passes per round | 100-109 (10) | 300 | f4d2b8c | 9.3 |

Analysis outputs: `analysis/by_dose.csv`, `analysis/per_seed.csv`, `analysis/series.csv`, `analysis/summary.csv`.

## Thread count

**1**: every run record here states `ncf.num_threads = 1`.
