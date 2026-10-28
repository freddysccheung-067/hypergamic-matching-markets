# horizon/e16_fp/

Written by `scripts/analyze_extras.py` from the run records in this directory. The design of each experiment is its entry in `docs/PREREGISTRATION.md`.

| registration | question | seeds | rounds | commits | CPU hours |
|---|---|---|---|---|---|
| R2-E16 | floating-point sensitivity, ten runs per seed | 100-104 (5) | 300 | 9b11a36 | 81.3 |

Analysis outputs: `analysis/spread.csv`, `analysis/summary.csv`.

## Thread count

**1**: every run record here states `ncf.num_threads = 1`.
