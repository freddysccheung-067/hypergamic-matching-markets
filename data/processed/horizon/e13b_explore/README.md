# horizon/e13b_explore/

Written by `scripts/analyze_extras.py` from the run records in this directory. The design of each experiment is its entry in `docs/PREREGISTRATION.md`.

| registration | question | seeds | rounds | commits | CPU hours |
|---|---|---|---|---|---|
| R2-E13b | exploration slots | 100-109 (10) | 300 | f4d2b8c | 3.9 |
| R2-E13b-check (post hoc) | comparator check for exploration | 100-109 (10) | 300 | 9b11a36 | 4.5 |

Analysis outputs: `analysis/check.csv`, `analysis/paired.csv`, `analysis/series.csv`.

## Thread count

**1**: every run record here states `ncf.num_threads = 1`.
