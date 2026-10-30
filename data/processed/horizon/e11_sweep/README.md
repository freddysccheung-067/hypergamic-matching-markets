# horizon/e11_sweep/

Written by `scripts/analyze_extras.py` from the run records in this directory. The design of each experiment is its entry in `docs/PREREGISTRATION.md`.

| registration | question | seeds | rounds | commits | CPU hours |
|---|---|---|---|---|---|
| R2-E11 | one-knob sweep, eight cells | 100-109 (10) | 300 | 349b484 | 15.5 |

Analysis outputs: `analysis/cells.csv`, `analysis/per_seed.csv`, `analysis/series.csv`.

## Thread count

**1**: every run record here states `ncf.num_threads = 1`.
