# horizon/: the second revision's experiments

Every number in the paper comes from a directory under this one. Each experiment
directory holds one `seed<N>.csv` per seed (one row per arm and round) and, beside
it, `seed<N>.json`, the run record: the commit the run came from, whether that tree
was clean, the full configuration and its hash with the seed removed, the library
versions and the wall time. `analysis/` inside each directory holds what
`scripts/analyze_horizon.py` and `scripts/analyze_extras.py` computed from the seed
files; the paper reads only those.

| directory | registration | what it is |
|---|---|---|
| `e09_canonical/` | R2-E9, R2-E12 | four arms, 300 rounds, seeds 42-61 and 100-119, with diagnostics |
| `e10_long/` | R2-E10 | NCF and oracle, 1000 rounds, seeds 100-109 |
| `e11_sweep/` | R2-E11 | eight one-knob cells, 300 rounds, seeds 100-109 |
| `e13a_dose/` | R2-E13a | 0.5, 2 and 4 training passes per round |
| `e13b_explore/` | R2-E13b, R2-E13b-check | exploration slots; oracle and control prefixes |
| `e13c_offpolicy/` | R2-E13c | shadow rankers trained on oracle and random markets |
| `e13d_warm/` | R2-E13d | warm start, 40 seeds |
| `e14_arrivals/` | R2-E14 | replacement at 0.5, 1 and 2 percent per round, 600 rounds |
| `e15_institutional/` | R2-E15 | 10 x 4 institutional grid, rounds 250-299 |
| `e15i_inert/` | R2-E15i | the c = 0.01 slice at the inert gate, 5 rounds |
| `e15s_five/` | R2-E15s | the grid read over rounds 0-4 (post hoc) |
| `e16_fp/` | R2-E16 | ten 1e-7 weight perturbations of each of five seeds |
| `e17_size/` | R2-E17 | N = 500 and 2000 |
| `h1r/`, `h5r/` | R2-H1r, R2-H5r | the appendix hypotheses at the shipped gate |

Files at this level: `run_index.csv` (one row per experiment, from the run
records), `units.csv` (T(0.05) in rounds, rows per agent and optimizer steps),
`predictions.csv` (every registered prediction, scored), `parameters.csv` (the
model's parameters, from `experiments/base.yaml`) and `registrations.csv` (the
commit at which each registration entered `docs/PREREGISTRATION.md`).

## Thread count

**1**, for every run under this directory: every run record states
`ncf.num_threads = 1` and `OMP_NUM_THREADS = 1`, and the analysis scripts refuse
a directory whose records disagree.
