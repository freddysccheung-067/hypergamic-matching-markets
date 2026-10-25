# rigor/long_horizon_1t — the 300-epoch shipped cell, regenerated at 1 thread

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_ablation.py --config experiments/long_horizon_1t.yaml \
        --seeds 42,43,44,45,46 --epochs 300 \
        --output-dir data/processed/rigor/long_horizon_1t

## Why this exists

`../long_horizon/` is the committed 300-epoch run, and its NCF arm does not
reproduce from this code at 1, 2, 4, 8, 16 or 24 threads, nor from the source as
it stood before the 2026-10-24 scoring change (`docs/REVIEW_RESPONSE.md` A3). Its
oracle, random and popularity arms reproduce exactly. The paper's central claim
rests on that directory's trajectory, so a regenerated series was needed.

This directory is that series, and it is a **measurement taken today**, not a
reproduction of the committed one.

## Regime

`experiments/long_horizon_1t.yaml`: the shipped configuration with
`simulation.n_epochs: 300` and `ncf.num_threads: 1`. Setting `n_epochs` explicitly
is deliberate — it makes `epochs_per_gen = 300 // 5 = 60`, which is the generation
schedule a 300-epoch design implies.

## The comparison was pre-registered

`docs/PREREGISTRATION.md` E7 fixed the decision rule before this run's output
existed: if the tail mean is within 0.001 of the committed +0.000374 and the
epoch-99 gap within one committed seed sd, the two agree at the resolution the
paper claims and the paper switches to this directory; otherwise the paper reports
this one as the measurement, states that the committed directory cannot be
reproduced and disagrees, and does not average the two or pick the convenient one.

`scripts/_compare_long_horizon.py` applies that rule mechanically. Its output is
recorded in `docs/PREREGISTRATION.md` E7a and in the paper's
`\S`~cold-start section.

## Thread count

**1.** Which is why this directory is comparable to `../fresh/`,
`../exploratory_1t/`, `../interventions/` and `../c4_shock0.5/` and to nothing
else in the repository.

## Files

`<arm>_seed<seed>.csv`, one row per epoch (300), four arms, the trimmed column set
of `docs/INTERFACES.md` §6; `summary.csv` and `gaps.csv` at the final epoch.
