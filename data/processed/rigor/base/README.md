# rigor/base — twenty seeds, full metrics table, all four arms

Generated 2026-10-25 on branch `rigor/statistical-hardening`.

    python scripts/run_rigor_base.py --all \
        --output-dir data/processed/rigor/base

## What this holds

`metrics_seed<seed>.csv`, one file per seed, seeds 42–61, holding **every row for
every epoch and all four arms** — the full `Simulation.run()` frame rather than
the trimmed nine-column set `run_ablation.py` writes. The extra columns are the
point: `algorithmic_loss` and `welfare_full` are computed on every epoch and were
discarded at write time before this directory existed, so the statistical work in
`docs/STATS.md` §1–§2 had no committed source for them.

Keeping the arms in one file per seed makes the pairing explicit: the arms in a
row set share a population, so a gap is a row-wise subtraction and never a
difference of means.

## Regime

`experiments/base.yaml` as shipped at the time of generation: `φ = 0.02`,
100 epochs, N = 1000, k = 20. Regime label `base20` in
`../intervals.csv`. It duplicates `live100`'s regime deliberately, so §2 can
compare five seeds against twenty on one regime rather than across two.

## This directory does not reproduce

Re-running seed 42 today at 1 thread gives an epoch-99 NCF Gini of **0.6976498**
against the committed **0.6973985** — a difference of 2.5e-4 at the same thread
count and the same regime.

## Thread count

**1.** It predates `ncf.num_threads`, so the count was whatever torch read from
the machine; the development machine reports 24, but the committed epoch-99 Gini
of 0.6973985 is matched by a run at **1**, not at 24 (0.6935411), which is why 1
is the recorded value and why the difference from today's 1-thread run is a code
difference rather than a thread one.

That 2.5e-4 gap is small enough that every conclusion built on this directory
survives, and large enough that this is an archival record rather than a fixture.
`docs/INTERFACES.md` §3.1(c) says as much for the whole of `data/processed/`, and
this directory is the clearest case of it.

**Consequence for the C1 comparison.** `../fresh/` is compared against
`../exploratory_1t/` — seeds 42–61 regenerated in the same session — and not
against this directory, because comparing a fresh run to a directory that no
longer regenerates would confound the seed set with the code version.
`../intervals.csv` still carries intervals computed over this directory's values;
those are what the paper quotes for `base20`, and they are the numbers to treat as
committed rather than as reproducible.

## Files

| file | contents |
|---|---|
| `metrics_seed<seed>.csv` | one row per (arm, epoch), every column of `docs/INTERFACES.md` §6 |
