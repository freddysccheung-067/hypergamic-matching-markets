# rigor/c4_shock0.5 — the low-chemistry cell, run to 300 epochs

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_ablation.py --config experiments/c4_shock0.5.yaml \
        --seeds 42,43,44,45,46 --epochs 300 \
        --output-dir data/processed/rigor/c4_shock0.5

## Why this exists

`data/processed/rigor/sensitivity.csv` reports reach 0.4616 at `shock_scale` 0.5
against 0.9566 shipped, and the paper read that as the cell where the filter never
learns coverage — the one place the cold-start explanation should fail. That
reading rests on **epoch 99**. This directory asks what the cell does at 300.

Prediction registered in `docs/PREREGISTRATION.md` E4 before the run; the result
is recorded in E3c.

## Regime

`experiments/c4_shock0.5.yaml`: `hypergamy.shock_scale: 0.5`, everything else
shipped, `simulation.n_epochs: 300`, 1 thread. The config sets `n_epochs` to 300
explicitly so `epochs_per_gen = 300 // 5 = 60`, matching the generation schedule a
300-epoch design implies.

## Result: reach does climb here, so the premise was false

| epoch | 0 | 19 | 49 | 99 | 199 | 299 |
|---|---|---|---|---|---|---|
| gap, mean | +0.0246 | +0.0520 | +0.0350 | +0.0111 | −0.0009 | −0.0012 |
| reach, mean | 0.041 | 0.073 | 0.269 | **0.457** | 0.488 | **0.490** |

The gap peaks at +0.0565 (epoch 9) and decays to −0.0012 — proportionally as
complete a decay as the shipped cell's, from a peak a fifth as large.

**0.4616 was reach at epoch 99, not a ceiling.** Over 300 epochs reach reaches
0.490, so the cell does not isolate a market without coverage learning; it isolates
a slower instance of the same process. E4's stated premise was therefore false and
its prediction (ratio of the epoch-99 gap to the epoch-299 gap below 10) is
undefined because the epoch-299 gap is negative.

**Consequence for the paper:** the sweep's "conclusion-reversing" claim is
withdrawn. The cold-start explanation is not rescued by this cell and not
contradicted by it — the cell does not test it either way.

## Thread count

**1.**

## Files

`<arm>_seed<seed>.csv`, one row per epoch (300), four arms, the trimmed column set
`docs/INTERFACES.md` §6; `summary.csv` and `gaps.csv` at the final epoch.
