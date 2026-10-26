# rigor/latefreeze — the frozen ranker at epochs 20 and 40

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_interventions.py --late-freeze 20,40 \
        --seeds 42,43,44,45,46 --epochs 100 --jobs 5 \
        --output-dir data/processed/rigor/latefreeze

## Why this exists

`../interventions/`'s frozen arm froze at epoch 6, the peak of the *gap*. Reach at
epoch 6 is 0.041, so the ranker could barely rank and the arm was degenerate in
two ways at once. `docs/PREREGISTRATION.md` E3b registered this follow-up — freeze
where the ranker can already rank — before it was run.

## Conditions

| condition | freeze epoch | control reach there |
|---|---|---|
| `frozen20` | 20 | 0.182 |
| `frozen40` | 40 | 0.607 |

Plus `control` from `../interventions/`, reproduced here so every comparison is
within one file.

## Result: the arms are flat by construction

Both frozen arms are constant to the last digit from the freeze epoch onward
(`frozen20`: 0.939596 at every epoch 21–99; `frozen40`: 0.883149 at every epoch
41–99). The within-arm change has an sd of **exactly zero**.

That is structural, not a result. Freezing the weights freezes the score matrix,
which freezes the shortlists, which fixes exposure; the arm cannot decay because
nothing in it can move. **The frozen-ranker intervention cannot test the
coverage-learning mechanism in this model at any freeze point**, which E3c records
as the session's clearest negative result about its own design.

What the two freeze points show cross-sectionally is at least consistent with the
coverage story: a ranker frozen at higher reach sits at a lower gap (0.8831 Gini at
epoch 40 against 0.9396 at epoch 20), in the right order. Consistent is not tested.

## Thread count

**1.**

## Files

`latefreeze_seed<seed>.csv`, one row per (condition, epoch), same columns as
`../interventions/`.
