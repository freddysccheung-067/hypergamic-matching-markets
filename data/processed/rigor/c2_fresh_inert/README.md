# rigor/c2_fresh_inert — the H4 grid on fresh seeds, at the original gate

Generated 2026-10-25 on branch `rev/revision`.

    python scripts/run_institutional.py --only grid \
        --config experiments/c2_inert.yaml \
        --seeds 100,101,...,119 \
        --output-dir data/processed/rigor/c2_fresh_inert

## Why this exists

The H4 optimum at `(c = 0.01, size_fraction = 0.65)` was chosen as the maximum of
forty cells on the same five seeds (42–46) that reported it, and its per-cell seed
sd was the ninth smallest of forty. The clean answer is to re-estimate on seeds
that took no part in the choice. That is what this directory is, registered as
`docs/PREREGISTRATION.md` E2 (and corrected to the right gate in E2a) before the
run.

`experiments/c2_inert.yaml` is `experiments/base.yaml` with
`population.admissible_fraction` set back to **0.2**, because the committed
`../institutional/` grid was generated at that gate. Comparing fresh seeds at the
shipped 0.02 against a grid measured at 0.2 would confound the seed set with the
regime — the error an earlier attempt at this run made, recorded in E2a.

## Result: location replicates, magnitude and precision do not

`c = 0.01`, sorted by mean residual welfare difference, twenty seeds:

| rank | size_fraction | mean | sd |
|---|---|---|---|
| 1 | **0.65** | **+54.45** | **46.42** |
| 2 | 0.80 | +39.64 | 40.91 |
| 3 | 0.50 | +37.43 | 45.92 |
| 4 | 0.35 | +14.27 | 26.30 |
| 5 | 1.00 | +2.90 | 28.14 |
| 6 | 0.25 | +1.73 | 32.65 |
| 7–10 | 0.02, 0.05, 0.10, 0.15 | −4.98 to −14.69 | |

The optimum cell ranks first, as it did originally, and the non-monotone shape is
preserved. Its fresh interval is **[+32.72, +76.17]** (t, 19 df), excluding zero at
t = 5.25, p = 4.6e-5.

Two things did not replicate. The point estimate is 10.17 lower than the original
+64.62, which narrowly misses E2's one-sd criterion (9.86) — and that criterion
was itself built on a five-seed sd that turns out to be **4.7 times too small**:
the fresh sd at the same cell is 46.42 against 9.86. The published precision was
overstated by a factor of about five, the same low-variance pathology
`docs/STATS.md` §2.2 found for algorithmic loss and clearing.

## Regime caveat

This directory is at the inert gate, like the grid it replicates. It therefore
says nothing about whether the optimum exists at the shipped gate — and
`../h4_shipped_gate/` answers that: it does not.

## Thread count

**1, and it cannot matter.** A is the `oracle` arm and B the `complete` arm;
neither trains the NCF.

## Files

`grid.csv` (200 rows: 10 sizes × 4 costs × 20 seeds) and `grid_summary.csv`
(mean and sd per cell).
