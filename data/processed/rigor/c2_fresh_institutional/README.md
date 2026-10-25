# rigor/c2_fresh_institutional — superseded; kept for the record

Generated 2026-10-25 on branch `rev/revision` **at the wrong gate**.

    python scripts/run_institutional.py --only grid --seeds 100,...,119 \
        --output-dir data/processed/rigor/c2_fresh_institutional

## Why this directory is wrong, and why it is kept

The committed `../institutional/` grid was measured at
`admissible_fraction: 0.2`. This run used `experiments/base.yaml` as shipped,
which is `0.02`. The two are different regimes, so the comparison it enables —
fresh seeds at one gate against five seeds at another — cannot speak to the H4
claim. `docs/PREREGISTRATION.md` E2a records the error.

It is kept rather than deleted because it is a real run at a real configuration
and deleting it would hide the mistake. **Do not quote it against the committed
grid.** The correct version of this experiment is `../c2_fresh_inert/`, and a
version at the shipped gate is `../h4_shipped_gate/`.

## What it does show

Run at 0.02, the cell at `size_fraction = 0.65` reads **−12.98** on twenty fresh
seeds, and seeds 42–46 read **−17.23** — against the committed **+64.62** at
0.2. Both facts are consistent with the gate being the whole of the difference,
and the exact reproduction at 0.2 of all 200 committed rows confirms it.

## Thread count

**1, and it cannot matter:** neither arm trains the NCF.
