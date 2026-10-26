# rigor/h4_shipped_gate — the institutional grid at the shipped gate

Generated 2026-10-25 on branch `rev/revision`.

    python scripts/run_institutional.py --only grid \
        --seeds 42,43,44,45,46 \
        --output-dir data/processed/rigor/h4_shipped_gate

## Regime

`experiments/base.yaml` **as shipped**: `admissible_fraction: 0.02`. That is the
whole point of the directory — the committed `../institutional/` grid was measured
at `0.2`, the inert gate, and this one asks whether its result survives at the
gate the rest of the paper runs.

Everything else is identical to `../institutional/grid.csv`: ten
`size_fraction` levels by four `signal_cost` levels, seeds 42–46, five epochs per
paired run, Environment A as the oracle arm and B as the complete arm.

## Result

**No cell of the forty beats Environment A.** At `c = 0.01` the residual welfare
difference at each size:

| size_fraction | mean diff | sd |
|---|---|---|
| 0.02 | −0.23 | 0.50 |
| 0.05 | −1.45 | 2.47 |
| 0.10 | −4.89 | 3.90 |
| 0.15 | −9.64 | 5.07 |
| 0.50 | −10.56 | 12.55 |
| 0.25 | −11.39 | 6.39 |
| 0.65 | −17.23 | 13.32 |
| 0.35 | −19.47 | 14.75 |
| 0.80 | −22.66 | 32.52 |
| 1.00 | −40.22 | 27.21 |

At `c = 0.05`, `0.10` and `0.25` the best cell is negative too (−0.25, −0.28,
−0.37). The ordering is close to monotone in size: the thin-market penalty
dominates the information gain at every size tested.

## Regime check

Re-running seeds 42–46 of the **committed** grid at `admissible_fraction: 0.2`
reproduces all 200 rows of `../institutional/grid.csv` exactly, to the last digit,
on `welfare_common_a`, `welfare_diff_residual` and `gini_a`. At `0.02` it does
not. The gate is the whole of the difference, which is why this directory exists
and why `../institutional/README.md`'s "pin the inert regime to reproduce"
instruction was right.

## Thread count

**1, and it cannot matter: this directory trains no model.** A is the `oracle`
arm and B is the `complete` arm; neither runs the NCF, so no torch reduction is
involved. See `docs/INTERFACES.md` §3.1.

## What it licenses

Refuting H4 as stated. The interior optimum reported in `../institutional/` is a
property of a market whose threshold gate never binds — the configuration
`paper/sections/design.tex` describes as one where the market never responds to
its own outcomes. At the shipped gate there is no interior optimum at all.

The run was **not pre-registered**; `docs/PREREGISTRATION.md` E6 records it with
that label and says why it was run anyway.
