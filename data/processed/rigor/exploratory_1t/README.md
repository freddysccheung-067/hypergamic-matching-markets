# rigor/exploratory_1t — seeds 42–61 regenerated at 1 thread

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_rigor_base.py --seeds 42,...,61 \
        --epochs 100 --jobs 8 --output-dir data/processed/rigor/exploratory_1t

## What this exists for

It is the comparison arm for `../fresh/`. The committed `../base/` holds the same
twenty seeds at the same regime, but it was generated before the current code and
does not reproduce: seed 42's epoch-99 NCF Gini reads **0.6973985** there against
**0.6976498** from a run today at 1 thread. The difference is 2.5e-4 — small, same
thread count, same regime — which means the committed directory is archival rather
than a fixture, exactly as `docs/INTERFACES.md` §3.1(c) warns.

A comparison of fresh seeds against an archival directory would confound the seed
set with the code version, so both arms of the C1 comparison were generated inside
one session.

## Regime

`experiments/base.yaml` as shipped, 100 epochs, four arms, seeds 42–61,
1 thread.

## Result

See `../fresh/README.md` for the side-by-side table. Headlines at n = 20:

| quantity | value |
|---|---|
| exposure Gini, NCF | 0.67267 |
| exposure Gini, oracle | 0.65015 |
| gap, NCF − oracle | +0.02252 [+0.01868, +0.02635] |

## Files

`metrics_seed<seed>.csv`, one per seed, all four arms, every column of
`docs/INTERFACES.md` §6.
