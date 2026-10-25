# rigor/fresh — twenty fresh seeds, the C1 confirmation set

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_rigor_base.py --seeds 100,101,...,119 \
        --epochs 100 --jobs 8 --output-dir data/processed/rigor/fresh

## Regime

`experiments/base.yaml` as shipped, 100 epochs, four arms, seeds **100–119**.
These twenty seeds were registered as a confirmation set in
`docs/PREREGISTRATION.md` E1 **before** the run, on the reasoning that seeds
42–61 had been used to choose the H4 optimum and to anchor every exploratory
table.

## Why one session and one thread count

`ncf.num_threads` is part of a run's identity (`docs/INTERFACES.md` §3.1(b)): the
same seed at different counts gives different numbers over 100 epochs of this
feedback loop. So the comparison set was regenerated in the same session, at the
same count, into `../exploratory_1t/` — **not** against the committed
`../base/`, which predates the current code and does not reproduce (seed 42 reads
0.6973985 there against 0.6976498 here).

## Result

Twenty-seed point estimates, 1 thread, final epoch:

| quantity | exploratory (42–61) | fresh (100–119) |
|---|---|---|
| exposure Gini, NCF | 0.67267 | 0.67217 |
| exposure Gini, oracle | 0.65015 | 0.64684 |
| gap, NCF − oracle | +0.02252 [+0.01868, +0.02635] | **+0.02533 [+0.02169, +0.02897]** |
| clearing rate, NCF | 0.24350 | 0.23850 |
| rank satisfaction, NCF | 0.46285 | 0.46269 |
| reach, NCF | 0.95625 | 0.95645 |
| algorithmic loss | 0.66364 | 0.67178 |

Intervals are 95% Student-t over seeds. **Every fresh estimate falls inside the
exploratory t interval**, and every ordering holds on all twenty seeds
(popularity > ncf > oracle > random), which is what E1 predicted. Welch tests
against the exploratory set give |t| ≤ 1.12 on every quantity.

The fresh set has a smaller spread on the Gini quantities and a larger one on
algorithmic loss, which is the quantity-specific behavior `docs/STATS.md` §2.2
describes rather than a contradiction of it.

## Files

`metrics_seed<seed>.csv`, one per seed, all four arms, every column of
`docs/INTERFACES.md` §6.
