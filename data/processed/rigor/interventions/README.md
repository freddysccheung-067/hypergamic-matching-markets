# rigor/interventions — C3, frozen ranker and warm start

Generated 2026-10-25 on branch `rev/revision`, at `ncf.num_threads: 1`.

    python scripts/run_interventions.py --seeds 42,43,44,45,46 \
        --epochs 100 --warm-passes 50 --jobs 5 \
        --output-dir data/processed/rigor/interventions

## Why these exist

The paper's weakest sentence was its mechanism: the exposure gap shrinks while
coverage climbs, and reach had never been manipulated. These two arms manipulate
it. Predictions registered in `docs/PREREGISTRATION.md` E3 and E3a before the run;
the result is discussed in E3c.

## Conditions, all on one population and seed

| condition | what it does |
|---|---|
| `control` | the shipped loop, 100 epochs |
| `frozen` | trains to epoch 6 (the gap's peak), then stops training and keeps the market running |
| `warm` | pretrains on the epoch-0 swipe log for 50 passes, then runs the shipped loop |

The warm-start pass count was calibrated by running epoch 0 only: 0 passes gives
reach 0.041, 10 gives 0.082, 50 gives 0.821. Disclosed in E3a.

## Result, and why the frozen arm is uninformative

The frozen arm's NCF Gini is **constant to the last digit** from the freeze epoch
to epoch 99 (seed 42: 0.959983 → 0.959983; identically on the other four). That is
not a finding about training: freezing the weights freezes the score matrix, which
freezes the top-$k$ shortlists, which fixes exposure. Any freeze point gives a flat
arm. The epoch-6 freeze was chosen because that is the gap's peak, but reach there
is 0.041 — the ranker can barely rank — so the arm is doubly uninformative.
`../latefreeze/` re-runs the frost at epochs 20 and 40 and gets the same exact
constancy, which is what identifies the cause as structural.

**The warm arm produced a real result, and it is half against the prediction.**
Reach at epoch 0 rises from 0.041 (control) to 0.861. The epoch-0 gap falls from
+0.3055 to +0.1079, as predicted. But the terminal gap also falls, from +0.0224 to
+0.0132 (paired difference −0.00924, t(4) = −4.78, sd 0.00432), where E3(ii)
predicted it would be roughly unchanged. And **67% of the decay survives**
(warm decay −0.1889 against control −0.2831). Starting informed does not explain
most of the decay.

Gaps above are NCF Gini minus the oracle Gini from `../exploratory_1t/` at the
same seed and thread count, never against a committed 24-thread directory.

## Thread count

**1.** Every condition pinned inside the script, so the arms are comparable to each
other and to `../fresh/`.

## Files

`interventions_seed<seed>.csv`, one row per (condition, epoch), columns
`condition, seed, epoch, gini, reach, mcr, rsi, top1_share, welfare_alg,
noise_ratio`.
