# Pre-registration

Append only. Each entry is written and committed **before** the run it describes,
so that the prediction exists in history before the result does. Never edit an
entry after its run has produced data. Corrections are appended as new entries
that name the entry they supersede.

Predictions are stated as a decision rule with a number in it, so that "the
prediction held" is checkable rather than a matter of reading.

---

## E1 — Fresh seeds 100–119 (commitment C1)

**Written:** 2026-10-25, before the run. **Status:** committed before data existed.

**Question.** The existing 20-seed interval (`base20`, seeds 42–61) rests on seeds
that were also the ones used to choose the H4 optimum and to anchor every
exploratory table. Do the headline quantities reproduce on twenty seeds that took
no part in any earlier choice?

**Design.** Seeds 100–119, four arms (`ncf`, `oracle`, `random`, `popularity`),
100 epochs, `ncf.num_threads = 1`, `experiments/base.yaml` otherwise as shipped.
Output: `data/processed/rigor/fresh/`. One session, one thread count, both the
new set and the comparison set regenerated at that count.

**Prediction.** For every quantity and arm, the fresh 20-seed point estimate falls
inside the exploratory 20-seed 95% interval; and no ordering reverses (the `ncf`
arm exceeds `oracle` on every seed; `popularity` > `ncf` > `oracle` > `random` at
the final epoch holds).

**Decision rule, fixed now.** The interval the prediction is judged against is the
`t` construction at twenty seeds, `mean ± 2.093·sd/√20`, because that is the
construction with nominal coverage and it is the one `docs/STATS.md` §1 says to
read. The BCa interval is reported beside it but is not the pass/fail criterion;
the brief's wording ("the existing 20-seed interval") is ambiguous between the two
constructions and this entry resolves it before the data exists. **The prediction
is judged against the t interval.** Had it been judged against the BCa interval,
the lower bound would have been `+0.0204` rather than `+0.0199`, and that
distinction is recorded here rather than chosen after seeing which one the data
favours.

**Disclosure.** One seed of this set, seed 100, was run during sizing (one seed,
100 epochs, 1 thread) to measure the cost of the run; it took 1m40s and reported
a gap of `+0.0176`. That single seed is disclosed because it was observed before
this entry was written. It is inside neither the BCa nor the t interval. The
prediction above is about the twenty-seed estimate, not about any one seed, and it
is left as written: a single seed below the interval is expected when the
per-seed sd is about 0.008 and the interval half-width about 0.004. **If the
twenty-seed estimate also falls below `+0.0199`, the prediction failed and that is
the finding** — it would mean seeds 42–61 sit high relative to fresh seeds, which
is exactly the selection effect §2.2 of `docs/STATS.md` warns about.

**What would falsify it.** A fresh point estimate outside the t interval for any
headline quantity, or any reversal of the arm ordering on any seed.

---

## E2 — H4 re-estimated on fresh seeds (commitment C2)

**Written:** 2026-10-25, before the run. **Status:** committed before data existed.

**Question.** The institutional optimum at `(c = 0.01, size_fraction = 0.65)` was
chosen as the maximum of forty cells on the same five seeds (42–46) that report
it, and its per-cell seed sd is the ninth smallest of forty. Is the location of
the optimum a property of the design or of those five seeds?

**Design.** The full 10 × 4 institutional grid at `c = 0.01` re-estimated on
seeds 100–119 — i.e. the ten `size_fraction` levels at `c = 0.01` for twenty fresh
seeds — plus the two neighbours of the original optimum (`size_fraction` 0.50 and
0.80) called out explicitly. Same thread count as E1.

**Prediction.** (i) The fresh point estimate at the optimum cell falls by less
than one seed sd of the exploratory estimate (sd = 9.86, so by less than 9.86
welfare units). (ii) The optimum cell still ranks in the top five of the ten
`size_fraction` levels at `c = 0.01` on the fresh grid.

**What would falsify it.** A fall of more than one sd, or a rank outside the top
five. Either is the result and goes in the paper.

---

## E3 — The two interventions (commitment C3)

**Written:** 2026-10-25, before the run. **Status:** committed before data existed.

**Question.** The paper's weakest sentence is its mechanism: the gap shrinks
while coverage rises, and reach was never manipulated. Two interventions
manipulate it.

**Design.** Five seeds (42–46), 100 epochs, `ncf.num_threads = 1`, control and
treatment in the same session. Control is the shipped configuration.

- **(i) Frozen ranker.** Train normally to the peak (epoch 6), then stop training
  and keep the market running to epoch 100. New config key `ncf.freeze_after`,
  default `None` (no freeze), so the shipped behaviour is unchanged.
- **(ii) Warm start.** Pretrain the NCF on a synthetic swipe log until reach at
  epoch 0 exceeds 0.8, then start the loop. New config key
  `ncf.warm_start_epochs`, default `0` (no warm start).

**Predictions.**

- (i) The gap stays at its peak. Concretely: the epoch-99 gap under the frozen
  ranker is within one final-epoch seed sd of the epoch-6 gap, i.e. the frozen
  trajectory does not decay.
- (ii) The epoch-0 gap is much smaller than the control's epoch-0 gap
  (`+0.2633`), and most of the decay disappears, while the terminal gap at epoch
  99 is roughly unchanged (within one seed sd of the control's `+0.0241`).

**What would falsify them.** For (i): the frozen gap decays anyway, by more than
one seed sd from its peak to epoch 99. That would mean something other than
training closes the gap, and it is a finding in itself. For (ii): the terminal
gap moves by more than one seed sd, or the epoch-0 gap is not substantially
smaller.

**Reported as.** Paired per-seed differences against the control run in the same
session, never against the committed 24-thread directories.

---

## E4 — Long horizon at `shock_scale` 0.5 (commitment C4)

**Written:** 2026-10-25, before the run. **Status:** committed before data existed.

**Question.** At `shock_scale` 0.5 the filter never learns coverage (reach 0.4616
against 0.9566 shipped). Does the gap still converge there, where the mechanism
the paper offers for convergence is absent?

**Design.** 300 epochs, `shock_scale: 0.5`, five seeds (42–46), 1 thread, shipped
`admissible_fraction` otherwise.

**Prediction.** The gap is small throughout (below the 0.05 ceiling at every
epoch after the first few) and shows no comparable decay, because the mechanism
claimed to close it is absent there. Concretely: the ratio of the epoch-99 gap to
the epoch-299 gap is below 10, against the 55 measured at the shipped cell.

**What would falsify it.** A decay comparable to the shipped cell's (ratio ≥ 10).
That would mean the mechanism story is wrong for a second reason.

---

## E5 — Equivalence test on the tail (commitment C5)

**Written:** 2026-10-25, before the run. **Status:** committed before data existed.

**Question.** "Below 0.001 and indistinguishable from flat" is not a statement
with a defined meaning. What margin, and what does the tail support?

**Design.** On the existing 300-epoch series (no new simulation): declare an
equivalence margin of **0.002 Gini**, run two one-sided tests (TOST) against
±0.002 on the tail mean over epochs 250–299, and report. Also compute the first
epoch at which the gap stays below 0.002 for the rest of the run, with a range
over seeds.

**Prediction.** The tail is equivalent to zero at the 0.002 margin: both one-sided
tests reject at α = 0.05. The time-to-threshold (first epoch from which the gap
stays below 0.002) is between 120 and 200 epochs on every seed.

**What would falsify it.** Failing TOST at 0.002, or a time-to-threshold outside
120–200 on any seed. Either is reported as the result.

---

## E3a — calibration of the warm start (appended before the C3 run)

**Written:** 2026-10-25, before `scripts/run_interventions.py` was first run.

E3(ii) says "pretrain until reach at epoch 0 exceeds 0.8" without saying how many
passes that is. The number was calibrated by running epoch 0 only, at the shipped
`N = 1000`, one seed, at 1 thread: 0 passes gives reach 0.041, 10 gives 0.082,
50 gives 0.821. **`--warm-passes 50` is the setting**, chosen because it is the
first count tried that clears the threshold, and it is recorded here rather than
left in a shell history.

Disclosed because it is a choice made with data in hand: the threshold came from
E3, the number of passes was picked to hit it. What that risks is the same
criticism as any tuning — but the intervention is a manipulation of the
initial condition, not an estimate of a parameter, and the prediction it is used
to test (the epoch-0 gap is much smaller) is unaffected by whether 50 or 60
passes were used. A reader who wants the arm run at a different pass count can
pass `--warm-passes`; the pass count is a column in the output.

---

## E3b — the frozen-ranker result was degenerate, and what to do about it

**Written:** 2026-10-25, after the C3 run and **before** the follow-up it
describes. Appended because E3(i) did not test what it was meant to test.

**What happened.** E3(i) froze the ranker after epoch 6, the peak of the *gap*.
Reach at epoch 6 is 0.041: the model has barely trained, so every agent's
shortlist is nearly the same before and after the freeze, and the frozen arm's
NCF Gini is constant to the last digit from epoch 6 to epoch 99
(0.959983 -> 0.959983 on seed 42, and identically on the other four). The
prediction "the gap stays at its peak" is therefore satisfied **trivially**: a
ranker that cannot rank cannot change what it ranks, so the arm says nothing
about whether continued training is what closes the gap. Reporting it as
confirmation would be reporting an identity as a mechanism.

**The informative version.** Freeze at epochs where the ranker can already
rank, and where the gap is still falling. Two arms, same five seeds, 1 thread,
100 epochs:

- **frozen20** — `freeze_after = 20`. Reach is expected to be well above 0.5 by
  then (a trajectory property to be measured, not asserted), so the ranker's
  scores are informative and the market keeps both shortlists and matching.
- **frozen40** — `freeze_after = 40`, a later point, where the control gap is
  roughly half its peak.

**Prediction.** If continued training is what closes the gap, then in each arm the
gap at epoch 99 stays within one final-epoch seed sd of its own value at the
freeze epoch, and the paired (frozen − control) difference at epoch 99 is
positive and large. Concretely: for `frozen40` the difference is at least half
the control's remaining decay from epoch 40 to epoch 99.

**What would falsify it.** A frozen arm whose gap continues to fall by more than
one seed sd after the freeze. That would mean the decay is driven by something
that runs whether or not the ranker trains — θ relaxation, turnover, or the
matching itself — and the paper's mechanism story is wrong.

**What C3's warm arm already establishes, and does not.** Reach at epoch 0 rises
from 0.041 (control) to 0.861 (warm). The epoch-0 gap falls from +0.3055 to
+0.1079, a 65% reduction — so starting informed does shrink the initial gap, as
predicted. But the terminal gap also falls, from +0.0224 to +0.0132 (paired
difference -0.00924, t(4) = -4.78), which E3(ii) predicted would be *roughly
unchanged*. And 67% of the decay survives (warm decay -0.1889 against control
-0.2831). So E3(ii) is **half confirmed and half refuted**: the start is
manipulable, the terminal level is not invariant, and most of the decay is not
explained by starting from ignorance.

---

## E2a — the C2 run was at the wrong gate, and the rerun's prediction

**Written:** 2026-10-25, after the first C2 run and before the corrected one.

**What happened.** C2 was run with `experiments/base.yaml` as shipped
(`admissible_fraction: 0.02`). The committed institutional grid it was meant to be
compared against was generated in October at `admissible_fraction: 0.2`, and
`data/processed/institutional/README.md` says so. The two are different regimes,
so the first C2 run compares a fresh twenty seeds at one gate against five
seeds at another, and its numbers cannot speak to the H4 claim at all. That run
also produced a second, more alarming fact on the way: run at 0.02, seeds 42--46
do not reproduce the committed `+64.62` at any cell — they give `-17.23`. The
reproduction at 0.2 is exact to the last digit on `welfare_common_a`,
`welfare_diff_residual` and `gini_a` across all 200 rows, which identifies the
gate as the whole of the difference.

**What that means for H4.** The stated optimum, `+64.62` at
`(c=0.01, size_fraction=0.65)`, is a measurement of the *inert* institutional
regime: the one the paper's own §4 calls deaf to the market's outcomes. The paper
currently presents it as a property of the model. That is a larger defect than
the seed question C2 asked.

**The corrected design.** Ten `size_fraction` levels at `c = 0.01`, seeds 100--119,
at `admissible_fraction: 0.2`, which is the regime the original was measured in.
This is the like-for-like version of C2.

**Prediction, unchanged from E2 and still made before this run:** the fresh point
estimate at `size_fraction = 0.65` falls by less than one exploratory seed sd
(9.86 welfare units) and the cell still ranks in the top five of the ten levels.

**Added prediction, made now and before the data:** the cell will rank first or
near-first in the inert regime, because it did so on the original five seeds and
the regime — not the seed set — is what the first C2 run changed.

---

## E5a — result of the equivalence test, appended after the run

**Written:** 2026-10-25, after E5 was computed on the existing series and before
anything was written into the paper.

**TOST at the declared margin of 0.002.** The replicate is the seed (n = 5), not
the epoch: the fifty tail values are one trajectory, so their spread is not the
sampling scale. Seed sd = 0.000291, SE = 0.000130, df = 4, one-sided
$t_{0.05} = 2.132$.

- $t(\mu > -0.002) = +18.26$ — rejects.
- $t(\mu < +0.002) = +12.50$ — rejects.
- **TOST passes.** The 90% interval over seeds is $[+0.000097, +0.000651]$,
  wholly inside $\pm 0.002$.

So the tail *is* equivalent to zero at a 0.002 margin, in the sense of a
pre-declared equivalence test, which is a stronger statement than "the interval
includes zero". This is what E5 predicted and it holds.

**Time-to-threshold: the prediction failed on part of its range.** E5 predicted the
first epoch from which the gap stays below 0.002 lies between 120 and 200 on every
seed. Measured: 268, 237, 210, 198, 225 for seeds 42–46 — **range 198 to 268, with
four of five seeds above 200.** The point prediction was 120–200 and the data sit
one to three hundred epochs later; only seed 45 lands at the boundary.

What that means: the tail is equivalent to zero, and the trajectory takes about
200–270 epochs to get there, not the 120–200 estimated. The paper accordingly says
"below 0.002 from about epoch 200 to 270 depending on seed", not "from about epoch
200". E5's window was a guess about where the crossing would fall; the measurement
is the finding, and it is reported as a miss.
