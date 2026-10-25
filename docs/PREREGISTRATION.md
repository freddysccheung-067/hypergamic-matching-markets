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
