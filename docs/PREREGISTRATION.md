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
favors.

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
  default `None` (no freeze), so the shipped behavior is unchanged.
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

---

## E3c — the E3b prediction was met for a structural reason, and E4 failed

**Written:** 2026-10-25, after the E3b and C4 runs, before either is written into
the paper.

### E3b: the frozen arms are constant by construction

E3b predicted that a frozen ranker's gap would stay within one seed sd of its
value at the freeze epoch. It did, exactly — the gap is constant to every decimal
at both freeze points (frozen20: 0.939596 Gini at epochs 21 through 99,
identically; frozen40: 0.883149 from epoch 41 on). **The sd across seeds of the
within-arm change is exactly zero.**

That is not a mechanism finding. Freezing the weights freezes the score matrix,
which freezes the top-$k$ shortlists, which freezes exposure, which fixes the Gini
apart from the oracle arm's own (already stationary) value. **Any freeze point
produces a flat arm by construction**, so a within-arm test cannot distinguish
"training drives the decay" from "nothing in this model moves without training".
E3b was pre-registered as the informative version of E3(i) and it is not: the
degeneracy is structural, not a matter of the freeze landing too early.

What the two freeze points *do* show, cross-sectionally, is the thing worth
reporting: at the freeze epoch the gap was +0.27534 (epoch 20, reach 0.182) and
+0.21890 (epoch 40, reach 0.607). A ranker frozen at higher reach sits at a lower
gap, in the right order and at the right magnitude relative to the control's own
trajectory. That is consistent with the coverage story and is not a test of it.

**Conclusion, stated plainly: this session did not succeed in manipulating
training independently of reach.** The honest summary for the paper is that the
frozen-ranker intervention is uninformative in this model, and that the mechanism
claim remains conditional and untested rather than tested and supported.

### E4: the gap converges at `shock_scale` 0.5 too — prediction failed

E4 predicted no comparable decay at `shock_scale` 0.5, on the reasoning that
reach never climbs there. Measured, 300 epochs, five seeds, 1 thread:

| epoch | 0 | 6 | 19 | 49 | 99 | 199 | 299 |
|---|---|---|---|---|---|---|---|
| gap | +0.0246 | +0.0578 | +0.0520 | +0.0350 | +0.0111 | −0.0009 | −0.0012 |
| reach | 0.041 | — | 0.073 | 0.269 | 0.457 | 0.488 | 0.490 |

**Reach does climb here — from 0.041 to 0.490 over 300 epochs.** The 0.4616 in
`sensitivity.csv` is reach at epoch 99; it is not a ceiling. So the premise of E4
("the cell where reach never climbs") is false once the horizon is extended, and
the cell is not the mechanism-free control the paper treats it as.

The prediction as stated (ratio of the epoch-99 gap to the epoch-299 gap below 10)
fails: the epoch-299 gap is negative, so the ratio is undefined, and the gap decayed
from a peak of +0.0565 (epoch 9) to −0.0012 — proportionally as complete a decay
as the shipped cell's. **The mechanism story is not rescued by this cell and not
contradicted by it; the cell turns out not to test it either way.**

This has a consequence for the paper: `\S`~sensitivity's "the explanation is
conditional, and here is the cell where it fails" must be rewritten. The condition
that fails is reach at *100 epochs*, not reach at all.

---

## E2b — result of the corrected C2, appended after the run

**Written:** 2026-10-25, after seeds 100--119 were run at the inert gate and
before any of it was written into the paper.

**Design as pre-registered in E2a:** ten `size_fraction` levels at `c = 0.01`,
seeds 100--119, `admissible_fraction: 0.2`, the regime the committed grid was
measured in.

| rank | size_fraction | mean residual welfare diff | sd (20 seeds) |
|---|---|---|---|
| 1 | **0.65** | **+54.45** | 46.42 |
| 2 | 0.80 | +39.64 | 40.91 |
| 3 | 0.50 | +37.43 | 45.92 |
| 4 | 0.35 | +14.27 | 26.30 |
| 5 | 1.00 | +2.90 | 28.14 |
| 6--10 | 0.25, 0.02, 0.05, 0.10, 0.15 | +1.73 to −14.69 | |

**Prediction (i), the point estimate falls by less than one seed sd: FAILED, just.**
The fresh estimate is +54.45 against the original +64.62, a fall of 10.17 against
the exploratory seed sd of 9.86. The prediction is narrowly missed. It is worth
saying what "narrowly" means: 10.17 against 9.86 is a 3% margin, well inside the
uncertainty on either number, and the sd that the criterion uses (9.86) is itself
now known to be a five-seed underestimate of the population's — the fresh sd at
the same cell is 46.42, 4.7 times larger. A criterion built on the wrong scale was
missed by an amount that scale cannot resolve.

**Prediction (ii), still top five of ten: HELD.** The cell ranks first, which is
where it ranked in the original grid, and the shape is preserved: positive at
0.35--1.00 with a maximum at 0.65 and negative below 0.25.

**Added prediction from E2a ("will rank first or near-first"): HELD.**

**What this does not fix.** The cell still survives at five seeds and at twenty,
and its fresh twenty-seed interval [+32.72, +76.17] still excludes zero
(t = 5.25, p = 4.6e-5). But the selection problem E2 existed to address is
*worse* than the original analysis suggested rather than better: the original
five-seed sd (9.86) was 4.7 times too small, which is the same pathology
`docs/STATS.md` §2.2 found for `algorithmic_loss` and `mcr`. The optimum was
selected partly on a low-variance accident, and the fresh sd confirms it.

**Conclusion for the paper.** H4's location replicates; its magnitude does not,
and its precision was overstated by a factor of nearly five. Combined with E2a's
gate finding, the H4 result is: an interior optimum at `size_fraction` 0.65 that
replicates on fresh seeds, in a regime where the market does not learn, with a
magnitude of roughly +54 rather than +65 and a much wider uncertainty than the
published interval carries.

---

## E6 — H4 at the shipped gate. **NOT PRE-REGISTERED. Recorded after the fact.**

**Written:** 2026-10-25, after the run. This entry breaks the file's own rule and
says so at the top, because the alternative — presenting a post-hoc run as though
it had been predicted — is worse than the breach.

**Why it was run without a prediction.** E2a established that the whole
institutional grid was measured in the inert regime, and the paper was edited to
say that whether the optimum survives at the shipped gate "needs the full grid
re-run and is left open". Leaving a two-hour question open while eight hours of
budget remained was not defensible, so the grid was re-run. **No prediction was
committed first.** A reader should weight this result accordingly: it is a
measurement, and it is the first measurement of this quantity, so there was no
prior to be biased toward; but it did not pass through the discipline the rest of
this file imposes.

**Design.** The full 10 × 4 grid, seeds 42–46, `experiments/base.yaml` as shipped
(`admissible_fraction: 0.02`), output to
`data/processed/rigor/h4_shipped_gate/`. Identical to the committed institutional
run in every respect except the gate.

**Result: H4 does not survive at the shipped gate.** No cell in any of the forty
beats Environment A. The best cell at each cost level, and the whole c = 0.01
slice:

| rank | size_fraction | mean residual welfare diff | sd |
|---|---|---|---|
| 1 | 0.02 | −0.23 | 0.50 |
| 2 | 0.05 | −1.45 | 2.47 |
| 3 | 0.10 | −4.89 | 3.90 |
| 4 | 0.15 | −9.64 | 5.07 |
| 5 | 0.50 | −10.56 | 12.55 |
| 6 | 0.25 | −11.39 | 6.39 |
| 7 | **0.65** | **−17.23** | 13.32 |
| 8 | 0.35 | −19.47 | 14.75 |
| 9 | 0.80 | −22.66 | 32.52 |
| 10 | 1.00 | −40.22 | 27.21 |

At c = 0.05, 0.10 and 0.25 the best cell is also negative (−0.25, −0.28, −0.37).
The ordering at c = 0.01 is close to monotone in size: welfare falls as the
subgraph grows, and the cell that was the optimum under the inert gate sits
seventh of ten.

**What this means.** The interior optimum is a property of a market whose
threshold gate never binds — the configuration `design.tex` describes as one where
the market "never responds to its own outcomes". At the shipped gate the
institutional comparison has no interior optimum at all: B loses everywhere, and
increasingly so with size. H4, as stated, is refuted at the configuration the
paper actually runs.

**Not reconciled.** The paper does not now search for a cost level or a size at
which B wins; that would be the move this file exists to prevent. The result is
reported as a refutation in a regime, and the inert-regime result is reported
beside it as a fact about that regime.

---

## E7 — regenerating the 300-epoch trajectory at 1 thread

**Written:** 2026-10-25, after the run was started and **before its output
exists.** The run is `data/processed/rigor/long_horizon_1t/`; the config is
`experiments/long_horizon_1t.yaml`. This entry fixes the decision rule in advance
because the temptation, once numbers are on screen, is to prefer whichever set
agrees with the paper.

**Why.** `docs/REVIEW_RESPONSE.md` A3 records that the committed
`rigor/long_horizon/` NCF arm does not reproduce from this code at any thread
count tested (1, 2, 4, 8, 16, 24), nor from the pre-`af24bc5` source at 24. Its
oracle, random and popularity arms reproduce exactly. The paper's central claim
rests on that directory's trajectory.

**Design.** Shipped configuration, `simulation.n_epochs: 300`,
`ncf.num_threads: 1`, seeds 42–46, four arms, written to
`data/processed/rigor/long_horizon_1t/`. Same seeds, same horizon, one thread
count, generated today.

**Decision rule, fixed now.**

1. If the new tail mean (epochs 250–299) is within **0.001** of the committed
   `+0.000374` in absolute value, and the epoch-99 gap is within **one seed sd**
   of the committed value, the two agree at the resolution the paper claims, and
   the paper switches its quoted trajectory to the reproducible directory while
   saying that the committed one differs.
2. Otherwise the two disagree materially. The paper then reports the new
   trajectory as the measurement, states that the committed one cannot be
   reproduced and disagrees, and does **not** average them or choose the one that
   supports a conclusion.
3. In both cases the old directory stays in the repository, unedited, with its
   README's A3 note intact. Nothing is deleted to make the record tidy.

**What this cannot fix.** Whatever the outcome, the committed long-horizon numbers
quoted in `docs/STATS.md` §3 and in the paper's earlier drafts came from a run this
repository cannot regenerate. Replacing them changes which numbers the paper
claims; it does not change the fact that A3 happened, and A3 is recorded
separately and stays recorded.

---

## E7a — result: branch 1

**Written:** 2026-10-25, after the run, applying E7's rule mechanically via
`scripts/_compare_long_horizon.py`.

| | committed `long_horizon` | regenerated `long_horizon_1t` |
|---|---|---|
| peak | +0.30554 at epoch 6 | +0.30554 at epoch 6 |
| epoch 99 | +0.02414, sd 0.00544 | +0.02240, sd 0.00831 |
| final | +0.00044, sd 0.00090 | +0.00010, sd 0.00042 |
| tail 250–299 | +0.000374, sd 0.000291 | +0.000231, sd 0.000340 |

**Rule applied.** Tail difference 0.000144, well inside the 0.001 threshold. Epoch-99
difference 0.001748, inside one committed seed sd (0.005439). **Branch 1: the two
agree at the resolution the paper claims.**

**What the agreement does and does not mean.** The paper's *qualitative* claims —
a peak at epoch 6, a decay to a tail equivalent to zero, a hundred-epoch figure
that describes a transient — hold on both series. Its *quoted figures* differ in
the third decimal: epoch 99 is +0.0241 committed against +0.0224 regenerated, and
the tail is +0.000374 against +0.000231.

Two things follow. **The paper switches its quoted trajectory to
`long_horizon_1t`**, because that is the directory a reader can regenerate, and
says in the text that the committed one differs at that magnitude. And **A3 stands
unchanged**: the fact that the committed directory cannot be regenerated is still a
fact, and switching which numbers are quoted does not repair it.

One difference is worth noting because it cuts against the paper's rhetoric: the
regenerated epoch-99 gap is *smaller* (+0.0224 against +0.0241), which weakens
rather than strengthens the "the hundred-epoch number is far from where it settles"
contrast. The paper reports the regenerated value, not the larger one.

## E7b — what changes when the regenerated series is quoted

**Written:** 2026-10-25, after E7a, before the paper is edited.

E7a said the two series agree at the tolerance fixed in advance. They do, and
switching which one the paper quotes still changes three sentences:

| quantity | committed | regenerated | consequence |
|---|---|---|---|
| epoch-99 gap | +0.02414 | **+0.02240** | the "fifty-five times the tail" contrast shrinks to about 97× on the ratio of means, and the ratio was never a measured quantity anyway |
| tail mean | +0.000374 [+0.000013, +0.000735] | **+0.000231 [−0.000192, +0.000653]** | the interval **includes zero**; 3 of 5 seeds positive, p = 0.204 |
| first epoch with the mean below 0.002 | 249 | **211** | the transient ends sooner on this series |

**The tail interval changing from excluding to including zero is the substantive
one.** The paper says the tail is "small and positive but barely distinguishable
from zero" and leans on the t test excluding zero. On the regenerated series it
does not: the same test gives p = 0.204. The honest statement becomes that the
tail is indistinguishable from zero on this series, which is *closer* to what the
paper's own caution ("a flat tail is also what a very slow decay looks like")
already said.

**The TOST still passes** at the pre-declared 0.002 margin (t = 14.66 and 11.63
against 2.132), so the equivalence claim is unaffected and is now the only tail
claim the data support. The paper drops the "all five seeds positive, t test
excludes zero" sentence and quotes the equivalence test alone.

**Which series the rest of the paper now uses.** The gap-over-time claims, the
trajectory table and the tail statistics switch to `long_horizon_1t`.
`intervals.csv`'s `long300` row is **not** regenerated: it is a bootstrap product
over the committed directory, and rebuilding the whole interval pipeline to move
one row is a larger change than this session should make unannounced. The ledger
therefore carries the committed row and says which directory it is from, and the
text quotes the regenerated series beside it.

## E8 — which gap number the paper leads with, and why it does not change

**Written:** 2026-10-25, at the end of the revision, before the choice is made in
the text.

The headline quantity is $\gapShipped = +0.0251$, five seeds, from
`intervals.csv` row `live100`. Three other numbers measure the same thing:

| source | value | seeds | thread | reproducible? |
|---|---|---|---|---|
| `intervals.csv` `live100` | +0.0251 [+0.0155, +0.0347] | 5 (42–46) | 24 | yes, from `ablation_h100/` |
| `intervals.csv` `base20` | +0.0237 [+0.0199, +0.0274] | 20 (42–61) | 1 | no — `base/` does not regenerate |
| `rigor/fresh/` (this session) | +0.0253 [+0.0217, +0.0290] | 20 (100–119) | 1 | yes |
| `rigor/exploratory_1t/` (this session) | +0.0225 [+0.0187, +0.0264] | 20 (42–61) | 1 | yes |

**The choice, made explicit.** The paper leads with +0.0251 because it is the
number the committed `ablation_h100/` directory supports and that directory
reproduces exactly at 24 threads — it is the only *archival* value whose
provenance is clean. The two twenty-seed numbers are +0.0253 and +0.0225, which
bracket it.

**What is not done.** The headline is not switched to the fresh value, even though
+0.0253 is from the pre-registered confirmation set and is arguably the better
number. Switching it would change every interval in the ledger, every comparison
in §3.2's seed-count discussion, and the H3 verdict's quoted bounds — a
twenty-place edit at the end of a session, with the check being re-reading every
one of them. The paper instead quotes all four and says which is which
(`\S`~h3, the fresh-seed paragraph). If a future session switches the headline, it
should switch `intervals.csv` first, regenerate, and let the macros follow.

**The one thing that is recorded as a limitation rather than a choice:** `base20`'s
row is still quoted in §3.2 for the five-versus-twenty width comparison, and that
row comes from a directory that no longer regenerates. It is quoted because it is
the only twenty-seed run at 24 threads, and the width comparison is about seed
count rather than about the value, but a reader deserves to know the provenance.

---

## E8a — supersedes E8: the headline follows the brief's seed rule

**Written:** 2026-10-25, after E8, before the text is changed.

E8 kept the five-seed `live100` value (+0.0251) as the paper's headline on the
ground that its directory reproduces exactly. That conflicts with an instruction
the revision was given and E8 did not weigh: **100-epoch claims quote the
twenty-seed interval and say "twenty seeds"; horizon and sweep claims keep five
and say so.** E8 chose provenance over that rule without saying it was overriding
a rule. This entry corrects that.

**The change.** Every 100-epoch headline — abstract, introduction, conclusion, and
H3's opening — now quotes `base20`: **+0.0237 [+0.0199, +0.0274], twenty seeds
(42–61), positive on all twenty** (checked from `rigor/base/` directly: 20 of 20
final-epoch gaps positive). The five-seed +0.0251 stays in the ledger row it has
always occupied, labelled five seeds, and in H3 as the comparison the width
statement needs.

**The provenance caveat travels with it rather than deciding it.** `rigor/base/`
does not regenerate bit-for-bit (seed 42: 0.6973985 committed against 0.6976498
today, same thread count), and the paper says so in H3's four-measurements
paragraph. That caveat is a fourth-decimal one; the brief's rule is about not
letting a five-seed number stand where a twenty-seed one exists, which is a
first-order question about precision. The rule wins.

**What does not change.** The fresh confirmation set (+0.0253) is still not the
headline, for E8's reason: it would replace a committed artefact's value with one
generated in the session that is also testing it. It is reported beside the
headline as the replication.
