# Response to review

`docs/REVIEW.md` is the adversarial review written against `paper/draft`, on branch
`review/redteam`, recommending rejection. This file answers it item by item.

The review's own framing is that the manuscript "states its conclusions more strongly
than its own evidence". Part of that is fair and has been fixed. Part of it rests on
three misreadings, which are set out here with line references so that a reader can
check them rather than take this document's word.

Two conventions below. **Fixed** means the manuscript or the code changed and the
change is in this branch. **Declined** means the finding was checked and the text is
right; the reason is given. Nothing is declined without a citation into the
manuscript.

---

## Summary

| # | Finding | Verdict |
|---|---|---|
| 1.1 | "Converges to zero" is not what was measured | **Fixed** — the word is gone from abstract, intro, conclusion |
| 1.2 | The ledger prints the widest interval without naming it | **Fixed** — both constructions now printed |
| 1.3 | "Monotone decay" includes points inside one seed sd | **Fixed** — the section now says so explicitly |
| 1.4 | Overlapping-sample agreement called a replication | **Fixed** |
| 1.5 | The mechanism is contradicted by its own counterexample | **Superseded** — the cell is not a counterexample (reach reaches 0.49 by epoch 299), but the mechanism is now reported as untested, which is what the review asked for |
| 1.6 | No stated rule for BCa over $t$ | **Fixed** — rule stated, both columns printed |
| 1.7 | The optimum was selected partly on low variance | **Fixed, and worse than claimed** — fresh seeds 100–119 replicate the location but the seed sd is 46.42 against 9.86; and the optimum vanishes at the shipped gate |
| 2.1 | Five seeds found biased; correction applied only where noticed | **Fixed as disclosure** — see below |
| 3.1 | The abstract overstates; "fifty-five times" is a ratio of imprecise quantities | **Fixed** |

---

## 1.1 "Converges to zero" — fixed

The review is right, and this was the paper's worst sentence.

What was measured is a flat tail: $+0.00037$ over epochs 250--299, with a $t$
interval of $[+0.000013, +0.000735]$ and Wilcoxon $p = 0.063$. A limit is not
observed; a tail is. And a flat tail is also what a very slow decay looks like.

The abstract, the introduction and the conclusion now say the gap is below the
pre-declared $0.002$ margin from about epoch 211 and that the tail is no longer
distinguishable from flat, and `paper/sections/results.tex` carries a paragraph
naming the distinction between a bounded residual and a convergence rate. The
phrase "fifty-five times its terminal value" is gone: a ratio of two quantities
measured to roughly $\pm30\%$ is an order of magnitude, not a constant.

**Numbers updated in the revision of 2026-10-25.** The committed 300-epoch
directory cannot be regenerated from this code at any thread count
(§A3 below), so the paper now quotes `data/processed/rigor/long_horizon_1t/`, which
was regenerated in one session at 1 thread and compared against a rule fixed in
advance (`docs/PREREGISTRATION.md` E7, E7a). On that series the tail mean is
$+0.00023$ with a $t$ interval of $[-0.00019, +0.00065]$ that **includes zero**
(three of five seeds positive, $p = 0.204$), and the epoch-99 gap is $+0.0224$.
The equivalence test at a 0.002 margin still passes ($t = 14.66$ and $11.63$ on
four degrees of freedom), and it is now the only tail claim the paper makes. This
is stricter than what this section originally asserted: the paper no longer claims
a positive residual at all.

The review's alternative sentence — "below 0.001 by epoch 250 and no longer
distinguishable from flat" — is what the manuscript now uses.

## 1.2 The ledger interval — fixed

Correct on both counts. The H4 row printed the Bonferroni-$t$ interval
$[+28.80, +100.43]$ and never said in the table which construction it was, so a
reader comparing it to `multiple_comparisons.csv`'s Bonferroni-BCa
$[+55.07, +78.46]$ would think the effect was imprecisely estimated when it is not.

`paper/tables.py` now emits both columns for every row, so the choice is visible
rather than implied, and the caption states which rows carry a multiplicity
correction and over what family. The "two cells wide" verdict was ambiguous and now
reads "best of 40 cells", with the caption saying explicitly that this is a count of
surviving cells and not a width in `size_fraction`.

## 1.3 Monotone decay — fixed

Partly fair, and the fix was easy once stated. The reported points after epoch 124
are all inside one final-epoch seed sd ($0.0111$) of one another, so ordering them
by epoch number is not evidence of anything.

The section now says the decay is established to about epoch 124 and that the later
entries are ordered noise, and it no longer uses monotonicity as evidence for
convergence.

## 1.4 Overlapping-sample agreement — fixed

The review is right that seeds 42--46 are the first five of 42--61, so the two
figures agreeing is expected and is not a test. The text no longer says the
twenty-seed run "rules out a five-seed accident". It reports what the twenty-seed
run actually establishes — the interval's width, and that the point estimate moves
by less than its own interval when fifteen seeds are added — and points the reader
at the sensitivity sweep for the robustness claim.

## 1.5 The counterexample — superseded: both sides were wrong

This was the review's headline finding. It was declined in an earlier round on the
ground that the cell does not isolate coverage learning, because the oracle arm
concentrates there too. **That defence rested on a premise the revision has measured
and found false, so the disposition changes.**

The earlier defence: reach is 0.4616 at `shock_scale` 0.5 against 0.9566 shipped, so
the filter "never learns coverage" there, and the gap is small because *both* arms
concentrate — the oracle's own Gini rises from 0.6502 to 0.9019 when the population
loses heterogeneity — so the cell is not a counterexample.

**What the revision established:** the 0.4616 is reach at **epoch 99**. The same
cell run to 300 epochs reaches **0.490** (`data/processed/rigor/c4_shock0.5/`, five
seeds, 1 thread). So "reach never climbs here" is false at a longer horizon, and the
cell is a slower instance of the same process rather than an exception to it.

**Both the review and the earlier response are therefore wrong about this cell, in
different directions.** The review read it as a counterexample and it is not one,
because reach does climb. The earlier response read it as a cell outside the
mechanism's scope, and it is not that either, for the same reason. What it is: a
place where the same process runs at half speed, which neither confirms nor refutes
the mechanism.

**The mechanism is now reported as untested, which is what the review's second
paragraph asked for.** The earlier response conceded that reach is never manipulated
and said no intervention had been run. Two have now been run
(`docs/PREREGISTRATION.md` E3). The frozen ranker is **uninformative**: freezing the
weights freezes the shortlists, so the arm is flat by construction and the within-arm
test cannot separate training from drift (E3b, E3c). The warm start **meets half its
prediction and refutes the other half**: the epoch-0 gap falls from +0.3055 to
+0.1079 as predicted, the terminal gap also falls (paired −0.00924, t(4) = −4.78)
where it was predicted unchanged, and two thirds of the decay survives. The paper
reports the mechanism as a hypothesis this study does not establish.

So the disposition is: **the review is right about the conclusion and wrong about
the reason.** The mechanism claim should not be made, and it is not made — but not
because the `shock_scale` 0.5 cell contradicts it. Nothing in this grid contradicts
it. It has not been tested in a way that could be.


## 1.6 No stated rule for BCa over t — fixed

Fair. The rule is now stated once, in the paragraph above the ledger
(`paper/sections/results.tex:20--33`), applied in every table by printing both
columns, and the sentence claiming the paper applied it inconsistently no longer
has a target: there is no longer a place where one construction is chosen without
the other being shown.

## 1.7 Selection on low variance — partly fixed

Fair, and not fully repairable. The optimum cell at (c = 0.01, size\_fraction =
0.65) is the maximum of 40 cells by mean, and it is also the ninth smallest by sd.
The review is right that Bonferroni widens for multiplicity without un-selecting.

What the revision does: states this in `\S\ref{sec:h4}` rather than leaving it to be
inferred, reports the range of the per-cell seed sd
(`intervals.csv`, `welfare_diff_residual_within_cell_sd`), and says plainly that the
clean answer is re-estimation on fresh seeds, which this study does not have.

What it did not do at the time: re-estimate the optimum on disjoint seeds.

**Revision of 2026-10-25: done, and the review was more right than this section
allowed.** The ten size levels at c = 0.01 were re-run on seeds 100–119, which took
no part in choosing the cell, after a prediction was committed
(`docs/PREREGISTRATION.md` E2, corrected to the original gate in E2a). The location
replicates — the cell ranks first of ten, +54.45 [+32.72, +76.17] — but its seed sd
is **46.42 against the published 9.86**, so the published interval was built on a
standard deviation about 4.7 times too small, exactly the selection effect the
review described. And a second finding the review did not raise is larger: the
whole grid was measured at the inert gate (`admissible_fraction` 0.2), and at the
shipped gate no cell of the forty beats A (`data/processed/rigor/h4_shipped_gate/`,
E6). H4 is now reported as an inert-regime result, refuted where the paper runs.

## 2.1 Seed-block bias — fixed as disclosure, and the review is right that it is uncomfortable

The review's sharpening is fair and the paper now carries it. `docs/STATS.md` §2.2
shows seeds 42--46 are the minimum-variance contiguous block of sixteen for
algorithmic loss (sd ratio 0.31) and second-smallest for clearing rate (0.41). The
review's point is that the correction was applied to the interval arithmetic
everywhere but to the seed choice nowhere, and that "the ratio came out near 1.0 for
the other five" is not evidence they are fine.

That is correct as a statement about inference. `paper/sections/limits.tex` now says
so directly: the seeds were inherited, not chosen; the two affected quantities carry
no headline; a future study should choose seeds for the purpose. It does not claim
the other five are unaffected.

The review suggests re-running on disjoint seeds. `data/processed/rigor/base/`
already covers seeds 42--61 and the twenty-seed intervals are reported beside the
five-seed ones, so a reader can see the width change. A fully disjoint design is a
new study.

## 3.1 The abstract — fixed

"The gap decays to below 0.001 ... so the hundred-epoch figure is fifty-five times
its terminal value" presupposed the convergence and divided two imprecise numbers.
Rewritten, as in 1.1.

---

## What the review got right that is not in the table above

Three findings are accepted without a fix, because they are limitations rather than
defects and the manuscript already owns them: no calibration, one population size,
and a single $k$. The review's §2 treatment of these is accurate and is the reason
its overall verdict is harsh rather than wrong.

## A defect the review did not find

Checking the reproducibility claim turned up something the review missed and the
manuscript got wrong: **the simulation was not thread-count invariant.** Torch's
intra-op reductions are not bitwise associative, so one seed at 1 thread and at 8
threads diverges at about $2\times10^{-7}$ per training step; over a hundred epochs
of the feedback loop that becomes $0.006$--$0.015$ in exposure Gini, against a
between-seed sd of $0.011$. `docs/INTERFACES.md` §3 claimed byte-identical results
"across thread counts" and the C++ half of that was true and tested; the torch half
was neither.

Fixed: `ncf.num_threads` defaults to 1, is applied before any weight is drawn, and
is tested (`tests/python/test_reproducibility.py`). The contract now says what it
can defend. `data/processed/rigor/thread_sensitivity.csv` records the measurement.

This does not change any conclusion, because every thread-count variation moves the
values inside the seed intervals already reported. It does mean the reproducibility
claim in the shipped documentation was stronger than the code supported, which is
the kind of error a review is for and this one did not catch.

---

## A3 — three values for one horizon: diagnosis

**Date:** 2026-10-25. **Branch:** `rev/revision`. **Diagnosis only; no published
number changed.**

### The three values

| source | NCF Gini, seed 42, epoch 99 | gap over five seeds |
|---|---|---|
| `data/processed/ablation_h100/` (24 threads, verified) | 0.6935411 | **+0.0251** |
| `data/processed/rigor/long_horizon/`, epoch 99 row | 0.6967353 | **+0.0241** |
| `sensitivity.csv`, shipped cell (repeated across three axes) | — | **+0.0260** |

The oracle arm is bit-identical across the first two (`0.6642505920886874` at
every epoch from 14 on), so the difference is entirely in the NCF arm.

### What was tested, and what it ruled out

**(1) A source change to `ncf.py` between the two runs — ruled out.** The last
change to `src/python/ncf.py` is `af24bc5` (2026-10-24 19:04:10). `ablation_h100`
was generated after it (committed `a065cc3`, 2026-10-24 20:04). `git diff --stat
a065cc3 070e137 -- src/` shows no change to `ncf.py`, `orchestrator.py` or
`config.py` between the two data commits. And running the *pre*-`af24bc5`
`ncf.py` (from `9dc3bf6`) at 24 threads reproduces 0.6935411 exactly, the same as
current code — so the perf commit is numerically inert here and is not the cause.

**(2) A different generation schedule from `n_epochs` — ruled out.** `epochs_per_gen
= n_epochs // n_generations` (orchestrator.py:1228), so a 300-epoch config would
put turnover at epochs 60/120/180/240 rather than 20/40/60/80. This was tested
directly: seed 42 at 24 threads, 300 epochs, with `simulation.n_epochs` set to
300 (epochs_per_gen 60) and with it left at 100 (epochs_per_gen 20). Both give
0.6935411 at epoch 99 and 0.6651935 at epoch 299 — identical. The schedule is
not the cause.

**(3) Thread count — this is the mechanism, but the long run does not land on any
count tested.** Seed 42, 100 epochs, NCF arm, final-epoch Gini:

| threads | Gini | matches |
|---|---|---|
| 1 | 0.6976498 | `thread_sensitivity.csv`, `rigor/base` (0.6973985 — close, not equal) |
| 2 | 0.7037464 | — |
| 4 | 0.6922273 | — |
| 8 | 0.7075976 | `thread_sensitivity.csv` |
| 16 | 0.7037017 | — |
| 24 | 0.6935411 | `ablation_h100`, exactly |

The long-horizon value at epoch 99 is **0.6967353**, which matches none of these.
A 300-epoch rerun at 24 threads reaches 0.6651935 at epoch 299 against the
committed 0.6648466 — close, not equal.

### The divergence signature

The two directories' NCF arms are **bit-identical through epoch 4**, diverge by
$1.0\times10^{-7}$ at epoch 5, $2.3\times10^{-5}$ at epoch 12, $6.1\times10^{-4}$
at epoch 15 and $2.8\times10^{-4}$ at epoch 19. The other three arms are
bit-identical at every epoch checked. A divergence that starts at the seventh
decimal and compounds through the feedback loop is the signature of a different
floating-point reduction order — a different BLAS/threading path — not of a code
change, which would move the first tracked epoch grossly.

### Which branch, and what it means

**Branch (a) applies to the sentence, branch (b) applies to the trajectory.**

The sentence in `design.tex` 38–39 — the first 100 epochs are "identical row for
row" — is **false** and is corrected. The two runs differ from epoch 5.

The trajectory is the more serious half, and it is (b) in substance: the
300-epoch NCF arm **cannot be reproduced from the shipped code at any thread
count tested (1, 2, 4, 8, 16, 24) nor from the pre-perf code at 24**. It is the
output of a numerical environment this repository no longer has, and every
derived quantity (decay fit, asymptote, stationarity, the abstract's
"+0.3055 peak" and "below 0.001 from about epoch 250") inherits that provenance.
The oracle arm reproduces, so whatever differs is confined to the torch training
path.

**Not changed here.** Re-running the 300-epoch horizon is a decision, not an
edit: it is ~25 minutes of compute at 1 thread for five seeds and would replace
the series every convergence claim rests on. It is escalated rather than taken.
`docs/REVIEW_RESPONSE.md` records it; the paper's numbers are left as committed
until that decision is made.

### An incidental finding: `rigor/base` is not reproducible either

Seed 42 in `rigor/base/` reads 0.6973985; a fresh 1-thread run today reads
0.6976498. The difference is 2.5e-4 — small, same thread count, same regime. So
the twenty-seed `base20` intervals that §2 and §4 quote rest on a directory that
no longer regenerates bit for bit either. This is why C1 compares against
`exploratory_1t` (seeds 42–61 regenerated today, 1 thread, same session) rather
than against `base/`.
