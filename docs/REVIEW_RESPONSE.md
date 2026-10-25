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
| 1.5 | The mechanism is contradicted by its own counterexample | **Declined** — see below; the paper states this twice |
| 1.6 | No stated rule for BCa over $t$ | **Fixed** — rule stated, both columns printed |
| 1.7 | The optimum was selected partly on low variance | **Partly fixed** — disclosed; not repairable without fresh seeds |
| 2.1 | Five seeds found biased; correction applied only where noticed | **Fixed as disclosure** — see below |
| 3.1 | The abstract overstates; "fifty-five times" is a ratio of imprecise quantities | **Fixed** |

---

## 1.1 "Converges to zero" — fixed

The review is right, and this was the paper's worst sentence.

What was measured is a flat tail: $+0.00037$ over epochs 250--299, with a $t$
interval of $[+0.000013, +0.000735]$ and Wilcoxon $p = 0.063$. A limit is not
observed; a tail is. And a flat tail is also what a very slow decay looks like.

The abstract, the introduction and the conclusion now say the gap is **below
$0.001$ from about epoch 250 and no longer distinguishable from flat**, and
`paper/sections/results.tex` carries a paragraph naming the distinction between a
bounded residual and a convergence rate. The phrase "fifty-five times its terminal
value" is gone: a ratio of two quantities measured to roughly $\pm30\%$ is an order
of magnitude, not a constant, and the text now says "roughly fifty times" only in
the one place where it draws the order-of-magnitude point.

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

## 1.5 The counterexample — declined

This is the review's headline finding and it does not survive checking.

**The review's claim.** "If learning coverage caused the gap to close, the cell
without coverage learning should retain a larger gap. It does not." Therefore the
mechanism "has not been tested — it has been contradicted."

**Why that is not a contradiction.** Reach falling is not the only thing that
changes at `shock_scale` 0.5. The oracle's Gini also rises there, from $0.6502$ to
$0.9019$: the oracle arm concentrates because the population has lost the
heterogeneity that let a true-utility ranking spread exposure. The gap is a
*difference between two arms*, so a cell where both arms concentrate is a cell
where the difference is small for a reason that has nothing to do with the NCF
catching up. The review's inference assumes the oracle arm is held fixed across
cells; it is not, and the paper's Table 2 shows that.

**The paper already says this, twice.** `paper/sections/results.tex:359--366`:

> "This is the mechanism switching off, not a small deviation: at low chemistry
> noise the filter never learns coverage, so the process §[cold-start] identifies as
> the reason the gap closes is unavailable. The gap is still small and positive
> there ($+0.0097$), but because *both* arms concentrate and the ranker has not
> caught up."

And `paper/sections/results.tex:186--188`:

> "Not that the mechanism is the one H3 offers (§[sensitivity] finds a cell where
> coverage never climbs and the gap closes anyway)."

The review says the paper "acknowledges the reversal but does not confront that the
direction is wrong for its mechanism". The sentence it is quoting is the
confrontation. The review's own text cites it and then sets it aside.

**What is genuinely wrong, and is now fixed.** The mechanism is asserted from a
correlation across epochs; reach is never manipulated. The paper says so in
`results.tex:186` but not prominently enough, and the *causal* claim ("the filter
learning coverage accounts for the decay") is stronger than an observational
trajectory supports. The conclusion now says the explanation is conditional and that
no intervention on reach was run. That is a real weakening and the review earns it.

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

What it does not do: re-estimate the optimum on disjoint seeds. That is a new run,
not a revision, and pretending a correction fixes a selection effect would be worse
than disclosing it.

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
