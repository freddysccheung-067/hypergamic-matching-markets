# Review: "Exposure Concentration Under a Learned Ranker Is a Cold-Start Effect"

**Recommendation: reject.**

I read the manuscript, `docs/STATS.md`, `docs/THESIS.md` and the READMEs under
`data/processed/`. The paper is unusually well documented and its authors have
plainly thought about what their measurements will bear. That is precisely what
makes the review unpleasant: the transparency is genuine, and it exposes how
little of the paper's central claim the design can carry. The manuscript states
its conclusions more strongly than its own evidence, and in two places the
inference is contradicted by a table in the same document.

My concerns are ordered by severity. Section 5 refers to the manuscript's §5;
equation and line references are to `paper/sections/results.tex` unless stated.

---

## 1. The inference

### 1.1 "Converges to zero" is not what was measured (major)

§5.4, and the abstract. The paper's headline claim is that the exposure gap
"converges to zero". What was measured is that a fifty-epoch mean of five seed
trajectories sits at $+0.00037$, with a $t$ interval of
$[+0.000013, +0.000735]$ and Wilcoxon $p = 0.063$.

$0.000013$ is three orders of magnitude below the point estimate. The lower
bound of the interval is not "zero" in any useful sense; it is a number the
authors would not report if they believed it. And $p = 0.063$ on the
distribution-free test will be read by most referees as "consistent with zero at
the 10% level", not as evidence of convergence.

Then the stationarity test. Epochs 290–299 against 280–289 gives $p = 0.324$.
The paper reads this as showing the trajectory has stopped moving. It shows
something weaker: that a ten-epoch window is too short, at five seeds, to detect
a change of the size that would still matter. A convergence test requires either
a model of the decay or a horizon long enough that the residual is bounded away
from the scale of interest. The paper has neither. It has a flat tail, and a flat
tail is what a very slow decay looks like.

The authors anticipate this in §5.4 — "the comparison answers which curve fits
better, not what the asymptote is" — and then use the word "converges" anyway, in
the abstract and in the conclusion. **The honest sentence is "the gap is below
0.001 by epoch 250 and no longer distinguishable from flat".** That sentence
supports a much smaller claim than the title does.

Note also what the paper does with the model comparison. AIC and BIC prefer
decay-to-constant by $\Delta\mathrm{BIC} = 23.9$. The paper sets that aside
because the fitted $K$ is negative. Fine — but then it cannot appeal to a fitted
asymptote at all, and it quotes a non-parametric tail mean whose precision is set
by five points. The paper wants the credibility of the model comparison without
its result, and the robustness of the non-parametric estimate without its
imprecision.

### 1.2 The H4 interval printed in the ledger is the widest one available (major)

Table 1 prints $+64.62$ with interval $[+28.80, +100.43]$ and the verdict
"survives Bonferroni". The interval is the Bonferroni $t$ interval. The
Bonferroni BCa interval is $[+55.07, +78.46]$ (`multiple_comparisons.csv`,
`named_cell_lo_bonferroni` / `named_cell_hi_bonferroni`).

The paper is therefore displaying a $t$-multiplier-and-family correction against
a BCa family correction, and it never says in the table which it used. §5.5 in
the text does say, and is careful — "the $t$ interval is much wider because at
five seeds it is the honest floor" — but a reader who reads Table 1 and not
§5.5 will take $[28.80, 100.43]$ as the corrected interval and conclude the
effect is imprecisely estimated. It is not; the BCa interval is 40% narrower.
**Either print both columns or print one and say which, in the caption.**

Worse, the verdict column says "two cells wide" without saying that this is a
*count of surviving cells*, not a width in size\_fraction. §5.5 says it properly.
The table does not.

### 1.3 A null result described as a trend (major)

§5.4, first paragraph: "The mean gap rises to a peak shortly after the start,
then decays monotonic in the reported points."

The table gives epochs 0, 6, 19, 49, 99, 124, 199, 250–299. Between 199 and
250–299 the reported values are $+0.0025$ and $+0.00037$ — a fall. But the paper
does not report an interval on the epoch-199 value, and at five seeds with the
seed sd of the final epoch at $0.0111$ (`long_horizon_stationarity.csv`,
`seed_sd_final`), a difference of $0.0021$ between epochs 199 and 275 is
unmeasurable. **"Monotone in the reported points" is true and misleading: the
points after 124 are all inside one seed sd of each other.** The decay is
established up to about epoch 124. After that the paper is reporting noise
ordered by epoch number.

This matters because the paper uses "monotone decay to zero" as its evidence for
convergence. If the last two reported points are indistinguishable, the decay
claim ends at epoch 124, where the gap is $+0.0072$ — seven times the claimed
terminal value.

### 1.4 The H3 gap's own interval is treated as a settled fact (moderate)

§5.3: "the paired gap is $+0.0251$ with a $t$ interval of $[+0.0155, +0.0347]$".

This is correct and well sourced. But the same subsection then says the
twenty-seed replication "rules out a five-seed accident", which is a stronger
inference than the data supports. The twenty-seed run gives a *different*
quantity: seeds 42–61 rather than 42–46, hence a different point estimate
($+0.0237$). Agreement between two overlapping samples is not a test; it is
expected. What would rule out an accident is a test of the difference, or a
pre-registered replication on disjoint seeds. The paper has neither.

### 1.5 The mechanism is asserted from a correlation between arms (major)

This is the paper's deepest problem, and it is only partly acknowledged.

§5.4 concludes that the gap closes because "the filter learns coverage", citing
reach climbing from 4% to 96%. Nowhere is reach manipulated. The evidence is that
reach and the gap move together across epochs, and that in the one sweep cell
where reach does not climb (`shock_scale` 0.5, reach $0.4616$), the gap is small
for a different reason.

That is one observation against one counterexample. At `shock_scale` 0.5 the gap
is $+0.0097$ — *smaller* than the shipped cell's $+0.0260$. So the cell where the
proposed mechanism is unavailable has a **smaller gap**, not a larger one. The
paper's own table (Table 2) shows this. If learning coverage caused the gap to
close, the cell without coverage learning should retain a larger gap. It does
not.

The paper acknowledges the reversal ("both arms concentrate") but does not
confront that the direction is wrong for its mechanism. **A mechanism that
predicts the gap stays large where reach fails, and instead finds it small, has
not been tested — it has been contradicted.**

### 1.6 Where an interval includes zero and the text does not say so (moderate)

§5.5, exogenous term: the $t$ interval $[-52.53, +10.32]$ includes zero while
the BCa interval excludes it. The paper says the BCa interval is "optimistic" and
reads the sign as negative. That is a defensible choice given §1's argument that
BCa undercovers at five seeds. But the paper applies that argument
*inconsistently*: in §5.3 it quotes the BCa interval for `gini_gap` at 20 seeds
(where it is fine), and in §5.4 it quotes the $t$ interval for the tail mean
(where $t$ is wider and needed). There is no stated rule for when BCa is
preferred over $t$, and the reader is left to infer that the paper used whichever
supported the reading.

Recommendation: state one rule in §4, apply it in every table, and print both
columns so the reader can check.

### 1.7 A selection effect in the paper's own best case (major)

§5.5 on H4 reports the optimum's standard deviation as the ninth smallest of
forty, "a little under a third of the median cell's spread", and concludes that
"five seeds happened to agree". This is an admission that the optimum cell was
selected partly on low variance as well as high mean. The paper then quotes its
$t$ interval as the one to read.

But a $t$ interval on the maximum of forty cells, where the maximum was selected
on both mean and low variance, is not corrected by the Bonferroni multiplier. The
Bonferroni correction widens for multiplicity; it does not un-select. The paper
applies the former and claims to have addressed the latter. If much of the
optimum's $+64.62$ is an artefact of selecting a low-variance cell, the corrected
interval is not honest — it is wide around a biased point.

A reader who wanted to be convinced would ask for the optimum re-estimated on
fresh seeds. That is the only clean answer, and the paper does not have it.

---

## 2. The design

### 2.1 Five seeds (major)

Every headline rests on five seeds. The paper defends this with a power
calculation showing fewer than five seeds are needed to detect each effect at 80%
power. That defence is circular in a specific way: the power calculation uses the
twenty-seed sd, and the five-seed sd is the quantity at issue. It is also
irrelevant to the paper's actual problem. §2.2 of `STATS.md` shows the five seeds
are the *minimum-variance contiguous block* of sixteen for algorithmic loss. The
paper reports this and then continues to use five seeds everywhere except for the
two quantities where the problem was detected. **The paper found evidence that its
seed choice is biased and applied the correction only where it happened to look.**

If seeds 42–46 are unrepresentative for two of seven quantities, the prior that
they are representative for the other five is not established by the fact that
the ratio came out near 1.0. With sixteen overlapping blocks and one minimum,
some block has to be smallest; that the chosen block is smallest for two
different quantities is exactly the signal that it is not a random five. The
right response is to re-run the headline on seeds 47–61. That is cheap and the
paper does not do it.

### 2.2 One population size (major)

$N = 1000$ is the only size at which the A/B comparison is run. The paper
concedes this in §4.6. It is disqualifying for the clearing-rate results: the
paper itself says in §6 that clearing "falls roughly as $1/N$" and "does not
transport". A quantity that the paper knows does not transport across the one
dimension it varied is reported in §5 as though it were a property of the
mechanism.

The A-side ablation at N=1000 is defensible on its own terms. The institutional
comparison is not, because the subgraph's whole purpose is to be *small*, and the
paper varies its size only as a fraction of a single population. Whether the
interior optimum at size\_fraction 0.65 is a property of fraction or of absolute
size $|B| = 650$ is unanswerable from this design. A referee will ask, because
figure 3's x-axis invites exactly that question.

### 2.3 The parameter sweep's thresholds were chosen by the authors (major)

§4.5 states that C2's ceiling (0.05) and C3's floor (0.75) were fixed before the
cells were read, and that this is "the two judgments in the paper most exposed to
the charge of being chosen after the fact", with "the only mitigation is that
they were written down first".

Writing them down first is not a mitigation when the same authors chose them
without pre-registration, and when the results sit close enough to the
thresholds to matter. The `shock_scale` 3.0 cell has a gap point estimate of
$+0.0470$ against a ceiling of 0.05. That is a 6% margin on a hand-chosen
threshold, and the *largest* cell in a grid of twelve. A ceiling of 0.04 would
have failed C2 there; a ceiling of 0.06 would have passed everything
comfortably. The paper's claim that "C2 holds in all twelve cells" is a statement
about the number 0.05, not about the phenomenon.

C3 is worse. The reach floor is 0.75. The failing cell has reach $0.4616$ and the
next-lowest cell has $0.8904$ — there is no cell anywhere near the threshold. A
floor anywhere in $[0.5, 0.89]$ gives the same verdict. So C3's "one reversal"
is robust, and its threshold choice is harmless. C2's is not robust and its
threshold is load-bearing. **The paper treats both thresholds as equally
defensible, and they are not.**

### 2.4 The engineering realism of the shortlist (moderate)

$k = 20$ with $N = 1000$ per side means an agent sees 2% of the opposite side.
The paper never justifies $k$, never sweeps it in the sensitivity grid (it sweeps
four knobs and $k$ is not among them), and the ROTH & ROTHBLUM truncation result
it cites in §2 says truncation is a strategic object. So the design fixes the one
parameter that most directly determines the visibility constraint and does not
test it.

The `flooding` README shows $k \in \{20, 40\}$ was run for the H2 study, so the
infrastructure exists. Its absence from the sensitivity grid is a gap, not a
limitation.

### 2.5 Nothing is calibrated (accepted, not a defect)

The paper is admirably explicit that no parameter is estimated. I do not score
this against it; a simulation of a mechanism is a legitimate instrument and the
paper says what it can and cannot support. I note it only because it sets the bar
for everything else: with no external anchor, the paper's contribution has to be
the *internal* rigour of its design, and §§2.1–2.4 are where that rigour is
thinnest.

---

## 3. Reporting and presentation

### 3.1 The abstract overstates (major)

The abstract says the gap "decays to below $0.001$ ... so the hundred-epoch
figure is fifty-five times its terminal value". "Terminal value" presupposes the
convergence the paper has not established (§1.1). And "fifty-five times" is a
ratio of $+0.0241$ to $+0.00044$ — both of which are numbers whose own intervals
are large relative to the numerator's distance from zero. A ratio of two
quantities each measured to ±30% is not a number to four significant figures.

The abstract also says the mechanism "fails in the one cell where coverage never
climbs", which is correct, but omits that the gap is *smaller* there (§1.5),
which is the fact that damages the mechanism.

### 3.2 Figure captions argue rather than describe (minor)

Figures 1–4 each carry a sentence beginning "Earns its place because...". This is
the drafting convention of the paper's own tooling leaking into the manuscript.
A reader does not need to be told why a figure is present; they need to be told
what it shows and what the axes mean. Figure 3's caption is 90 words and does not
state the units of the y-axis relative to the baseline in the left panel.

### 3.3 The validation table has no committed artefact (major, disclosed)

The paper says plainly, in the caption and in §4.5, that these numbers are
asserted in the test suite and transcribed in `docs/validation.md` with no CSV
behind them. I credit the disclosure. But it is a table of results in a results
section that cannot be regenerated from committed data, in a paper whose stated
virtue is that every number traces to a committed file. Either commit the
artefact or move the table to an appendix marked as untraceable.

### 3.4 Uncited bibliography entries

`refs.bib` carries 35 entries and 35 are cited, so this is clean. Earlier drafts
had orphans; the authors fixed it. Noted so they know it was checked.

### 3.5 The paper is uncompiled

I cannot assess typesetting, float placement, or whether the three `tabular`
environments fit the page. Table 2 has 8 columns in a single-column layout at
11pt; I doubt it fits within `\textwidth` without `\small` or a `sidewaystable`,
and neither is present. The authors should compile before resubmission.

---

## 4. What would make this publishable

The dataset is better than the paper. Three changes would move it substantially:

1. **Replace "converges to zero" with a bounded statement** and test the bound
   directly: report the epoch after which the gap is below 0.001 on all five
   seeds and the width of the interval on that claim. Drop the ratio.
2. **Re-run the headline on disjoint seeds (47–61)** and report the difference
   from seeds 42–46 with its interval. This is the single most valuable
   experiment the paper could add and it costs one run.
3. **Sweep $k$.** The visibility constraint is the paper's whole subject and the
   one knob it holds fixed is the one that defines it.

Even with all three, the mechanism claim of §1.5 has to be weakened to a
description: the gap closes as the filter trains, and in one configuration it
does not close the same way. That is a finding. It is not an explanation.

---

## 5. Summary

| # | criticism | severity |
|---|---|---|
| 1.1 | "Converges to zero" exceeds a flat tail and $p = 0.063$ | major |
| 1.2 | Ledger prints the widest corrected interval without saying which | major |
| 1.3 | "Monotone decay" includes points inside one seed sd | major |
| 1.5 | Mechanism contradicted by the smaller gap in its counterexample cell | major |
| 1.7 | Optimum selected on low variance; Bonferroni does not un-select | major |
| 2.1 | Five seeds found biased for 2 of 7 quantities, correction not applied | major |
| 2.2 | Single population size for a subgraph-size result | major |
| 2.3 | C2's ceiling is load-bearing and hand-chosen | major |
| 3.1 | Abstract overstates, and a ratio of imprecise quantities | major |
| 3.3 | Validation table has no committed artefact | major |
| 1.4 | Overlapping-sample agreement called a replication | moderate |
| 1.6 | No stated rule for choosing BCa over $t$ | moderate |
| 2.4 | $k$ fixed and not swept | moderate |
| 3.2 | Captions argue; units unclear | minor |
| 3.5 | Uncompiled; Table 2 likely overflows | minor |

The paper's strongest element is its willingness to report H2 and H4 as failures
at full length, and its `docs/STATS.md`, which is more rigorous than most
published appendices. The weakest is that its central claim is stated in the
vocabulary of convergence when its evidence is vocabulary of a small flat tail.
The gap between those two vocabularies is the review.
