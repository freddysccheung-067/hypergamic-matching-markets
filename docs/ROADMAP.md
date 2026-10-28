# Roadmap: what this study does not do

Written at the end of the revision of 2026-10-25 and updated at the end of the
second revision (2026-10-28), so that a future session can start cold. Each item
states what is missing, why it matters, what would have to change, and what
result would count. **No speculative numbers appear here**: where a quantity would
come from a run, the entry says so and does not guess it.

Ordered by how much each would change what the paper can claim.

---

## 0. Raised by the second revision's results

These are questions the second revision's measurements opened and did not answer.
Each is stated in the paper as unexplained.

- **Within-list order.** At round 299 the ranker's shortlists hold the oracle's
  agents (Jaccard about 0.99) while its algorithmic loss stays well above the
  oracle's. Deferred acceptance reads list order, so the residual welfare loss is
  consistent with lists in a different order, but within-list order was not
  measured. What would count: a per-round rank correlation between the NCF order
  and the oracle order *inside* the shared top-k, logged with the R2-E12
  diagnostics, and its relation to algorithmic loss.
- **A position-weighted exposure measure.** Exposure counts list membership. A
  measure that discounts lower positions could follow a different trajectory.
  What would count: the R2-E9 trajectory recomputed with a position discount, on
  recorded lists (`Simulation.record_lists`), no new runs needed.
- **The four-pass overshoot.** At four training passes per round the gap ends
  below the margin (the ranker less concentrated than the oracle). The experiment
  does not say why. A candidate is overfitting the resampled negatives; it is not
  tested.
- **The institutional sign change.** The B-minus-A comparison points one way at
  five rounds and the other after the transient, for an oracle A whose thresholds
  settle early. The cause was not isolated. What would count: per-round welfare of
  both environments over all 300 rounds, which `scripts/run_institutional_horizon.py`
  records but does not save.
- **Open-market stationarity.** The registered stationarity criterion fails in
  the open market because each round's arrivals move each round's mean; the levels
  over rounds 500 to 599 are reported as what the gap fluctuates around. What
  would count: more seeds and a longer horizon at one rate, with the trend test of
  R2-E14-check.

---

## 1. An open market: arrivals

**Status after the second revision.** Replacement is implemented
(`simulation.arrival_rate = departure_rate`, uniform departure, the newcomer in the
same slot with fresh traits, embedding and threshold; R2-E14). What remains open is
everything below that replacement does not cover: a market whose size changes, and
departure that depends on outcomes (item 2).

**What is missing.** The population has generational turnover
(`simulation.turnover_fraction = 0.2`, `simulation.n_generations = 5`), which
replaces the bottom fifth by cumulative match count between generations. That is
attrition-free churn: no agent joins except as a replacement, no agent leaves in
response to an outcome, and the population size is constant by construction.

**Why it matters.** The paper's central result is that the exposure gap is a
*cold-start* property — a cost of a ranker that has not yet learned whom it can
represent. A market with genuine arrivals has a permanent cold start for its
newest members, so the question "does the gap persist?" has a different answer in
an open market: some share of the population is always at epoch zero. The
steady-state gap is therefore the right object there, and the paper cannot
compute it.

**What would have to change.** A C++ change to the population layer
(`src/cpp/population.cpp`, the `Agent` struct and `sample_population`) plus config
surface (`population.arrival_rate`, `population.departure_rate`), plus turnover in
the orchestrator to draw arrivals rather than replacements. This is not a script:
the state matrix layout, the θ calibration, the recommender's embedding rows and
the swipe-stream seeding all assume the population is fixed at construction.

**The prediction to test.** The steady-state gap increases in the arrival rate.
With a constant inflow, the fraction of agents within their first *n* epochs rises
with the rate, and if the gap is a function of ranker inexperience then so does
the equilibrium gap.

**What would count.** A 300-epoch run at three arrival rates, five seeds, 1
thread: the gap at epoch 299 as a function of the rate, with an interval per
rate. If it is flat, the cold-start story does not extend to an open market and
that is the finding.

---

## 2. Attrition: exit on outcomes

**What is missing.** No agent leaves in response to anything. `docs/THESIS.md`
H6 asks whether high-intent agents leave Environment A at a higher rate; the
model cannot pose the question because it has no exit margin.

**Why it matters.** Two things follow. H6 is untestable as shipped, which the
paper reports as a finding rather than fixing. And the welfare numbers are
unbounded in a specific way: a mechanism that repelled participants would look
free, because nobody can leave.

**What would have to change.** An exit rule read inside the epoch loop: after
matching, remove agents meeting a condition and replace them through the
generation mechanism, or shrink the market. The latter breaks far more than the
former, since the state matrix is fixed-size; a placeholder row is the cheaper
path.

**Pre-register two rules, because the sign depends on the rule.** This is the
part that matters and the part a future session must not skip:

- *Exit on no-match*: leave after *k* consecutive epochs unmatched. Under this
  rule the association with intent is predicted to be null, because matching does
  not depend on τ in the shipped code — the cost is spent, not sorted on.
- *Exit on net payoff*: leave when cumulative payoff falls below a threshold.
  Under this rule the association is predicted to be strongly positive, and made
  positive by **cost burn** rather than by rejection: high-intent agents propose
  far more often and are matched at the same rate, so they spend more.

The sign of "differential attrition by intent" is a property of which rule is
chosen, and a study that pre-registers one rule and reports its sign has measured
the rule, not the market.

**What would count.** H6 becomes testable: exit rate as a function of τ, per
rule, five seeds, with an interval. A null under the first rule and a positive
association under the second is the result already predicted in
`docs/THESIS.md`, and would confirm that the channel is cost burn.

---

## 3. Calibration: method of simulated moments

**What is missing.** Nothing is calibrated. Every parameter is a modeling choice
and `docs/LIMITATIONS` says so at length. That is an honest position and it is
also why the results are statements about a mechanism under a stated utility
rather than about any population.

**Why it matters.** It is the single largest gap between this work and a claim
about markets. Without it, "the gap is +0.054 at a twenty-seed estimate" is a fact
about the simulator.

**What would have to change.** An MSM procedure: choose a small set of target
statistics observable in both the model and a public data source, simulate at
candidate parameter vectors, and minimise the weighted distance between simulated
and observed moments. Candidates worth examining — **none verified for this
document, and each must be checked for license and availability before use**:

- A large-scale online-dating panel, for the assortative-matching and
  message-concentration moments. Licensing is the binding constraint for most of
  these and is not established here.
- Public marriage-register or census microdata, for the status-assortative
  matching moments.
- Published summary statistics from a speed-dating study, for trait weights.

**Standing constraints, to be honoured rather than revisited:**

- **No dataset enters the repository.** Code that reads a path supplied at run
  time, yes; a committed extract, no.
- **Verify license and availability before use**, per source, and record the
  verification.
- **The honest claim is "not arbitrary", never "calibrated"** — unless the fit is
  actually good, in which case say what it fits and how well.

**What would count.** A parameter vector whose simulated moments land within a
stated tolerance of the targets, with the tolerance declared before fitting, and
an honest statement of which moments are matched and which are not. A failed fit
is a result and should be reported as one.

---

## 4. Global sensitivity

**What is missing.** The sweep moves one knob at a time: twelve cells, four axes,
three levels each, the rest shipped. One-at-a-time designs cannot detect
interactions — the effect of `shock_scale` at `taste_sigma` 0.2 may differ from
its effect at 1.0, and the design cannot see it.

**Why it matters.** The paper already found one interaction by accident (the gate
rule, `admissible_fraction < 2k/N`, couples φ and k) and one conditional result
(the cold-start explanation depends on `shock_scale`). A design built to see
interactions would find the ones the paper missed rather than the ones it
stumbled on.

**What would have to change.** A Latin hypercube or Sobol design over the four
knobs (`taste_sigma`, `shock_scale`, `beta/alpha`, `delta`), with a variance
decomposition — first-order and total Sobol indices — on the headline gap.
Practical size: a Sobol design needs $N(2d+2)$ points for $d = 4$, so 200–500
cells; at the measured cost of the sweep that is hours, not days.

**What would count.** Total-order indices that sum above one (interaction
present) or a first-order index for a knob the one-at-a-time design called inert
(the `delta` axis is the candidate: it moves the gap by 0.0034 against intervals
0.012 wide, and an interaction could hide that).

---

## Done in the second revision

Re-running the 300-epoch shipped cell at 1 thread, which the first version of this
file listed as a repair: R2-E9 ran it on forty seeds, and seeds 42 to 46 reproduce
the earlier one-thread run exactly. The paper's numbers now come only from
`data/processed/horizon/`.
