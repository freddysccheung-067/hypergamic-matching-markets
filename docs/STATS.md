# Statistics

What was measured, at what n, with what interval, and what a reader may and may
not conclude.

Every table below traces to a file under `data/processed/rigor/`. Every line
states its seed count. No claim appears without an interval.

This document exists because the repository's earlier tables read `mean ± sd`,
and a standard deviation is not a confidence interval. At five seeds it is also
a badly estimated one. An sd describes how far the seeds fell apart; an interval
describes where the quantity being estimated lies. Reporting the first as if it
were the second makes a result look better determined than it is — by a factor of
about 1.24 at n = 5, compensating errors aside.

The method throughout is the BCa bootstrap over seeds, 10,000 resamples, with the
percentile interval and a Student-t interval reported beside it. The t interval
is not a formality: §2 measures it covering 94.9% of the time at n = 5 where BCa
covers 84.3%, and that is why the wider of the two is the one to read at small n.

---

## §1. The seven headline quantities

Source: `data/processed/rigor/intervals.csv`. Regimes as named in
`data/processed/rigor/README.md`. BCa intervals, 10,000 resamples, bootstrap seed
20261024.

| # | quantity | regime | seeds | point | 95% BCa interval | percentile | half-width ÷ 1 sd |
|---|---|---|---|---|---|---|---|
| 1 | exposure Gini, NCF | live100 | 5 | 0.6796 | [0.6643, 0.6922] | [0.6653, 0.6938] | 0.76 |
| 1b | exposure Gini, NCF | live20 | 5 | 0.9412 | [0.9383, 0.9435] | [0.9386, 0.9436] | 0.80 |
| 1c | exposure Gini, NCF | base20 | 20 | 0.6738 | [0.6675, 0.6810] | [0.6676, 0.6808] | 0.42 |
| 2 | Gini gap, NCF − oracle | live100 | 5 | +0.0251 | [+0.0186, +0.0307] | [+0.0189, +0.0309] | 0.78 |
| 2b | Gini gap, NCF − oracle | long300 | 5 | +0.00044 | [−0.00029, +0.00105] | [−0.00001, +0.00126] | — |
| 2c | Gini gap, NCF − oracle | base20 | 20 | +0.0237 | [+0.0204, +0.0271] | [+0.0201, +0.0268] | 0.42 |
| 3 | clearing rate, NCF | live100 | 5 | 0.2404 | [0.2324, 0.2528] | [0.2324, 0.2528] | 0.77 |
| 4 | rank satisfaction, NCF | live100 | 5 | 0.4502 | [0.4362, 0.4633] | [0.4390, 0.4650] | 0.74 |
| 5 | algorithmic loss, NCF | base20 | 20 | 0.6627 | [0.6508, 0.6741] | [0.6508, 0.6741] | 0.43 |
| 6 | welfare_common, A | grid_pooled | 5 (40 cells) | 973.70 | [973.70, 973.70] | [973.70, 973.70] | — |
| 6b | welfare_common, B | grid_pooled | 5 (40 cells) | 787.50 | [618.0, 884.9] | [628.2, 878.8] | — |
| 7 | welfare_diff_residual | grid_pooled | 5 (40 cells) | −245.8 | [−421.3, −145.5] | [−400.6, −155.3] | — |
| 7b | welfare_diff_residual, optimum cell | c=0.01, sf=0.65 | 5 | +64.62 | [+57.47, +73.17] | [+57.47, +72.53] | 0.80 |

### The t intervals, which are the ones to read at five seeds

| # | quantity | regime | seeds | 95% Student-t interval |
|---|---|---|---|---|
| 1 | exposure Gini, NCF | live100 | 5 | [0.6569, 0.7023] |
| 1c | exposure Gini, NCF | base20 | 20 | [0.6664, 0.6813] |
| 2 | Gini gap, NCF − oracle | live100 | 5 | [+0.0155, +0.0347] |
| 3 | clearing rate, NCF | live100 | 5 | [0.2240, 0.2568] |
| 4 | rank satisfaction, NCF | live100 | 5 | [0.4277, 0.4728] |
| 5 | algorithmic loss, NCF | base20 | 20 | [0.6501, 0.6753] |
| 7b | welfare_diff_residual, optimum cell | c=0.01, sf=0.65 | 5 | [+52.37, +76.86] |

At twenty seeds the two agree closely (rows 1c and 5 differ by 0.002 at the
bounds), because the bootstrap has enough points to see its own tails. At five
they differ by about 20%, and it is the t interval that has the nominal coverage.

**The last column is the headline of this whole document, and it points the
alarming way.** It is the interval's half-width divided by the sample's own sd —
that is, the 95% interval measured in the units the old `mean ± sd` tables used.
If `mean ± sd` were a 95% interval the number would be 1.00. Normal theory says it
should be **2.78** at n = 5 (t with 4 df) and **2.09** at n = 20.

It is **0.74–0.80 at five seeds**. The bootstrap interval is not wider than the
sd the repository already quoted; it is *narrower*, by about a quarter.

That is the small-sample bootstrap defect §2 measures directly (§1 and
`test_five_seeds_bootstrap_interval_is_narrower_than_the_t_interval`): five points
give the resampling distribution no tails to draw from, so it produces an interval
that attains 84% coverage rather than 95%. Emitting it as "the 95% interval" and
quitting would have replaced one wrong number with another that looks more
rigorous.

**So the t column is the headline interval at five seeds, and STATS.md quotes it
wherever the two disagree.** The BCa interval is the one that is asymptotically
correct and is quoted at twenty seeds; the t interval is the one with nominal
coverage at five. A reader who wants one number should read the wider of the two,
which at n = 5 is always the t interval.

**Converting an old `mean ± sd`.** The t interval is the right conversion. At five
seeds a 95% interval on the mean is `mean ± 2.776·sd/√5`, i.e. `mean ± 1.241·sd`
— the sd is divided by √n before the t multiplier applies, which is the step the
old tables omitted. Equivalently: **multiply the quoted sd by 1.24 to get the
95% half-width at five seeds**, and by 0.47 at twenty. The BCa interval cannot be
converted from an sd at all, because it is not a scaled sd — and at five seeds it
is *narrower* than 1 sd, which is the defect described above.

### Reading each row

**Row 1 (exposure Gini).** The NCF's exposure concentration at 100 epochs is
0.68; the t interval is [0.657, 0.702] and the BCa interval [0.664, 0.692]. The
20-seed run puts it at 0.674 with both intervals agreeing to 0.002, which is the
expected direction and confirms the 5-seed interval is not hiding anything here.
The 20-epoch figure (1b) is a different regime, not a noisier estimate of the
same thing: 0.941 is the cold-start state.

**Row 2 (the H3 gap).** This is the quantity the thesis's central algorithmic
claim rests on. At 100 epochs it is +0.025, with the t interval [0.0155, 0.0347]
and BCa [0.0186, 0.0307] — small, positive, and clearly separated from zero on
either reading. At 300 epochs (2b) it is +0.0004 with a BCa interval that
*includes* zero, while the t interval on the tail mean excludes it (§3). Both
facts are true and the resolution is in §3: it converges to essentially zero, and
whether that residual is exactly zero is not resolved at five seeds and does not
matter.

**Row 3 (clearing).** 24% of agents match under the NCF at 100 epochs, t interval
[0.224, 0.257]. The interval is tight because clearing is an integer count over
1000 agents; the seed-to-seed variation is a few agents.

**Row 4 (rank satisfaction).** 0.450, t interval [0.428, 0.473]. This is the
*acceptor* side under M-proposing, which `docs/INTERFACES.md` §5 establishes is
the acceptor-pessimal stable matching. So this is a lower bound across stable
matchings, not a central estimate, and it must never be compared against a
W-proposing baseline without saying so.

**Row 5 (algorithmic loss).** 0.663 means the NCF-mediated matching realizes 34%
of the full-information welfare. This quantity had no committed column before
this branch — it was computed on every epoch and discarded at write time — so it
is new. It also carries the worst small-sample behaviour of anything measured:
§2 shows its five-seed sd understates the twenty-seed sd by a factor of 3.2. The
interval here rests on 20 seeds for that reason.

**Row 6 (welfare by environment).** Environment A's welfare is *identical* in all
40 institutional cells, because the oracle arm reads neither `c_B` nor
`size_fraction`: every cell is the same run repeated. The zero-width interval is
therefore structural, not a failure, and A's row is one measurement over five
seeds rather than forty over five. B's welfare varies across cells by design; its
interval resamples the 40 design cells, and the five seeds inside a cell are
replicates of one cell rather than 200 independent draws. In `intervals.csv` these
rows carry `n_seeds = 5` and `n_units = 40, unit = cell` — read the first as the
replication and the second as what the interval was computed over.

**Row 7 (residual welfare difference).** Pooled over the grid, B loses to A by
246 welfare units [−421, −146]. This is the *pooled* figure and is not the H4
claim: H4 is about a specific region of size_fraction, and row 7b gives that
cell. The two must not be quoted interchangeably — one says "on average B loses",
the other says "at size_fraction 0.65 with c_B = 0.01, B wins by 65".

**Row 7b and §5.** The optimum's interval is the *unadjusted* one. It was selected
from 40 cells, and the interval that survives that selection is reported in §5.

### What a reader may not conclude

- These are intervals for a **model's** quantities, over its seed distribution.
  They say nothing about any real population. A narrow interval means the
  simulation reproduces across populations; it does not mean the answer is true.
- An interval that excludes zero is not a claim of practical significance. Row 1b
  is tightly bounded around 0.94 but describes a 20-epoch transient.
- Comparing two rows with overlapping intervals is not a test. Gaps must come
  from the *paired* per-seed difference (as rows 2, 2c, and every sensitivity cell
  do), which is narrower than either arm's own interval.

---

## §2. Is five seeds enough?

Sources: `data/processed/rigor/power_analysis.csv`, `power_widths.csv`,
`power_conclusions.csv`. 20 seeds (42–61) at the shipped regime, 100 epochs.

### §2.1 Width scaling

Interval width by seed count, averaged over every contiguous block of that
length, against the 1/√n law (`power_widths.csv`):

| quantity | n=5 | n=8 | n=10 | n=15 | n=20 | ratio(5)/ratio(20) | √(20/5) | ratio ÷ law |
|---|---|---|---|---|---|---|---|---|
| `gini_ncf` | 0.0232 | 0.0200 | 0.0178 | 0.0159 | 0.0136 | 1.71 | 2.00 | 0.85 |
| `gini_gap` | 0.0112 | 0.0101 | 0.0095 | 0.0083 | 0.0068 | 1.65 | 2.00 | 0.82 |
| `mcr_ncf` | 0.0316 | 0.0293 | 0.0257 | 0.0211 | 0.0165 | 1.92 | 2.00 | 0.96 |
| `algorithmic_loss` | 0.0500 | 0.0412 | 0.0362 | 0.0296 | 0.0233 | 2.15 | 2.00 | 1.07 |

**The 1/√n law holds.** Every quantity's ratio falls within 18% of the predicted
1.71–2.15 against 2.00, and the deviation is systematic rather than erratic: the
ratio is slightly *below* the law for the Gini quantities and slightly above for
`mcr` and `algorithmic_loss`. That pattern is the finite-sample behaviour of a
bootstrap interval, not a defect — at n = 5 the bootstrap cannot see the tails it
has not observed (§1), so its width grows more slowly than 1/√n as n falls. The
`algorithmic_loss` ratio of 2.15 is on the conservative side of the law, which is
the opposite of what its sd ratio in §2.2 would predict.

**The discrepancy between §2.1 and §2.2 is the point.** The width follows 1/√n
faithfully; the *sd estimate at n = 5* does not estimate the twenty-seed sd. Both
are true because they answer different questions. §2.1 asks "as I add seeds from a
fixed pool, how fast does the interval shrink" — and it shrinks as it should. §2.2
asks "are the five seeds I have a representative sample of the population of
seeds" — and for two quantities they are not. A table of widths by n cannot detect
that; a comparison of the 5-seed sd against the 20-seed sd can.

### §2.2 Seed-to-seed sd, five seeds against twenty

| quantity | sd over 5 | sd over 20 | ratio | 5-block sd range | first block's rank (of 16) |
|---|---|---|---|---|---|
| `algorithmic_loss_ncf` | 0.0084 | 0.0270 | **0.31** | 0.0084–0.0409 | **1 (minimum)** |
| `mcr_ncf` | 0.0079 | 0.0192 | **0.41** | 0.0076–0.0283 | 2 |
| `gini_oracle` | 0.0111 | 0.0121 | 0.92 | — | — |
| `gini_gap_ncf_minus_oracle` | 0.0079 | 0.0079 | 1.00 | — | — |
| `gini_ncf` | 0.0179 | 0.0159 | 1.13 | 0.0035–0.0218 | 12 |
| `reach_ncf` | 0.0079 | 0.0071 | 1.11 | 0.0056–0.0084 | 15 |
| `rsi_ncf` | 0.0367 | 0.0323 | 1.14 | 0.0169–0.0428 | 12 |

**This is the finding of §2, and it is uncomfortable.** Seeds 42–46 are not a
typical five: for `algorithmic_loss` they are the *minimum-variance* contiguous
five of the sixteen available, and for `mcr` they are second. Every committed
number in this repository rests on those five.

Note the asymmetry that makes this more than a curiosity. The two quantities whose
five-seed sd understates the population's are both *low-ranked* on that measure —
the five lands at the bottom of the range. The two quantities whose sd is slightly
*conservative* (`gini` 1.13×, `rsi` 1.14×) are mid-ranked (12 of 16). A five-seed
sample can land anywhere in that range; that the two worst cases both landed at
the bottom, and that both are the same five seeds, is why the problem is
quantities-specific rather than uniform noise.

The consequence is quantity-specific:

- For `gini`, `rsi`, `reach` and the gap, the five-seed sd is within about 14% of
  the twenty-seed sd and slightly conservative. The five were enough.
- For `algorithmic_loss` and `mcr`, they were not. The five-seed sd is 0.31× and
  0.41× the twenty-seed sd, so any interval built on it is about three times too
  narrow. No quoted headline rests on either quantity's tight interval today —
  `algorithmic_loss` had no committed column until this branch — but any future
  claim that does should be re-derived from the twenty-seed run first.

A caution on the reading: with 16 overlapping blocks the "minimum of 16" is not
an independent draw, and some block must be smallest. What makes 42–46 worth
naming is not the rank alone but that they were chosen years before the twenty
more were run, and that two different quantities both put them at the bottom.

### §2.3 Does any conclusion flip?

No. Across every quantity and every n ∈ {5, 8, 10, 15, 20}, the fraction of
blocks whose interval excludes zero is 1.0 and the point estimate's sign is
stable (`power_conclusions.csv`). No headline changes sign or crosses zero at any
seed count from five to twenty.

This is a weak statement, and its weakness is the point: every headline in this
project is a ratio or a difference with an effect size many times its seed sd
(Gini gaps of 0.02 against sd 0.008; welfare differences of 65 against sd 10). A
claim that needed 20 seeds to separate from zero would show up here as a
fraction below 1.0 at n = 5. None does.

### §2.4 The power analysis

Seeds required for a two-sided test at α = 0.05 to detect each effect at 80%
power, from `n = ((1.96 + 0.84)·sd/|effect|)²` using the twenty-seed sd:

| quantity | effect | sd (20 seeds) | seeds needed | binding? |
|---|---|---|---|---|
| `gini_ncf` | 0.674 | 0.0159 | 5 | no |
| `gini_gap` | 0.0237 | 0.0079 | 5 | no |
| `mcr_ncf` | 0.243 | 0.0192 | 5 | no |
| `rsi_ncf` | 0.458 | 0.0323 | 5 | no |
| `reach_ncf` | 0.957 | 0.0071 | 5 | no |
| `algorithmic_loss` | 0.663 | 0.0270 | 5 | no |

The "seeds needed" column reports 5 for every row because the raw requirement is
below 5 in every case and has been clamped to the minimum feasible run
(`power_analysis.csv`, column `n_requirement_binding` = False throughout). The raw
values are 0.0044 for `gini_ncf`, 0.879 for the gap, 0.049 for `mcr`, 0.039 for
`rsi`, 0.0004 for `reach` and 0.013 for `algorithmic_loss` — all below one, which
is another way of saying the effects are many standard errors wide. There is no
row where twenty seeds would be insufficient.

Every requirement falls below five, so five seeds is more than enough for the
effects as measured, and **no claim in this project needs more than 20 seeds to
separate from zero.** The seed count was never the binding constraint on any
headline.

That is not the same as saying the seed count does not matter. It matters for the
*precision* of a quoted number — §2.2 shows it matters a great deal for
`algorithmic_loss` — and it would matter for any claim about a small effect. The
honest summary is: the effects are large enough that even a badly-chosen five
could not miss their sign, which is why no conclusion flips; but the interval
widths on two quantities are wrong by a factor of three, which is why the
intervals in §1 should be quoted and the old sds should not.

---

## §3. The cold-start result, pushed to convergence

Sources: `data/processed/rigor/long_horizon/` (300 epochs, five seeds, shipped
φ = 0.02), and `long_horizon_gap_series.csv`, `long_horizon_decay_fits.csv`,
`long_horizon_asymptote.csv`, `long_horizon_stationarity.csv`.

### §3.1 Does the gap converge to zero or to an asymptote?

It converges to **zero**, within what five seeds can resolve.

| epoch | 0 | 6 | 19 | 49 | 99 | 124 | 199 | 250–299 mean |
|---|---|---|---|---|---|---|---|---|
| gap | +0.2633 | +0.3055 | +0.2869 | +0.1718 | +0.0241 | +0.0072 | +0.0025 | **+0.00037** |

The tail mean over epochs 250–299 is +0.00037 with a 95% t interval of
[+0.000013, +0.000735], five seeds, all five positive (t = 2.88, p = 0.045;
Wilcoxon p = 0.063). The BCa interval on the final epoch's gap is
[−0.00029, +0.00105], which includes zero.

**Which to quote:** the effect is at most seven parts in ten thousand of the Gini
scale, and its sign is not settled. State it as "converges to zero; any residual
is below 0.001 and is not resolved at five seeds." Do not quote +0.00037 as a
positive floor.

**The curve comparison, and why it is reported but not trusted.** Both models
were fitted to the five-seed mean series, from epoch 0 and again from the peak at
epoch 6:

| fit window | model | params | RSS | AIC | ΔAIC | ΔBIC | K | K's interval |
|---|---|---|---|---|---|---|---|---|
| from epoch 0 | `A·exp(−λt)` | 2 | 0.1411 | −2294.6 | 30.7 | 26.9 | — | — |
| from epoch 0 | `K + A·exp(−λt)` | 3 | 0.1266 | −2325.2 | **0** | **0** | −0.0111 | [−0.0152, −0.0070] |
| from peak (6) | `A·exp(−λt)` | 2 | 0.0832 | −2398.1 | 27.6 | 23.9 | — | — |
| from peak (6) | `K + A·exp(−λt)` | 3 | 0.0752 | −2425.7 | **0** | **0** | −0.0079 | [−0.0109, −0.0050] |

AIC and BIC both prefer the decay-to-constant model, decisively. **But the K it
fits is negative**, with an interval excluding zero, while the observed tail is
positive. A single exponential cannot describe a series that rises to a peak at
epoch 6 and then decays — the constant absorbs the misfit on the rising side
rather than estimating an asymptote. The model comparison therefore answers
"which of these two curves fits better", not "what is the asymptote", and it is
reported for completeness rather than as the answer. The non-parametric tail mean
is the number in §3.1's first paragraph.

### §3.2 Is 100 epochs enough for the NCF arm to be stationary?

No. At epoch 99 the gap is +0.024, which is 55× its value at epoch 299. A
100-epoch run of this model is measuring a transient.

At 300 epochs, however, the arm **is** stationary:

| test | statistic | p |
|---|---|---|
| epochs 290–299 vs 280–289, paired t across seeds | +0.00010 | 0.324 |
| same, Wilcoxon signed-rank | — | 0.313 |
| seeds with the late window higher | 3 of 5 | — |
| largest single step in epochs 290–299 | 0.00041 (epoch 297) | — |
| that step as a share of the final-epoch seed sd | 3.7% | — |
| mean absolute step over the last 10 epochs | 0.00018 | — |

The largest epoch-to-epoch move in the last ten epochs is 3.7% of the spread
across populations at the final epoch. The trajectory has stopped resolving above
the seed noise, so 300 epochs is enough for this configuration.

### What a reader may conclude

- THESIS H3's "cold-start effect that relaxes" is confirmed and stronger than
  stated: the gap does not merely shrink, it disappears.
- The 100-epoch horizon understates the convergence by a factor of 55 and should
  be quoted as a horizon, never as a steady state.
- The convergence is to zero *at this configuration*. §4 shows one cell
  (`shock_scale` 0.5) where the mechanism H3 offers for the convergence — the
  filter learning coverage — is not available, and the same question has not been
  run to 300 epochs there.

---

## §4. Does any of it survive a different point in parameter space?

Sources: `data/processed/rigor/sensitivity.csv`, `sensitivity/README.md`,
`sensitivity/delta_power.csv`. Twelve cells: four knobs at three levels each,
N = 1000, k = 20, 5 seeds, 100 epochs, the shipped cell repeated across three
axes and identical there.

Bootstrap seed 20261024 throughout. All thresholds below were fixed before the
cells were read; they are stated so a reader can disagree with them.

### §4.1 Claim-by-claim

| claim | test | holds |
|---|---|---|
| **C1** Gini(NCF) > Gini(oracle) | paired gap's 95% interval excludes zero, positive | **12 / 12** |
| **C2** the gap is small by epoch 100 | point estimate ≤ 0.05 | **12 / 12** |
| **C3** the NCF reaches most agents | reach ≥ 0.75 | **11 / 12** |
| **C4** clearing is not degenerate | mcr's interval excludes zero | **12 / 12** |

**C1 never reverses.** This is the robust result of the sweep. The smallest gap is
`shock_scale` 0.5 at +0.0097 [+0.0062, +0.0116]; the largest is `shock_scale` 3.0
at +0.0470 [+0.0192, +0.0665]. A factor of five in size, no change in sign.

**C2 comes closest at `shock_scale` 3.0**, +0.0470, where the upper bound reaches
0.0665. At that noise level the cold-start story is at its weakest in this grid.

**C3 fails at `shock_scale` 0.5**, and that is the one reversal:

| shock_scale | gap | reach | mcr |
|---|---|---|---|
| 0.5 | +0.0097 [+0.0062, +0.0116] | **0.4616** | 0.0916 |
| 1.5 (shipped) | +0.0260 [+0.0164, +0.0348] | 0.9566 | 0.2380 |
| 3.0 | +0.0470 [+0.0192, +0.0665] | 0.9976 | 0.3512 |

At `shock_scale` 0.5 the filter never learns coverage. The gap is small there
too — smaller than at the shipped setting — but for the opposite reason: both
arms concentrate, and the NCF has not caught up. THESIS H3 attributes the gap's
closure to reach climbing from 4% to 96%. In this cell reach does not climb, so
the explanation does not apply, and the same gap number would be reached by a
different route. **A conclusion reverses here: "the gap is a cold-start effect
that relaxes as the filter learns who it can represent" is conditional on
`shock_scale` being high enough for the filter to learn.**

### §4.2 The β/α axis, where the documented guardrail does not hold

`experiments/base.yaml` states, and `tests/cpp/test_delta_power.cpp` pins, that
δ has power only when β/α ≥ 2. The task for this section was to say plainly that
the hypergamy results are conditional on β/α ≥ 2. **The measurement does not
support that statement**, and saying so is the finding.

The overlap table in the C++ test — the fraction of a woman's top-20 retained
when δ moves 0 → 1 — reproduces exactly at the settings it was measured at
(`taste_sigma` 0, `shock_scale` 0, N = 2000): 0.9373 / 0.7615 / 0.7365 against
the documented 0.9374 / 0.7615 / 0.7366. The test is correct.

The sweep runs at `taste_sigma` 0.5 and `shock_scale` 1.5. Re-measured there
(`delta_power.csv`, N = 1000, δ = 1, top-20):

| β/α | overlap at taste_sigma 0, shock 0 | overlap at the swept settings | δ has power? |
|---|---|---|---|
| 1.0 | 0.9374 | **0.8440** | yes |
| 2.0 | 0.7615 | **0.8540** | yes |
| 3.0 | 0.7366 | **0.8674** | yes |

The ordering **reverses**. At the shipped heterogeneity and chemistry noise,
β/α = 1 gives δ *more* power than β/α = 3, and all three clear the 0.75 the
guardrail treats as meaningful. The rule of thumb describes a noise-free
homogeneous population, not the one the experiments run on. This is not a bug in
the test, which measures what it says; it is a statement about the shipped
configuration that the config file does not make.

**The condition as it can actually be stated.** δ moves no headline in this grid:
the gap is +0.0229 at δ = 0, +0.0260 at 0.5 and +0.0263 at 1.0, a spread of
0.0034 against intervals about 0.012 wide. β/α moves it three times as much
(0.0116 across its levels), and `shock_scale` ten times as much (0.0373). So the
defensible claim is not "the hypergamy results require β/α ≥ 2" but **"no
hypergamy claim in this work rests on δ, at any β/α tested; the δ axis is
flat."**

### §4.3 What a reader may conclude

- The NCF-versus-oracle ordering is not an artifact of the shipped corner: it
  holds in every cell and reverses nowhere.
- Its size is not stable. Quote +0.010 to +0.047, or quote the shipped cell with
  its regime; do not quote +0.025 as "the" gap.
- One mechanism claim is conditional: the cold-start explanation requires the
  filter to learn coverage, and at `shock_scale` 0.5 it does not.
- The β/α condition as written in `experiments/base.yaml` is not supported at the
  shipped settings. This contradiction is recorded here rather than silently
  resolving either side; `experiments/base.yaml` and
  `tests/cpp/test_delta_power.cpp` were not edited, as `docs/THESIS.md` and
  `src/cpp/` are owned elsewhere tonight.

---

## §5. Multiple comparisons on the factorial grid

Sources: `data/processed/rigor/multiple_comparisons.csv`,
`sensitivity_summary.csv`. `data/processed/institutional/grid.csv`.

### §5.1 The family

The grid holds **200 rows**, which are **40 distinct cells**
(10 size_fraction × 4 signal_cost) × **5 seeds**. A cell is the unit that was
selected over; its five seeds are replicates inside it, not five more chances to
win. The family is therefore **40**.

Family-wise error rate at α = 0.05 uncorrected, if the 40 tests were independent:
**1 − 0.95⁴⁰ = 0.871**. The cells are not independent — adjacent size_fractions
share populations and the same five seeds are reused across all 40 — so the
realised rate is lower than this and is not claimed. The number is the nominal
upper bound and is labelled as such.

### §5.2 What survives

| correction | rejects, of 40 |
|---|---|
| Benjamini–Hochberg, q ≤ 0.05 | **28** |
| Holm, α = 0.05 | **19** |

Seven cells have a positive mean (B beats A). Only two of them survive BH:

| cell (c, size_fraction) | mean | sd | p | q (BH) | Holm | rank by mean |
|---|---|---|---|---|---|---|
| **0.01, 0.65** | **+64.62** | 9.86 | 1.3e-4 | **3.9e-4** | reject | 1 |
| 0.01, 0.50 | +58.49 | 36.66 | 0.018 | 0.027 | — | 2 |
| 0.01, 0.80 | +36.73 | 47.21 | 0.16 | 0.20 | — | 3 |
| 0.01, 0.25 | +24.58 | 51.04 | 0.34 | 0.42 | — | 4 |
| 0.01, 0.35 | +5.43 | 52.04 | 0.83 | 0.88 | — | 5 |
| 0.01, 0.02 | +0.69 | 5.72 | 0.80 | 0.88 | — | 6 |
| 0.05, 0.02 | +0.23 | 5.76 | 0.93 | 0.93 | — | 7 |

### §5.3 Does the optimum survive?

**Yes, decisively.** The cell at (c = 0.01, size_fraction = 0.65) ranks first of
40 by mean, its one-sample t against zero is 14.65 on 4 df (p = 1.3e-4), and its
BH q-value is 3.9e-4. Holm rejects it too.

Its interval after correction for having been chosen as the maximum of 40:

| interval | lo | hi |
|---|---|---|
| unadjusted BCa | +57.47 | +73.17 |
| Bonferroni BCa (40 cells) | +55.07 | +78.46 |
| Bonferroni t | +28.80 | +100.43 |

The point estimate does not move — BH adjusts decisions, not estimates. The BCa
interval widens by about 13% to hold simultaneously across the family, which is
cheap because the effect is eleven standard errors wide. The t interval is much
wider because at five seeds it is the honest floor (§1), and it is the one to
read if a reader wants the conservative bound.

**The narrower finding.** That the optimum survives is not the same as the
optimum being a plateau. Only two cells where B beats A survive correction, at
size_fraction 0.65 and 0.50. The other five positive cells — 0.80, 0.35, 0.25,
0.02 — do not. So H4's interior optimum is, at this seed count, a **two-cell
feature**, not a broad region, and THESIS H4's "optimum at size_fraction 0.65"
should be read with that width attached. The curve's shape beyond those two cells
is not distinguishable from zero.

There is a second reason for caution that the correction does not address: the
optimum's sd (9.86) is the 9th smallest of the 40, against a median of 35.7. A
cell that is both the highest mean and unusually low-variance is a cell where
five seeds happened to agree; that is a property of the sample, not of the
design, and it is why the t interval above is the one to prefer.

### §5.4 The H5 interaction term

Family: the three terms of the decomposition (`exogenous`, `algorithmic`,
`interaction`), BH over three, five seeds each.

| term | mean | sd | t | p | q (BH) | rejected? | 95% BCa interval |
|---|---|---|---|---|---|---|---|
| exogenous | −21.10 | 25.31 | −1.86 | 0.136 | 0.204 | **no** | [−41.13, −3.42] |
| algorithmic | −717.61 | 92.58 | −17.33 | 6.5e-5 | 2.0e-4 | **yes** | [−795.77, −651.37] |
| **interaction (H5)** | **+16.70** | 40.37 | 0.93 | 0.407 | 0.407 | **no** | [−13.30, +53.05] |

**The interaction does not survive, and it does not need to.** H5's own statement
is that the interaction may be near zero and that reporting that is the
measurement. +16.7 [−13.3, +53.0] is indistinguishable from zero, exactly as
THESIS H5 anticipated, and the interval is reported whatever its sign.

The exogenous term is the one where the methods disagree: its BCa interval
excludes zero ([−41.13, −3.42]) while its t interval ([−52.53, +10.32]) and its
BH q-value (0.204) do not. At five seeds with this skew, the BCa interval is
optimistic (§1) and the honest reading is that the sign is negative and the
magnitude is not resolved.

### What a reader may conclude

- The institutional optimum survives correction for the 200 cells it was chosen
  from, and survives it comfortably.
- "The optimum" is two cells wide at five seeds, not a region. Quote it as a
  location with a narrow support, not as a curve with a maximum.
- The H5 interaction is zero within its interval, which is what H5 said it might
  be.

---

## Summary of what changed, and what did not

| result | before | after |
|---|---|---|
| headline intervals | mean ± sd (spread, not an interval) | 95% t at n=5; BCa at n=20 |
| five seeds adequate? | assumed | yes for Gini/RSI/gap; **no** for algorithmic loss and clearing |
| cold-start gap at 100 epochs | +0.025, horizon unstated | converges to zero by ~250 epochs; 100 is a transient |
| gap at 300 epochs | unknown | +0.0004 [−0.0003, +0.0011]; zero within resolution |
| NCF–oracle ordering across the grid | untested | holds 12/12, size varies 5× |
| cold-start *mechanism* | reach climbs 4%→96% | conditional: fails at shock_scale 0.5 (reach 46%) |
| δ's power below β/α 2 | documented as inert | **not reproduced at shipped settings; ordering reverses** |
| institutional optimum | +64.62 ± 9.86 | survives BH (q=3.9e-4); two cells wide |
| H5 interaction | +16.7 ± 40.4 | +16.7 [−13.3, +53.0], not distinguishable from zero |

## Files

All under `data/processed/rigor/`. See its `README.md` for the full index and the
regime table.

| § | file |
|---|---|
| 1 | `intervals.csv` |
| 2 | `power_analysis.csv`, `power_widths.csv`, `power_conclusions.csv` |
| 3 | `long_horizon/`, `long_horizon_gap_series.csv`, `long_horizon_decay_fits.csv`, `long_horizon_asymptote.csv`, `long_horizon_stationarity.csv` |
| 4 | `sensitivity.csv`, `sensitivity/` |
| 5 | `multiple_comparisons.csv`, `sensitivity_summary.csv` |
