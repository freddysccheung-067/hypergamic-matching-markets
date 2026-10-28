# Drafting report

> **Archival.** The report of the first drafting session. The second revision's
> report is [docs/REVISION_REPORT.md](../docs/REVISION_REPORT.md); the tables and
> figures this file names were replaced and are no longer in `paper/`.

Branch `paper/draft`, created from `main` with `rigor/statistical-hardening`
merged in (`--no-ff`). `lit/positioning` was already an ancestor of that line —
its tip was `main` itself — so the second merge was a no-op and made no commit.
That is worth knowing before the history is read: the literature documents
(`LITERATURE.md`, `NOVELTY.md`, `RELATED_WORK.md`) arrived with the rigor
branch.

Nothing under `src/`, `data/` or the existing `docs/` was modified. This
session created `paper/` and nothing else.

## 1. Page count and word count per section

**The document is uncompiled.** No TeX distribution is installed on this
machine — `pdflatex`, `xelatex`, `tectonic` and `latexmk` are all absent from
`PATH`, and the brief forbids installing one. There is therefore no page count
and no typeset word count. Everything below is a source-level count.

| section | words | target | delta |
|---|---|---|---|
| §1 Introduction | 1,009 | 800 | +209 |
| §2 Related work | 1,138 | 1,000 | +138 |
| §3 Model | 1,678 | 1,500 | +178 |
| §4 Design | 1,128 | 800 | +328 |
| §5 Results | 2,971 | 2,000 | +971 |
| §6 Limitations | 951 | 600 | +351 |
| §7 Conclusion | 354 | 300 | +54 |
| **total** | **9,229** | **7,000** | **+2,229** |
| abstract | 190 | ≤200 | — |

Counts are running prose only: LaTeX markup stripped, float bodies and captions
excluded, inline math symbols counted. Produced by `paper/wordcount.py`; the
abstract figure comes from `paper/wordcount.py --abstract`, which additionally
drops inline math, so it reads 190 where a plain whitespace count of the same
passage reads 199.

The abstract was cut from 210 to 190 by deleting words rather than by adding a
caveat: "finite" came off the market description, "because" became a colon so the
clause stopped re-explaining the opening claim, "that is not a floor" became "no
floor", and "where it stops being distinguishable from flat" became
"indistinguishable from flat". Every number is unchanged.

**The length question, settled.** The draft was written to per-section targets
that summed to 7,000 words against a stated goal of 8–10 pages; as the earlier
version of this report noted, those two constraints are not simultaneously
satisfiable at 11pt single column, and the draft has since grown to 9,229 words
because the review response added sourced passages that the review asked for.

That is the right trade for a working manuscript and the wrong shape for a
submission. Nothing is trimmed here, because trimming is a decision about what
the paper is arguing, which is the author's. Before submission: 9,229 words of
body text sets to roughly 16–17 pages, and no venue this project can reach wants
that. The realistic cut is to about 6,000 words — §5 loses the per-hypothesis
recap in favour of the table, §3 moves the derivation detail to an appendix, and
§2 keeps one citation per claim rather than two. None of that removes a number
that traces to a committed file.

## 2. Numbers I could not source, and therefore did not write

The rule was: a number needs both a file under `data/processed/` and a section of
`docs/STATS.md`. Most numbers met it. These did not, and the sentences were
rewritten without them.

**H1 has no registered interval.** `data/processed/separating/h1.csv` exists and
carries `mean_tau` per (cost, seed), five seeds over eleven cost levels, but
`docs/STATS.md` has no section for H1 and no interval is registered. The paper
therefore reports H1's direction qualitatively and states that no interval is
quoted. **A reader can compute an interval from `h1.csv`; the drafting session
chose not to, because computing a new statistic during drafting would put a
number in the paper with no STATS.md section behind it.** The numbers I could
have written and deliberately did not: mean received intent of 0.5024 at
`c = 0` rising to 0.6846 at `c = 2.0`, with seed sd between 0.0013 and 0.0090.

**H2 has no interval either.** `data/processed/flooding/` has coefficients with
bootstrap-free standard errors, and `docs/THESIS.md` reports slopes with `±`
values drawn from them. `docs/STATS.md` registers nothing for H2. The subsection
reports the mechanism's absence — which needs no number — and describes the
hand-added flooding term's behaviour without quoting a coefficient.

**H6 has no committed artefact at all.** No file under `data/processed/`
contains an attrition or intent-correlation result. §5.7 states this and quotes
no number. `docs/THESIS.md`'s figure (correlation ≈ 0.85 at `c = 2`) comes from
`THEORY.md` §10.8, which is not `STATS.md` and is not a `data/processed/` file,
so it was not written.

**The validation table is the one exception, and it is flagged in the paper.**
`docs/validation.md` quotes 0.2433 / 0.3367 / 0.2200 / 0.2833 / 0.1800, 0.2519,
0.7254, 0.7433 and the pair counts 22/8/11/12/8. There is **no CSV under
`data/processed/` holding any of them** — they are asserted in
`tests/python/test_validation.py` and transcribed in the doc. The brief requires
the validation table from `docs/validation.md`, so I included it and said so in
its caption and in §4.5: it is marked as the only table in the paper that does
not trace to a committed artefact. **This is the one place where I followed the
"include this table" instruction over the "every number needs a source"
instruction, and the paper discloses the conflict rather than hiding it.**

**Two numbers in `docs/RELATED_WORK.md` were dropped for a different reason.**
It cites Chiappori & Salanié (2016) and Koren, Bell & Volinsky (2009), neither of
which appears in `docs/LITERATURE.md`. Since the brief forbids introducing a
source not in that file, both were removed from §2 along with the sentences that
carried them. The related-work section is still complete without them; the
prose around each was rewritten rather than deleted.

**One internal inconsistency worth reporting.** `docs/RELATED_WORK.md` says the
exposure gap is "+0.287 at epoch 19 and +0.025 at epoch 99". The committed series
(`long_horizon_gap_series.csv`) gives +0.286878 at epoch 19 and +0.024145 at epoch
99. Both round to the doc's figures, so nothing is wrong, but the paper quotes
the committed values at four decimal places rather than the doc's rounded ones.

## 3. Did it compile?

**No.** No TeX engine is installed. The error is not a LaTeX error; there is no
binary to produce one.

What was done instead, per the brief, is a structural check
(`paper/verify.py`), which passes:

- every `\input` path resolves (10 inputs);
- every `\ref` has a `\label` (18 distinct labels, 0 missing);
- every `\cite` key has a `refs.bib` entry (35 cited, 0 missing);
- every `\includegraphics` target exists (4 figures);
- no duplicate labels;
- brace balance is zero in all 11 files.

Additional checks the brief did not ask for but that a compile would have
caught: unused labels are reported (8, all intentional — section labels and the
`eq:tau-star` equation, which is referenced by name in prose rather than by
`\ref`), and uncited bib entries are reported (0 after revision).

**What this cannot catch.** Package-level errors, float placement, overfull
hbox, a `tabular` whose column count disagrees with its body, and anything that
depends on font metrics. The three generated tables avoid `booktabs` and use
only `tabular`/`hline` for that reason, but they have not been rendered.

## 4. The three titles

1. **Exposure Concentration Under a Learned Ranker Is a Cold-Start Effect: A
   Convergence Test in a Simulated Matching Market** — set in the document.
   Claims the finding and names the method. The word "simulated" is doing real
   work: it prevents the title from being read as a claim about platforms.
2. **Horizon Dependence in Recommender-Mediated Matching: Exposure
   Concentration, Institutional Comparison, and What Five Seeds Support** —
   broader, and the last clause advertises the paper's main weakness rather than
   hiding it.
3. **How Long Does a Recommender's Exposure Gap Last? Evidence from a
   Three-Hundred-Epoch Matching-Market Simulation** — the most concrete and the
   most modest; a question title, which some venues dislike.

None claims more than the design supports. All three avoid "solved", "proves",
and "causes". Title 1 is the most defensible; title 3 is the most honest about
scope.

## 5. The single weakest claim, and why

**That the fit comparison can be dismissed while the tail mean is trusted, in
§5.4.**

The paper reports that AIC and BIC decisively prefer a decay-to-constant model,
then sets that comparison aside because the fitted constant is negative, and
quotes the non-parametric tail mean instead. The reasoning is sound as far as it
goes — a single exponential genuinely cannot represent a series that rises before
it decays, so the constant is absorbing misfit rather than estimating an
asymptote. But the consequence is that the paper's strongest single finding rests
on a quantity with no model behind it: a mean over fifty epochs of five seed
trajectories, whose own $t$ interval has a lower bound of $+0.000013$, three
orders of magnitude below the point estimate, and whose Wilcoxon test gives
$p = 0.063$.

The specific vulnerability is this. The paper says the gap "converges to zero",
and it says the residual is "below 0.001 and unresolved at five seeds". Both
sentences are supported. But "converges to zero" is a statement about a limit,
and the only evidence for the limit is that the last fifty epochs are flat and
small. A flat tail is also what a very slow decay looks like, and the paper's own
stationarity test — epochs 290–299 against 280–289, $p = 0.324$ — cannot
distinguish "stationary at zero" from "still falling at a rate too small to
detect in ten epochs with five seeds". The paper would need either a longer run
or a correctly specified functional form to close that gap, and it has neither.

I would expect a referee to concede the empirical point — the gap is small and
flat — and reject the inferential one — that it has converged. The honest
formulation is "the gap is below 0.001 by epoch 250 and no longer distinguishable
from flat", which is weaker than the abstract's "converges". The abstract and
§5.4 currently use the stronger word.

## 6. What was produced

```
paper/
  main.tex                   article, 11pt, natbib + hyperref, abstract, title
  refs.bib                   35 entries, all from docs/LITERATURE.md
  tables.py                  generates tables/*.tex from data/processed/
  verify.py                  structural checks (all pass)
  wordcount.py               per-section counts
  sections/                  intro, related, model, design, results, limits, conclusion
  tables/                    ledger.tex, sensitivity.tex, validation.tex (generated)
  figures/                   4 PNGs copied from docs/figures/
```

Four figures, each with a comment in its caption saying why it earned its place:
`arm_comparison` (H3 against both controls), `separating_curve` (the only H1
artefact, and it shows the mechanism's discrepancy), `thin_market_curve` (all
three elements of the H4 verdict in one frame), `decomposition` (the H5
arithmetic and the scale difference). `gini_evolution` was considered and
rejected: it duplicates `arm_comparison`'s final-epoch numbers, so it would have
spent a figure on no new information.

The paper is a complete draft, not a finished paper. It has not been compiled, it
has not been read by anyone else, and two of its sections exceed their specified
length.
