# Revision report: the second revision

Branch `rev/horizon`, from `main` at 48807de. Every item of the revision brief,
what was done, where, and in which commit. Items whose last step has not run are
marked **pending**; nothing pending is reported as done.

## Phase 0: baseline and compute plan

| item | done | where |
|---|---|---|
| baseline: tests, ruff, verify.py, gen_numbers no diff | yes, all clean at 48807de | |
| ROADMAP and STATS read | yes | |
| timings at one thread, 100 and 300 rounds; 22-way load test | yes | `docs/COMPUTE_PLAN.md` |
| compute plan per experiment and tier | yes; Tier 1 estimated 1.9 h, all tiers 5 to 6 h; measured about 4 h | 480cd6c |

## Phase 1: the arithmetic error (C1)

| item | done | where |
|---|---|---|
| 1a four decay fractions per seed with intervals | yes | `scripts/analyze_warm_start.py`, f8f8940 |
| 1b primary measure stated and registered | yes, gap from round 0, with the order disclosed (it was fixed after the four had been computed) | PREREGISTRATION E3d, 7dfcd8c |
| 1c every inherited reading rewritten | yes: abstract, introduction, results, conclusion, related work, REVIEW_RESPONSE, interventions README | f8f8940 |
| 1d macros renamed by units | yes (`warmLevel*`, `warmGap*`) | f8f8940 |
| 1e unit test: decay fraction from the CSVs equals the macro | yes | `tests/python/test_paper_numbers.py`, f8f8940 |

**What changed.** The draft printed exposure Gini levels as gaps and a level,
peak-basis ratio as the share of decay surviving a warm start. On the registered
measure the informed start removes about four fifths of the decay; the draft said
a third. The reading reversed: the warm start supports the coverage reading. The
brief's own "epoch-0 basis, 33.5%" was the level basis; the registration's
prediction was about the gap, which gives 21.7% on the original five seeds. The
"epoch-0 gaps" of +0.3055 and +0.1079 in older documents were round-0 levels minus
the round-99 oracle level, a third construction.

## Phases 2 to 4: the experiments

All registered at 480cd6c before any ran; run from detached worktrees of committed
trees, at one thread, each seed with a provenance record. 331 jobs in the first
pass (2026-10-27, 21:14 to 01:15).

| experiment | status | result that changed a claim |
|---|---|---|
| R2-E9 canonical 300 rounds, 40 seeds | done | replaces every archived horizon number; all 40 seeds enter the margin |
| R2-E10 1000 rounds | done | exponential decay predicts held-out rounds best; small residual inside the margin, not a floor by the registered rule |
| R2-E11 sweep at 300 rounds | done | re-registered as trajectory claims; C2t fails at chemistry 0.5 and 3.0; negative tails at chemistry 0.5 and β/α 3 |
| R2-E12 diagnostics | done | registered rank-correlation prediction failed in sign; the gap tracks shortlist overlap and the breakup of the common list |
| R2-E13a dose | done | the transient's length is set by optimizer steps |
| R2-E13b exploration | done; comparator check run, **analysis pending** | registered "no difference" failed |
| R2-E13c off-policy | done | feedback through thresholds is real and small; registered prediction failed |
| R2-E13d warm start, 40 seeds | done | about four fifths of the decay removed, on both sets |
| R2-E14 arrivals | done; trend check registered (d91a312), **computation pending** | the gap does not close in an open market |
| R2-E15 institutional, rounds 250 to 299 | done | B beats both A arms at the lowest cost: reverses the first revision's "refuted at the shipped gate", which read five rounds |
| R2-E15s five-round grid (post hoc) | registered d91a312, jobs built ec0eb83, **not run** | |
| R2-E15i inert gate | done | size 0.65 ranks first, as registered |
| R2-E16 floating point | done after a rerun (R2-E16a) | within-seed spread reaches the between-seed spread |
| R2-E17 population size | done | transient not monotone in N; registered "peak rises with N" failed |
| R2-E18 data units | **table pending** (`scripts/analyze_extras.py`) | neither data nor steps make the length transfer across N |
| R2-H1r, R2-H5r | done | H1 holds at the shipped gate; H5 interaction includes zero |

R2-E14 needed no C++ change: replacement in a fixed-capacity slot keeps the state
matrix, embedding rows and swipe seeding valid; `arrival_rate = 0` is tested as a
strict no-op against committed one-thread output.

## Phase 5: specification and consistency

| item | done | where |
|---|---|---|
| S1 to S9: equations, parameter table, ranker, log, round and pass, metrics, turnover, the gate inequality, algorithm box | written | `paper/sections/model.tex`; the parameter table is generated from `experiments/base.yaml` |
| S8 gate inequality: candidate set N/2, no factor-of-two sentence | written | model.tex |
| consistency fixes (section order, one statement of the mechanism, "Supplementary note S#", seed counts, one value per gap, regime names, entry round defined once, Table 1's by-construction row removed, TOST at 90%) | written | the restructured sections |
| claim recalibration (no "inverts"; no regulate/wait dichotomy; the oracle's concentration named as preference-driven; hypergamy one variant; no "thirty-eight sources"; each related paper's specific claim; "no closed form is known") | written | intro, related, limits |

## Phase 6: literature and structure

| item | done | where |
|---|---|---|
| every named reference verified | yes, against Crossref or the publisher's page | `paper/check_refs.py`, `paper/refs_check.csv` |
| Bilderbeek removed from paper and every document | yes; the owner confirmed it is not a real source | 1a80cf2, `docs/UNVERIFIED_REFS.md` |
| restructure around the horizon result | written | `paper/sections/` |
| figures | written; **final render pending** | `paper/figures.py` |
| style pass | written to the brief's rules; **spell check pending** | |
| front matter: author, date, code and data, scope | written; DOI pending a release | `paper/main.tex` |

## Phase 7: verification — pending

The analysis extras, number and table generation, figure render, compile, test
suite, cross-reference check, reviewer passes and arxiv source check have not run.
They are scripted (`python paper/build.py --analysis`). `docs/ARITHMETIC_CHECK.md`
is done by hand from the analysis CSVs.

## Registered predictions that failed

Recorded in the paper, not repaired: R2-E11 C5t (a reversal also at β/α 3);
R2-E12 reach at round 0, the correlation sign, the item share by round 100;
R2-E13b; R2-E13c; R2-E13d round-0 reach on one seed; R2-E14 newcomers at 0.5%;
R2-E15 (B beats the oracle A); R2-E17 peak with N. The scored table is
`data/processed/horizon/predictions.csv` (**pending**).

## Where the brief was wrong

- C1's "epoch-0 basis, 33.5%" is on levels; the registration's prediction was on
  the gap.
- E13c assumes the ranker trains on its own shortlists. The training log comes
  from true utilities; the shortlists reach it only through thresholds.
- At 300 rounds δ moves the peak gap by about its own interval width, not an
  eighth of it; still far less than chemistry noise does.
