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

All registered at 480cd6c before any ran, except the post-hoc rows, which are
labeled; run from detached worktrees of committed trees, at one thread, each seed
with a provenance record. 331 jobs in the first pass (2026-10-27, 21:14 to 01:15),
then the 50 R2-E16 reruns, 20 R2-E13b-check jobs and 10 R2-E15s jobs (2026-10-28).

| experiment | status | result that changed a claim |
|---|---|---|
| R2-E9 canonical 300 rounds, 40 seeds | done | replaces every archived horizon number; all 40 seeds enter the margin |
| R2-E10 1000 rounds | done | exponential decay predicts held-out rounds best; small residual inside the margin, not a floor by the registered rule |
| R2-E11 sweep at 300 rounds | done | re-registered as trajectory claims; C2t fails at chemistry 0.5 and 3.0; negative tails at chemistry 0.5 and β/α 3 |
| R2-E12 diagnostics | done | registered rank-correlation prediction failed in sign; the gap tracks shortlist overlap and the breakup of the common list |
| R2-E13a dose | done | the transient's length is set by optimizer steps |
| R2-E13b exploration | done; comparator check (post hoc, b8e34e4) done | registered "no difference" failed; against a same-length head the round-299 excess is +0.0024 [-0.0017, +0.0065], so it was the comparator's, while the faster closing at rounds 49 and 99 survives |
| R2-E13c off-policy | done | feedback through thresholds is real and small; registered prediction failed |
| R2-E13d warm start, 40 seeds | done | about four fifths of the decay removed, on both sets |
| R2-E14 arrivals | done; trend check (post hoc, d91a312) done | the gap has not closed by round 600 at any rate; the registered stationarity check fails at every rate, and the trend over rounds 400 to 599 is -0.0029, +0.0040 and +0.0196 (every interval excludes zero), so the levels are readings at a horizon, not steady states |
| R2-E15 institutional, rounds 250 to 299 | done | B beats both A arms at the lowest cost: reverses the first revision's "refuted at the shipped gate", which read five rounds |
| R2-E15s five-round grid (post hoc) | done; registered d91a312, run at ec0eb83 | over rounds 0 to 4 no cell has B ahead of the oracle A, and A wins after correction in 28 of 40; at size 0.65 and c_B = 0.01 the difference is -9.9 [-25.9, +6.1], against +58.3 over rounds 250 to 299 |
| R2-E15i inert gate | done | size 0.65 ranks first, as registered |
| R2-E16 floating point | done after a rerun (R2-E16a) | within-seed spread reaches the between-seed spread |
| R2-E17 population size | done | transient not monotone in N; registered "peak rises with N" failed |
| R2-E18 data units | done (`horizon/units.csv`, the paper's units table) | neither data nor steps make the length transfer across N; a warm start's steps on one log are worth less than steps on fresh logs (62 against 87 rounds, but 2,642 against 2,131 steps) |
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
| figures | done: nine figures rendered from the committed CSVs, each checked as an image; byte-identical when rebuilt from unchanged data | `paper/figures.py` |
| style pass | done as a pattern scan of the TeX sources for British spellings, em dashes, stock phrases, repeated words and process traces; fixed what it found (one British spelling, six sentences about the revision process). No dictionary spell checker is installed, so this is not a full spell check | |
| front matter: author, date, code and data, scope | done; the paper cites the Zenodo concept DOI, which resolves to the newest version once release v2.0.0 is published | `paper/main.tex` |

## Phase 7: verification

| item | done | where |
|---|---|---|
| 7a every number traces to a CSV | yes: every quoted number is a macro written by `paper/gen_numbers.py` (225 macros, each block headed by its source files), which raises when a prose claim stops matching its file; a test renders it afresh against the committed copy, recomputes the headline numbers from the raw seed files, and checks that every macro the prose uses is defined | `tests/python/test_paper_numbers.py` |
| 7b one command from a clean checkout | yes: `python paper/build.py` in a detached worktree of 67d16b6 regenerates the numbers, tables and figures byte-identical to the committed files, the only file it changes being `main.pdf` itself, and compiles 27 pages with no overfull box and no undefined reference | README, "Building the paper" |
| 7c hand arithmetic check | yes, and then recomputed by script from the same CSVs: every printed value reproduces to its last digit. The one apparent disagreement was a unit slip in the check itself, a share of 0.2075 compared against the printed 20.8 percent, now noted there | `docs/ARITHMETIC_CHECK.md` |
| 7d cross-references | yes: 49 labels, none duplicated; 77 references, none to a missing label; 36 cited keys, all in `refs.bib`; nine figures included, none unused | |
| 7e references against DOIs | yes: 47 entries with a DOI match their Crossref record on title, year and first author; the 8 without one (PMLR, NeurIPS, JASSS, arXiv) were checked against the page in their `url` field. The checker's surname comparison dropped non-ASCII letters, which made Almås a false mismatch; it now folds accents on both sides | `paper/check_refs.py`, `paper/refs_check.csv` |
| 7f two referee passes on the PDF alone | pass 1 done (a fresh agent given only the PDF; its first attempt died on a provider error and was rerun): **59/100, major revision**. Methods 16/30, novelty 9/20, evidence 9/20, clarity 11/15, reproducibility 14/15. Its framing point is correct and is now stated in the abstract, introduction and model: once the gate stops binding the ranker is trained on the oracle's own top-k lists, so the gap is the convergence rate of an imitator. Also fixed: "indistinguishable from zero" (the long residual is +0.00046, above zero and inside the margin); the peak gap described as a near closed form; the dose claim scoped to fresh logs at fixed N; the warm-start range; the institutional sign change shown to depend on the gate; the random and popularity arms, which coincide with the oracle at round 0 only; the open market's ordering explained as a tenure ratio; two missing references added after Crossref checks. Not fixed, and why: a rate below 0.5 percent per round and a within-list rank measurement would each need new runs; the 45 cell-level tests without multiplicity control, the AIC/BIC on an autocorrelated mean series and the verification table's tolerances are left as stated limitations. Two of its points were wrong (Figure 2a is not clipped; the 4-round figure is read on the steep part of the curve). Pass 2 (a second fresh agent, PDF alone): **64/100, major revision, no fatal issue**. Fixed: the conclusion stated the optimizer-steps rule without its conditions and asserted the within-list-order explanation of the welfare gap as fact, and both now carry their conditions; the abstract's inert-gate clause claimed more than the five-round run shows and now says the five-round comparison is positive; a limitations paragraph lists what was not varied (optimizer settings, the margin as a convention, the status bonus never removed, size and gate target moving together, uncorrected sweep tests, within-list order); process sentences in the supplement and introduction were rewritten. Not fixed, and why: a lock-in arm whose log comes from the ranker's own shortlists, a sweep of the optimizer settings and k, a run with the status bonus removed, an inert-gate run at the late horizon, and an ablation of B's defection penalty would each be new registered experiments. The two passes scored 59 and 64; the scores come from a model reading a PDF and are not a measure of acceptability | |
| 7g this report | yes | |
| 7h arXiv source | yes: `paper/arxiv-source.tar.gz` carries exactly the files `main.tex` uses plus the generated `main.bbl`; extracted into an empty directory, it compiles with TeX alone and no BibTeX, 27 pages, no undefined reference or citation; a test fails if it is stale against the sources | `paper/make_arxiv.py`, `tests/python/test_arxiv_source.py` |

## Registered predictions that failed

Recorded in the paper, not repaired: R2-E11 C5t (a reversal also at β/α 3);
R2-E12 reach at round 0, the correlation sign, the item share by round 100;
R2-E13b; R2-E13c; R2-E13d round-0 reach on one seed; R2-E14 newcomers at 0.5%;
R2-E15 (B beats the oracle A); R2-E17 peak with N. Of the 39 scored predictions,
29 held and 10 failed; the scored table is
`data/processed/horizon/predictions.csv`, printed as the paper's predictions table.

## Where the brief was wrong

- C1's "epoch-0 basis, 33.5%" is on levels; the registration's prediction was on
  the gap.
- E13c assumes the ranker trains on its own shortlists. The training log comes
  from true utilities; the shortlists reach it only through thresholds.
- At 300 rounds δ moves the peak gap by about its own interval width, not an
  eighth of it; still far less than chemistry noise does.
