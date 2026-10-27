# Compute plan: second revision

Written 2026-10-27, before any experiment of the second revision was run. Every
estimate below is built from timings measured on this machine that day. Where a
run type was not timed directly, the line says which measurement it is scaled
from.

## Machine

Intel Core Ultra 9 275HX, 24 cores (no simultaneous multithreading), 32 GB RAM,
Windows 11. Every run is at `ncf.num_threads = 1` and `OMP_NUM_THREADS = 1`.

## Measurements

One seed (42), shipped configuration, `N = 1000`, 1 thread, idle machine:

| run | ncf | oracle | random | popularity | four arms |
|---|---|---|---|---|---|
| 100 rounds | 85.0 s | 7.1 s | 10.0 s | 4.9 s | 107 s |
| 300 rounds | 247.9 s | 21.1 s | 23.9 s | 12.7 s | 306 s |

Marginal cost per round, from the difference of the two: NCF 0.81 s, oracle
0.070 s, random 0.070 s, popularity 0.039 s. One NCF round splits into swipe
generation 0.02 s, one training pass 0.38 s and scoring both sides 0.22 s, so a
round at training dose d costs about 0.43 + 0.38 d seconds.

The R2-E12 diagnostics add 0.28 s per NCF round (6.9 s against 5.2 s over six
rounds, same seed).

Other run types, ten rounds each:

| run | seconds per round |
|---|---|
| NCF at `N = 500` | 0.43 |
| NCF at `N = 2000` | 2.16 |
| oracle at `N = 2000` | 0.24 |
| oracle A plus Environment B, paired, `size_fraction` 0.25 and 1.0 | 0.29 and 0.13 |

**Concurrency.** 22 NCF runs of 20 rounds started together each took 51 to 53 s,
against about 20 s alone. Each process slows by about 2.6 times, and the machine
delivers about 8.5 single-process seconds per second of wall time. Every estimate
below divides single-process seconds by 8.5.

The first attempt at that measurement read 13,634 s per run, because the machine
slept for 3 h 47 min while it ran: all 22 processes finished within three seconds
of one another, at the moment it woke. A keep-awake guard
(`SetThreadExecutionState`; no power setting is changed) now runs for as long as
the revision's jobs do.

## Estimates

Single-process seconds, then wall time at 8.5 times that rate. Seed sets:
exploratory 42 to 61, confirmation 100 to 119.

### Tier 1

| experiment | design | single-process | wall |
|---|---|---|---|
| R2-E9 | 40 seeds x 300 rounds, four arms, diagnostics on the NCF arm (1.09 s/round) | 27,800 s | 0.91 h |
| R2-E11 | 8 off-shipped cells x 10 seeds x 300 rounds, NCF and oracle (shipped cell from R2-E9) | 21,100 s | 0.69 h |
| R2-E13d | warm start, 40 seeds x 300 rounds, NCF only (controls from R2-E9) | 9,900 s | 0.32 h |
| R2-E12 | computed inside R2-E9 | included | |
| **total** | | **58,800 s** | **1.9 h** |

Tier 1 is well inside the 8-hour ceiling, so no Tier 1 sample size is cut. R2-E11
runs at ten seeds per cell, the brief's preferred count.

### Tier 2

| experiment | design | single-process | wall |
|---|---|---|---|
| R2-E10 | 10 seeds (100 to 109) x 1000 rounds, NCF and oracle | 10,100 s | 0.33 h |
| R2-E13a | doses 0.5, 2 and 4 x 10 seeds x 300 rounds (dose 1 from R2-E9) | 11,300 s | 0.37 h |
| R2-E13b | exploration 0.05 and 0.2 x 10 seeds x 300 rounds, plus oracle prefix runs | 14,000 s | 0.46 h |
| R2-E13c | shadow ranker on oracle and random data, 10 seeds x 300 rounds | 5,600 s | 0.18 h |
| R2-E15 | 40 cells x 10 seeds x 300 rounds, oracle A and NCF A | 15,000 s | 0.49 h |
| R2-E15i, R2-H1r, R2-H5r | inert-gate grid check, H1 curve, H5 factorial, shipped gate | 1,800 s | 0.06 h |
| **total** | | **57,800 s** | **1.9 h** |

### Tier 3

| experiment | design | single-process | wall |
|---|---|---|---|
| R2-E14 | arrivals at 0.005, 0.01 and 0.02 x 10 seeds x 600 rounds, NCF and oracle | 19,000 s | 0.62 h |
| R2-E16 | 5 seeds x 10 perturbed runs x 300 rounds, NCF | 12,200 s | 0.40 h |
| R2-E17 | `N` = 500 and 2000 x 10 seeds x 300 rounds, NCF and oracle | 8,600 s | 0.28 h |
| R2-E18 | re-expression of existing runs; no simulation | 0 | 0 |
| **total** | | **39,800 s** | **1.3 h** |

### All tiers

About 156,000 single-process seconds, 5.1 hours of wall time with the queue full,
and about 6 hours once the gaps between jobs at the end of each tier are counted.
The longest single job is an R2-E10 seed at 1000 rounds, about 40 minutes under
load.

## Order and stopping

Jobs run from one queue (`scripts/run_queue.py`) at 22 workers, ordered by tier
and then by listing order. A tier's jobs are added to the queue only after its
registrations are committed. If the work runs out of budget it stops at a tier
boundary, and this file and `docs/REVISION_REPORT.md` say which experiments did
not run.

## Where things are

Each job runs from a git worktree of the commit it was registered at, so a job
cannot import code that was being edited when it started. Output goes to
`data/processed/horizon/<experiment>/seed<seed>.csv` with a provenance record
beside each file (`src/python/provenance.py`).
