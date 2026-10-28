# Validation against closed-form results

Each check below is a theorem or an exact identity, so a disagreement with the simulator would be
a bug in this project. None was found. The checks live in `tests/python/test_validation.py`, one
test per check; `scripts/run_validation.py` writes their numbers to
`data/processed/rigor/validation.csv`, which the paper's verification table reads, and a test
fails if that file differs from a fresh computation. Every number in this note is in that file.

Two different tools are used because there are two kinds of result. The deficit checks concern
the largest matching under a *compatibility* constraint (`s_m ≥ s_w + δ`). That is a
maximum-cardinality bipartite matching problem, solved with
`scipy.sparse.csgraph.maximum_bipartite_matching`; it is not a stability question and
`deferred_acceptance` does not compute it. The uniqueness check is a stability result, and that
is the deferred-acceptance engine.

## 1. The deficit identity (Hall's theorem, threshold form)

When every woman's admissible set is an upper set of men (nested neighborhoods), the deficiency
form of Hall's marriage theorem (Hall 1935) says the minimal unmatched share of W is
`D = sup_t (G_W(t) − G_M(t + δ))₊`, with G the survival functions. On a finite sample the
identity is exact.

**Setup.** N = 300 per side, M ~ N(0, 1), W ~ N(0.6, 1.3²), δ = 0. The variances differ so the
survival functions cross and D > 0. Measured: unmatched W from the exact maximum matching,
divided by N. Predicted: the supremum over the empirical survival functions.

| seed | matching | sup (G_W − G_M)₊ |
|---|---|---|
| 0 | 0.2433 | 0.2433 |
| 1 | 0.3367 | 0.3367 |
| 2 | 0.2200 | 0.2200 |
| 3 | 0.2833 | 0.2833 |
| 4 | 0.1800 | 0.1800 |

**A case with a known population value.** M ~ Unif[0, 1], W ~ Beta(2, 1), for which D = 1/4.
N = 20,000, one seed: measured 0.2519; the exact matching equals the empirical supremum to within
one agent, and the 0.0019 excess over 1/4 is sampling error in the distributions.

## 2. The Gaussian closed form

For two normal status distributions with equal variance σ² and mean gap Δm, the supremum is
attained at the midpoint, so `D(δ) = (2Φ((δ − Δm) / 2σ) − 1)₊`. Setup: M ~ N(1, 1), W ~ N(0, 1),
δ = 2, N = 5,000 per side, five seeds.

| case | population value | matching, mean of 5 seeds | seed sd |
|---|---|---|---|
| equal variances | 0.3829 (closed form) | 0.3864 | 0.0102 |
| W ~ N(0, 0.9²) | 0.4031 (exact supremum, no closed form) | 0.4135 | 0.0090 |

Both sit above the population value by about one seed sd. That is the upward bias of a sample
supremum at this N, which is of order N^(-1/2); the per-seed tolerance asserted is 0.03.

## 3. Uniqueness under common rankings

Complete, strict, status-ranked lists: every agent ranks the other side by status alone. With
rankings common to a side, the proposer-optimal and acceptor-optimal stable matchings coincide
(Gale & Shapley 1962), because the lattice of stable matchings collapses to a point (Knuth 1996).
N = 300, five seeds.

- `deferred_acceptance` with M proposing and with W proposing returned the **same matching on
  every seed**, with zero blocking pairs in both directions.
- That matching is the pairing of the k-th highest M with the k-th highest W, computed
  independently from `argsort(argsort(s))`.

**The precondition matters.** Adding N(0, 0.01²) noise to each agent's own view of the other
side's status leaves status as the dominant term but makes the lists agent-specific. Both
directions still return stable matchings, yet they differ. Pairs that differ between the two
directions, by seed: 22, 8, 11, 12, 8. Stability survives the perturbation and uniqueness does
not.

## 4. The popularity closed form

A ranker that gives every agent on a side the same k candidates exposes exactly 2k of N agents,
each equally, so the exposure Gini is 1 − 2k/N and reach is 2k/N. The popularity arm, k = 20, at
N = 500, 1000 and 2000, rounds 1 to 4, three seeds per N: the largest error in either quantity is
2.2e-16.

## 5. Truncated deferred acceptance is stable on its own lists

On the top-k shortlists of a run (NCF and oracle arms, N = 300), an independent blocking-pair
counter finds no pair listed by both sides that prefers each other to their assigned partners:
the matching is stable with respect to the lists it was given. Against full preferences, about
17,900 blocking pairs remain per matching, which is what truncation costs.

## What these checks do and do not establish

They establish that the matching, survival-function and exposure machinery in this repository
reproduces known results on their home ground. They establish nothing about the recommender. The
deficit is measured under a compatibility constraint that is a fixed upper-set structure on a
single scalar, and that structure is what makes the deficit a one-dimensional supremum.
Environment A's constraint is *visibility*: the recommender's truncated top-k lists, which depend
on an estimated multi-attribute utility and differ by agent. No closed form is known for that
setting, and the project's recommender results remain simulation results about this model.
