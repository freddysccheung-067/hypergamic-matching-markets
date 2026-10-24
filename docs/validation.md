# Validation against Bilderbeek (2026)

Reference: E. Bilderbeek, *Hypergamy Is Solved*, 23 October 2026. The paper's results are
closed-form and proved, so a disagreement between it and this project would be a bug here.
None was found. The checks live in `tests/python/test_validation.py`, one test per check;
run with `pytest tests/python/test_validation.py -s` to print the numbers below.

Two different tools are used because the paper states two different kinds of result.
Theorems 2.2 and 3.3 concern the largest matching under a compatibility constraint
(`s_m ≥ s_w`, or `s_m ≥ s_w + δ`). That is a maximum-cardinality bipartite matching problem,
solved here with `scipy.sparse.csgraph.maximum_bipartite_matching`. It is not a stability
question and `deferred_acceptance` does not compute it. Theorem 5.1 is the stability result,
and that is the Gale–Shapley engine.

## 1. Theorem 2.2: the deficit D

`D = sup_t (G_W(t) − G_M(t))₊` is the minimal mass of W unmatched when every union must satisfy
`s_m ≥ s_w`.

**Setup.** N = 300 per side, M ~ N(0, 1), W ~ N(0.6, 1.3²). The variances differ so the survival
functions cross and D > 0. Measured deficit is the unmatched-W count from the exact maximum
matching divided by N; predicted is the supremum over the empirical survival functions.

| seed | matching | sup (G_W − G_M)₊ | difference |
|---|---|---|---|
| 0 | 0.2433 | 0.2433 | 0 |
| 1 | 0.3367 | 0.3367 | 0 |
| 2 | 0.2200 | 0.2200 | 0 |
| 3 | 0.2833 | 0.2833 | 0 |
| 4 | 0.1800 | 0.1800 | 0 |

The two agree exactly on every seed, inside the 1/N tolerance asserted. This is what the
theorem says: on a sample the identity is exact, not asymptotic.

**Example 2.4.** M ~ Unif[0, 1], W ~ Beta(2, 1), N = 20,000, one seed. Measured 0.2519
against the predicted 1/4 (tolerance 0.02). The exact matching again equals the empirical
supremum to within one agent. The 0.0019 excess over 1/4 is sampling error in the
distributions, not matching error.

## 2. Theorem 3.3: the height deficit

`D(δ) = (2Φ((δ − Δm) / 2σ) − 1)₊` for equal variances. US calibration m_M = 175.3,
m_W = 161.3, σ = 7.6, so Δm = 14.0. N = 5,000, five seeds, exact matching on
`s_m ≥ s_w + δ`.

| quantity | formula | matching, mean over 5 seeds | seed sd |
|---|---|---|---|
| D(0) | 0 | 0.0000 (0.0000 on every seed) | 0 |
| D(14.0) | 0 | not run (formula is exactly 0 at the boundary) | — |
| D(30.48), σ_M = σ_W = 7.6 | 0.7217 | 0.7254 | 0.0054 |
| D(30.48), σ_M = 7.6, σ_W = 7.1 | 0.7380 | 0.7433 | 0.0053 |

**The 0.74 figure.** The task brief and the paper's abstract quote D(30.48) ≈ 0.74 at the
σ = 7.6 calibration. Theorem 3.3 gives 0.7217 there. The 0.74 belongs to the unequal-variance
population (σ_W = 7.1), which the paper uses for Figure 1 but which Theorem 3.3 does not cover:
its proof assumes σ_M = σ_W. Theorem 2.2 does cover it, and the exact supremum there is 0.7380,
which rounds to the paper's 0.74. So the paper is internally right and the abstract's closed
form is slightly loose. The test asserts both numbers separately rather than letting one stand
for the other.

The matching runs about 0.004–0.005 above the population value in both rows. That is the
expected upward bias of a sample supremum at this N, not a discrepancy; the tolerance is
0.03 per seed and 0.01 on the mean.

## 3. Theorem 5.1: assortative uniqueness

Complete, strict, status-ranked lists: every agent ranks the other side by status alone.
N = 300, five seeds.

- `deferred_acceptance` with M proposing and with W proposing returned the **same matching on
  every seed**, with zero blocking pairs in both directions.
- That matching equals the pairing of the k-th highest M with the k-th highest W, computed
  independently from `argsort(argsort(s))`.

**The precondition is load-bearing.** Adding i.i.d. N(0, 0.01²) noise to each agent's own view
of the other side's status leaves status as the dominant term but makes the lists
agent-specific. Both directions still return stable matchings (zero blocking pairs), yet they
differ. Pairs that differ between the M-proposing and W-proposing result, by seed:
22, 8, 11, 12, 8. So stability survives the perturbation and uniqueness does not. A
perturbation as small as 1% of the status scale is enough, which is the point: the theorem's
conclusion rests on the lists being common, not merely correlated with status.

## What a matching D does and does not establish

Agreement on D establishes that the matching and survival-function machinery in this repository
reproduces the paper's closed forms, and that the deferred-acceptance engine behaves as
Theorem 5.1 requires on its home ground. It establishes nothing about the recommender. The
deficit in Theorems 2.2 and 3.3 is measured under a *compatibility* constraint, `s_m ≥ s_w + δ`,
which is a fixed upper-set structure on a single scalar status, and that structure is exactly
what makes D a one-dimensional supremum. Environment A's constraint is *visibility*: the
recommender's truncated top-k lists, which depend on an estimated, multidimensional utility and
differ by agent. That is a different constraint structure, so a matching D is not evidence that
the recommender's unmatched mass has a closed form. It does not appear to have one, and finding
one is the open problem the paper names in Section 9, where it lists multidimensional status
and search frictions as unsolved. The project's own recommender results (the NCF-versus-oracle
Gini gap in THESIS.md H3) remain simulation results about this model.

## Not covered

Theorems 2.1, 3.4, 4.3, 4.5, 6.1 and 7.1, Proposition 3.1 and Corollary 5.2 were not tested.
The six-foot theorem (3.4) and the stochastic-dominance feasibility condition (2.1) follow from
the same Hall bound as 2.2 and would be cheap to add.
