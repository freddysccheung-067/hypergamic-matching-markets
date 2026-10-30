# Unverified references

References that could not be verified against a publisher record, a DOI resolver or
arXiv, and are therefore not cited anywhere in the paper.

## A previously cited source that could not be located

**"E. Bilderbeek (2026), *Hypergamy Is Solved: Feasibility, Deficit, and Equilibrium in
Status-Ordered Matching Markets, with an Application to the Six-Foot Rule*",** listed in
earlier drafts as self-published with no DOI. It has no DOI, no arXiv identifier and no
indexed record, it could not be found by web search, and the repository's owner has
confirmed it is not a real source. It was removed from the bibliography, the paper and
every document in the repository in the second revision.

What the paper had attributed to it is either classical or was dropped:

| claim | now |
|---|---|
| the minimal unmatched mass under a nested compatibility constraint is `sup_t (G_W(t) − G_M(t + δ))₊` | the deficiency form of Hall's marriage theorem (Hall 1935) |
| the Gaussian closed form of that deficit for equal variances | a one-line consequence of the threshold form, stated without citation |
| uniqueness and assortativeness of the stable matching under common rankings | Gale and Shapley (1962); Knuth (1996) |
| a stochastic-dominance feasibility condition, invariance theorems, a height calibration, a list of "open problems" | removed; nothing in the paper depends on them |

The validation table's rows had carried that source's theorem numbers; they now name the
classical results, and its height-calibrated setting was replaced by a neutral Gaussian
one, so no number in the table comes from the removed source.

Earlier commits in the repository's history still contain the old text. History was not
rewritten.

## Candidates considered and not used

None. Every reference the brief named was found and verified (`paper/refs_check.csv`
for DOI entries; the publisher's page, given in each entry's `url` field, for PMLR,
NeurIPS and JASSS papers).
