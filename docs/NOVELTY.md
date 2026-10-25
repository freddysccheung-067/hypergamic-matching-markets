# Novelty

## 1. The question

If a reviewer read only Bilderbeek (2026) and this repository, what would they say is new?

## 2. The answer

Two things, both measurements rather than theorems.

**A numerical object the closed form cannot reach.** Bilderbeek's Theorem 2.2 gives unmatched mass
under a *compatibility* constraint on one scalar — `s_m ≥ s_w + δ`, an upper-set structure, which
is what makes `D` a one-dimensional supremum. Environment A's constraint is a *visibility*
condition: the feasible set is the top-k of an estimated multi-attribute utility, lists differ by
agent, and estimator error is correlated in a way no marginal captures. Theorem 2.2 does not
apply. This repository computes what happens instead, and finds the answer depends on the horizon
at which the estimator is evaluated: the exposure gap is +0.287 at epoch 19, +0.025 at epoch 99.
No closed form in the paper has a horizon argument.

**An institutional comparison with the selection artifact priced.** Environment B is a subgraph of
the same population with a positive signalling cost, and the welfare rule scores agents B excludes
at their counterfactual A outcome. Without that, an entry filter looks like a gain for the
arithmetic reason that it removes the agents it cannot serve. B beats an oracle A at `c_B = 0.01`
with an interior optimum at `size_fraction` 0.65, and loses at `c_B ≥ 0.05`. Neither the rule nor
the result is in the paper.

## 3. The strongest objection, and the reply

*This is §9 read as an invitation, and §9 names three open problems: multidimensional status,
search frictions, endogenous status. The repository addresses none. Its "multi-trait" utility is
an additive sum of four synthetic traits — a one-dimensional utility of a multidimensional type,
not multidimensional status. Its visibility constraint is top-k truncation, which Roth & Rothblum
(1999) analyzed. And Chen, Hsieh & Lin (2023) already ran a matching-model-based recommender on a
live platform. What survives is a simulation whose parameters have no empirical anchor.*

The first half is a genre claim and it wins: no theorem is produced. But `D` is a supremum because
the constraint is *nested*, the step Theorem 2.2's proof uses explicitly. Top-k lists from an
estimated utility are not nested and their error depends on training time, so the structure that
makes `D` tractable is absent rather than merely harder. The second half bites, and the claim has
narrowed because of it: not "recommender-mediated matching on multiple traits" but the conjunction
of three specific absences in Chen et al. — no characterisation of unmatched mass against the
visibility constraint, no institutional comparison, no horizon.

---

# §4 Claims in README.md and docs/THESIS.md that the literature already establishes

Each sentence below asserts a result that is proved or measured elsewhere and should carry a
citation. **Neither file has been edited** — the list is for the session that owns them.

## README.md

**README.md:11**
> on a truncated list is not the same preference that operates on the whole population.

**Roth & Rothblum (1999)**, *Truncation Strategies in Matching Markets — in Search of Advice for
Participants*, Econometrica 67(1), 21–43. Truncation changing outcomes under deferred acceptance is
a 1999 theorem, presented here as a premise.

**README.md:95**
> At `c = 0` every type proposes, so a proposal says nothing about intent: Akerlof's pooling outcome.

**Akerlof (1970)**, QJE 84(3), 488–500. Attributed in the sentence, which is right, but this is a
cost-channel analogue, not Akerlof's quality-unobservability result; `THESIS.md:74` states the
distinction correctly and the README should match it.

**README.md:96–97**
> At `c > 0` only types with `τ > c / (V̄ · p)` propose, so a proposal carries information: Spence's
> separating outcome.

**Spence (1973)**, QJE 87(3), 355–374. The single-crossing condition is Spence's; the threshold is a
one-dimensional instance.

**README.md:235–236**
> returns the proposer-optimal matching, which is simultaneously the acceptor-pessimal one: among all
> stable matchings, every `W` receives her worst.

**Gale & Shapley (1962)** for proposer-optimality; **Knuth (1996)**, CRM Proceedings and Lecture
Notes 10, for the lattice duality that makes "proposer-optimal" and "acceptor-pessimal" one
statement; **Roth (1982)** for the incentives side. Currently framed as an observation about the
project's own test.

**README.md:251**
> a feedback loop through preferences.

**Chaney, Stewart & Engelhardt (2018)** and **Mansoury et al. (2020)**. The repository's negative
result — its loop does not compound — is only meaningful against that literature.

**README.md:262–264**
> **Market clearing rate is not comparable across population sizes** at fixed `k`: it falls roughly as
> `1/N`, 0.368 at N=500 against 0.101 at N=4000 with k=20.

**Ashlagi, Kanoria & Leshno (2017)**, JPE 125(1), 69–98, prove the competition effect. The `1/N`
scaling should be attributed; the measurement is the repository's.

## docs/THESIS.md

**THESIS.md:69–70**
> This is a **pooling equilibrium**, and it is Akerlof's outcome reached through a signalling-cost
> channel rather than through unobservable quality alone.

**Akerlof (1970)**. Correctly attributed and qualified; add the reference only.

**THESIS.md:72**
> The posterior rises. This is **Spence's separating equilibrium**.

**Spence (1973)**. Add the reference.

**THESIS.md:82**
> here it is *derived* from the pooling result rather than assumed

The derivation is the project's; the result derived from is **Akerlof (1970)**.

**THESIS.md:138–141**
> Without this, B would look better simply by removing the agents it cannot serve — a
> selection artifact, not a welfare gain.

**Halaburda, Piskorski & Yıldırım (2018)**, Management Science 64(8), 3574–3594, formalise the
restricted-choice version of this concern. The comparability rule should be presented as a response
to a known objection rather than a safeguard the project devised.

**THESIS.md:179**
> **H2 — Threshold inflation is a defence against pooling, not a contagion.**

No literature claim; the section reports the hypothesis as untestable. Listed only because a reader
arriving from the Akerlof/Spence material will expect a reference here and there is nothing to cite.

---

## Summary of the novelty test

Thirty-five sources were asked whether they already do recommender-mediated matching on multiple
traits. Thirty-three do not. Three require the claim to be restated:

1. **Chen, Hsieh & Lin (2023)** — matching-model recommender, live platform, measured matches.
   Narrows the claim most: same intersection, different question.
2. **Pizzato et al. (2010)** — reciprocal recommender, deployed, 23%→42% top-ten success.
3. **Tomita, Togashi & Moriwaki (2022)** — matching-theory recommender deployed on a dating
   platform. Together with (2) this rules out any novelty claim resting on the ranker.

The Bilderbeek §9 gap survives, but not as `README.md:276` states it. §9 names *multidimensional
status*, *search frictions*, and *endogenous status*. This project addresses a fourth thing §9 does
not list — a learned visibility constraint — and none of the three it does. The accurate sentence:

> What it adds is the recommender-intermediated setting that the paper's Section 9 does not treat,
> where visibility is a truncated top-k list over an estimated multi-attribute utility and no closed
> form is known; it does not claim progress on the multidimensional, search-friction, or
> endogenous-status problems the section names.
