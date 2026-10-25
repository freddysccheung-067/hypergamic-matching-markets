# Related work

## 1. The closed-form one-dimensional result, and what it settles

For a marriage market where status is a single scalar and unions must be hypergamous, the
aggregate questions have closed-form answers and they were settled in October 2026. Bilderbeek
(2026) models the market as a coupling of two Borel probability measures and proves the whole
picture: universal hypergamy is feasible exactly when one side's status distribution first-order
stochastically dominates the other's (Theorem 2.1, via Strassen), and when it is not, the minimal
unmatched mass is the Hypergamy Deficit `D = sup_t (G_W(t) − G_M(t))₊` (Theorem 2.2, a
continuum Hall marriage theorem). No behavioural strategy changes either fact (Theorem 4.3),
status gains are transfers of rank rather than creations of it (Theorem 4.5), and under
status-ranked preferences the unique stable matching is positive assortative (Theorem 5.1, the
Gale–Shapley result of Gale & Shapley 1962 stated on the continuum). A Gaussian calibration to US
heights gives a closed form for a required status gap, `D(δ) = (2Φ((δ − Δm)/2σ) − 1)₊`, which
vanishes to about 14 cm and reaches roughly 0.74 at one foot.

The consequences for anyone modelling this are severe. Unmatched mass is a functional of the two
marginal status distributions alone. Effort, apps, and every other intervention that rearranges
who meets whom without changing those marginals cannot move it (Theorem 4.3 makes this exact, and
the paper says so in as many words). Two things the literature had treated as substantive — the
feasibility of "he must be taller than me" and the size of the resulting deficit — turn out to be
properties of histograms. Any new work in a one-dimensional status setting is, at best, a check
on Theorem 8.1.

## 2. The empirical literature, and why a model is the right instrument

The empirical record is consistent and mostly about marginals. Becker (1973) supplies the
complementarity condition: supermodular surplus implies positive assortative matching, so
measured sorting is evidence about the surplus function. Schwartz & Mare (2005) trace US
educational homogamy from 1940 to 2003, falling then rising, driven first by graduates marrying
each other and later by declining intermarriage at both tails. Eika, Mogstad & Zafar (2019) find
that assortative mating accounts for a non-negligible share of cross-sectional household income
inequality across five countries, while *changes* in assortative mating barely move inequality
trends — a level effect without a trend effect, and a warning about which of the two a
cross-sectional model can speak to. Esteve et al. (2016) show the educational gender-gap reversal
shifting marriage patterns globally from hypergamy toward hypogamy with no preference change
required, which is Bilderbeek's Remark 8.2 measured.

On the micro side, Fisman et al. (2006) randomise speed-dating encounters and find trait weights
differing by sex and selectivity rising with group size; Lin & Lundquist (2013) show trait
salience is not separable. In online settings, Hitsch, Hortaçsu & Ariely (2010) estimate mate
preferences, feed them into Gale–Shapley, and find predicted stable matches track observed ones
closely — concluding that preferences, not search frictions, explain most sorting. Bruch & Newman
(2018) find a desirability hierarchy consistent across cities, both sexes pursuing partners about
25% more desirable than themselves, with reply probability falling sharply in the gap.

Why model rather than estimate? Because every instrument above observes *realised* matches inside
one institution. None can answer a counterfactual about a different institution, and the question
here is institutional: what happens to exposure, clearing and welfare when the list an agent sees
is produced by an engagement-optimising estimator rather than given. Chiappori & Salanié (2016)
make the estimation side precise — identifying matching parameters from observable partner
characteristics alone is not credible — which is an argument that a structural model, not a
reduced-form estimate, is the vehicle for a mechanism question. Kanoria & Saban (2021) and
Halaburda, Piskorski & Yıldırım (2018) take the same view from the design side: both derive
platform welfare from a model of restricted choice, and neither estimates it from a market.

## 3. Rankers, feedback loops, and the visibility constraint

The recommender literature has its own version of the question. Koren, Bell & Volinsky (2009) and
He et al. (2017) establish the collaborative-filtering architecture this project uses — matrix
factorisation, then a neural generalisation over implicit feedback — and neither is a claim about
markets. Pizzato et al. (2010) introduce the reciprocal recommender and show that accounting for
both sides lifts top-ten success from 23% to 42%; Tomita, Togashi & Moriwaki (2022) describe
deploying a matching-theory-based recommender on a live dating platform. Reciprocity-aware
ranking is therefore solved engineering, and a project that claimed novelty for its ranker would
be claiming novelty for something built in 2010.

The loop literature is more directly adversarial. Chaney, Stewart & Engelhardt (2018) show that
training on data confounded by the system's own recommendations homogenises users *without*
raising utility. Mansoury et al. (2020) show a feedback loop amplifying popularity bias, cutting
aggregate diversity, and hitting minority-group users hardest. Jiang et al. (2019) separate the
echo chamber from the filter bubble theoretically. Abdollahpouri (2019) frames popularity bias as
a fairness problem, and Sonboli et al. (2022) show that in multisided settings fairness criteria
for different stakeholders conflict, so no single objective is available.

Every one of these predicts a loop that *persists or compounds*. The result here points the other
way — the NCF-versus-oracle exposure gap is +0.287 at epoch 19 and +0.025 at epoch 99, positive on
all five seeds — and the explanation is that the ranker's deficiency is coverage, which training
fixes, not preference distortion, which it does not. That distinction is not in the loop
literature and is the one claim here a reader should check hardest. Two further results bound what
mediation can be credited with: Roth & Rothblum (1999) show truncating a preference list changes
outcomes strategically, so a top-k list is not a neutral convenience; and Ashlagi, Kanoria & Leshno
(2017) show that in an unbalanced random market even a slight imbalance produces an essentially
unique stable matching with the long side matched near-randomly, so some of any measured asymmetry
is competition rather than curation.

## 4. The gap, stated narrowly

What this project does that the above does not: it compares **two institutions on a common
welfare basis inside one model** — a recommender-mediated global market and a small
signalling-cost subgraph — where the mediating mechanism is a *learned ranker whose estimation
error is the treatment*, and where the welfare comparison scores excluded agents at their
counterfactual outcome so selection cannot masquerade as gain. Concretely, it reports what
unmatched mass and welfare do when visibility is a top-k list over an estimated multi-attribute
utility, as a function of the horizon at which the estimator is evaluated.

That is narrower than "recommender-mediated matching on multiple traits", deliberately.
Chen, Hsieh & Lin (2023) already run a matching-model-based recommender on a live dating platform
and measure realised matches, establishing that such a ranker improves outcomes. What they do
not do is characterise unmatched mass against the visibility constraint, or price an excluded
agent against an alternative institution. It explicitly does not claim: any closed form for the
multidimensional deficit; that its ranker is novel; that its parameters correspond to any
population; or that Bilderbeek's one-dimensional results extend to the multi-attribute case.
