# Thesis

## What this project is, and what it is not

This is a computational model of **matching-market institution design**. It asks a question that
is well defined inside a specified model: when a matching market has a signaling cost `c`, an
information asymmetry over intent, a network size `N`, and an enforcement penalty `d`, how does
total welfare depend on those four parameters?

It is not a proof about any real population. A simulation of a specified utility model
demonstrates what follows from the assumptions; it cannot establish that the assumptions hold.
Any sentence of the form "this proves modern dating is collapsing because of X" is outside what
the method can deliver, and the project does not make it.

What the model *can* do, and what the earlier version of this project was already doing, is
produce **comparative statics**: how welfare, clearing, and inequality move when one institutional
parameter changes with everything else held fixed. That is a claim about the model. It becomes a
claim about the world only through the separate empirical question of whether the parameters are
calibrated correctly, which this project does not attempt and should not pretend to.

## The pivot: from one institution to two

The earlier design modeled a single institution — an algorithmically mediated hypergamic market —
and asked how algorithmic curation amplified concentration inside it. That is a good question
with a defect: with only one institution there is no contrast condition. A rising Gini under a
recommender can always be answered with "compared to what?", and the answer was either a random
baseline (which measures distance from uniformity, not harm) or an oracle (which is not an
achievable institution).

The pivot adds a second institution and makes the comparison the object of study.

| | Environment A | Environment B |
|---|---|---|
| signalling cost `c` | ≈ 0 | positive |
| visibility | global, algorithmically mediated | local subgraph, complete within it |
| intent observable | no | no, but *partially separated* by cost |
| value alignment | unobservable | partially observable via entry filter |
| defection penalty `d` | 0 | positive |
| population | large `N` | small `N_sub` |

Neither is a strawman, and neither is the "good" one. Environment A has liquidity and no
information; Environment B has information and no liquidity. The thesis is that **welfare is
maximized at an interior point**, and the model's job is to locate it.

## The mechanism, stated precisely

### Signalling cost separates intent

Each agent carries a private type `τ_i ∈ (0, 1]`: willingness to commit, invisible to others. A
match with `j` yields `τ_i · V_ij` to `i`, where `V_ij` is the compatibility value. Proposing
costs `c` and succeeds with probability `p`. Then `i` proposes only when

    τ_i · V_ij · p > c

so the set of proposers is exactly `{i : τ_i > c / (V_ij · p)}`. Define the separating threshold

    τ*(c) = c / (V̄ · p)

where `V̄` is the expected compatibility value over the candidate pool.

This is the entire mechanism, and it is worth being explicit about what it is and is not.

- **At `c → 0`, `τ* → 0`.** Every type proposes, including the lowest. A received proposal then
  carries no information about intent: the posterior `E[τ | proposal]` equals the population mean
  `E[τ]`. This is a **pooling equilibrium**, and it is Akerlof's outcome reached through a
  signalling-cost channel rather than through unobservable quality alone.
- **At `c > 0`, `τ* > 0`.** Proposals from low-intent agents are filtered out by their own
  incentive to avoid the cost. The posterior rises. This is **Spence's separating equilibrium**.

The proposal's framing treats Akerlof as the whole story. It is the `c = 0` case of a two-regime
model, and stating it that way is what lets the simulation produce a boundary rather than a
one-directional claim.

**What follows for the acceptor.** An agent who cannot infer intent from a proposal must use
something else. In the model that something is the observable trait bundle, and the rational
response to a pool of uninformative proposals is to raise the bar on observables: `θ_i` rises not
because the acceptor is irrational, but because the signal she would need is absent. That is the
"asymmetric preference inflation" of the proposal, and here it is *derived* from the pooling
result rather than assumed. The alternative — that inflation is a separate, exogenous contagion —
is a competing hypothesis, and the model can distinguish them (H2 below).

### Algorithmic mediation changes the pool, not the preferences

`generate_swipes` computes utilities from traits and the seed; it takes no shown-set argument. So
under the current architecture the recommender changes **who is visible and therefore who is
reached**, not what anyone wants. This must be said plainly because the popular version of the
"algorithmic amplification" story claims the algorithm *changes preferences*, and this model does
not do that.

What the recommender does do, and what the ablation already measured, is **estimation-error
concentration**: a recommender that only approximates preferences puts the agents it cannot
represent into no shortlist at all. Measured at N=1000, k=20, five seeds: Gini 0.939 ± 0.006
under the NCF against 0.664 for an oracle with the same objective. The loss of representational
fidelity, not a status bias, is what concentrates exposure.

### Thin markets: the cost of Environment B

Environment B has complete information within the subgraph and pays for it in size. If an agent
accepts only candidates above some compatibility bar `u`, the number of acceptable partners in a
pool of size `n` is `Binomial(n, 1 − F(u))`, so the probability that at least one exists is

    P(acceptable partner exists) = 1 − F(u)^n

which collapses geometrically in the wrong direction. The critical mass

    N* = ln(1 − ρ) / ln F(u)

is the smallest `n` at which the pool clears with probability `ρ`. This is the **thin-market
penalty**, and it is the reason Environment B cannot simply be declared superior.

## Welfare, and why the comparison is not rigged

Welfare must be defined on a common basis or the two environments cannot be compared. Define, per
epoch, over the whole population `N`:

    W = Σ_{matched pairs (i,j)} [ τ_i·V_ij + τ_j·V_ji ]              (value created)
        − Σ_{proposals} c                                            (signalling cost spent)
        − Σ_{dissolved matches} [ (1−τ_defector)·L_partner + d ]      (defection loss)

**The comparability rule.** Environment B is a subgraph of the same population: agents excluded by
its entry filter are not deleted, and agents who fail to match in B are not dropped. Every agent
appears in `W` exactly once, and an agent that B excludes is scored at whatever A would have given
them. Without this, B would look better simply by removing the agents it cannot serve — a
selection artifact, not a welfare gain. Both accounting modes are reported:

- `welfare_common` — all `N` agents, B's excluded agents scored at their A outcome. **The number
  the thesis rests on.**
- `welfare_selected` — entrants only. Reports the selection artifact honestly rather than hiding
  it, and must never be quoted as the headline.

## The exogenous shift, and what the decomposition buys

Median female status rises exogenously over generations in this model as it has empirically. Under
the hypergamous indicator `1[s_m ≥ s_w + δ]`, a higher `s_w` mechanically shrinks the eligible
male pool with no app, algorithm, or cultural change involved. That is a **compositional** effect
and it should be modeled as one.

The decomposition is a factorial design on the three sources:

    ΔW_total = ΔW_exogenous + ΔW_algorithmic + ΔW_interaction

estimated from four runs: no drift and no NCF; drift only; NCF only; both. The interaction term is
the only quantity that supports a claim of the form "the algorithm amplifies the demographic
shift", and it is the one most likely to be near zero, because in the current architecture the two
enter through separate channels. **Saying so in advance is the point** — a decomposition that can
report "no interaction" is a measurement; one that can only report amplification is a story.

## Hypotheses, each with a kill condition

The project is only as good as its ability to fail. Each hypothesis below specifies what result
would refute it.

**H1 — Signalling cost separates intent.** The mean intent of received proposals, `E[τ | proposal]`,
increases with `c`; and the acceptance rate of high-intent proposers rises while low-intent
proposers exit.
*Killed if:* `E[τ | proposal]` is flat in `c` over `c ∈ {0, 0.05, 0.25, 1.0}`.

**H2 — Threshold inflation is a defence against pooling, not a contagion.** `θ` rises with
received proposal volume **net of** the mean received intent, and stops rising when `c` makes the
signal informative.
*Killed if:* `θ` is unrelated to volume, or continues to rise at high `c`.

**H3 — Algorithmic estimation error concentrates exposure.** `Gini(NCF) > Gini(oracle)` on a
shared population and seed; already measured at 0.939 vs 0.664.
*Killed if:* the NCF matches or beats the oracle. This one has already survived.

**H4 — There is an interior optimum.** `welfare_common(B)` exceeds `welfare_common(A)` for
`N_sub` above a critical size and falls below it for small `N_sub`, at fixed `c`.
*Killed if:* the ordering is monotone in `N_sub` — either B always wins (the thin-market penalty
is not doing work) or always loses (the information gain is not doing work).
*Measured (THEORY.md §10.5):* survives at `c_B = 0.01` against an oracle A, with the optimum at
size_fraction 0.65 (+64.6 ± 9.9 over five seeds, residual accounting). Killed on the
`c_B ≥ 0.05` slice, where B always loses, and against an NCF-mediated A, where B always wins. The
curve's right-hand slope comes from top-k truncation, not from the thin-market penalty.

**H5 — Exogenous drift and algorithmic mediation are separable.** The interaction term is
estimated, not assumed, and is reported whatever its sign.
*Killed if:* the interaction cannot be estimated because the factorial cells are not comparable —
which would itself be a finding about the model.
*Measured (THEORY.md §10.6):* the cells are comparable (asserted per seed), so H5 survives, and the
interaction is +16.7 ± 40.4 (standard error 18.1, three of five seeds negative): indistinguishable
from zero, as this section anticipated.

**H6 — High-intent agents exit the global market first.** Attrition in Environment A is negatively
correlated with `τ`.
*Killed if:* attrition is flat in `τ`.

## What this design cannot show

Stated here so that no downstream section has to hedge:

- **Nothing about real populations.** The parameters are modeling choices; THEORY.md §2.3 already
  states that neither heterogeneity knob has an empirical anchor, and the same holds for `c`, `d`,
  and the entry filter.
- **Nothing about intent as measured in the world.** `τ` is a construct with no instrument.
- **No genuine closed loop.** The recommender does not change preferences, only visibility. A
  reader who wants the feedback story will not get it here, and the README says so.
- **No claim that one institution is ethically superior.** Environment B carries an entry filter
  that excludes agents; that is a cost, and the model prices it rather than endorsing it.
- **The defection penalty deters nothing.** `d` is charged to a defector's welfare after a match
  dissolves. No agent proposes differently, sets a different threshold, or stays in a match longer
  because of it: survival depends on intent and alignment only, and the proposal decision does not
  read `d`. So B's penalty lowers B's welfare when defection happens and never prevents defection.
  The proposal's "social penalty for casual defection" is modelled as a cost, not as a deterrent.
  Making it deter would mean feeding `d` into the proposal decision, so that an agent likely to
  defect stops proposing — a different model, and one whose H4 result has not been measured.

## Structure of the work

| phase | what it establishes |
|---|---|
| Environment A (built) | preference structure, heterogeneity, the four ablation arms, attribution limits |
| Extension (new) | `τ`, `g`, signalling cost, defection, the Environment B subgraph |
| Comparative (new) | `welfare_common` A-vs-B, the critical-mass curve, the drift decomposition |
| Presentation | the three headline figures, each with its control |
