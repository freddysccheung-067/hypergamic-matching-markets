# Mathematical Formulation

## 1. Agent State Space

Each agent $a_i$ is a column vector:

$$\mathbf{v}_i = [s_i,\ a_i,\ c_i,\ t_i]^\top$$

| Symbol | Meaning | Distribution |
|--------|---------|--------------|
| $s_i$ | Socio-economic status | $\text{LogNormal}(\mu_s, \sigma_s)$ or Pareto($\alpha_s$) |
| $a_i$ | Physical attractiveness score | $\mathcal{N}(\mu_a, \sigma_a)$, clipped to $[0,1]$ |
| $c_i$ | Sociability metric | $\mathcal{N}(\mu_c, \sigma_c)$, clipped to $[0,1]$ |
| $t_i$ | Temporal age | $\mathcal{N}(\mu_t, \sigma_t)$ at initialization |

The population is partitioned into two disjoint demographic sets $M$ and $W$ with $|M| = |W| = N/2$.

### 1.1 Every trait here is a modeling construct, not a measurement

$s$, $a$ and $c$ are synthetic draws. Nothing in this model estimates a real quantity, and the
results say nothing about any actual person. That is true of all three traits; it is worth
spelling out for $a$ in particular, because attractiveness is the trait the model handles least
honestly if left unexplained.

**Attractiveness is not measurable on the scale this model uses.** Three distinct problems, only
one of which the model can absorb:

1. **Self-report compression.** Asked to rate themselves, most people cluster near the top of the
   scale — a 7/10, not a 5/10, for someone who considers themselves average. The reported
   distribution is therefore shifted and truncated relative to whatever the underlying variable
   is. This does *not* necessarily bias the ranking: a shift shared by everyone is an intercept,
   and an intercept cancels when you compare candidates. What it destroys is *usable variance*.
   If everyone reports 7, the term carries no information.

2. **Rank scales are not interval scales.** A "7" and a "5" are ordinal labels. Treating them as
   reals — as this model does — assumes the gaps are comparable, which nothing in the
   elicitation supports. Averaging ordinal ratings is not meaningful in the way averaging
   heights is.

3. **Ratings are relative to a reference population.** A 7 in one city, age group or app is a 5
   in another. There is no absolute scale. Any empirical calibration would need to name its
   reference population, and none of the commonly cited sources do.

**What this model does and does not claim.** It assumes $a_i$ is a real-valued score on an
interval scale with a roughly Gaussian population distribution. That is an assumption of
convenience, chosen so the utility function is differentiable and the simulation tractable. It is
not an empirical claim, and no result in this project should be read as an estimate of a real
attractiveness distribution.

**What survives the assumption.** Because traits are standardized before entering the utility
(§2), $a_i$'s *units* cancel: only its rank order matters. So the model's conclusions depend on
attractiveness being ordinally comparable, not on the specific 0–1 scale or on the ratings being
self-reported. What it cannot survive is a distribution with no variance, or one whose ordering
is noise — which is exactly what heavy self-report compression would produce.

**Why the default is not a self-report distribution.** Setting $a \sim \mathcal{N}(0.7, 0.08)$ to
mimic self-report would look more faithful while making the term behave as near-constant, since
almost all its variance would fall inside a narrow band. A model in which attractiveness barely
varies cannot test anything about attractiveness. The honest choices are to declare the
assumption, report results as an upper bound on the term's influence, and treat any empirical
calibration as a separate exercise requiring data this project does not have.

---

## 2. Utility Functions

### 2.0 Traits are standardized first

Every trait is z-scored before it enters a utility:

$$\tilde{x}_j = \frac{x_j - \mu_x}{\sigma_x}$$

with $(\mu_x, \sigma_x)$ computed once across the whole population after sampling. Raw
values stay on the agent, in the state matrix and in every plot; the standardization is
only the lens the utility looks through.

The reason is that the raw traits are not comparable. At the base config, status is
$\text{LogNormal}(0, 0.8)$ — sample sd $1.31$, max near $44$ — while attractiveness is
$\mathcal{N}(0.5, 0.15)$ clipped to $[0,1]$, sd $0.15$. Weights of $\alpha = 1.0$ and
$\beta = 0.8$ therefore weight status $11.2\times$, not $1.25\times$, and the stated
preference ordering is not the one the code computes. After standardization every trait
has unit variance, so a weight means what it says.

Two consequences worth stating:

- **$\delta$ is in standard deviations of log-status.** A $\delta$ of $1.0$ means "one
  population sd above her own status". On the raw scale it meant one raw log-status unit,
  which is why the same $\delta$ behaved completely differently across configs.
- **The distance term is scale-free**, which softens the collinearity in §2.1: $s$ no
  longer enters $d$ on a scale $11\times$ that of $a$, so $\gamma$ carries a comparable
  share of each trait instead of being almost purely a status penalty.

A degenerate trait ($\sigma_x = 0$) is left unstandardized rather than divided by zero.
Its contribution is then constant across agents, which is the correct behavior for a
trait that carries no information.

### 2.1 Hypergamous Utility (agents in $W$)

$$U_w(m) = \alpha \cdot \tilde{s}_m + \beta \cdot \tilde{a}_m - \gamma \cdot d(\mathbf{\tilde v}_w, \mathbf{\tilde v}_m) + \lambda \cdot \mathbb{1}[\tilde{s}_m \geq \tilde{s}_w + \delta_w]$$

- $\alpha, \beta, \gamma$ — preference weights, now against unit-variance traits
- $\delta_w \geq 0$ — status surplus threshold, in standard deviations of log-status ($\delta_w = 0$ is neutral)
- $\mathbb{1}[\cdot]$ — indicator function that penalises/rewards the status gap
- $d(\cdot,\cdot)$ — Euclidean distance over standardized traits, expanded below

The distance term must be written out, because an unspecified $d$ hides two modeling decisions
that change the results:

$$d(\mathbf{\tilde v}_i, \mathbf{\tilde v}_j) = \sqrt{(\tilde{s}_i - \tilde{s}_j)^2 + (\tilde{a}_i - \tilde{a}_j)^2 + (\tilde{c}_i - \tilde{c}_j)^2 + (\tilde{t}_i - \tilde{t}_j)^2}$$

**Age scaling is gone.** The former divisor of 10 existed because age was measured in
years (sd $\approx 6$) while the other traits lived near unit scale (sd $\approx 0.15$).
Once all four traits are z-scored the gap is already dimensionless and any divisor would
be a second, undocumented weighting. Note that this *changes the relative weight of age*:
at the old settings a one-sd age gap contributed $6/10 = 0.6$ against a one-sd
attractiveness gap of $0.15$, so age was actually the heavier term; now both contribute 1.

**A collinearity that must be acknowledged.** $s$ enters twice: as the linear reward
$\alpha \tilde{s}_m$ and again inside $d$. The model therefore rewards high status *and*
penalises status dissimilarity, so $\gamma$ carries part of the status weight with the
opposite sign. Standardization does not remove this coupling, but it makes the two terms
comparable, so "$\alpha$ is the weight on status" now holds up to a distance penalty
measured on the same scale rather than a much smaller one. Two defensible resolutions,
unchanged: drop $s$ from $d$, or report $\alpha$ and $\gamma$ jointly rather than
interpreting either in isolation.

### 2.2 $\delta$ has power only when $\beta/\alpha \geq 2$

The indicator $\mathbb{1}[\tilde{s}_m \geq \tilde{s}_w + \delta]$ does nothing unless the
cutoff **bisects the competitive set** — the handful of men who could plausibly enter
$w$'s top-$k$. If status dominates the linear terms, the top-$k$ is a status prefix, every
man in it is above the cutoff (or every one is below), and the indicator is a *constant* on
the shortlist. Adding $\lambda$ to a constant reorders nothing, so $\delta$ is inert no
matter what value it takes. For the cutoff to bite, non-status traits have to be able to
compensate for a status deficit — an attractive man below the cutoff must be able to beat a
plain one above it — and that requires $\beta$ to be a substantial fraction of $\alpha$.

Measured top-1000 overlap between the top-$k$ computed at $\delta = 0$ and at $\delta = 1$,
on the base trait distributions, with $k = 20$, $\gamma = 0.3$, $\lambda = 1.5$:

| $\alpha$ | $\beta$ | overlap | reading |
|---|---|---|---|
| 1.0 | 0.1 | 0.9978 | inert |
| 1.0 | 0.8 | 0.9661 | nearly inert — the old pre-standardization default |
| 1.0 | 1.0 | 0.9374 | |
| 1.0 | 1.5 | 0.8301 | |
| 1.0 | 2.0 | 0.7615 | some power — **base config** |
| 1.0 | 3.0 | 0.7366 | |
| 0.6 | 1.5 | 0.6278 | |
| 0.3 | 1.5 | 0.4994 | strong |

Overlap falls monotonically in $\beta/\alpha$, slowly at first and then steeply between
$\beta/\alpha = 1$ and $2$ — the region where non-status traits begin to compensate for a
status deficit across the whole shortlist rather than only in its tail.

Rule of thumb: $\delta$ has real power when $\beta/\alpha \geq 2$. `experiments/base.yaml`
ships $\alpha = 1.0$, $\beta = 2.0$, where $\delta$ is no longer structurally dead. This is
guarded by `tests/cpp/test_delta_power.cpp`, which fails if a retune drives the overlap back
above 0.95, and by its inverse, which asserts near-inertness at $\beta \ll \alpha$ and exact
inertness when status is the only signal, so the guardrail cannot pass vacuously.

A correction to the sentence this paragraph used to carry. It said status "stays the dominant
linear term, which preserves the Matthew-effect story". The first half is false at these
weights: `docs/attribution.md` measures zeroing `beta` as moving *more* of the shortlist than
zeroing `alpha` (overlap 0.607 against 0.748 at $k = 20$), which is what $\beta = 2\alpha$ on
standardized traits implies. The Matthew-effect narrative in the README is about concentration
under a recommender, and it does not depend on which of the two linear terms dominates — but any
prose that says "status drives selection" is wrong at the shipped configuration and is corrected
in §9.2.

**Why this note exists, and why the numbers above were re-measured.** An earlier version of
the model ran on raw traits, where status sd was $1.31$. At the then-default $\alpha = 1.0$,
$\beta = 0.8$ the effective status weight was $11.2\times$ attractiveness, and raising
$\delta$ from $0.0$ to $1.0$ moved **0 of 199** women's positive sets — every sweep over
$\delta$ in that configuration measured nothing. Standardization removed the scale bug.
The overlap figures published before standardization are not comparable to these and should
not be quoted alongside them; the values in the table were computed by the hidden
`[.measure]` case in `test_delta_power.cpp` against the current code, and that case
regenerates them after any retune.

### 2.3 Preference heterogeneity

The utilities above make every agent a rescaling of one preference ordering: two women with
the same traits rank the whole market identically. That costs the model two things it cannot
buy back by tuning weights.

First, **there is nothing to concentrate**. If every shortlist is the same shortlist, the
visibility Gini follows from the trait distribution alone and no algorithm can move it.
Second, **the control arm has nothing to control**: with homogeneous preferences a
recommender and an oracle select the same people, so "algorithmic loss" would measure the
NCF's approximation error and never the market effect the project is about.

Two independent sources of heterogeneity fix this. Both are **off by default** and, with
both off, every number the project published before this section is reproduced bit for bit.

#### Taste multipliers

Each agent carries three multipliers, drawn once at sampling:

$$m^{(\alpha)}_i,\ m^{(\beta)}_i,\ m^{(\gamma)}_i
\sim \text{LogNormal}\!\left(-\frac{\sigma^2}{2},\ \sigma\right),
\qquad \sigma = \texttt{population.taste\_sigma}$$

The $-\sigma^2/2$ drift term makes $\mathbb{E}[m] = 1$ for every $\sigma$, so the knob moves
the *dispersion* of preferences rather than their average strength. The utility becomes

$$U_i(j) = m^{(\alpha)}_i \alpha \tilde{s}_j + m^{(\beta)}_i \beta \tilde{a}_j
- m^{(\gamma)}_i \gamma\, d(\tilde{\mathbf v}_i, \tilde{\mathbf v}_j)
+ \lambda\,\mathbb{1}[\tilde{s}_j \geq \tilde{s}_i + \delta] + \text{shock}(i,j)$$

with the indicator present only for $i \in W$.

$\lambda$ and $\delta$ stay homogeneous. A per-agent $\lambda$ is a defensible extension but
is out of scope here, because it would make $\delta$'s population-level effect
uninterpretable: a sweep over $\delta$ would then mix "the cutoff moved" with "the reward for
clearing it differs by agent", and no single number would describe either.

The multipliers multiply the *weights*, so agent $i$'s preference vector is
$(\alpha m^{(\alpha)}_i,\ \beta m^{(\beta)}_i,\ \gamma m^{(\gamma)}_i)$ — three numbers per
agent. That is the low-rank part of the heterogeneity.

#### Persistent pairwise chemistry

Every utility gains

$$\text{shock}(i,j) = \sigma_{\text{shock}} \cdot L(i,j), \qquad L(i,j) = \operatorname{logit}(u)$$

where $u \in (0,1)$ is the top 53 bits of a splitmix64 chain over the **ordered** triple
$(\text{seed}, \text{key}_i, \text{key}_j)$, offset by half an ulp so $u$ never reaches $0$
or $1$. $L$ is therefore standard logistic, with $\operatorname{sd} = \pi/\sqrt{3} \approx
1.8138$. The heavier-than-normal tail is deliberate: a small minority of pairs are strongly
drawn to each other, and a normal shock understates that.

Four properties are load-bearing, and each has a test:

- **Deterministic and persistent.** $L$ is a pure function of the pair, so
  `generate_swipes`, `calibrate_thresholds`, `build_full_info_pref_lists` and
  `compute_welfare` all see the same value. If any two disagreed, the swipe log and the
  welfare number would describe different markets and neither would be a result.
- **Asymmetric.** $L(i,j) \neq L(j,i)$ — the hash covers the ordered pair, so $i$'s
  chemistry with $j$ is not $j$'s with $i$. Symmetric noise would be a pair effect, a much
  more tractable object and not the one being modeled.
- **Computed, not stored.** $O(1)$ memory per pair; no $N \times N$ matrix exists anywhere.
- **Skipped at zero.** At $\sigma_{\text{shock}} = 0$ the branch is not taken at all. A hash
  and a log per pair is real work, and the no-heterogeneity path keeps its old cost.

$\sigma_{\text{shock}}$ is in **absolute utility units**, not a ratio, and $\alpha$ and
$\beta$ get retuned — the same value is a large shock at one weighting and a whisper at
another. So the orchestrator computes the noise-to-signal ratio once at construction:

$$\text{NSR} = \frac{\sigma_{\text{shock}} \cdot 1.8138}{\operatorname{sd}(u_{\text{noise-free}})}$$

`utility_sd(states, config)` supplies the denominator as the standard deviation of the
noise-free utility over at most 4000 ordered pairs, sampling **both** orientations: the $M$
and $W$ utilities differ by the indicator, and $M{\to}W$ alone understates the spread
(measured 2.19 against 2.39 pooled). At the base weights the denominator is $\approx 2.39$
with no tastes and $\approx 2.70$ at $\sigma_{\text{taste}} = 0.5$, so
$\sigma_{\text{shock}} = 1.5$ is a ratio near 1.0 — measured at runtime and logged, never
asserted in the config.

#### What a recommender can and cannot learn

| | structure | learnable |
|---|---|---|
| taste multipliers | 3 numbers per agent (low-rank) | yes, from enough swipes |
| pairwise chemistry | one number per ordered pair (full-rank) | no |

A collaborative filter estimates a user embedding from observed swipes. The multipliers live
in that embedding and are recoverable. The chemistry does not: it is an independent draw for
every pair, so nothing learned about $(i,j)$ transfers to $(i,j')$.

Two consequences that change how a result is read:

- **The oracle arm of §6.1 is an upper bound that includes unlearnable chemistry.** It ranks
  by $U = \text{learnable} + \text{shock}$, and the second term is invisible to any learner.
  Algorithmic loss therefore carries an **irreducible component even for a perfect
  recommender**. It is not a clean measure of the NCF's quality and must not be reported as
  one.
- **That component is determined by NSR, not fit.** Reporting $\Delta$ at one NSR reports one
  point on a curve.

#### Neither knob has an empirical anchor

$\sigma_{\text{taste}}$ and $\sigma_{\text{shock}}$ are **modeling knobs, not estimates**.
Nothing here measures how much real preference weights vary, and no reference population is
named. They were set to be visible at the base weights and nothing more. Therefore:

- every headline result is reported **as a function of** them, never at a single setting;
- the concentration-versus-heterogeneity curve — reach and Gini against
  $(\sigma_{\text{taste}}, \sigma_{\text{shock}})$ — is itself a result worth showing,
  because it is what says how much of the Matthew effect is preference heterogeneity rather
  than algorithm;
- a claim that holds at one setting only is a hypothesis, not a finding.

Measured endpoints, $N = 1000$, $k = 20$, oracle shortlists (top-$k$ among θ-admissible
candidates), seed 42:

| setting | reach | Gini |
|---|---|---|
| both off | 0.069 | 0.958 |
| base ($\sigma_{\text{taste}} = 0.5$, $\sigma_{\text{shock}} = 1.5$) | 0.955 | 0.664 |

With both off, 93% of the market appears on nobody's shortlist and the most-exposed agent is
shortlisted by all 500 members of the opposite side — every shortlist holds the same ~20
people. At the base settings 95.5% of agents appear on someone's shortlist. The distance
between those two rows is the quantity the project is about, and it is a function of two free
parameters, which is why the sweep matters more than either endpoint.

### 2.4 Standard Utility (agents in $M$)

$$U_m(w) = \alpha' \cdot \tilde{s}_w + \beta' \cdot \tilde{a}_w - \gamma' \cdot d(\mathbf{\tilde v}_m, \mathbf{\tilde v}_w)$$

The status surplus constraint is absent ($\delta_m = 0$), operationalizing the asymmetry.

---

## 3. Adaptive Expectation Updates

After epoch $t$, each agent updates its acceptance threshold $\theta_i^{(t)}$:

$$\theta_i^{(t+1)} = \theta_i^{(t)} + \eta \left( \frac{\text{matches}_i^{(t)}}{\text{proposals}_i^{(t)}} - \theta_i^{(t)} \right)$$

$\theta_i$ doubles as the agent's selectivity and its running estimate of its own match yield, so
this is a Robbins–Monro stochastic approximation step with rate $\eta$. Agents experiencing
persistent rejection lower $\theta_i$; those with high yield raise it.

An earlier draft called this step *Bayesian* and wrote the target as a separate prior
$\hat{p}_i$. That was wrong on two counts: no prior or likelihood is defined anywhere, and
`include/agent.h` uses $\theta_i$ itself as the estimate. The formula above is what the code
computes. A genuine Bayesian treatment would need an explicit Beta prior over match probability —
a different model, not a rename.

**$\theta_i$ is a utility cutoff, and it gates swipes.** A candidate $j$ is swipeable by $i$ only
when $U_i(j) \geq \theta_i$, and the top-$k$ is then taken among the admissible candidates; this is
what closes the loop, because a $\theta_i$ pushed upward by a high match yield shrinks the positive
set the positive
set the NCF trains on. Because the update target is a *yield* in $[0,1]$ while $\theta_i$ is
compared against utilities on the scale of $\alpha\tilde{s} + \beta\tilde{a} - \gamma d +
\lambda\mathbb{1}[\cdot]$ (standardized units, §2.0), $\theta_i$ is
calibrated at $t=0$ as the $(1-\phi)$ quantile of each agent's own utility distribution, with
$\phi$ = `population.admissible_fraction` (default 0.2), so that roughly $\phi N/2$ candidates per
agent are admissible at the start rather than all or none.

**A known inconsistency, not fixed here.** The update target is a yield in $[0,1]$ and
$\theta_i$ is a utility, so the recursion drives $\theta_i$ toward a number whose *scale*
is wrong even though its *direction* is right. Standardization changed the numeric range of
$\theta$ but not this mismatch; it is recorded here rather than repaired because repairing
it means redefining the update, which changes every result the project reports.

**When the gate binds, and when it is inert.** $\theta$ can only affect behaviour when it
truncates the admissible set below $k$: an agent with at least $k$ admissible candidates
takes the top $k$ of them regardless of where the cutoff sits. The condition is

$$\phi < \frac{k}{N/2},$$

i.e. `admissible_fraction` below $2k/N$. At the shipped $\phi = 0.2$ and $k=20$, $N=1000$
that is $0.2 < 0.04$ — false, so every agent has roughly 100 admissible candidates against a
shortlist of 20 and $\theta$ never acts. Measured: at $\phi = 0.02$ (about 10 admissible at
$k=20$) every one of 1000 agents is truncated and 6709 positives move under a one-unit shift
in $\theta$; at $\phi = 0.2$, none do. The feedback loop of §3 is therefore *live only above
this threshold*, which is a genuine property of the model rather than a defect to hide: it
says the market only responds to its own outcomes when selectivity is tight relative to list
length. Every ablation result below reports the regime it was run in.

---

## 4. NCF Recommendation Engine

### 4.1 Architecture

The engine maps user embeddings $\mathbf{p}_u \in \mathbb{R}^k$ and item embeddings $\mathbf{q}_i \in \mathbb{R}^k$ into a predicted interaction score:

$$\hat{y}_{ui} = f\left(\mathbf{h}^\top \left[\phi_{\text{GMF}}(\mathbf{p}_u, \mathbf{q}_i)\ \|\ \phi_{\text{MLP}}(\mathbf{p}_u, \mathbf{q}_i)\right]\right)$$

**GMF branch:**
$$\phi_{\text{GMF}} = \mathbf{p}_u \odot \mathbf{q}_i$$

**MLP branch:**
$$\phi_{\text{MLP}} = \text{ReLU}\left(W_L \cdots \text{ReLU}(W_1 [\mathbf{p}_u \| \mathbf{q}_i] + b_1) \cdots + b_L\right)$$

### 4.2 Training Objective

Binary cross-entropy over implicit feedback (swipe = 1, pass = 0):

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{D}} \left[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \right] + \lambda_{\text{reg}} \|\Theta\|_2^2$$

Negative samples are drawn uniformly at random from unobserved pairs.

---

## 5. Truncated Preference Lists

At the close of each epoch, the NCF engine generates a ranked list of top-$k$ candidates for each agent $i$:

$$\mathcal{P}_i = \text{top-}k_{j \neq i}\left\{ \hat{y}_{ij} \right\}$$

These truncated lists replace full preference orderings in the matching stage. The key structural property: because hypergamous agents concentrate swipes on high-$s$ nodes, the NCF loss converges to funnel exposure toward the same small cluster, so most agents' $\mathcal{P}_i$ lists overlap heavily at the top.

---

## 6. Gale-Shapley Deferred Acceptance

### 6.1 Standard (Full-Information) Benchmark

Run with complete preference orderings derived from utility functions. Produces a stable matching $\mu^*$ — no blocking pair $(m, w)$ exists such that both prefer each other to their current partner.

### 6.2 Algorithmically-Constrained Variant

Run with truncated lists $\mathcal{P}_i$. Because lists are incomplete:
- Proposals may exhaust the list before a match is found → agent goes unmatched
- Congestion at top nodes causes cascading rejections

Output: partial matching $\mu_{\text{alg}}$ with unmatched residual $\mathcal{U}$.

---

## 7. Primary Metrics

### 7.1 Visibility Gini Coefficient

$$G = \frac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2 \bar{x}}$$

where $x_i$ = total profile exposures received by agent $i$ across an epoch.

### 7.2 Market Clearing Rate

$$\text{MCR} = \frac{|\mu_{\text{alg}}|}{N/2}$$

Fraction of agents who receive a match. A value of 1.0 = full market clearance.

#### The scaling law, and why MCR is not comparable across $N$

Two regimes, and the boundary between them is where the market stops being sparse.

**Sparse regime: $\text{MCR} \sim k^2/N$.** A proposer is matched when some acceptor both
received his proposal and ranked him highly enough to keep him. With lists of length $k$
spread out over a pool of $N/2$, a given pair is mutually visible with probability of order
$k/(N/2)$ on each side, and there are $N/2$ proposers, so the expected number of mutually
visible pairs scales like $N \cdot (k/N)^2 \cdot N = k^2$ — an expected matched count of
order $k^2$, hence a rate of order $k^2/N$. Measured at $N = 2000$, base heterogeneity:

| $k$ | MCR | $k^2/N$ | ratio |
|---|---|---|---|
| 2 | 0.0040 | 0.0020 | 2.00 |
| 5 | 0.0160 | 0.0125 | 1.28 |
| 10 | 0.0580 | 0.0500 | 1.16 |
| 15 | 0.1280 | 0.1125 | 1.14 |
| 20 | 0.1760 | 0.2000 | 0.88 |
| 30 | 0.2770 | 0.4500 | 0.62 |
| 40 | 0.3520 | 0.8000 | 0.44 |
| 80 | 0.5840 | 3.2000 | 0.18 |

The law is accurate to within about 15% for $k \leq 15$, i.e. while $k^2/N < 0.12$, and then
saturates: once lists are long enough that most agents are mutually visible, the ceiling is no
longer how many pairs exist but how many agents there are, and MCR approaches 1. The constant
is also not 1 but near 1.2, which is the part a thumb-suck derivation would miss.

**The $2k/N$ figure is wrong in general.** It is the ceiling only in the fully concentrated
market, where every list holds the same $\sim k$ people: there, each acceptor keeps the best
of the $k$ who proposed, roughly $k$ proposers match out of $N/2$, giving $\text{MCR} \approx
k/(N/2) = 2k/N$. Spread-out lists destroy that identity, and the ceiling grows to $k^2/N$.
Since heterogeneity is what spreads the lists, $\sigma_{\text{taste}}$ and
$\sigma_{\text{shock}}$ move the *scaling law*, not just its constant — a second reason
every result has to be reported as a function of them.

**Consequence for comparability.** MCR falls roughly as $1/N$ at fixed $k$. Measured at
$k = 20$: 0.368 at $N = 500$, 0.270 at 1000, 0.176 at 2000, 0.101 at 4000. So an MCR reported
at one $N$ says nothing about another, and comparing a small population to a large one
without matching the list length is comparing two different markets. The fix is to scale
$k$ with $\sqrt{N}$: at $k \approx 0.9\sqrt{N/2}$ the rate is roughly flat — 0.268, 0.270,
0.255, 0.238 at $N$ = 500, 1000, 2000, 4000 — with the mild downward drift coming from the
saturation regime and from ties.

The cost of that scaling: $k \propto \sqrt{N}$ means the swipe log, which has
$N \cdot k \cdot (1 + n_{\text{negatives}})$ rows, grows as $N^{1.5}$ rather than $N$. At
$k \approx 0.9\sqrt{N/2}$ and $n_{\text{negatives}} = 4$, $N = 1000$ produces 100,000 rows and
$N = 4000$ produces 800,000 — a factor of 8 for a 4× population, exactly $N^{1.5}$, against
4× if $k$ were held fixed. The NCF's per-epoch cost follows the row count. This is a Day 5
budget constraint, not a modelling one.

Do not read a lower MCR as a worse market without fixing $N$ and $k$: it is mostly an
arithmetic consequence of longer candidate pools, and the substantive metric at fixed
exposure is the welfare gap and the concentration, not the clearing rate.

**A caveat on these measurements.** They use oracle shortlists — top-$k$ by true utility —
which is the right object for a statement about the matching ceiling, but it is not quite
the object an experiment produces: `generate_swipes` also applies the θ gate, so the lists a
run hands to Gale-Shapley are truncated by admissibility as well.

The gate is not simply a discount, though, and the direction is worth stating because the
obvious guess is wrong. θ is calibrated as a *quantile* of each agent's own utility, so a
tighter gate keeps the **top** candidates — precisely the ones every agent already ranks
highly. Measured at $N = 2000$, $k = 20$: `admissible_fraction` 0.2 gives MCR 0.176, and
0.02 — ten admissible candidates per agent, half of $k$ — still gives 0.176, because
truncating to the top ten removes only candidates nobody was going to match with anyway.
The rate collapses only when the gate cuts below the mutual-visibility core: 0.005 (two or
three admissible) gives 0.016. So the ceiling above is the right order of magnitude for a
realistic gate, and a result that changes when θ is tightened is describing the gate, not
the ranking.

### 7.3 Rank Satisfaction Index

$$R = \frac{1}{|M'|} \sum_{i \in M'} \frac{\text{rank}_i(\text{partner}(i))}{|\mathcal{P}_i|}$$

Lower is better; $R=1$ means every agent ended up matched to their least-preferred visible candidate.

### 7.4 Algorithmic Loss

$$\Delta = \frac{\text{SW}(\mu^*) - \text{SW}(\mu_{\text{alg}})}{\text{SW}(\mu^*)}$$

where $\text{SW}(\mu) = \sum_{(m,w) \in \mu} [U_m(w) + U_w(m)]$ is total social welfare.

### 7.5 Blocking Pair Ratio

$$\text{BPR} = \frac{|\text{blocking pairs in } \mu_{\text{alg}}|}{|\mu_{\text{alg}}|}$$

Measures instability introduced by incomplete information.

### 7.6 Reach and top-1% share

Two companions to the Gini, because a Gini alone cannot distinguish *concentration*
from *starvation* — both show as a high number.

$$\text{reach} = \frac{|\{i : x_i > 0\}|}{N}
\qquad
\text{top1} = \frac{\sum_{i \in \text{top } \lceil 0.01N \rceil} x_{(i)}}{\sum_i x_i}$$

A market that shows everybody once and a market that shows twenty people five hundred
times each can have similar reach and wildly different Gini. Reported together they cannot
be read as one thing: the popularity arm below has $G = 0.96$ *and* reach $4\%$, which is
the signature of a non-market rather than of an unequal one.

### 7.7 The ablation arms

Four rules produce the preference lists. All four share the population, seed, $\theta$,
utility parameters and `deferred_acceptance` call; only the list-construction rule differs,
which is what makes the comparison an ablation rather than four experiments.

| arm | rule |
|---|---|
| `oracle` | top-$k$ of the $\theta$-admissible set by true utility |
| `random` | $k$ uniform draws **without replacement** from the $\theta$-admissible set |
| `popularity` | one shared list per side: top-$k$ of the other side by in-degree; epoch 0 falls back to the oracle ranking |
| `ncf` | swipe log → NCF → top-$k$ |

**The popularity identity is a closed form, not a tendency.** Giving every chooser on a
side the same $k$-element list means exactly $2k$ agents appear on every list and the other
$N - 2k$ appear on none, each of the $2k$ receiving exactly $N/2$ exposures. That vector's
Gini is algebraically

$$G_{\text{pop}} = 1 - \frac{2k}{N}, \qquad \text{MCR}_{\text{pop}} = \frac{2k}{N},$$

with **zero seed variance**. Both are asserted to $10^{-9}$ in
`tests/python/test_ablation.py` at several $k$, because an implementation that gave agents
*different* lists while calling itself a popularity baseline would pass a loose ordering
check and fail this one. The epoch-0 fallback is excluded from the identity and tested
separately: with no in-degree history the arm ranks by true utility, so epoch 0 equals the
oracle arm. Starting from all-zero in-degrees instead would rank the market by id — a
different and accidental baseline that hands exposure to whoever was sampled first.

**Measured, seed-replicated.** $N = 1000$, $k = 20$, base config
($\sigma_{\text{taste}} = 0.5$, $\sigma_{\text{shock}} = 1.5$, NSR $\approx 1.00$,
`admissible_fraction` 0.2), final epoch of a 20-epoch run, five seeds (42–46):

| arm | Gini (mean ± sd) | reach | top-1% share |
|---|---|---|---|
| `random` | 0.3780 ± 0.0058 | 99.4% | 2.8% |
| `oracle` | 0.6544 ± 0.0109 | 95.8% | 12.2% |
| `ncf` | 0.9393 ± 0.0065 | 18.1% | 23.8% |
| `popularity` | 0.9600 ± 0.0000 | 4.0% | 25.0% |

Per-seed gaps, paired by seed (both arms in a difference run on the same population, so the
difference at one seed is one observation):

| gap | mean ± sd |
|---|---|
| `gini(ncf) − gini(oracle)` | **+0.2849 ± 0.0074** |
| `gini(oracle) − gini(random)` | +0.2764 ± 0.0117 |
| `gini(popularity) − gini(random)` | +0.5820 ± 0.0058 |

**The NCF lands above the oracle, and that is the finding.** A rank-8 replica in the session
preamble landed at 0.925 against an oracle of 0.653 — an earlier measurement of the same
sign. Estimation error pushes agents the model is uncertain about out of every shortlist, so
a *lossy personalisation concentrates more than perfect ranking does*. The mechanism is not
"the algorithm favours status" but "the algorithm cannot represent everyone, so it drops the
uncertain ones". The NCF's 0.9393 also sits within 0.02 of the popularity ceiling of 0.96:
at these settings the learned arm is close to degenerate — reach 18% against the shared-list
arm's 4% is what separates them, not the Gini.

That proximity is not a coincidence and is worth recording, because it shows up on epoch 0
exactly. After a single training epoch on 1000 agents the NCF's exposure Gini is 0.96000 —
the popularity identity to seven digits — with 38 agents on all 500 lists. One epoch is not
enough for the user embeddings to separate, so the score is still dominated by the item side
and every user receives nearly the same global ranking. The model does learn its way out of
that: reach climbs from 4% to 18% over 20 epochs while the Gini falls from 0.960 to 0.939.
An arm that never left 0.96 would be a bug wearing an ablation's clothes; this one leaves it,
slowly, and the slow exit is itself a statement about how little one epoch of implicit
feedback buys.

**$G_{\text{ncf}} - G_{\text{random}}$ is not the headline.** The random arm has the lowest
Gini of the four because uniform exposure is near-equal by construction, so that difference
mostly measures distance from uniformity. The quantity that carries the argument is
$G_{\text{ncf}} - G_{\text{oracle}}$: what the recommender's estimation error costs relative
to a ranker that solves the same task perfectly. It is reported that way, and no
`gini_excess_over_random` headline exists in the output.

**These numbers are a function of the heterogeneity setting, not of the model alone.** NSR
runs 0.99–1.03 across the five seeds and is recorded on every row of the ablation CSVs. The
oracle arm's utility includes unlearnable pairwise chemistry (§2.3), so it is a bound that
no learner can reach even in principle, and the size of the gap to it is set by NSR. A
result at one NSR is one point on a curve.

**A correction to this session's own premise.** The four arms above were measured with
`admissible_fraction = 0.2`, where §3's gate is inert: no agent is truncated by $\theta$, so
the outcomes cannot feed back into the swipe set and the run is a repeated static experiment
rather than a simulation. The arms still separate, because they separate on ranking rather
than on feedback — but the regime is stated rather than left implicit, and the live-gate
regime ($\phi < 2k/N$, §3) is a separate measurement.

**What this model does and does not contain.** It is exposure-mediated *in one direction
only*. `generate_swipes` takes no shown-set argument: an agent's preferences are a pure
function of (states, params, seed, epoch), so what the recommender surfaced never shapes what
anyone wants. The Matthew effect here is therefore **not compounded by a feedback loop
through preferences** — the recommender's concentration is a property of who it can rank
confidently, iterated over a market whose tastes do not move in response to it. What *does*
feed back is selectivity: the matching outcome moves $\theta$ (§3), and $\theta$ gates the
next epoch's positives. That loop is real but, at the shipped calibration, inert.

---

## 8. Multi-Generational Pareto Fit

Over $T$ epochs, track the distribution of cumulative match counts per agent. Fit a Pareto distribution:

$$P(X > x) = \left(\frac{x_m}{x}\right)^{\alpha_P}$$

A decreasing $\alpha_P$ across generations indicates heavier tail concentration — fewer agents capturing progressively more matches. This is the macro-level signature of micro-level hypergamy amplified through the NCF loop.

---

## 9. Identification limits

This section records what the model can and cannot answer. It exists because the rest of the
document describes a mechanism precisely enough that its output is easy to over-read, and a
reader who takes a correlation out of the swipe log will reach a conclusion the model does not
support.

The short version: **the model has no causal structure between traits.** It answers "how much does
each term of the utility drive selection" and nothing about why any trait is distributed the way
it is. Everything below is a consequence of that.

### 9.1 Traits are sampled independently, so no trait causes another

`sample_population` draws status, attractiveness, sociability and age from four separate
distributions using one seeded engine (§1). No draw is conditioned on any other. The generative
correlation between every pair of traits is therefore zero up to sampling error — measured at
±0.005 across all six pairs on a 50,000-agent draw, and asserted over 20 seeds in
`tests/python/test_attribution.py`.

This is a design commitment rather than a finding. It means:

- There is no path in this model by which status *causes* attractiveness or the reverse. A
  regression of one on the other inside the simulation recovers zero, as it must.
- Any nonzero association that appears downstream was created by the selection mechanism, not
  inherited from the population.
- The independence makes the selection effects *cleaner* than they would be in real data, where
  the traits are plausibly correlated for reasons the model has no representation of. It does not
  make the model more realistic; it makes the selection mechanism legible, which is the point.

### 9.2 The model cannot answer "does status reflect attractiveness"

Because no such relationship is generated, the model has nothing to estimate. A question of the
form "among people who end up together, is the high-status partner also the better-looking one"
is a question about a joint distribution the model does not contain. It would need an assumed
copula between the traits, and the answer would be a restatement of that assumption.

What the model *can* answer is a question about the mechanism rather than the population: **how
much does each term of the utility drive who reaches the top of the market?** That is what
leave-one-term-out measures — zero one weight, rebuild every agent's ranking, and see how much of
the baseline shortlist survives.

The measurement is `src/python/attribution.py::leave_one_term_out_effect`, and its output is in
`docs/attribution.md`. At the shipped configuration (N=1000, seven seeds, k=20, share-in-common
overlap) it reads:

| term zeroed | weight | overlap with the baseline shortlist |
|---|---|---|
| `attract` | `beta` = 2.0 | 0.607 ± 0.014 |
| `status` | `alpha` = 1.0 | 0.748 ± 0.010 |
| `indicator` | `lambda` = 1.5 | 0.918 ± 0.002 |
| `distance` | `gamma` = 0.3 | 0.929 ± 0.005 |

Two things about that table are worth stating rather than leaving to the reader. Attractiveness
reorders more of the shortlist than status does, which is the opposite of what the project's prose
about the "Matthew effect" invites a reader to assume — and it is exactly qualification 2 below:
`beta` is twice `alpha`, so it should. And the two linear terms are separated from the other two
by a wide margin with spreads an order of magnitude smaller, so the grouping is solid even where
the within-group ordering is a property of the weights.

Three qualifications travel with every number in it:

1. **It is one weight at a time, at one configuration.** Removing `alpha` while `beta`, `gamma`
   and `lambda` stay put measures the marginal contribution of `alpha` *at these weights*. It is
   not an intrinsic importance ranking, and it does not transfer to another weighting — an
   interaction as basic as the indicator's dependence on `delta` (§2.2) already breaks additivity.
2. **The weight decides much of it.** Weights are expressed against standardized traits (§2.0), so
   at `beta = 2·alpha` the attractiveness term has twice the linear influence by construction.
   The ablation reports that, correctly, and it is not a claim about what anyone values.
3. **The distribution decides the rest.** Status is lognormal and heavy-tailed; attractiveness is
   normal and clipped. A term multiplying a heavy tail can recruit a large share of a top-k
   without its weight being large, purely because its extreme upper tail is where the ranking
   concentrates. A Pareto status would move these numbers again.

So the honest form of the claim is: *at this weighting and these distributions, removing term X
changes this share of the shortlist.* Report the configuration alongside it or the number means
nothing.

### 9.3 Correlation among matched pairs is a selection artifact

An analyst who computes a correlation from a swipe log, or from a table of matched pairs, sees a
number that is not zero. It is **not** evidence of a relationship between the traits. It is what
happens when you condition on a shortlist.

The market ranks candidates on several traits at once and keeps a small surviving subset. Measured
on the same seven populations as the table above, for the status/attractiveness pair:

| shortlist size | population | among shortlisted pairs |
|---|---|---|
| top-10 | +0.008 ± 0.010 | −0.248 ± 0.125 |
| top-100 | +0.008 ± 0.010 | −0.177 ± 0.050 |
| top-500 (whole pool) | +0.008 ± 0.010 | +0.008 ± 0.010 |

Two consequences follow:

- **The selected set separates from the population.** `selection_correlation` reports the
  population value next to the selected-set value for each trait pair, and the gap is the
  artifact. It is largest at small shortlists and decays to zero as the shortlist approaches the
  whole candidate pool — where "selected" and "population" are the same set and the two columns
  agree exactly. That decay is the diagnostic: a genuine generative correlation would be flat in
  $k$, because a shortlist would merely sample it.
- **The sign is not the intuitive one.** With two positively-weighted traits and a shortlist far
  smaller than the candidate pool, the induced correlation is **negative**: among candidates that
  clear a high bar, a higher status comes with a lower attractiveness on average, because a
  candidate can be admitted on either. This is the same structure as conditioning on a sum, and it
  is the opposite of what a reader expecting "the survivors are better on everything" would guess.
  The sign is measured, not assumed. Over 20 independent populations at N=1000 it was negative in
  every one, so the sign is a property of the mechanism; the magnitude is not, varying from −0.06
  to −0.38 across those markets.

Two practical rules follow, both cheap to follow and both easy to violate:

- **Never report a correlation computed on matched or shortlisted pairs without the population
  value beside it.** Alone it describes the search, and will be read as describing people.
- **Never quote a selection statistic from one market without its spread.** The per-market
  magnitude above varies by a factor of six. `docs/attribution.md` reports every figure as a mean
  and standard deviation over independent populations, which is the only form in which it can be
  quoted.

A note on what the $n$ is, because it is routinely got wrong in both directions. The selected-set
correlation is computed over pooled (agent, candidate) pairs — 10,000 at top-10 with 1000 agents
— so its binomial standard error is tiny. The uncertainty that matters is *between markets*: two
populations drawn with different seeds give the same sign but different magnitudes, because the
identity of the endogenous top of the market changes with the draw. Bootstrapping pairs inside one
market would resample the wrong thing and would report a confidence interval far too narrow.

### 9.4 What this section protects

The paragraphs above are the difference between a project that demonstrates a specified mechanism
and one that appears to demonstrate something about real people. Everything in §1–§8 is a
statement about a simulation with declared inputs. The moment a selected-set correlation is
reported as a finding about individuals, the project has made a claim it did not measure and
cannot measure — and the reviewers most likely to catch it are the ones most likely to be reading
the README first.

---

## 10. The institutional model: signalling, intent, and two environments

This section formalizes the pivot described in [THESIS.md](THESIS.md). Everything above describes
Environment A — a large, algorithmically mediated market with no signalling cost. Section 10 adds
the parameters that make an *institution* a variable rather than a constant, and defines the
welfare accounting that lets two institutions be compared.

Nothing in §1–§9 changes. The new state is appended, the new parameters default to values that
reproduce current behaviour exactly, and every existing test must keep passing. That is a hard
requirement, not a preference: §9.4's argument about what the project can claim depends on the
current results staying reproducible.

### 10.1 New state: intent, value alignment, network membership

Three fields are appended to the agent state. They are **not** traits in the §2 sense — none
enters `utility_toward`, so none changes any existing number.

| field | symbol | range | role |
|---|---|---|---|
| `intent` | $\tau_i$ | $(0, 1]$ | private willingness to commit; never observable |
| `values` | $g_i$ | $[0, 1]$ | value alignment; observable only inside Environment B |
| `community` | $k_i$ | $\{0,1\}$ | which subgraph the agent belongs to; 1 = eligible for B |

The state matrix grows from 11 to 14 columns, appended in that order. `kStateCols` is the single
constant every shape guard derives from, so the change is mechanical — but four hard-coded
`shape[1] != 7` guards have already been converted to `kStateCols` once, and the same discipline
applies here.

`intent` is drawn from a Beta distribution with configurable `(a, b)`, rescaled to $(0,1]$ so no
agent has exactly zero intent. `values` from a Beta on $[0,1]$. `community` is assigned by
membership probability, not uniformly: a subgraph is defined by an entry filter (§10.3), and
`community` marks agents who *would* pass it.

**The shipped values.** All five parameters live in `experiments/base.yaml` and are validated on
both sides of the boundary:

| key | shipped | what it does |
|---|---|---|
| `population.intent_beta_a` | 2.0 | shape of $\tau$ |
| `population.intent_beta_b` | 2.0 | shape of $\tau$ |
| `population.values_beta_a` | 2.0 | shape of $g$ |
| `population.values_beta_b` | 2.0 | shape of $g$ |
| `population.community_prob` | 0.5 | share with $k_i = 1$ |

Beta(2, 2) has mean 0.5 and standard deviation 0.224, so intent is two-sided and no cost is so
small that it separates nothing — which is the property that makes $\tau^*(c)$ a curve rather
than a step. The **measured** columns at seed 42, N = 1000, with the heterogeneity and chemistry
knobs at their shipped settings: mean $\tau$ = 0.5153 (min 0.0227, max 0.9916), mean $g$ = 0.5043,
and a community share of 0.515 — within binomial error of the 0.5 target. Changing any of the
five moves only the three new columns: the four trait columns and the chemistry key are drawn from
a separate engine stream (the implementation note in docs/INTERFACES.md §1) and are bit-identical
at every setting of these knobs, which `tests/python/test_signal.py` asserts.

None of the three enters `utility_toward`. That is the guard against the failure mode this section
is most exposed to: putting $\tau$ or $c$ inside the utility would silently change every welfare
number, because `compute_welfare` reads the same function. A separate test perturbs every intent
value and asserts the pair utilities do not move by one bit; a second test asserts intent *does*
change the proposal decision, so the first cannot pass vacuously.

**Why intent must be a declared latent rather than inferred.** There is no instrument for it, in
this model or in the world. The model's claim is conditional: *if* intent is distributed as
specified, *then* the welfare consequences follow. §9.1 already makes the analogous point about
traits and it applies with more force here.

### 10.2 The proposal decision, and the separating threshold

An agent proposes to $j$ when

$$\tau_i\, V_{ij}\, p \;>\; c$$

where $V_{ij}$ is the compatibility value of the pair, $p$ the probability that a proposal is
accepted, and $c$ the signalling cost. Rearranged, the set of proposers is
$\{i : \tau_i > c/(V_{ij}p)\}$, giving the separating threshold

$$\tau^*(c) \;=\; \frac{c}{\bar V p}$$

with $\bar V$ the expected compatibility over the candidate pool.

**$\bar V$ is the mean over θ-admissible candidates, not over the whole market.** This is a
measurement decision with a large effect. `utility_toward` is unbounded and many of its values are
negative — the distance penalty and the absence of a bonus both push that way — so a
whole-market mean is pulled toward zero by pairs the θ gate has already refused, and a cost
compared against it would almost never bite. Measured at the shipped weights, seed 42, N = 1000:
the admissible-pool mean is $\bar V_M = 4.43$ and $\bar V_W = 4.94$, against a whole-market mean
near zero. The threshold is therefore computed against the pool a proposal is actually chosen from,
and the binding constraint remains the per-pair test $\tau_i V_{ij} p > c$: the pool mean appears
only in the reported $\tau^*$, never in the decision.

**The shipped parameters and the measured thresholds.** `hypergamy.signal_cost` defaults to 0.0,
which is the pooling regime and reproduces every pre-existing number bit for bit. The separating
threshold at the H1 grid, seed 42, N = 1000, $p = 0.5$:

| $c$ | $\tau^*_M$ | $\tau^*_W$ | $E[\tau \mid \text{proposal}]$ | proposals |
|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 0.5153 | 20,000 |
| 0.05 | 0.0226 | 0.0203 | 0.5153 | 20,000 |
| 0.25 | 0.1130 | 0.1013 | 0.5209 | 19,762 |
| 1.00 | 0.4519 | 0.4050 | 0.5797 | 16,605 |

The first two rows are **identical**, and that is a property of the model rather than a defect:
$\min(\tau_i V_{ij} p)$ over the admitted pairs at these weights is above 0.05, so a cost that
small selects on nothing. A cost must be of the order of the utility it is compared against to do
work, and the utilities here are of order 4. H1 is therefore stated as non-decreasing in $c$ with a
strict increase over the whole grid, which this satisfies; the kill condition — a flat curve
across all four settings — is not met. Asserting strict increase at every step would be asserting
a property of the utility scale, not of the mechanism.

The market closes at $c \geq \tau_{\max} \bar V p$. Measured over five seeds at N = 1000 with
$p = 0.5$: $c_M \approx 2.07$ and $c_W \approx 2.47$ (per-side ranges 1.98–2.15 and 2.41–2.57).
Above those the log contains no positive row and the clearing rate is zero. Measured on the log: at
$c = 4$ it still holds 1,082 proposals, at $c = 8$ it holds 19. Through the orchestrator at $c = 40$
every epoch reports `mcr = 0`, `proposal_cost_total = 0`, and a `separating_threshold` above 1 —
a closed market, reported rather than hidden as a missing value.

**$p$ is measured, not set.** The accept probability is the previous epoch's empirical rate, taken
per side from the swipe log: of the positive swipes a side made, the share the other agent
reciprocated. Epoch 0 has no history and uses $p = 0.5$ for both sides, which is the neutral prior
rather than a claim about the market. A side that made no positive swipe in the previous epoch
falls back to 0.5 too, for the same reason: "unmeasured" is not "impossible". A hard-coded $p$
would make the cost non-responsive and the whole mechanism decorative, so the orchestrator
recomputes it every epoch and writes it into the config the C++ decision reads.

**The cost constrains the matching through the proposer, not through both sides.** A proposal costs
the proposer $c$, so it is the proposer's own act: a recommender can rank the candidates in front
of an agent but cannot spend on their behalf. At $c > 0$ the deferred-acceptance lists are
restricted to the pairs the proposer actually proposed. Intersecting *both* sides' lists would
require a mutual top-$k$ positive, which at $k = 20$ of 100 candidates thins the market by a factor
of five before the cost bites — measured as `mcr` falling from 0.21 to 0.00 at $c = 0.5$, a fifth
of the way to the cutoff. That is a mutual-consent market, a different experiment, and not this
one. The acceptor's list stays whole: her willingness is what "accept" means in deferred
acceptance, and the restriction is about who paid.

A refused pair is **reported with label 0**, not omitted. The label records that the pair was
considered and declined, which is information the NCF can learn from; a deleted row would be
indistinguishable from a candidate the agent never saw. Those rows sit after the positives and
before the sampled negatives within the agent's block.

Three regimes, and they are the spine of the whole comparison:

| condition | equilibrium | what an acceptor learns |
|---|---|---|
| $c = 0$ | pooling | nothing: $\mathbb{E}[\tau \mid \text{proposal}] = \mathbb{E}[\tau]$ |
| $0 < c < \tau_{\max}\bar V p$ | partial separation | $\tau \geq \tau^*(c)$ |
| $c \geq \tau_{\max}\bar V p$ | no proposals | the market is closed |

The middle regime is Spence's; the first is Akerlof's result reached through a cost channel rather
than through unobservable quality alone. **The earlier framing cited Akerlof for both** and that
was imprecise: Akerlof's mechanism is that quality cannot be certified at any price, so the market
pools and unravels. Here the certification device — cost — exists, and the question is what
setting of it suffices. That distinction is what makes an interior optimum possible.

**Implementation note.** `c` is subtracted from the *proposer's* utility at the proposal decision
and is **not** part of `utility_toward`, which stays the pair's value. Putting `c` inside the
utility function would silently change every welfare number, since `compute_welfare` reads the
same function. The cost is spent, not disliked.

### 10.3 Environment B: the constrained subgraph

Environment B is a subgraph of the same population, characterized by four parameters:

$$\mathcal{B} = \big(N_{\text{sub}},\; c_B,\; g_{\min},\; d\big)$$

- $N_{\text{sub}}$ — subgraph size
- $c_B$ — signalling cost inside the subgraph, $c_B > c_A \approx 0$
- $g_{\min}$ — entry filter on value alignment; agents with $g_i < g_{\min}$ are ineligible
- $d$ — defection penalty

Inside B, visibility is complete among eligible agents: every eligible agent ranks every other
eligible agent of the opposite side, truncated only at $k$. That is the information advantage, and
it is bought with size.

**Defection.** A match $(i,j)$ survives with probability

$$\sigma_{ij} \;=\; \sigma_0 \cdot \big(\tau_i \tau_j\big)^{\alpha_\sigma} \cdot g_{ij}^{\beta_\sigma}$$

where $g_{ij}$ is pairwise alignment, not individual. A dissolved match imposes loss
$L\cdot(1-\tau_{\text{defector}})$ on the partner. In Environment A, $d = 0$ so defection is free
and $\sigma$ is applied without penalty; in B, $d > 0$ is subtracted from the defector's welfare.
This is the "cost of casual defection" of the proposal, made explicit rather than asserted.

**The shipped parameters.** `environment_b` in `experiments/base.yaml`, validated in
`EnvironmentBConfig`:

| key | shipped | role |
|---|---|---|
| `size_fraction` | 0.5 | share of each side drawn into the pool, before the filter |
| `values_min` | 0.3 | $g_{\min}$; Beta(2,2) puts ~78% of agents above it |
| `signal_cost` | 0.25 | $c_B$; must exceed `hypergamy.signal_cost` (0.0), checked where B runs |
| `defection_penalty` | 0.5 | $d$ |
| `sigma_0`, `sigma_alpha`, `sigma_beta` | 1.0, 0.5, 0.5 | $\sigma_{ij}$ |
| `partner_loss` | 1.0 | $L$ |

None of the last five has an empirical anchor; they are modeling knobs, and results are reported
as functions of them rather than at a point. $c_B = 0.25$ sits in the partially separating range
of §10.2 (it removes about 1% of proposals), so B's clearance is not dominated by the cost.

**Construction.** Each side's pool is a seeded permutation prefix of that side, then filtered on
$g \geq g_{\min}$. The filter does not preserve the sex ratio, so the pool is balanced by dropping
the excess from the larger side, **highest ids first**. That rule depends on the ids alone, so one
population gives one subgraph. The drop is returned in a `SubgraphReport` and logged at WARNING. At
seed 42, N = 1000, the shipped config passes 194 M and 200 W and drops 6 W to leave |B| = 388. Across
the thin-market sweep a pool loses between 1 and 7 agents on average (column `n_dropped`).

**One code path.** Survival is $\sigma_{ij} = \sigma_0(\tau_i\tau_j)^{0.5}g_{ij}^{0.5}$ with
$g_{ij} = 1 - |g_i - g_j|$ (values live on $[0,1]$, so the gap already does). $d$ is not an input to
$\sigma$: it prices a dissolution and does not change its probability, which is what lets A run the
same function with $d = 0$ and no penalty term. The survival draw is a hash of
$(\text{seed}, \text{epoch}, \min(i,j), \max(i,j))$, so it is reproducible, independent of
iteration order, and symmetric in the pair. The defector of a dissolved pair is the lower-intent
member (ties to the lower row id); §10.3 does not say who defects, so that is a stated modeling
choice. `resolve_defection` is wired into the welfare columns through
`comparison.agent_payoffs`: both environments draw survival every epoch, and B alone charges $d$
(§10.4).

### 10.4 Welfare accounting, and the comparability rule

Per epoch, over the whole population:

$$W \;=\; \underbrace{\sum_{(i,j)\in\mu} \big(\tau_i V_{ij} + \tau_j V_{ji}\big)}_{\text{value created}}
\;-\; \underbrace{\sum_{\text{proposals}} c}_{\text{signal cost}}
\;-\; \underbrace{\sum_{\text{dissolved}} \big(L(1-\tau_{\text{defector}}) + d\big)}_{\text{defection loss}}$$

**The rule that makes A and B comparable.** Environment B is a subgraph: some agents are excluded
by $g_{\min}$ and some fail to match inside it. Neither group is deleted from the population. Every
agent appears in $W$ exactly once, and an agent that B excludes or fails to match is scored at the
outcome Environment A would have given them.

Two numbers are reported and they answer different questions:

- **`welfare_common`** — all $N$ agents, excluded ones scored at their A outcome. This is the
  number the thesis rests on, because it is the only one that is not contaminated by selection.
- **`welfare_selected`** — entrants only. Reported for completeness and as a warning: it is
  systematically higher than `welfare_common` by exactly the value of the agents B removed, and
  quoting it as a headline is the single easiest way to produce a spurious result in this design.

The difference between them,

$$W_{\text{selection artifact}} \;=\; W_{\text{selected}} - W_{\text{common}} \;\geq\; 0$$

is itself a diagnostic. If it is large, Environment B's apparent advantage is mostly the agents it
declined to serve.

#### As implemented, and three corrections to the statement above

`compute_welfare` returns the three terms separately (`value_created`, `signal_cost`,
`defection_loss`) and their total `welfare_total`; INTERFACES.md §2.2. The accountings are built in
`src/python/comparison.py` from one per-agent payoff vector, $\tau_i V_{i,\mu(i)}$ minus the
agent's proposal spend minus whatever defection cost lands on them, whose sum is `welfare_total`.

**1. $W$ equals the pre-§10.4 welfare only at $\tau \equiv 1$.** §10.7 said the per-match welfare
is unchanged at $c = 0$, $d = 0$. That is true of the column `welfare` (and so of `welfare_alg`,
`welfare_full` and every §7 number), which is still the unweighted $\sum U_{ij} + U_{ji}$. It is not
true of $W$, whose value term is weighted by intent: at Beta(2, 2) intent, `value_created` is 0.516 ×
`welfare` at seed 42 and 0.530 × at seed 43 (oracle arm, epoch 0). The identity that does hold, and
is asserted bit for bit against values recorded before the change, is $W = $ `welfare` when every
$\tau_i = 1$, $c = 0$ and nothing dissolves. With the shipped survival parameters something always
dissolves, so even in A at $c = 0$, `welfare_total` carries a defection term (about 50–60 at
N = 1000).

**2. `welfare_selected` $\geq$ `welfare_common` is not a property of the design.** To be
comparable, `welfare_selected` is reported as the per-entrant mean scaled to $N$ (a raw sum over
fewer agents would be smaller for a trivial reason). The artifact is then
$N(\bar w_{\text{entrants}} - \bar w_{\text{all}})$, which is positive only if entrants do better
than the population under the common rule. Two things make them do worse: $c_B$, which every
entrant pays per proposal and recovers at most once, and thinness, when the filter leaves B too
small to match its own entrants. Measured at seed 42, N = 1000: the artifact is $-1308$ at
size_fraction 0.5, $c_B = 0.25$, and $-804$ at size_fraction 0.25, $c_B \approx 0$, $g_{\min} = 0.8$.
It is also not monotone in $g_{\min}$: at size_fraction 0.5 and $c_B \approx 0$ it runs
11.5, 105.7, 241.8, 188.6, $-106.5$ at $g_{\min}$ = 0, 0.2, 0.4, 0.6, 0.8 — rising while the filter
removes low-alignment agents and falling once B is thin. `tests/python/test_comparison.py` asserts
the identity, the region where the sign holds, and these counterexamples. A negative artifact is
still a warning: it means the entrant-only number *understates* B, and quoting it would be wrong in
the other direction.

**3. The literal rule double-books, so `welfare_common` is an upper bound for B.** An agent B does
not match is scored at their full-A outcome. In full A that agent's partner may be someone B *did*
match, and the rule then counts both: one at a B match, the other at an A match to a partner who is
not available. At seed 42, $c_B = 0.01$, between 9 and 44 such agents per configuration. The
feasible version, `welfare_common_residual`, re-runs A's matching with B's matched agents struck
from every list and scores the agents B left behind there. Both are reported, and the residual one
is the conservative headline.

They usually bracket the comparison but are not guaranteed to. Striking B's matched agents removes
competitors as well as partners, so A's re-run can match a left-behind agent *better* than full A
did. Across the 200 grid cells the residual difference exceeds the literal one in exactly one
($c_B = 0.25$, size_fraction 0.05, seed 45: −8.5 against −13.0), by 4.5 where the two typically
differ by 13 at that size. So the residual figure is the feasible accounting, not a strict lower
bound, and should be called conservative rather than a bound.

An unmatched entrant is scored at their A outcome **minus what they spent proposing in B**. §10.4
does not say this, but leaving it out would make a failed proposal in B free, and failed proposals
are the thin-market penalty the comparison is meant to price.

### 10.5 The thin-market penalty

An agent accepts only candidates whose compatibility exceeds a personal bar $u$. In a pool of $n$
eligible candidates the number acceptable is $\text{Binomial}(n, 1-F(u))$, so

$$P(\text{at least one acceptable partner}) \;=\; 1 - F(u)^{n}$$

and the critical mass at which the pool clears with probability $\rho$ is

$$N^* \;=\; \frac{\ln(1-\rho)}{\ln F(u)}$$

This is why Environment B cannot be declared superior a priori. It has better information per
candidate and fewer candidates, and the two effects oppose each other. **H4 in THESIS.md is the
claim that an interior optimum exists**; the model's job is to test it, and the kill condition is
that the ordering is monotone in $N_{\text{sub}}$, which would mean one of the two effects is not
operating.

Note that $F(u)$ is not free: it is the CDF of the compatibility distribution *conditional on the
entry filter*. A stricter $g_{\min}$ raises mean alignment and therefore raises $F$ at any fixed
bar — B gets denser in quality as it gets thinner in count. The two knobs interact, and a sweep
must vary them jointly rather than one at a time.

**Measured.** `scripts/run_thin_market.py`, N = 1000, five seeds (42–46), 100 pools per size,
epoch 0, k = 20, shipped $g_{\min}$ and $c_B$. $|B|$ is the mean eligible count after filtering and
balancing; the bar $u$ for the acceptability columns is each agent's own θ.

| size_fraction | $\lvert B\rvert$ | clearance | clearance sd (seeds) | $p_{\text{acc}}$ measured | binomial | exact |
|---|---|---|---|---|---|---|
| 0.01 | 7.0 | 0.121 | 0.008 | 0.577 | 0.556 | 0.558 |
| 0.02 | 14.6 | 0.186 | 0.013 | 0.813 | 0.804 | 0.807 |
| 0.04 | 29.9 | 0.284 | 0.009 | 0.966 | 0.963 | 0.965 |
| 0.08 | 60.9 | 0.391 | 0.008 | 0.999 | 0.999 | 0.999 |
| 0.15 | 115.9 | 0.489 | 0.008 | 1.000 | 1.000 | 1.000 |
| 0.25 | 193.9 | 0.547 | 0.007 | 1.000 | 1.000 | 1.000 |
| 0.35 | 272.7 | 0.504 | 0.006 | 1.000 | 1.000 | 1.000 |
| 0.50 | 391.8 | 0.425 | 0.005 | 1.000 | 1.000 | 1.000 |
| 1.00 | 790.0 | 0.303 | 0.017 | 1.000 | 1.000 | 1.000 |

Clearance is not monotone in size: it **rises through 0.5 at $|B| \approx 126$ (size_fraction 0.163),
peaks near 0.55 at $|B| \approx 194$, and falls back through 0.5 at $|B| \approx 278$ (0.357)**. The
seed-to-seed crossings run 0.148–0.174 and 0.341–0.368. Clearance is below 0.5 for $|B| < 126$ and
for $|B| > 278$.

**The two halves of that curve have different causes, and only one is the thin-market penalty.**
The rise is acceptability: below $|B| \approx 60$ the probability of having any acceptable partner
is below 1. The fall is not. $p_{\text{acc}}$ is 1.000 from $|B| \approx 100$ on, and clearance
declines anyway, because the lists are truncated at $k = 20$: as the pool grows, each agent's top 20
is a thinner slice of it, and proposals concentrate on the same few agents. Raising k to 60 at
seed 42 (no filter, negligible cost) moves the peak from size_fraction 0.15 to 0.6 and the peak
value from 0.56 to 0.74; at size_fraction = 1 clearance equals the oracle arm's exactly (0.270 at
k = 20, 0.622 at k = 60). So the interior optimum H4 predicts exists in B's clearance, but its
right-hand slope is set by $k$ and the proposal structure rather than by the two effects H4 names.
That is a result about this model's truncation, not support for H4 as stated, and it should not be
quoted as such.

#### H4 measured on welfare_common

`scripts/run_institutional.py`, N = 1000, seeds 42–46, $g_{\min} = 0.3$, each cell a paired A/B run
of 5 epochs (the matching is static under both arms at these settings, so the epochs differ only in
the defection draw). **A is the oracle arm**: the comparison is then between two institutions with
nothing estimated, and B's advantage is not inflated by the NCF's estimation error. Mean ± sd over
seeds of $W_B - W_A$, with $W_A$ = 973.7 ± 78.5 throughout:

| size_fraction | $\lvert B\rvert$ | literal, $c_B$ = 0.01 | residual, $c_B$ = 0.01 | residual, $c_B$ = 0.05 | residual, $c_B$ = 0.25 |
|---|---|---|---|---|---|
| 0.02 | 15 | +3.9 ± 3.5 | +0.7 ± 5.7 | +0.2 ± 5.8 | −1.7 ± 5.7 |
| 0.05 | 39 | −0.1 ± 4.2 | −10.8 ± 4.6 | −13.9 ± 4.5 | −24.2 ± 9.1 |
| 0.10 | 74 | +13.7 ± 11.4 | −9.4 ± 15.9 | −21.7 ± 16.0 | −69.1 ± 11.2 |
| 0.15 | 115 | +31.7 ± 23.6 | −9.1 ± 25.4 | −40.3 ± 26.7 | −175.7 ± 34.7 |
| 0.25 | 198 | +91.4 ± 42.8 | +24.6 ± 51.0 | −51.1 ± 52.3 | −392.1 ± 52.4 |
| 0.35 | 275 | +101.5 ± 42.5 | +5.4 ± 52.0 | −104.1 ± 55.4 | −612.8 ± 73.5 |
| 0.50 | 389 | +148.6 ± 37.8 | +58.5 ± 36.7 | −96.9 ± 38.8 | −837.1 ± 57.6 |
| 0.65 | 512 | **+166.3 ± 15.9** | **+64.6 ± 9.9** | −140.1 ± 12.3 | −1124.5 ± 28.1 |
| 0.80 | 634 | +137.8 ± 33.4 | +36.7 ± 47.2 | −216.5 ± 46.9 | −1446.9 ± 54.0 |
| 1.00 | 790 | +63.5 ± 29.8 | −4.4 ± 43.9 | −320.0 ± 43.4 | −1861.4 ± 40.2 |

"Literal" is §10.4's rule as written; "residual" re-runs A without B's matched agents (§10.4,
correction 3). The full grid, including $c_B = 0.1$, gini and clearance per cell, is in
`data/processed/institutional/grid_summary.csv`.

**What this says about H4.** At $c_B = 0.01$ the ordering is not monotone in size: B loses or ties
below size_fraction ≈ 0.2, wins in the middle, and the advantage shrinks toward the right edge.
The measured optimum is **size_fraction 0.65 ($|B| \approx 512$)** under both accountings, and it is
the only cell where the residual difference is positive at all five seeds with a spread smaller than
its mean (+64.6 ± 9.9; per seed +54.6 to +78.5). The residual curve's left side is negative at all
five seeds at size_fraction 0.05 (−18 to −6), so a crossing exists. By H4's own kill condition — an
ordering monotone in $N_{\text{sub}}$ — H4 survives at $c_B = 0.01$.

Three qualifications, each of which changes how it may be quoted:

- **It is cost-dependent.** At $c_B \geq 0.05$ the residual difference is negative at every size
  above 0.02 and falls monotonically in size, which is H4's kill condition on that slice: B always
  loses to an oracle A. The cost is charged per proposal, every entrant proposes to up to $k$
  candidates and is matched at most once, so the signalling bill grows linearly in $|B|$ while
  matches grow more slowly. The interior optimum exists only where c_B is small against the value of
  a match.
- **Its right-hand slope is the truncation effect of the clearance table above, not H4's
  thin-market mechanism.** B's clearance over its own entrants peaks at 0.61 at size_fraction 0.25
  and falls to 0.39 at 0.65 and 0.31 at 1.0 as the top-20 lists concentrate, and that is what bends
  the welfare curve down. (Gini over the whole population is not comparable across sizes here: B
  gives every excluded agent zero exposure, so its Gini runs 0.99 at size_fraction 0.02 down to 0.72
  at 1.0, against A's 0.65.) The left-hand slope *is* the
  thin-market penalty. So the optimum is real in this model, but only one of its two sides is caused
  by what H4 names.
- **It depends on which A.** Against an NCF-mediated A (seed 42, epochs 30–39, so the NCF has
  trained), $W_A$ is 234.0 rather than ~1078, and B wins at every size at $c_B = 0.05$: residual
  differences +39.8, +264.5, +347.6 at size_fraction 0.1, 0.5, 1.0, monotone increasing. Against that
  A the ordering is monotone, the other half of H4's kill condition. The oracle A is the comparison
  between institutions; the NCF A folds the recommender's estimation error (H3) into A's side.
  Neither is reported without the other.

**values_min, as a separate axis.** Size_fraction 0.5, $c_B = 0.01$, same seeds
(`values_min_summary.csv`). A stricter filter shrinks B in count and raises its alignment, and the
net effect on welfare_common is monotone down: the residual difference is +91.7 ± 20.4 at
$g_{\min} = 0$, +58.5 ± 36.7 at 0.3, +7.4 ± 25.7 at 0.5 and −11.5 ± 15.6 at 0.7. The selection
artifact over the same axis rises from −31.2 ± 52.4 to +192.5 ± 56.8 at 0.5 and falls back to
+74.8 ± 198.3 at 0.7. That pair is the warning of §10.4 in one line: tightening the filter makes
B's *entrants* look better and makes the *population* worse off. Because $g_{\min}$ moves $|B|$
(500 → 94 across this axis), its effect is partly a size effect, and the size grid above is run at
fixed $g_{\min} = 0.3$ for that reason; how the size optimum moves with $g_{\min}$ is not measured
here.

**Measured against analytic acceptability.** At θ as the bar the two agree within 0.03 everywhere
(worst gap 0.021 at size_fraction 0.01). That check is nearly vacuous: θ is calibrated at the 0.8
quantile, so $F = 0.8$, $N^* = 3.1$, and any pool of a few dozen clears it. To make pool size
matter, `thin_market_curve` takes `bar_quantile`, which replaces θ with each agent's own
$q$-quantile of utility over the filtered reference set. There the formula of this section and the
measurement come apart:

| bar | $N^*$ ($\rho = 0.5$) | worst gap, binomial $1-F^n$ | worst gap, exact |
|---|---|---|---|
| θ | 3.1 | 0.021 (at $\lvert B\rvert$ = 7) | 0.019 (at 7) |
| q = 0.95 | 13.3 | 0.026 (at 7) | 0.025 (at 7) |
| q = 0.99 | 65.6 | **0.071** (at 312) | 0.009 (at 7) |

Two different things are in that table. At $|B| \approx 7$ every bar shows a gap of about 0.02–0.03
under *both* forms: that is the smallest pool, where the formula is evaluated at the mean pool size
while the measurement averages over pools of varying size (and 0.2% of them are empty), so it is
sampling and Jensen's-inequality error, not a distributional one, and it sits close to the 0.03
tolerance. The 0.071 at q = 0.99 is the other thing, and is the one that matters.

The cause of that is **not** an evaluation on the wrong distribution (F is computed per agent over
the agent's own filtered reference set, and the exact form uses the same numbers). It is that
$1-F^n$ is the with-replacement law, and a pool is a subset of a finite reference set of ~230
candidates per side. The exact law is hypergeometric, $1 - \binom{B}{n}/\binom{K}{n}$, which is
exactly 1 at $n = K$ where the binomial gives 0.984 (measured 1.000). The binomial is the large-$K$
limit and is adequate while $n \ll K$. The stated 0.03 criterion therefore holds for the binomial
only while $n/K$ is small, and fails (0.071) at the strict bar. `tests/python/test_environment_b.py`
asserts 0.03 against the exact form and reports the binomial column alongside. $N^*$ itself is
computed from the binomial formula as specified, so at q = 0.99 it is an approximation whose error
grows with $N^*/K$.

`F` at a quantile bar sits slightly below $q$ (0.987 rather than 0.99 at ~230 candidates, since
"strictly below the $q$-quantile" covers slightly less than $q$ of a finite set), which moves $N^*$
from 229 to 181 at $q = 0.99$, $\rho = 0.9$. The test recomputes $F$ independently rather than
assuming it equals $q$.

### 10.6 The factorial decomposition

Median status drifts exogenously, $s_w(t) = s_w(0) + \gamma t$. Under the hypergamous indicator
this mechanically shrinks the eligible male pool with no algorithm involved — a **compositional**
effect that must be separated from the algorithmic one or the project would attribute to the app
what belongs to the labor market.

Four cells, each a full run:

| cell | exogenous drift | algorithmic mediation |
|---|---|---|
| (1) baseline | off | off |
| (2) drift only | on | off |
| (3) NCF only | off | on |
| (4) both | on | on |

$$\Delta W_{\text{exog}} = W_2 - W_1,\quad
\Delta W_{\text{alg}} = W_3 - W_1,\quad
\Delta W_{\text{interaction}} = W_4 - W_3 - W_2 + W_1$$

The interaction term is the only one that supports "the algorithm amplifies the demographic
shift". It is also the one most likely to be near zero, because in the current architecture the two
enter through separate channels — drift moves traits, the NCF moves visibility. Reporting it
whatever its sign is the difference between a measurement and a story. **A decomposition that can
only report amplification has assumed its own conclusion.**

#### Measured

`src/python/factorial.py`, driven by `scripts/run_institutional.py --only factorial`. N = 1000,
seeds 42–46, 40 epochs per cell, $W$ = `welfare_total` averaged over epochs 30–39.
"Drift on" is $\gamma = 0.02$ population status-sd per epoch applied to W's raw status from epoch
1, so by epoch 39 her status has moved 0.78 sd. "Mediation off" is the **oracle** arm, not an
unranked market: same $k$, same θ gate, same deferred acceptance, so the switch changes only
whether the ranking is estimated. The terms are formed per seed and summarised; the four cells at a
seed are asserted to start from one population (the only assertion the decomposition makes).

| | mean | sd | min | max |
|---|---|---|---|---|
| $W_1$ baseline | 974.1 | 78.5 | 907.9 | 1078.2 |
| $W_2$ drift only | 953.0 | 54.0 | 909.0 | 1029.7 |
| $W_3$ NCF only | 256.5 | 20.4 | 234.0 | 289.4 |
| $W_4$ both | 252.1 | 12.1 | 234.9 | 261.7 |
| $\Delta W_{\text{exog}}$ | −21.1 | 25.3 | −48.6 | +5.2 |
| $\Delta W_{\text{alg}}$ | −717.6 | 92.6 | −844.2 | −646.5 |
| $\Delta W_{\text{interaction}}$ | **+16.7** | **40.4** | −17.0 | +74.2 |

**The interaction is indistinguishable from zero.** Its mean is +16.7 with a standard error of 18.1
over five seeds ($t = 0.93$); three of the five seeds are negative and two positive (+74.2, +43.0,
−0.4, −17.0, −16.4 at seeds 42–46). Its magnitude is 2.3% of the algorithmic term. A positive sign
here would mean that drift hurts the NCF market *less* than the oracle market, which is the opposite
of amplification, and the data do not support even that.

This is the result §10.6 said in advance was likely, and THESIS.md H5 was stated so that it would
count as a result: **the interaction may be zero, and in this model it is.** Drift moves traits and
the NCF moves visibility; the two channels are separate by construction, and the measurement agrees
with the construction. The model therefore offers no support for "the algorithm amplifies the
demographic shift." It would be wrong to report the +16.7 as a small amplification or a small
dampening — its sign is not determined by these seeds.

Two caveats on scope. The exogenous term is itself small and of uncertain sign at this $\gamma$
(−21.1 ± 25.3); a larger drift would separate it from zero sooner than it would the interaction,
but it was not tuned to do so. And the algorithmic term is large because the NCF's market clears
far less than the oracle's (mcr ≈ 0.10 against 0.27 over the window), which is H3's
estimation-error concentration seen through welfare.

### 10.7 What §10 does not change

- `utility_toward` is untouched. Intent, values and community never enter it.
- `compute_welfare` gains terms, but the existing `welfare` key is unchanged at every setting. The
  §10.4 total `welfare_total` weights value by intent, so it equals `welfare` only when every
  $\tau_i = 1$, $c = 0$ and nothing dissolves — not at $c = 0, d = 0$ alone, as an earlier version of
  this line said (§10.4, correction 1).
- The four ablation arms of §7.7 keep their definitions and their measured values.
- Every number in §7 and §9 was produced under Environment A and stays valid as such. They are
  A-side results, and the README must say so rather than presenting them as general.

### 10.8 Attrition and intent (H6)

H6 claims high-intent agents leave Environment A at a higher rate. **There is no attrition
mechanism in the model**, so the hypothesis cannot be tested as stated: `turnover_fraction` is a
config field with no implementation, and `run_multigen` was never built. What can be measured is
the direction any given exit rule would see, and the answer depends entirely on which rule.

Measured at N = 1000, three seeds, oracle lists truncated at k = 20, epoch 0. Rank correlations
with $\tau$ (Spearman, because $\tau$ is Beta-shaped and the outcomes are 0/1):

| cost $c$ | proposes | proposal count | matched | pair survival |
|---|---|---|---|---|
| 0.00 | −0.015 | −0.015 | +0.012 | +0.180 |
| 0.25 | +0.010 | +0.051 | +0.012 | +0.180 |
| 1.00 | +0.200 | +0.612 | +0.012 | +0.180 |

Three things follow.

**Matching does not depend on intent.** The $\tau$-matched correlation is +0.012 at every cost,
because $\tau$ never enters `utility_toward`. So an exit rule keyed on "did not match this epoch" is
flat in $\tau$ by construction and kills H6 without needing evidence.

**Pair survival does, but in H6's favour only weakly.** A match between two high-intent agents
survives more often (+0.180), which makes high-intent agents *better off*, not worse — the
opposite of exiting first.

**Cost-burn is the one channel that produces H6's direction.** At $c > 0$ a high-intent agent
proposes far more often (rank correlation +0.612 at $c = 1$) and is matched no more often
(+0.012), because a proposal is paid whether or not it is accepted. Net payoff therefore inverts:

| cost $c$ | corr($\tau$, payoff) | mean payoff | low-$\tau$ quartile | high-$\tau$ quartile |
|---|---|---|---|---|
| 0.00 | +0.074 | +1.0 | +0.4 | +1.7 |
| 0.25 | +0.034 | −3.9 | −4.3 | −3.3 |
| 0.50 | −0.124 | −8.5 | −7.5 | −8.3 |
| 1.00 | −0.540 | −15.1 | −5.6 | −18.3 |
| 2.00 | −0.851 | −16.0 | −0.1 | −34.6 |

So H6 is **conditionally true**: an exit rule keyed on net payoff makes high-intent agents leave
first, at correlations up to −0.85, and the mechanism is not rejection but cost-burn. Under any
other rule it is false or undefined. This is the same cost dominance that kills H4 above
$c_B = 0.05$ (§10.4) — an entrant pays up to $k$ proposals and converts at most once.

**What would make H6 testable properly.** An actual exit rule, and a decision about which signal it
keys on. That decision is not a detail: it flips the hypothesis's sign. A model in which high-intent
agents leave first *because they are rejected* is not supported here; one in which they leave first
*because commitment is expensive and unreciprocated* is, and the two have different policy
implications. Recorded as an open question rather than resolved.

