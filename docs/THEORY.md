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
ships $\alpha = 1.0$, $\beta = 2.0$: status stays the dominant linear term, which preserves
the Matthew-effect story, while $\delta$ is no longer structurally dead. This is guarded by
`tests/cpp/test_delta_power.cpp`, which fails if a retune drives the overlap back above 0.95,
and by its inverse, which asserts near-inertness at $\beta \ll \alpha$ and exact inertness
when status is the only signal, so the guardrail cannot pass vacuously.

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

---

## 8. Multi-Generational Pareto Fit

Over $T$ epochs, track the distribution of cumulative match counts per agent. Fit a Pareto distribution:

$$P(X > x) = \left(\frac{x_m}{x}\right)^{\alpha_P}$$

A decreasing $\alpha_P$ across generations indicates heavier tail concentration — fewer agents capturing progressively more matches. This is the macro-level signature of micro-level hypergamy amplified through the NCF loop.
