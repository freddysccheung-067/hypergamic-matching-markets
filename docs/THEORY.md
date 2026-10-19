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

### 2.3 Standard Utility (agents in $M$)

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
