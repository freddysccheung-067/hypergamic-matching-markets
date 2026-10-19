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

### 2.1 Hypergamous Utility (agents in $W$)

$$U_w(m) = \alpha \cdot s_m + \beta \cdot a_m - \gamma \cdot d(\mathbf{v}_w, \mathbf{v}_m) + \lambda \cdot \mathbb{1}[s_m \geq s_w + \delta_w]$$

- $\alpha, \beta, \gamma$ — empirical preference weights
- $\delta_w \geq 0$ — status surplus threshold (the hypergamy parameter; $\delta_w = 0$ is neutral)
- $\mathbb{1}[\cdot]$ — indicator function that penalizes/rewards the status gap
- $d(\cdot,\cdot)$ — Euclidean distance, expanded below

The distance term must be written out, because an unspecified $d$ hides two modeling decisions
that change the results:

$$d(\mathbf{v}_i, \mathbf{v}_j) = \sqrt{(s_i - s_j)^2 + (a_i - a_j)^2 + (c_i - c_j)^2 + \left(\frac{t_i - t_j}{10}\right)^2}$$

**Age scaling.** Age is measured in years, with a population spread of roughly 20, while the other
three traits live at unit scale. Without the divisor the age gap would dominate $d$ and the
distance term would be almost purely an age penalty. The constant 10 is a modeling choice, not a
derived quantity.

**A collinearity that must be acknowledged.** $s$ enters twice: as the linear reward $\alpha s_m$
and again inside $d$. The model therefore rewards high status *and* penalises status dissimilarity,
so $\gamma$ carries part of the status weight with the opposite sign. At the defaults
$\alpha = 1.0$, $\gamma = 0.3$ the two do not cancel, but any claim of the form "$\alpha$ is the
weight on status" holds only up to this coupling. Two defensible resolutions: standardize the
traits before computing $d$ and drop $s$ from it, or keep $s$ in $d$ and report $\alpha$ and
$\gamma$ jointly rather than interpreting either in isolation.

### 2.2 Standard Utility (agents in $M$)

$$U_m(w) = \alpha' \cdot s_w + \beta' \cdot a_w - \gamma' \cdot d(\mathbf{v}_m, \mathbf{v}_w)$$

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
set the NCF trains on. Because the update target is a *yield* in $[0,1]$ while $\theta_i$ is
compared against utilities on the scale of $\alpha s + \beta a - \gamma d$, $\theta_i$ is
calibrated at $t=0$ as the $(1-\phi)$ quantile of each agent's own utility distribution, with
$\phi$ = `population.admissible_fraction` (default 0.2), so that roughly $\phi N/2$ candidates per
agent are admissible at the start rather than all or none.

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
