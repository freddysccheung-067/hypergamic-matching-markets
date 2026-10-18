# Mathematical Formulation

## 1. Agent State Space

Each agent $a_i$ is a column vector:

$$\mathbf{v}_i = [s_i,\ a_i,\ c_i,\ t_i]^\top$$

| Symbol | Meaning | Distribution |
|--------|---------|--------------|
| $s_i$ | Socio-economic status | $\text{LogNormal}(\mu_s, \sigma_s)$ or Pareto($\alpha_s$) |
| $a_i$ | Physical attractiveness score | $\mathcal{N}(\mu_a, \sigma_a)$, clipped to $[0,1]$ |
| $c_i$ | Sociability metric | $\mathcal{N}(\mu_c, \sigma_c)$, clipped to $[0,1]$ |
| $t_i$ | Temporal age | $\mathcal{U}(18, 45)$ at initialization |

The population is partitioned into two disjoint demographic sets $M$ and $W$ with $|M| = |W| = N/2$.

---

## 2. Utility Functions

### 2.1 Hypergamous Utility (agents in $W$)

$$U_w(m) = \alpha \cdot s_m + \beta \cdot a_m - \gamma \cdot d(\mathbf{v}_w, \mathbf{v}_m) + \lambda \cdot \mathbb{1}[s_m \geq s_w + \delta_w]$$

- $\alpha, \beta, \gamma$ — empirical preference weights
- $\delta_w \geq 0$ — status surplus threshold (the hypergamy parameter; $\delta_w = 0$ is neutral)
- $\mathbb{1}[\cdot]$ — indicator function that penalizes/rewards the status gap
- $d(\cdot,\cdot)$ — weighted Euclidean distance in feature space

### 2.2 Standard Utility (agents in $M$)

$$U_m(w) = \alpha' \cdot s_w + \beta' \cdot a_w - \gamma' \cdot d(\mathbf{v}_m, \mathbf{v}_w)$$

The status surplus constraint is absent ($\delta_m = 0$), operationalizing the asymmetry.

---

## 3. Bayesian Expectation Updates

After epoch $t$, each agent updates their internal acceptance threshold $\theta_i^{(t)}$ via:

$$\theta_i^{(t+1)} = \theta_i^{(t)} + \eta \cdot \left( \frac{\text{matches}_i^{(t)}}{\text{proposals}_i^{(t)}} - \hat{p}_i^{(t)} \right)$$

where $\hat{p}_i^{(t)}$ is the agent's prior belief about their match probability and $\eta$ is the learning rate. Agents experiencing persistent rejection lower $\theta_i$ (reduced selectivity); those with high yield raise it.

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
